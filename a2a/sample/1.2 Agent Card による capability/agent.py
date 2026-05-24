import argparse
import asyncio
import contextlib

import uvicorn
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from fastapi import FastAPI


class CapabilityAgentExecutor(AgentExecutor):
    """Agent Card の capability を確認するための最小エージェントです。"""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """受信したタスクに短い固定文を返します。"""
        # A2A では最初に Task 本体を通知し、その後に状態や成果物を更新します。
        await event_queue.enqueue_event(
            Task(
                id=context.task_id or "",
                context_id=context.context_id or "",
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message] if context.message else [],
            )
        )
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        await updater.add_artifact(
            parts=[Part(text="このエージェントは Agent Card で機能を公開しています。")],
            name="capability-result",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """キャンセル要求を受けたら、タスク状態を canceled にします。"""
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        await updater.cancel()


def build_agent_card(host: str, port: int) -> AgentCard:
    """Agent Card に、このエージェントの能力と接続先を定義します。"""
    return AgentCard(
        name="Capability Sample Agent",
        description="Agent Card の capabilities と skills を確認するサンプルです。",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        # capabilities はクライアントが機能対応を判断するための公開情報です。
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="echo",
                name="Echo",
                description="入力を受け取り、短い説明文を返します。",
                tags=["sample", "capability"],
                examples=["hello"],
                input_modes=["text"],
                output_modes=["text"],
            )
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=f"http://{host}:{port}/a2a/jsonrpc",
            )
        ],
    )


async def serve(host: str = "127.0.0.1", port: int = 41242) -> None:
    """Agent Card と JSON-RPC エンドポイントを公開します。"""
    agent_card = build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=CapabilityAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=agent_card))
    app.routes.extend(
        create_jsonrpc_routes(request_handler=request_handler, rpc_url="/a2a/jsonrpc")
    )

    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Card capability sample")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41242)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args.host, args.port))
