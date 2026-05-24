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


class TaskSendAgentExecutor(AgentExecutor):
    """クライアントから送られた Task を処理する最小実装です。"""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Task を submitted -> completed に進め、結果を artifact として返します。"""
        # 最初のイベントとして Task を送ると、クライアントは task_id を把握できます。
        await event_queue.enqueue_event(
            Task(
                id=context.task_id or "",
                context_id=context.context_id or "",
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message] if context.message else [],
            )
        )
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")

        user_text = context.get_user_input()
        await updater.add_artifact(
            parts=[Part(text=f"受信したタスク入力: {user_text}")],
            name="task-result",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """このサンプルではキャンセル要求をそのまま状態に反映します。"""
        await TaskUpdater(event_queue, context.task_id or "", context.context_id or "").cancel()


def build_agent_card(host: str, port: int) -> AgentCard:
    """Task を受け取れるエージェントとして Agent Card を作ります。"""
    return AgentCard(
        name="Task Send Sample Agent",
        description="A2A の Task 送信を確認するサンプルです。",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="receive_task",
                name="Receive Task",
                description="送信されたテキストを artifact として返します。",
                tags=["sample", "task"],
                examples=["ping"],
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


async def serve(host: str = "127.0.0.1", port: int = 41243) -> None:
    """JSON-RPC で Task を受け付ける A2A サーバーを起動します。"""
    agent_card = build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=TaskSendAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=agent_card))
    app.routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc"))
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task send sample")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41243)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args.host, args.port))
