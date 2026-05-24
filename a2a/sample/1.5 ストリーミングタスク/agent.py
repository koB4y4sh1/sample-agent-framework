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


class StreamingTaskAgentExecutor(AgentExecutor):
    """artifact を複数チャンクに分けて返すストリーミング用エージェントです。"""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """短い文を 1 語ずつ artifact_update として送信します。"""
        # ストリーミングでも、最初に Task 本体を通知します。
        await event_queue.enqueue_event(
            Task(
                id=context.task_id or "",
                context_id=context.context_id or "",
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message] if context.message else [],
            )
        )
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        await updater.start_work()

        words = ["A2A", "streaming", "sample", "completed"]
        for index, word in enumerate(words):
            # append=True は同じ artifact に続きのチャンクを追加する意味です。
            await updater.add_artifact(
                parts=[Part(text=word + " ")],
                artifact_id="stream-result",
                name="stream-result",
                append=index > 0,
                last_chunk=index == len(words) - 1,
            )
            await asyncio.sleep(0.5)

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """キャンセル要求を受けたら canceled を返します。"""
        await TaskUpdater(event_queue, context.task_id or "", context.context_id or "").cancel()


def build_agent_card(host: str, port: int) -> AgentCard:
    """streaming=True として Agent Card に公開します。"""
    return AgentCard(
        name="Streaming Task Sample Agent",
        description="A2A のストリーミングタスクを確認するサンプルです。",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="stream_words",
                name="Stream Words",
                description="結果を複数の artifact chunk として返します。",
                tags=["sample", "streaming"],
                examples=["start"],
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


async def serve(host: str = "127.0.0.1", port: int = 41245) -> None:
    """ストリーミングタスク用の A2A サーバーを起動します。"""
    agent_card = build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=StreamingTaskAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=agent_card))
    app.routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc"))
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming task sample")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41245)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args.host, args.port))
