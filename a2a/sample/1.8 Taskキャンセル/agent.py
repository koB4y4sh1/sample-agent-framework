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


class CancelTaskAgentExecutor(AgentExecutor):
    """cancel_task の動きを確認するため、途中で止められるエージェントです。"""

    def __init__(self) -> None:
        self.running_task_ids: set[str] = set()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """5 秒かかる処理を開始し、キャンセルされたら途中終了します。"""
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        self.running_task_ids.add(task_id)
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message] if context.message else [],
            )
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        for step in range(1, 6):
            if task_id not in self.running_task_ids:
                return
            message = updater.new_agent_message(parts=[Part(text=f"処理中 {step}/5")])
            await updater.update_status(TaskState.TASK_STATE_WORKING, message=message)
            await asyncio.sleep(1)

        self.running_task_ids.discard(task_id)
        await updater.add_artifact(
            parts=[Part(text="キャンセルされずに完了しました。")],
            name="cancel-result",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """cancel_task を受けたら実行中リストから外し、canceled を通知します。"""
        task_id = context.task_id or ""
        self.running_task_ids.discard(task_id)
        await TaskUpdater(event_queue, task_id, context.context_id or "").cancel()


def build_agent_card(host: str, port: int) -> AgentCard:
    """キャンセル可能な長期実行 Task として公開します。"""
    return AgentCard(
        name="Cancel Task Sample Agent",
        description="A2A の cancel_task を確認するサンプルです。",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="cancel_task_sample",
                name="Cancel Task Sample",
                description="実行中 Task を cancel_task でキャンセルできます。",
                tags=["sample", "cancel"],
                examples=["run"],
                input_modes=["text"],
                output_modes=["text", "task-status"],
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


async def serve(host: str = "127.0.0.1", port: int = 41248) -> None:
    """Task キャンセルサンプル用の A2A サーバーを起動します。"""
    agent_card = build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=CancelTaskAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=agent_card))
    app.routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc"))
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancel task sample")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41248)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args.host, args.port))
