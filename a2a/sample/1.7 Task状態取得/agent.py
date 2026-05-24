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


class GetTaskAgentExecutor(AgentExecutor):
    """get_task で状態取得しやすいよう、数秒間 working のままにするエージェントです。"""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Task を開始し、途中状態を保存しながら最後に完了します。"""
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message] if context.message else [],
            )
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            message=updater.new_agent_message(parts=[Part(text="処理中です")]),
        )
        await asyncio.sleep(3)

        await updater.add_artifact(
            parts=[Part(text="Task 状態取得サンプルが完了しました。")],
            name="get-task-result",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """キャンセル要求を受けた場合は canceled にします。"""
        await TaskUpdater(event_queue, context.task_id or "", context.context_id or "").cancel()


def build_agent_card(host: str, port: int) -> AgentCard:
    """get_task を試すための Agent Card を作成します。"""
    return AgentCard(
        name="Get Task Sample Agent",
        description="A2A の get_task による Task 状態取得を確認するサンプルです。",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="get_task_sample",
                name="Get Task Sample",
                description="処理中の Task を get_task で取得できます。",
                tags=["sample", "get-task"],
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


async def serve(host: str = "127.0.0.1", port: int = 41247) -> None:
    """Task 状態取得サンプル用の A2A サーバーを起動します。"""
    agent_card = build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=GetTaskAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=agent_card))
    app.routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc"))
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get task sample")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41247)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args.host, args.port))
