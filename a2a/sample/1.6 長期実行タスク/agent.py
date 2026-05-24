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


class LongRunningTaskAgentExecutor(AgentExecutor):
    """時間のかかる処理を status_update で見せるサンプルエージェントです。"""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """3 秒かかる処理として、進捗メッセージを 3 回送ります。"""
        # 長期実行でも、最初に Task を送ってから状態更新を流します。
        await event_queue.enqueue_event(
            Task(
                id=context.task_id or "",
                context_id=context.context_id or "",
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message] if context.message else [],
            )
        )
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")

        for step in range(1, 4):
            # status_update は「まだ処理中」という状態をクライアントへ伝えます。
            message = updater.new_agent_message(parts=[Part(text=f"処理中 {step}/3")])
            await updater.update_status(TaskState.TASK_STATE_WORKING, message=message)
            await asyncio.sleep(1)

        await updater.add_artifact(
            parts=[Part(text="長期実行タスクが完了しました。")],
            name="long-running-result",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """このサンプルではキャンセル要求を canceled 状態として返します。"""
        await TaskUpdater(event_queue, context.task_id or "", context.context_id or "").cancel()


def build_agent_card(host: str, port: int) -> AgentCard:
    """長期実行タスクを扱うエージェントとして Agent Card を作ります。"""
    return AgentCard(
        name="Long Running Task Sample Agent",
        description="A2A の長期実行タスクと status_update を確認するサンプルです。",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="long_running",
                name="Long Running",
                description="処理中ステータスを出しながら最後に結果を返します。",
                tags=["sample", "long-running"],
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


async def serve(host: str = "127.0.0.1", port: int = 41246) -> None:
    """長期実行タスク用の A2A サーバーを起動します。"""
    agent_card = build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=LongRunningTaskAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=agent_card))
    app.routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc"))
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long running task sample")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41246)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args.host, args.port))
