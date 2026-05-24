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


class ErrorHandlingAgentExecutor(AgentExecutor):
    """reject と failed の違いを確認するためのエージェントです。"""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """入力に応じて completed / rejected / failed のいずれかを返します。"""
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
        text = context.get_user_input().strip().lower()

        if text == "reject":
            # reject は「入力や条件が合わず、処理を受け付けない」場合に使います。
            message = updater.new_agent_message(parts=[Part(text="この入力は受け付けません。")])
            await updater.reject(message=message)
            return

        if text == "fail":
            # failed は「受け付けた後、処理中に失敗した」場合に使います。
            message = updater.new_agent_message(parts=[Part(text="処理中に失敗しました。")])
            await updater.failed(message=message)
            return

        await updater.add_artifact(
            parts=[Part(text="正常に完了しました。reject または fail も試せます。")],
            name="error-handling-result",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """キャンセル要求を受けた場合は canceled にします。"""
        await TaskUpdater(event_queue, context.task_id or "", context.context_id or "").cancel()


def build_agent_card(host: str, port: int) -> AgentCard:
    """エラー処理サンプル用の Agent Card を作ります。"""
    return AgentCard(
        name="Error Handling Sample Agent",
        description="A2A の rejected / failed / completed を確認するサンプルです。",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="error_handling",
                name="Error Handling",
                description="入力が reject なら rejected、fail なら failed を返します。",
                tags=["sample", "error"],
                examples=["ok", "reject", "fail"],
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


async def serve(host: str = "127.0.0.1", port: int = 41250) -> None:
    """エラー処理サンプル用の A2A サーバーを起動します。"""
    agent_card = build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=ErrorHandlingAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=agent_card))
    app.routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc"))
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error handling sample")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41250)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args.host, args.port))
