import argparse
import asyncio
import contextlib
from collections import defaultdict

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


class MultiTurnAgentExecutor(AgentExecutor):
    """同じ context_id で会話を継続する例を示すエージェントです。"""

    def __init__(self) -> None:
        self.turn_count_by_context: defaultdict[str, int] = defaultdict(int)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """context_id ごとに turn 数を数え、返答に含めます。"""
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

        self.turn_count_by_context[context_id] += 1
        turn = self.turn_count_by_context[context_id]
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.add_artifact(
            parts=[Part(text=f"context_id={context_id} の {turn} turn 目です。")],
            name="multi-turn-result",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """キャンセル要求を受けた場合は canceled にします。"""
        await TaskUpdater(event_queue, context.task_id or "", context.context_id or "").cancel()


def build_agent_card(host: str, port: int) -> AgentCard:
    """複数 turn の context 継続を説明する Agent Card を作ります。"""
    return AgentCard(
        name="Multi Turn Context Sample Agent",
        description="A2A の context_id による複数 turn 継続を確認するサンプルです。",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="multi_turn",
                name="Multi Turn",
                description="同じ context_id の turn 数を数えます。",
                tags=["sample", "context"],
                examples=["first", "second"],
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


async def serve(host: str = "127.0.0.1", port: int = 41249) -> None:
    """複数 turn / context 継続サンプル用の A2A サーバーを起動します。"""
    agent_card = build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=MultiTurnAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=agent_card))
    app.routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc"))
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi turn context sample")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41249)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args.host, args.port))
