import argparse
import asyncio
import contextlib
import logging

import grpc
import uvicorn
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler, GrpcHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
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
    a2a_pb2_grpc,
)
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class SampleAgentExecutor(AgentExecutor):
    """a2a-js のサンプルに近いサンプルエージェントの実行ロジックです。"""

    def __init__(self) -> None:
        # 実行中のタスク ID を覚えて、キャンセル判定に使います。
        self.running_tasks: set[str] = set()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """タスクをキャンセルします。"""
        task_id = context.task_id
        if task_id in self.running_tasks:
            self.running_tasks.remove(task_id)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """タスクをその場で実行します。"""
        # ユーザー入力とタスク情報を取り出します。
        user_message = context.message
        task_id = context.task_id
        context_id = context.context_id

        if not user_message or not task_id or not context_id:
            return

        # このタスクを実行中として記録します。
        self.running_tasks.add(task_id)

        logger.info(
            "[SampleAgentExecutor] メッセージ %s をタスク %s 用に処理中 (context: %s)",
            user_message.message_id,
            task_id,
            context_id,
        )

        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[user_message],
            )
        )

        # タスクの進行状況を更新するためのヘルパーです。
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        # 処理中に表示するメッセージを送ります。
        working_message = updater.new_agent_message(
            parts=[Part(text="Processing your question...")]
        )
        await updater.start_work(message=working_message)

        # ユーザーの入力内容を見て、返答文を決めます。
        query = context.get_user_input()

        agent_reply_text = self._parse_input(query)
        await asyncio.sleep(1)

        # 途中でキャンセルされた場合は、ここで終了します。
        if task_id not in self.running_tasks:
            return

        # 最終的な返答を成果物として返します。
        await updater.add_artifact(
            parts=[Part(text=agent_reply_text)],
            name="response",
            last_chunk=True,
        )
        await updater.complete()

        logger.info(
            "[SampleAgentExecutor] タスク %s は state: completed で終了しました",
            task_id,
        )

    def _parse_input(self, query: str) -> str:
        # 空入力なら、まずメッセージを促します。
        if not query:
            return "こんにちは。返答するためのメッセージを入力してください。"

        ql = query.lower()
        # あいさつ系の入力には、あいさつで返します。
        if "hello" in ql or "hi" in ql:
            return "Hello World! こんにちは。ご質問はありますか？"
        # 体調を聞かれたら、そのまま会話を続けます。
        if "how are you" in ql:
            return "元気です。お気遣いありがとうございます。今日は何をお手伝いしましょうか。"
        # 別れのあいさつには、丁寧に別れを返します。
        if "goodbye" in ql or "bye" in ql:
            return "さようなら。すてきな一日をお過ごしください。"
        # それ以外は、受け取った文をそのまま使って返します。
        return f"こんにちは。あなたは '{query}' と言いました。メッセージありがとうございます。"


async def serve(
    host: str = "127.0.0.1",
    port: int = 41241,
    grpc_port: int = 50051,
) -> None:
    """JSON-RPC、HTTP+JSON、gRPC の各トランスポートを組み込んだ Sample Agent サーバーを起動します。"""
    # Agent Card には、このエージェントの説明と接続方法を入れます。
    agent_card = AgentCard(
        name="Sample Agent",
        description="ストリーミング機能を試すためのサンプルエージェントです。",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="sample_agent",
                name="Sample Agent",
                description="あいさつを返します。",
                tags=["sample"],
                examples=["hi"],
                input_modes=["text"],
                output_modes=["text", "task-status"],
            )
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="GRPC",
                protocol_version="1.0",
                url=f"{host}:{grpc_port}",
            ),
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=f"http://{host}:{port}/a2a/jsonrpc",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
                url=f"http://{host}:{port}/a2a/rest",
            ),
        ],
    )

    # タスクの状態をメモリ上で管理します。
    task_store = InMemoryTaskStore()
    # 受け取ったリクエストを実際のエージェント処理につなぎます。
    request_handler = DefaultRequestHandler(
        agent_executor=SampleAgentExecutor(),
        task_store=task_store,
        agent_card=agent_card,
    )

    # REST 用のルートを作成します。
    rest_routes = create_rest_routes(
        request_handler=request_handler,
        path_prefix="/a2a/rest",
    )
    # JSON-RPC 用のルートを作成します。
    jsonrpc_routes = create_jsonrpc_routes(
        request_handler=request_handler,
        rpc_url="/a2a/jsonrpc",
    )
    # Agent Card を公開するルートを作成します。
    agent_card_routes = create_agent_card_routes(
        agent_card=agent_card,
    )
    # FastAPI アプリに各ルートを登録します。
    app = FastAPI()
    app.routes.extend(jsonrpc_routes)
    app.routes.extend(agent_card_routes)
    app.routes.extend(rest_routes)

    # 通常の gRPC サーバーを起動します。
    grpc_server = grpc.aio.server()
    grpc_server.add_insecure_port(f"{host}:{grpc_port}")
    servicer = GrpcHandler(request_handler)
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(servicer, grpc_server)

    # Uvicorn で HTTP サーバーを動かします。
    config = uvicorn.Config(app, host=host, port=port)
    uvicorn_server = uvicorn.Server(config)

    logger.info("Sample Agent サーバーを起動します:")
    logger.info(" - HTTP サーバー: http://%s:%s", host, port)
    logger.info(" - gRPC サーバー: %s:%s", host, grpc_port)
    logger.info(
        "Agent Card は http://%s:%s/.well-known/agent-card.json で利用できます",
        host,
        port,
    )

    await asyncio.gather(
        grpc_server.start(),
        uvicorn_server.serve(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Sample A2A agent server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41241)
    parser.add_argument("--grpc-port", type=int, default=50051)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(
            serve(
                host=args.host,
                port=args.port,
                grpc_port=args.grpc_port,
            )
        )
