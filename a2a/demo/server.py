from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI

from .agent import CodeReviewAgent, CodeReviewAgentConfig
from .card import build_agent_card
from .executor import CodeReviewExecutor

load_dotenv()
logger = logging.getLogger(__name__)


async def serve(*, host: str, port: int, model: str) -> None:
    """FastAPI 上に A2A JSON-RPC サーバを起動します。"""
    agent_card = build_agent_card(host, port, model)
    ai_agent = CodeReviewAgent(CodeReviewAgentConfig(model=model))
    request_handler = DefaultRequestHandler(
        agent_executor=CodeReviewExecutor(ai_agent),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=agent_card))
    app.routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc"))

    logger.info("A2A agent card: http://%s:%s/.well-known/agent-card.json", host, port)
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port)).serve()


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読み取ります。"""
    parser = argparse.ArgumentParser(
        description="OpenAI SDK で動くコードレビュー受付 A2A サーバを公開します。"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41250)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-nano"))
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(host=args.host, port=args.port, model=args.model))
