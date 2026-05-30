from __future__ import annotations

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)

from .agent import SPECIALIST_DESCRIPTION, SPECIALIST_NAME


def build_agent_card(host: str, port: int, model: str) -> AgentCard:
    """A2A クライアントへ公開する Agent Card を作成します。"""
    return AgentCard(
        name=SPECIALIST_NAME,
        description=SPECIALIST_DESCRIPTION,
        provider=AgentProvider(
            organization="sample_agent_framework",
            url="https://example.com/sample_agent_framework",
        ),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="code_review",
                name="コードレビュー受付",
                description=(
                    "OpenAI SDK で PR 概要、コード差分、設計メモをレビューし、"
                    f"Summary、Key Findings、Test Plan、Follow Up を {model} で返します。"
                ),
                tags=[
                    "code-review",
                    "pull-request",
                    "risk-analysis",
                    "streaming",
                    "follow-up",
                ],
                examples=[
                    "このPRのリスクをレビューして: 認証ミドルウェアを追加し、全APIに適用した",
                    "同じ context_id で続けて: 指摘を踏まえて修正したので再レビューしてください。",
                ],
                input_modes=["text"],
                output_modes=["text", "task-status"],
            ),
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=f"http://{host}:{port}/a2a/jsonrpc",
            )
        ],
    )
