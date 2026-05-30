from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

AZURE_OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"

# Azure CLI でログイン済みの資格情報を使い回します。
AZURE_CLI_CREDENTIAL = AzureCliCredential(process_timeout=30)


def _get_azure_openai_base_url() -> str:
    """Azure OpenAI の openai/v1 エンドポイントを返します。"""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
    return f"{endpoint}/openai/v1"


async def _get_azure_openai_access_token() -> str:
    """AsyncOpenAI が使う Azure AD アクセストークンを取得します。"""
    access_token = await AZURE_CLI_CREDENTIAL.get_token(AZURE_OPENAI_SCOPE)
    return access_token.token


SPECIALIST_NAME = "Code Review Desk A2A Agent"
SPECIALIST_DESCRIPTION = (
    "PR 概要、コード差分、設計メモ、障害報告を受け取り、"
    "リスク、レビュー観点、修正案、検証手順を返すコードレビュー受付エージェントです。"
)
SPECIALIST_INSTRUCTIONS = """あなたは A2A 経由で公開されるコードレビュー受付センターです。

専門性:
- PR 概要、コード差分、設計メモ、障害報告をレビューする
- 実装リスク、設計リスク、セキュリティリスク、運用リスクを指摘する
- 不足しているテスト観点と検証手順を提示する
- レビュー結果を、実務で使える指摘として簡潔にまとめる

回答ルール:
- コードレビュー、設計レビュー、PR リスク分析の範囲だけ回答する
- 事実と仮定を分ける
- 重要度の高い指摘から順に書く
- 不明な前提がある場合は、不明点を明示する
- 出力は「結論」「主な指摘」「検証手順」の順にする
"""

REVIEW_ARTIFACTS = (
    ("summary", "Summary"),
    ("key-findings", "Key Findings"),
    ("test-plan", "Test Plan"),
    ("follow-up", "Follow Up"),
)


def build_review_instructions(*, has_review_history: bool) -> str:
    """モデルに、artifact へ分解しやすい一定の出力形式を指示します。"""
    section_lines = "\n".join(f"## {title}" for _, title in REVIEW_ARTIFACTS)
    follow_up_rules = ""
    if has_review_history:
        follow_up_rules = """

再レビュー時の追加ルール:
- 前回レビュー内容を踏まえて比較する
- Key Findings では「Resolved:」「Remaining:」「New:」の接頭辞を使って整理する
- Follow Up には未解消の前提確認や、次に見るべき点だけを書く
"""

    return f"""{SPECIALIST_INSTRUCTIONS}

出力形式:
- 必ず次の見出しをこの順で出力する
{section_lines}
- 各見出しの本文は簡潔な箇条書きにする
- Summary は全体判断を短くまとめる
- Key Findings は重要度の高い指摘から並べる
- Test Plan は追加で確認したい検証項目を書く
- Follow Up は確認したい前提や次の依頼候補を書く
- 見出しは省略しない
{follow_up_rules}
"""


@dataclass(slots=True)
class CodeReviewAgentConfig:
    """コードレビュー用 LLM を実行するための設定です。"""

    model: str


class CodeReviewAgent:
    """コードレビュー用 LLM を呼び出すクラスです。"""

    def __init__(self, config: CodeReviewAgentConfig) -> None:
        self._config = config
        # Azure OpenAI の openai/v1 を AsyncOpenAI 互換で呼び出します。
        self._client = AsyncOpenAI(
            api_key=_get_azure_openai_access_token,
            base_url=_get_azure_openai_base_url(),
        )
        # context_id ごとに直前の response_id を覚えて、会話を継続します。
        self._previous_response_ids: dict[str, str] = {}

    def has_review_history(self, session_key: str) -> bool:
        """同じ context_id で前回レビューがあるかを返します。"""
        return session_key in self._previous_response_ids

    async def stream(self, *, user_input: str, session_key: str) -> AsyncIterator[str]:
        """Responses API のテキスト差分を順に返します。"""
        previous_response_id = self._previous_response_ids.get(session_key)
        stream = await self._client.responses.create(
            model=self._config.model,
            instructions=build_review_instructions(
                has_review_history=previous_response_id is not None
            ),
            input=user_input,
            previous_response_id=previous_response_id,
            stream=True,
        )

        async for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
            elif event.type == "response.completed":
                self._previous_response_ids[session_key] = event.response.id

    async def run(self, *, user_input: str, session_key: str) -> str:
        """レビュー依頼を OpenAI Responses API に渡し、レビュー結果を返します。"""
        chunks: list[str] = []
        async for chunk in self.stream(user_input=user_input, session_key=session_key):
            chunks.append(chunk)
        return "".join(chunks)
