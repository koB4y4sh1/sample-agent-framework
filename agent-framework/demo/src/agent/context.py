from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from agent_framework import (
    AgentSession,
    ContextProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
)
from message_converter import AnthropicMessageConverter
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """会話から継続的に抽出・更新するユーザープロファイル。"""

    summary: str | None = None
    goals: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    working_style: list[str] = Field(default_factory=list)
    communication_preferences: list[str] = Field(default_factory=list)
    recurring_topics: list[str] = Field(default_factory=list)


class CustomContextProvider(ContextProvider):
    """ユーザーの特徴・志向・制約を定期抽出して保持する ContextProvider。

    この Provider は単発の属性ではなく、会話を通じて見えてくる
    ユーザーの目的、好み、制約、仕事の進め方、伝達スタイルを保持する。

    動作:
    1. `before_run`
       既に抽出済みのプロファイルを instructions として注入する。
       これにより、エージェントは直近の入力だけでなく、継続的なユーザー特性を踏まえて応答できる。
    2. `after_run`
       一定ターンごとに履歴を見直し、プロファイルを再抽出・更新する。
       毎ターン抽出するとコストが高いため、`refresh_interval` で頻度を制御する。
    """

    DEFAULT_SOURCE_ID = "user_profile_memory"
    PROFILE_STATE_KEY = "user_profile"
    TURN_COUNT_STATE_KEY = "user_profile_turn_count"

    def __init__(
        self,
        anthropic_client: Any,
        model: str,
        *,
        history_source_id: str,
        source_id: str | None = None,
        refresh_interval: int = 3,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._anthropic_client = anthropic_client
        self._model = model
        self._history_source_id = history_source_id
        self._refresh_interval = refresh_interval
        self._converter = AnthropicMessageConverter()

    def _debug(self, message: str) -> None:
        print(f"[context] {message}")


    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """抽出済みプロファイルを instructions として注入する。"""
        profile = self._get_or_create_profile(state)
        instruction = self._build_profile_instruction(profile)
        if not instruction:
            self._debug("before_run: 注入するユーザープロファイルなし")
            return

        self._debug("before_run: ユーザープロファイルを instructions に注入")
        context.extend_instructions(self.source_id, instruction)


    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """一定ターンごとに履歴からユーザープロファイルを再抽出する。"""
        profile = self._get_or_create_profile(state)
        turn_count = int(state.get(self.TURN_COUNT_STATE_KEY, 0)) + 1
        state[self.TURN_COUNT_STATE_KEY] = turn_count

        if not self._should_refresh(profile=profile, turn_count=turn_count):
            self._debug(f"after_run: 抽出をスキップ (turn_count={turn_count})")
            return

        candidate_messages = context.get_messages(
            sources={self._history_source_id},
            include_input=True,
            include_response=True,
        )
        if not candidate_messages:
            self._debug("after_run: 抽出対象メッセージなし")
            return

        self._debug(
            "after_run: ユーザープロファイルを再抽出 "
            f"(messages={len(candidate_messages)}, turn_count={turn_count})"
        )
        extracted_profile = await self._extract_profile(candidate_messages)
        if extracted_profile is None:
            self._debug("after_run: 抽出結果の解析に失敗")
            return

        merged_profile = self._merge_profile(profile, extracted_profile)
        state[self.PROFILE_STATE_KEY] = merged_profile
        self._debug("after_run: ユーザープロファイルを更新")

    def _get_or_create_profile(self, state: dict[str, Any]) -> UserProfile:
        profile = state.get(self.PROFILE_STATE_KEY)
        if isinstance(profile, UserProfile):
            return profile
        profile = UserProfile()
        state[self.PROFILE_STATE_KEY] = profile
        return profile

    def _should_refresh(self, *, profile: UserProfile, turn_count: int) -> bool:
        if self._is_profile_empty(profile):
            return True
        return turn_count % self._refresh_interval == 0

    def _is_profile_empty(self, profile: UserProfile) -> bool:
        return not any(
            [
                profile.summary,
                profile.goals,
                profile.preferences,
                profile.constraints,
                profile.working_style,
                profile.communication_preferences,
                profile.recurring_topics,
            ]
        )

    async def _extract_profile(self, messages: list[Message]) -> UserProfile | None:
        analysis_messages = [
            *messages,
            Message(
                "user",
                [
                    "上記の会話履歴からユーザープロファイルを抽出してください。"
                    "回答は JSON オブジェクトのみを返してください。"
                ],
            ),
        ]

        response = await self._anthropic_client.messages.create(
            model=self._model,
            max_tokens=800,
            temperature=0,
            system=(
                "あなたは会話履歴からユーザープロファイルを抽出する分析器です。"
                "推測を広げすぎず、会話中に十分な根拠がある内容だけを抽出してください。"
                "回答は JSON オブジェクトのみを返してください。説明文、前置き、Markdown は禁止です。"
                'JSON のキーは "summary", "goals", "preferences", "constraints", '
                '"working_style", "communication_preferences", "recurring_topics" のみです。'
                "summary は文字列または null、他は文字列配列です。"
            ),
            messages=self._converter.convert_messages(analysis_messages),
            extra_headers={"anthropic-beta": "output-128k-2025-02-19"},
        )

        text = self._extract_text_response(response)
        json_text = self._extract_json_block(text)
        if not json_text:
            return None

        with suppress(Exception):
            payload = json.loads(json_text)
            return UserProfile.model_validate(payload)
        return None

    def _extract_text_response(self, response: Any) -> str:
        text_parts: list[str] = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_parts.append(block.text)
        return "\n".join(text_parts).strip()

    def _extract_json_block(self, text: str) -> str | None:
        if not text:
            return None
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        if "```json" in stripped:
            start = stripped.find("```json")
            if start >= 0:
                start += len("```json")
                end = stripped.find("```", start)
                if end >= 0:
                    return stripped[start:end].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return stripped[start : end + 1]
        return None

    def _merge_profile(self, current: UserProfile, extracted: UserProfile) -> UserProfile:
        return UserProfile(
            summary=extracted.summary or current.summary,
            goals=self._merge_unique(current.goals, extracted.goals),
            preferences=self._merge_unique(current.preferences, extracted.preferences),
            constraints=self._merge_unique(current.constraints, extracted.constraints),
            working_style=self._merge_unique(current.working_style, extracted.working_style),
            communication_preferences=self._merge_unique(
                current.communication_preferences,
                extracted.communication_preferences,
            ),
            recurring_topics=self._merge_unique(current.recurring_topics, extracted.recurring_topics),
        )

    def _merge_unique(self, existing: list[str], incoming: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for item in [*existing, *incoming]:
            normalized = item.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
        return merged

    def _build_profile_instruction(self, profile: UserProfile) -> str | None:
        lines: list[str] = []
        if profile.summary:
            lines.append(f"要約: {profile.summary}")
        if profile.goals:
            lines.append(f"目的: {self._format_items(profile.goals)}")
        if profile.preferences:
            lines.append(f"嗜好: {self._format_items(profile.preferences)}")
        if profile.constraints:
            lines.append(f"制約: {self._format_items(profile.constraints)}")
        if profile.working_style:
            lines.append(f"進め方の傾向: {self._format_items(profile.working_style)}")
        if profile.communication_preferences:
            lines.append(f"伝達の好み: {self._format_items(profile.communication_preferences)}")
        if profile.recurring_topics:
            lines.append(f"繰り返し出る話題: {self._format_items(profile.recurring_topics)}")

        if not lines:
            return None

        return "\n".join(
            [
                "以下は過去の会話から抽出したユーザープロファイルです。",
                "明示的な最新の指示がある場合はそちらを優先してください。",
                *lines,
            ]
        )

    def _format_items(self, items: list[str]) -> str:
        return " / ".join(items)
