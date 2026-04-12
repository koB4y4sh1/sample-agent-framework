from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent_framework import (
    AgentSession,
    ContextProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
)
from agent_framework.foundry import AnthropicFoundryClient
from pydantic import BaseModel, Field
from utils.print import print_gray

from agent.message_converter import CommonMessageConverter


class UserProfile(BaseModel):
    summary: str | None = None
    goals: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    working_style: list[str] = Field(default_factory=list)
    communication_preferences: list[str] = Field(default_factory=list)
    recurring_topics: list[str] = Field(default_factory=list)


class UserProfileContextProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "user_profile_memory"
    PROFILE_STATE_KEY = "user_profile"
    TURN_COUNT_STATE_KEY = "user_profile_turn_count"
    PROFILE_ROOT_DIR = Path(__file__).parent.parent.parent.parent / ".memory" 

    def __init__(
        self,
        analyser_client: AnthropicFoundryClient,
        model: str,
        *,
        history_source_id: str,
        source_id: str | None = None,
        refresh_interval: int = 3,
        message_converter: CommonMessageConverter | None = None,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._analyser_client = analyser_client
        self._model = model
        self._history_source_id = history_source_id
        self._refresh_interval = refresh_interval
        self._message_converter = message_converter or CommonMessageConverter()
        self.PROFILE_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    def _debug(self, message: str) -> None:
        print_gray(f"[context] {message}")

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        profile = self._get_or_create_profile(state)
        instruction = self._build_profile_instruction(profile)
        if not instruction:
            self._debug("before_run: no profile instruction")
            return

        self._debug("before_run: add profile instruction")
        context.extend_instructions(self.source_id, instruction)

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        profile = self._get_or_create_profile(state)
        turn_count = int(state.get(self.TURN_COUNT_STATE_KEY, 0)) + 1
        state[self.TURN_COUNT_STATE_KEY] = turn_count

        if not self._should_refresh(profile=profile, turn_count=turn_count):
            self._debug(f"after_run: skip refresh (turn_count={turn_count})")
            return

        candidate_messages = context.get_messages(
            sources={self._history_source_id},
            include_input=True,
            include_response=True,
        )
        if not candidate_messages:
            self._debug("after_run: no messages for profile update")
            return

        self._debug(
            "after_run: extract profile "
            f"(messages={len(candidate_messages)}, turn_count={turn_count})"
        )
        extracted_profile = await self._extract_profile(candidate_messages)
        if extracted_profile is None:
            self._debug("after_run: failed to extract profile")
            return

        merged_profile = self._merge_profile(profile, extracted_profile)
        state[self.PROFILE_STATE_KEY] = merged_profile
        self._save_profile(merged_profile)
        self._debug("after_run: profile updated")

    def _get_or_create_profile(self, state: dict[str, Any]) -> UserProfile:
        profile = state.get(self.PROFILE_STATE_KEY)
        if isinstance(profile, UserProfile):
            return profile
        profile = self._load_profile()
        if profile is None:
            profile = self._build_initial_profile()
            self._save_profile(profile)
        state[self.PROFILE_STATE_KEY] = profile
        return profile

    def _build_initial_profile(self) -> UserProfile:
        username = self._get_username()
        if not username:
            return UserProfile()
        return UserProfile(summary=f"User name is {username}")

    def _get_username(self) -> str | None:
        for env_name in ("USERNAME", "USER", "LOGNAME"):
            # 1. .env 環境変数
            value = os.getenv(env_name, "").strip()
            if value:
                return value
            
            # 2. OS 環境変数
            value = os.environ.get(env_name, "").strip()
            if value:
                return value
            
        return "demouser"

    def _get_profile_path(self) -> Path:
        username = self._get_username() or "default"
        safe_username = username.replace("/", "_").replace("\\", "_")
        return self.PROFILE_ROOT_DIR / f"{safe_username}.json"

    def _load_profile(self) -> UserProfile | None:
        path = self._get_profile_path()
        if not path.exists():
            return None
        return UserProfile.model_validate_json(path.read_text(encoding="utf-8"))

    def _save_profile(self, profile: UserProfile) -> None:
        path = self._get_profile_path()
        path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

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
        common_messages = self._message_converter.convert_messages(messages)
        if not common_messages:
            return None

        analysis_messages = [
            Message(
                "system",
                [
                "Extract a structured user profile from the conversation. ",
                "Only include facts supported by the messages. ",
                "Return only JSON with keys summary, goals, preferences, constraints, ",
                "working_style, communication_preferences, recurring_topics.",
                ],
            ),
            *common_messages,
            Message(
                "user",
                [
                    "Extract a user profile from the conversation history.",
                    "Return only the JSON object that matches the response schema.",
                ],
            ),
        ]

        response = await self._analyser_client.get_response(analysis_messages, options={"max_tokens": 800,"response_format": UserProfile})

        return response.value

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
            lines.append(f"Summary: {profile.summary}")
        if profile.goals:
            lines.append(f"Goals: {self._format_items(profile.goals)}")
        if profile.preferences:
            lines.append(f"Preferences: {self._format_items(profile.preferences)}")
        if profile.constraints:
            lines.append(f"Constraints: {self._format_items(profile.constraints)}")
        if profile.working_style:
            lines.append(f"Working style: {self._format_items(profile.working_style)}")
        if profile.communication_preferences:
            lines.append(
                f"Communication preferences: {self._format_items(profile.communication_preferences)}"
            )
        if profile.recurring_topics:
            lines.append(f"Recurring topics: {self._format_items(profile.recurring_topics)}")

        if not lines:
            return None

        return "\n".join(
            [
                "Use the following user profile as supplemental context.",
                "Prefer newer confirmed information when there is a conflict.",
                *lines,
            ]
        )

    def _format_items(self, items: list[str]) -> str:
        return " / ".join(items)
