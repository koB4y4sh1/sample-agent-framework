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


class UserProfile(BaseModel):
    summary: str | None = None
    goals: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    working_style: list[str] = Field(default_factory=list)
    communication_preferences: list[str] = Field(default_factory=list)
    recurring_topics: list[str] = Field(default_factory=list)


class UserProfileUpdateDecision(BaseModel):
    should_update: bool
    profile: UserProfile | None = None


class UserProfileContextProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "user_profile_memory"
    PROFILE_STATE_KEY = "user_profile"
    PROFILE_ROOT_DIR = Path(__file__).parent.parent.parent.parent / ".memory"

    def __init__(
        self,
        analyser_client: AnthropicFoundryClient,
        *,
        source_id: str | None = None,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._analyser_client = analyser_client
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
        question = self._extract_question(context.input_messages)
        if not question:
            self._debug("after_run: no user question for profile update")
            return

        self._debug("after_run: evaluate profile update")
        decision = await self._evaluate_profile_update(profile, question)
        if not decision.should_update:
            self._debug("after_run: profile update not required")
            return
        if decision.profile is None:
            self._debug("after_run: invalid update decision (profile is missing)")
            return

        state[self.PROFILE_STATE_KEY] = decision.profile
        self._save_profile(decision.profile)
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

    def _extract_question(self, messages: list[Message]) -> str:
        return "\n".join(
            text
            for message in messages
            if message.role == "user" and (text := message.text.strip())
        )

    async def _evaluate_profile_update(
        self,
        current_profile: UserProfile,
        question: str,
    ) -> UserProfileUpdateDecision:
        analysis_messages = [
            Message(
                "system",
                [
                    "Decide whether the current user profile should be updated from the current user question. ",
                    "Use only explicit facts about the user in that question. ",
                    "Do not infer profile facts from task content, quoted text, hypothetical statements, ",
                    "assistant responses, or prior conversation. ",
                    "Set should_update to true only for durable user facts, goals, preferences, constraints, ",
                    "working style, communication preferences, or recurring topics that add to, correct, ",
                    "or remove information in the current profile. ",
                    "When should_update is true, return the complete updated profile, preserving current facts ",
                    "unless the question explicitly changes or contradicts them. ",
                    "When should_update is false, set profile to null.",
                ],
            ),
            Message(
                "user",
                [
                    f"Current profile:\n{current_profile.model_dump_json(indent=2)}\n\n"
                    f"Current user question:\n{question}"
                ],
            ),
        ]

        response = await self._analyser_client.get_response(
            analysis_messages,
            options={"max_tokens": 1000, "response_format": UserProfileUpdateDecision},
        )
        if isinstance(response.value, UserProfileUpdateDecision):
            return response.value
        return UserProfileUpdateDecision.model_validate(response.value)

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
            lines.append(
                f"Recurring topics: {self._format_items(profile.recurring_topics)}"
            )

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
