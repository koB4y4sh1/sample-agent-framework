from __future__ import annotations

from typing import Any

from agent_framework import AgentSession, ContextProvider, SessionContext, SupportsAgentRun


class PreferencePolicyProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "preference_policy"

    def __init__(
        self,
        *,
        response_policies: list[str] | None = None,
        tool_policies: list[str] | None = None,
        source_id: str | None = None,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._response_policies = response_policies or [
            "State the conclusion first, then the rationale.",
            "Do not answer outside the question scope.",
            "If a premise is uncertain, mark it as unresolved.",
        ]
        self._tool_policies = tool_policies or [
            "Use pnpm for React and TypeScript tasks.",
            "Use uv for Python tasks.",
            "Avoid superficial explanations.",
        ]

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        context.extend_instructions(self.source_id, self._build_instruction())

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        return None

    def _build_instruction(self) -> str:
        lines = [
            "Always prioritize the following response policies.",
            "response_policies:",
            *[f"- {policy}" for policy in self._response_policies],
            "tool_policies:",
            *[f"- {policy}" for policy in self._tool_policies],
        ]
        return "\n".join(lines)
