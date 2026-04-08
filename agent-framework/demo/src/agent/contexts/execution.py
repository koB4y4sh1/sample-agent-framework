from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent_framework import AgentSession, ContextProvider, SessionContext, SupportsAgentRun


class ExecutionContextProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "execution_context"

    def __init__(
        self,
        *,
        model: str,
        history_source_id: str,
        source_id: str | None = None,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._model = model
        self._history_source_id = history_source_id

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        context.extend_instructions(self.source_id, self._build_instruction(session))

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        return None

    def _build_instruction(self, session: AgentSession) -> str:
        session_id = self._resolve_session_id(session)
        lines = [
            "Current execution context:",
            f"model: {self._model}",
            f"history_source_id: {self._history_source_id}",
            f"working_directory: {Path.cwd()}",
            f"platform: {os.name}",
        ]
        if session_id:
            lines.append(f"session_id: {session_id}")
        return "\n".join(lines)

    def _resolve_session_id(self, session: AgentSession) -> str | None:
        for attr_name in ("session_id", "id"):
            value = getattr(session, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
