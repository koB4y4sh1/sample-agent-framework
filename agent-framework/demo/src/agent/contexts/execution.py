from __future__ import annotations

from typing import Any

from agent_framework import AgentSession, ContextProvider, SessionContext, SupportsAgentRun

ExecutionMetadata = dict[str, Any]


class ExecutionContextProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "execution_context"

    def __init__(
        self,
        *,
        model: str,
        provider_family: str,
        history_source_id: str,
        source_id: str | None = None,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._model = model
        self._provider_family = provider_family
        self._history_source_id = history_source_id

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        metadata = self._build_metadata(agent)
        state["metadata"] = metadata
        context.metadata[self.source_id] = metadata
        context.extend_instructions(self.source_id, self._build_instruction(metadata))

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        return None

    def _build_metadata(self, agent: SupportsAgentRun) -> ExecutionMetadata:
        metadata: ExecutionMetadata = {
            "model": self._model,
            "provider_family": self._provider_family,
            "history_source_id": self._history_source_id,
        }
        agent_name = getattr(agent, "name", None)

        if isinstance(agent_name, str) and agent_name.strip():
            metadata["agent_name"] = agent_name.strip()

        return metadata

    def _build_instruction(self, metadata: ExecutionMetadata) -> str:
        lines = [
            "Current execution context:",
            f"model: {metadata['model']}",
            f"provider_family: {metadata['provider_family']}",
            f"history_source_id: {metadata['history_source_id']}",
        ]
        return "\n".join(lines)
