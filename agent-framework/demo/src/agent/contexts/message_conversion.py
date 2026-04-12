from __future__ import annotations

from typing import Any

from agent_framework import AgentSession, ContextProvider, Message, SessionContext, SupportsAgentRun

from ..message_converter import CommonMessageConverter


class MessageConversionContextProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "message_conversion"
    _METADATA_KEY_PREFIX = "message_conversion_original:"

    def __init__(
        self,
        *,
        history_source_id: str,
        message_converter: CommonMessageConverter,
        source_id: str | None = None,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._history_source_id = history_source_id
        self._message_converter = message_converter
        self._metadata_key = f"{self._METADATA_KEY_PREFIX}{self.source_id}:{history_source_id}"

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        messages = context.context_messages.get(self._history_source_id)
        if not messages:
            return

        context.metadata[self._metadata_key] = list(messages)
        context.context_messages[self._history_source_id] = self._message_converter.convert_messages(messages)

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        original_messages = context.metadata.pop(self._metadata_key, None)
        if isinstance(original_messages, list) and all(isinstance(item, Message) for item in original_messages):
            context.context_messages[self._history_source_id] = original_messages
