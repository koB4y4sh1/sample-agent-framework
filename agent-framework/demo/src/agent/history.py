from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, ClassVar
from uuid import uuid8

from agent_framework import (
    AgentSession,
    HistoryProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
)

from .contexts.message_converter import EXECUTED_INPUT_MESSAGES_METADATA_KEY
from .contexts.message_converter import MessageHistoryNormalizer
from .store.base import MessageStore


class LocalHistoryProvider(HistoryProvider):
    """会話履歴を MessageStore に永続化する Provider。"""

    DEFAULT_SOURCE_ID: ClassVar[str] = uuid8().hex

    def __init__(
        self,
        *,
        store: MessageStore,
        source_id: str | None = None,
        max_messages: int | None = None,
        execution_metadata_source_id: str = "execution_context",
        execution_metadata_message_key: str = "execution",
    ) -> None:
        super().__init__(
            source_id=source_id or self.DEFAULT_SOURCE_ID,
            load_messages=True,
            store_inputs=True,
            store_context_messages=False,
            store_outputs=True,
        )
        self._store = store
        self._max_messages = max_messages
        self._execution_metadata_source_id = execution_metadata_source_id
        self._execution_metadata_message_key = execution_metadata_message_key
        self._message_normalizer = MessageHistoryNormalizer()

    async def get_messages(
        self,
        session_id: str | None,
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Message]:
        """保存済みメッセージを取得する。"""
        stored_messages = await _maybe_await(self._store.read_messages(session_id))
        return stored_messages

    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """メッセージを既存履歴に追加して保存する。"""
        existing_messages = await _maybe_await(self._store.read_messages(session_id))
        combined_messages = [*existing_messages, *messages]
        combined_messages = self._message_normalizer.normalize_messages(
            combined_messages
        )
        if self._max_messages is not None:
            combined_messages = combined_messages[-self._max_messages :]
        await _maybe_await(self._store.write_messages(session_id, combined_messages))

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """実行後、入力・出力メッセージを履歴として保存する。"""
        messages_to_store: list[Message] = []
        messages_to_store.extend(self._get_context_messages_to_store(context))
        if self.store_inputs:
            messages_to_store.extend(self._get_input_messages_to_store(context))
        if self.store_outputs and context.response and context.response.messages:
            messages_to_store.extend(context.response.messages)

        if messages_to_store:
            metadata = context.metadata.get(self._execution_metadata_source_id)
            await self.save_messages(
                context.session_id,
                self._with_execution_metadata(messages_to_store, metadata),
                state=state,
            )

    def _with_execution_metadata(
        self,
        messages: Sequence[Message],
        metadata: Any,
    ) -> list[Message]:
        if not isinstance(metadata, dict):
            return list(messages)

        saved_messages = list(messages)
        for message in saved_messages:
            message.additional_properties[self._execution_metadata_message_key] = dict(
                metadata
            )
        return saved_messages

    def _get_input_messages_to_store(self, context: SessionContext) -> list[Message]:
        executed_input_messages = context.metadata.get(
            EXECUTED_INPUT_MESSAGES_METADATA_KEY
        )
        if isinstance(executed_input_messages, list) and all(
            isinstance(item, Message) for item in executed_input_messages
        ):
            return list(executed_input_messages)
        return list(context.input_messages)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
