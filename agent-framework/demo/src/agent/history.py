from __future__ import annotations
 
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid8
 
from agent_framework import (
    AgentSession,
    HistoryProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
)

from .message_normalizer import MessageHistoryNormalizer
 
MEMORY_ROOT_DIR = Path(__file__).parent.parent.parent / ".history"
 
 
class MessageStore(ABC):
    """永続化されたメッセージを読み書きするための抽象ストレージ。"""
 
    @abstractmethod
    def read_messages(self,  session_id: str | None) -> list[Message]:
        """保存済みメッセージを読み込む。"""
 
    @abstractmethod
    def write_messages(self, session_id: str | None, messages: Sequence[Message]) -> None:
        """メッセージを書き込む。"""
 
 
class LocalStore(MessageStore):
    """ローカルのデモ用ディレクトリ配下に JSON としてメッセージを保存する。"""
 
    def __init__(self) -> None:
        MEMORY_ROOT_DIR.mkdir(parents=True, exist_ok=True)
 
    def read_messages(self,  session_id: str | None) -> list[Message]:
        path = self._get_path(session_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Message.from_dict(item) for item in data]
 
    def write_messages(self, session_id: str | None, messages: Sequence[Message]) -> None:
        path = self._get_path( session_id)
        serialized = [message.to_dict() for message in messages]
        path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
 
    def _get_path(self, session_id: str | None) -> Path:
        safe_session_id = (session_id or "default").replace("/", "_").replace("\\", "_")
        return MEMORY_ROOT_DIR / f"{safe_session_id}.json"
 
 
class LocalHistoryProvider(HistoryProvider):
    """会話履歴を永続化 Provider"""
 
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
        """保存済みメッセージの取得"""
        stored_messages = self._store.read_messages(session_id)
        return stored_messages
 
    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """メッセージの保存"""
        existing_messages = self._store.read_messages(session_id)
        combined_messages = [*existing_messages, *messages]
        if self._max_messages is not None:
            combined_messages = combined_messages[-self._max_messages :]
        self._store.write_messages(session_id, combined_messages)

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Store messages with execution metadata from the run context."""
        messages_to_store: list[Message] = []
        messages_to_store.extend(self._get_context_messages_to_store(context))
        if self.store_inputs:
            messages_to_store.extend(context.input_messages)
        if self.store_outputs and context.response and context.response.messages:
            messages_to_store.extend(context.response.messages)

        if messages_to_store:
            metadata = context.metadata.get(self._execution_metadata_source_id)
            normalized_messages = self._message_normalizer.normalize_messages(messages_to_store)
            await self.save_messages(
                context.session_id,
                self._with_execution_metadata(normalized_messages, metadata),
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
            message.additional_properties[self._execution_metadata_message_key] = dict(metadata)
        return saved_messages
