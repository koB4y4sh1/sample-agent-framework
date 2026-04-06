from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from agent_framework import (
    AgentSession,
    ContextProvider,
    HistoryProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
)
from message_converter import AnthropicMessageConverter

MEMORY_ROOT_DIR = Path(__file__).parent / ".history"


class MessageStore(ABC):
    """永続化されたメッセージを読み書きするための抽象ストレージ。"""

    @abstractmethod
    def read_messages(self,  session_id: str | None) -> list[Message]:
        """保存済みメッセージを読み込む。"""

    @abstractmethod
    def write_messages(self, session_id: str | None, messages: Sequence[Message]) -> None:
        """メッセージを書き込む。"""


class LocalFileStore(MessageStore):
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


class HistoryManager(HistoryProvider):
    """生の会話履歴を永続化し、次回実行時に再投入する。"""

    DEFAULT_SOURCE_ID: ClassVar[str] = "demo_history"

    def __init__(
        self,
        *,
        store: MessageStore,
        source_id: str | None = None,
        max_messages: int | None = None,
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

    async def get_messages(
        self,
        session_id: str | None,
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Message]:
        stored_messages = self._store.read_messages(session_id)
        return self._expand_messages_for_replay(stored_messages)

    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        existing_messages = self._store.read_messages(session_id)
        combined_messages = [*existing_messages, *messages]
        if self._max_messages is not None:
            combined_messages = combined_messages[-self._max_messages :]
        self._store.write_messages(session_id, combined_messages)

    def _expand_messages_for_replay(self, messages: Sequence[Message]) -> list[Message]:
        replay_messages: list[Message] = []
        for message in messages:
            replay_messages.extend(self._expand_message_for_replay(message))
        return replay_messages

    def _expand_message_for_replay(self, message: Message) -> list[Message]:
        if message.role != "assistant":
            return [message]

        expanded_messages: list[Message] = []
        current_role = "assistant"
        current_contents: list[Any] = []

        for content in message.contents:
            target_role = self._resolve_role_for_content(content.type, default_role="assistant")
            if current_contents and target_role != current_role:
                expanded_messages.append(self._clone_message(message, current_role, current_contents))
                current_contents = []
            current_role = target_role
            current_contents.append(content)

        if current_contents:
            expanded_messages.append(self._clone_message(message, current_role, current_contents))
        return expanded_messages

    def _resolve_role_for_content(self, content_type: str, *, default_role: str) -> str:
        if content_type in {"function_result", "mcp_server_tool_result"}:
            return "user"
        if content_type in {"function_call", "mcp_server_tool_call", "text", "text_reasoning", "data", "uri", "hosted_file"}:
            return "assistant"
        return default_role

    def _clone_message(self, original: Message, role: str, contents: Sequence[Any]) -> Message:
        return Message(
            role=role,
            contents=list(contents),
            author_name=original.author_name,
            message_id=original.message_id,
            additional_properties=dict(original.additional_properties),
            raw_representation=original.raw_representation,
        )


class ConversationMemoryProvider(ContextProvider):
    """履歴を選別し、トークン数を見ながらメモリ用 instruction を注入する。"""

    DEFAULT_SOURCE_ID: ClassVar[str] = "demo_memory"

    def __init__(
        self,
        anthropic_client: Any,
        model: str,
        *,
        history_source_id: str = HistoryManager.DEFAULT_SOURCE_ID,
        source_id: str | None = None,
        max_history_messages: int = 12,
        max_input_tokens: int = 6000,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._anthropic_client = anthropic_client
        self._model = model
        self._history_source_id = history_source_id
        self._max_history_messages = max_history_messages
        self._max_input_tokens = max_input_tokens
        self._converter = AnthropicMessageConverter()

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        history_messages = context.get_messages(sources={self._history_source_id})
        selected_messages = await self._fit_history_to_budget(history_messages, context.input_messages)
        if not selected_messages:
            return
        context.extend_instructions(self.source_id, self._build_memory_instruction(selected_messages))

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        return None

    async def _fit_history_to_budget(
        self,
        history_messages: Sequence[Message],
        input_messages: Sequence[Message],
    ) -> list[Message]:
        selected_messages = list(history_messages[-self._max_history_messages :])
        while selected_messages:
            input_token_count = await self._count_tokens([*selected_messages, *input_messages])
            if input_token_count <= self._max_input_tokens:
                return selected_messages
            selected_messages.pop(0)
        return []

    async def _count_tokens(self, messages: Sequence[Message]) -> int:
        token_count = await self._anthropic_client.messages.count_tokens(
            model=self._model,
            messages=self._converter.convert_messages(messages),
        )
        return token_count.input_tokens

    def _build_memory_instruction(self, messages: Sequence[Message]) -> str:
        lines: list[str] = ["Use the following recent conversation history as reference."]
        for message in messages:
            text = (message.text or "").strip()
            if not text:
                continue
            lines.append(f"{message.role}: {text}")
        return "\n".join(lines)
