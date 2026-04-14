from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from agent_framework import (
    AgentSession,
    ContextProvider,
    HistoryProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
)

MEMORY_ROOT_DIR = Path(__file__).parent / ".memory"


class MessageStore(ABC):
    """永続化された会話メッセージを読み書きするための抽象基底クラス。"""

    @abstractmethod
    def read_messages(self, namespace: str, session_id: str | None) -> list[Message]:
        """指定した namespace と session のメッセージを読み込む。"""

    @abstractmethod
    def write_messages(self, namespace: str, session_id: str | None, messages: Sequence[Message]) -> None:
        """指定した namespace と session のメッセージを書き込む。"""


class LocalFileStore(MessageStore):
    """ローカルの ``.memory`` ディレクトリ配下に JSON で保存する MessageStore 実装。"""

    def __init__(self) -> None:
        """永続化先のルートディレクトリを作成する。"""
        MEMORY_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    def read_messages(self, namespace: str, session_id: str | None) -> list[Message]:
        """指定した namespace と session のメッセージをローカルストレージから読み込む。"""
        path = self._get_path(namespace, session_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Message.from_dict(item) for item in data]

    def write_messages(self, namespace: str, session_id: str | None, messages: Sequence[Message]) -> None:
        """指定した namespace と session のメッセージをローカルストレージへ保存する。"""
        path = self._get_path(namespace, session_id)
        serialized = [message.to_dict() for message in messages]
        path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_path(self, namespace: str, session_id: str | None) -> Path:
        """指定した namespace と session に対応する保存先パスを返す。"""
        safe_session_id = (session_id or "default").replace("/", "_").replace("\\", "_")
        namespace_dir = MEMORY_ROOT_DIR / namespace
        namespace_dir.mkdir(parents=True, exist_ok=True)
        return namespace_dir / f"{safe_session_id}.json"


# class SupabaseStore(MessageStore):
#     """Supabase を保存先として利用するための MessageStore 実装の雛形。"""

#     def __init__(
#         self,
#         *,
#         url: str,
#         key: str,
#         table_name: str = "agent_messages",
#     ) -> None:
#         """Supabase 用ストアを初期化する。

#         Args:
#             url: Supabase の Project URL。
#             key: Supabase の API Key。
#             table_name: 保存先テーブル名。
#         """
#         self.url = url
#         self.key = key
#         self.table_name = table_name

#     def read_messages(self, namespace: str, session_id: str | None) -> list[Message]:
#         """Supabase からメッセージを読み込む。

#         実運用ではここに Supabase SDK を使った取得処理を実装する。
#         """
#         raise NotImplementedError("SupabaseStore.read_messages is not implemented yet.")

#     def write_messages(self, namespace: str, session_id: str | None, messages: Sequence[Message]) -> None:
#         """Supabase へメッセージを書き込む。

#         実運用ではここに Supabase SDK を使った保存処理を実装する。
#         """
#         raise NotImplementedError("SupabaseStore.write_messages is not implemented yet.")


def create_message_store(
    store_type: Literal["local_file", "supabase"] = "local_file",
    **kwargs: Any,
) -> MessageStore:
    """保存先種別に応じて MessageStore を生成する。

    Args:
        store_type: 使用する保存先種別。
        **kwargs: 各保存先実装に渡す初期化引数。

    Returns:
        MessageStore 実装。
    """
    if store_type == "local_file":
        return LocalFileStore()
    raise ValueError(f"Unsupported store_type: {store_type}")


class HistoryManeger(HistoryProvider):
    """会話履歴の永続化層。

    - ストレージから会話履歴の読み取り
    - ストレージへの会話履歴の書き込み
    - Message の永続化処理の管理

    このクラスは具体的な保存先実装を知らず、MessageStore 抽象にのみ依存する。
    そのため LocalFileStore から SupabaseStore への差し替えは、生成時に渡す
    store を変更するだけで済む。
    """

    DEFAULT_SOURCE_ID: ClassVar[str] = "local_file_history"

    def __init__(
        self,
        *,
        store: MessageStore,
        source_id: str | None = None,
        max_messages: int | None = None,
    ) -> None:
        """履歴 manager を初期化する。

        Args:
            store: 利用するメッセージ保存先実装。
            source_id: provider の出所識別子。
            max_messages: 保存しておく最大メッセージ数。未指定時は上限なし。
        """
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
        """現在の session に対応する永続化済み履歴を返す。"""
        return self._store.read_messages(self.source_id, session_id)

    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """新しいメッセージを既存の会話履歴へ追記保存する。"""
        existing_messages = self._store.read_messages(self.source_id, session_id)
        combined_messages = [*existing_messages, *messages]
        if self._max_messages is not None:
            combined_messages = combined_messages[-self._max_messages :]
        self._store.write_messages(self.source_id, session_id, combined_messages)


class HistoryManager(ContextProvider):
    """SessionContext 上の履歴からモデル用コンテキストを組み立てる ContextProvider。"""

    DEFAULT_SOURCE_ID: ClassVar[str] = "local_file_context_memory"

    def __init__(
        self,
        *,
        source_id: str | None = None,
        history_source_id: str = HistoryManeger.DEFAULT_SOURCE_ID,
        max_messages: int = 10,
    ) -> None:
        """履歴ベースの context provider を初期化する。

        Args:
            source_id: provider の出所識別子。
            history_source_id: 参照対象となる HistoryProvider の source_id。
            max_messages: 参照対象とする直近メッセージ数。
        """
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._history_source_id = history_source_id
        self._max_messages = max_messages

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """SessionContext から履歴を取得し、モデル向け instruction として注入する。"""
        history_messages = context.get_messages(sources={self._history_source_id})
        selected_messages = self._select_messages(history_messages)
        if selected_messages:
            context.extend_instructions(
                self.source_id,
                self._build_memory_instruction(selected_messages),
            )

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """この provider は入力コンテキストの整形だけを担当するため後処理は行わない。"""
        return None

    def _select_messages(self, messages: Sequence[Message]) -> list[Message]:
        """次の応答に反映させる履歴メッセージを選別する。"""
        if not messages:
            return []
        return list(messages[-self._max_messages :])

    def _build_memory_instruction(self, messages: Sequence[Message]) -> str:
        """選別した履歴をコンパクトな instruction 文字列へ変換する。"""
        lines: list[str] = ["Use the following recent conversation history as reference:"]
        for message in messages:
            text = (message.text or "").strip()
            if not text:
                continue
            lines.append(f"{message.role}: {text}")
        return "\n".join(lines)
