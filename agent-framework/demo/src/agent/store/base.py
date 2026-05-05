from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from agent_framework import Message


class MessageStore(ABC):
    """永続化メッセージの読み書き契約。"""

    @abstractmethod
    async def read_messages(self, session_id: str | None) -> list[Message]:
        """保存済みメッセージを読み込む。"""

    @abstractmethod
    async def write_messages(
        self, session_id: str | None, messages: Sequence[Message]
    ) -> None:
        """メッセージを書き込む。"""
