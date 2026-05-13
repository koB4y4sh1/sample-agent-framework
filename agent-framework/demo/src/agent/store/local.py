from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from agent_framework import Message

from .base import MessageStore

MEMORY_ROOT_DIR = Path(__file__).parents[3] / ".history"  # demo/.history


class LocalStore(MessageStore):
    """ローカル JSON に会話履歴を保存するストア。"""

    def __init__(self) -> None:
        MEMORY_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    async def read_messages(self, session_id: str | None) -> list[Message]:
        path = self._get_path(session_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Message.from_dict(item) for item in data]

    async def write_messages(
        self, session_id: str | None, messages: Sequence[Message]
    ) -> None:
        path = self._get_path(session_id)
        serialized = [message.to_dict() for message in messages]
        path.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _get_path(self, session_id: str | None) -> Path:
        safe_session_id = (session_id or "default").replace("/", "_").replace("\\", "_")
        return MEMORY_ROOT_DIR / f"{safe_session_id}.json"
