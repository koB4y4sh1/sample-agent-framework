from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypeAlias

from agent_framework import Message

# Agent Framework 側では Gemini 系 provider を `google` として扱う。
ProviderFamily: TypeAlias = Literal["anthropic", "openai", "google"]


class MessageConverter(Protocol):
    def convert_messages(self, messages: Sequence[Message]) -> list[Message]:
        ...
