from __future__ import annotations

from typing import Literal, TypeAlias

# Agent Framework 側では Gemini 系 provider を `google` として扱う。
ProviderFamily: TypeAlias = Literal["anthropic", "openai", "google"]

# reasoning を別 provider に渡せない場合の扱い。
# `as_text`: 通常の text として残す
# `drop`: 再投入から外す
ReasoningPolicy: TypeAlias = Literal["as_text", "drop"]
