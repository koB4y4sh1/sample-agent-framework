from __future__ import annotations

from typing import Literal, TypeAlias

ProviderFamily: TypeAlias = Literal["anthropic", "openai", "google"]
ReasoningPolicy: TypeAlias = Literal["as_text", "drop"]
