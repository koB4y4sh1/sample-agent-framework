from __future__ import annotations

from .base import BaseProviderMessageConverter, ReasoningPolicy


class OpenAIMessageConverter(BaseProviderMessageConverter):
    """OpenAI に再投入するための Message converter。

    基本処理は `BaseProviderMessageConverter` に任せる。
    ここでは target provider が OpenAI であることだけを明示する。
    """

    def __init__(
        self,
        *,
        reasoning_policy: ReasoningPolicy = "as_text",
        reasoning_label: str = "[reasoning]",
    ) -> None:
        super().__init__(
            target_provider_family="openai",
            reasoning_policy=reasoning_policy,
            reasoning_label=reasoning_label,
        )


__all__ = ["OpenAIMessageConverter"]
