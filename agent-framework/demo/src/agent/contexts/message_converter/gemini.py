from __future__ import annotations

from .base import BaseProviderMessageConverter, ReasoningPolicy


class GeminiMessageConverter(BaseProviderMessageConverter):
    """Gemini に再投入するための Message converter。

    内部の provider family 名は Agent Framework 側の表現に合わせて `google` を使う。
    基本処理は `BaseProviderMessageConverter` に任せる。
    """

    def __init__(
        self,
        *,
        reasoning_policy: ReasoningPolicy = "as_text",
        reasoning_label: str = "[reasoning]",
    ) -> None:
        super().__init__(
            target_provider_family="google",
            reasoning_policy=reasoning_policy,
            reasoning_label=reasoning_label,
        )


__all__ = ["GeminiMessageConverter"]
