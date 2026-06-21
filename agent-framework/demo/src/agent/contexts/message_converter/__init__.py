"""Message converter パッケージの公開 API。

通常の利用者は、この `__init__.py` から公開されている名前だけを import すればよい。
`_normalizer.py` などの `_` 付きファイルは内部実装。
"""

from __future__ import annotations

from .anthropic import AnthropicMessageConverter, AnthropicReplayConverter
from .base import (
    BaseProviderMessageConverter,
    CommonMessageConverter,
    MessageHistoryNormalizer,
    ProviderFamily,
    ReasoningPolicy,
)
from .context_provider import EXECUTED_INPUT_MESSAGES_METADATA_KEY, MessageConversionContextProvider
from .gemini import GeminiMessageConverter
from .openai import OpenAIMessageConverter


def ProviderMessageConverter(
    *,
    target_provider_family: ProviderFamily | None = None,
    reasoning_policy: ReasoningPolicy = "as_text",
    reasoning_label: str = "[reasoning]",
) -> BaseProviderMessageConverter:
    """provider family 名から、対応する converter を返す互換 factory。

    既存コードが `ProviderMessageConverter(target_provider_family=...)` を使っているため、
    その呼び出しを保ちつつ、内部では provider 別 class に振り分ける。
    新しいコードでは `OpenAIMessageConverter` などを直接使ってもよい。
    """
    if target_provider_family == "anthropic":
        return AnthropicMessageConverter(
            reasoning_policy=reasoning_policy,
            reasoning_label=reasoning_label,
        )
    if target_provider_family == "openai":
        return OpenAIMessageConverter(
            reasoning_policy=reasoning_policy,
            reasoning_label=reasoning_label,
        )
    if target_provider_family == "google":
        return GeminiMessageConverter(
            reasoning_policy=reasoning_policy,
            reasoning_label=reasoning_label,
        )
    return BaseProviderMessageConverter(
        target_provider_family=target_provider_family,
        reasoning_policy=reasoning_policy,
        reasoning_label=reasoning_label,
    )


__all__ = [
    "AnthropicMessageConverter",
    "AnthropicReplayConverter",
    "BaseProviderMessageConverter",
    "CommonMessageConverter",
    "EXECUTED_INPUT_MESSAGES_METADATA_KEY",
    "GeminiMessageConverter",
    "MessageConversionContextProvider",
    "MessageHistoryNormalizer",
    "OpenAIMessageConverter",
    "ProviderFamily",
    "ProviderMessageConverter",
    "ReasoningPolicy",
]
