from __future__ import annotations

from agent_framework.foundry import AnthropicFoundryClient

from .auth import create_token_provider


def create_anthropic_chat_client(
    *,
    model: str,
    token_provider=None,
) -> AnthropicFoundryClient:
    resolved_token_provider = token_provider or create_token_provider()
    return AnthropicFoundryClient(
        model=model,
        azure_ad_token_provider=resolved_token_provider,
    )
