from __future__ import annotations

from agent_framework.foundry import FoundryChatClient

from .auth import create_token_provider


def create_gemini_chat_client(
    *,
    model: str,
    token_provider=None,
) -> FoundryChatClient:
    resolved_token_provider = token_provider or create_token_provider()
    return FoundryChatClient(
        model=model,
        credential=resolved_token_provider,
    )
