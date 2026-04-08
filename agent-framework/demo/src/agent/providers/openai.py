from __future__ import annotations

from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential


def create_openai_chat_client(
    *,
    model: str,
    credential=None,
) -> FoundryChatClient:
    resolved_credential = credential or AzureCliCredential(process_timeout=30)
    return FoundryChatClient(
        model=model,
        credential=resolved_credential,
    )