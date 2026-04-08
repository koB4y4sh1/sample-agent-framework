from __future__ import annotations

from azure.identity import AzureCliCredential, get_bearer_token_provider


def create_token_provider():
    return get_bearer_token_provider(
        AzureCliCredential(process_timeout=30),
        "https://ai.azure.com/.default",
    )
