from __future__ import annotations

from dataclasses import dataclass

from agent_framework.foundry import AnthropicFoundryClient
from azure.identity import AzureCliCredential, get_bearer_token_provider

from agent import DemoAgentConfig, DemoAgentFactory
from providers import ConversationMemoryProvider, HistoryManager, LocalFileStore
from skills import DemoSkills
from stream_renderer import ProviderFamily, StreamRendererResolver
from tools import DemoToolkit


@dataclass(slots=True)
class DemoConfig:
    provider_family: ProviderFamily = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 8000
    history_limit: int = 100
    memory_limit: int = 12
    memory_token_budget: int = 6000


class DemoApplication:
    """Anthropic デモを構成する依存オブジェクトを組み立てる。"""

    def __init__(self, config: DemoConfig | None = None) -> None:
        self.config = config or DemoConfig()
        token_provider = get_bearer_token_provider(
            AzureCliCredential(),
            "https://ai.azure.com/.default",
        )
        self.client = AnthropicFoundryClient(
            model=self.config.model,
            azure_ad_token_provider=token_provider,
        )
        self.store = LocalFileStore()
        self.history_provider = HistoryManager(
            store=self.store,
            max_messages=self.config.history_limit,
        )
        self.memory_provider = ConversationMemoryProvider(
            anthropic_client=self.client.anthropic_client,
            model=self.config.model,
            history_source_id=self.history_provider.source_id,
            max_history_messages=self.config.memory_limit,
            max_input_tokens=self.config.memory_token_budget,
        )
        self.skills = DemoSkills()
        self.skills_provider = self.skills.build_provider()
        self.toolkit = DemoToolkit(self.client)
        self.stream_renderer_resolver = StreamRendererResolver(self.config.provider_family)
        self.stream_renderer = self.stream_renderer_resolver.resolve()
        self.agent = DemoAgentFactory(
            config=DemoAgentConfig(
                model_name=self.config.model,
                max_tokens=self.config.max_tokens,
            ),
            client=self.client,
            history_provider=self.history_provider,
            memory_provider=self.memory_provider,
            skills_provider=self.skills_provider,
            toolkit=self.toolkit,
        ).create()

    def create_session(self):
        return self.agent.create_session()
