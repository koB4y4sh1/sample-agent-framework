from __future__ import annotations

from dataclasses import dataclass

from agent import (
    CustomContextProvider,
    DemoAgent,
    DemoAgentConfig,
    DemoCompactionConfig,
    DemoCompactionProvider,
    DemoSkills,
    DemoTools,
    LocalHistoryProvider,
    LocalStore,
)
from agent_framework.foundry import AnthropicFoundryClient
from azure.identity import AzureCliCredential, get_bearer_token_provider
from settings import load_model_settings
from stream_renderer import ProviderFamily, StreamResolver


@dataclass(slots=True)
class DemoConfig:
    provider_family: ProviderFamily = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 8000
    history_limit: int = 100
    compaction_token_budget: int = 16_000
    compaction_keep_last_tool_call_groups: int = 1
    compaction_summary_target_count: int = 4
    compaction_summary_threshold: int = 2
    compaction_keep_last_groups: int = 20


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
        self.store = LocalStore()
        self.history_provider = LocalHistoryProvider(
            store=self.store,
            max_messages=self.config.history_limit,
        )
        self.memory_provider = CustomContextProvider(
            anthropic_client=self.client.anthropic_client,
            model=self.config.model,
            history_source_id=self.history_provider.source_id,
        )
        self.compaction_provider = DemoCompactionProvider(
            history_source_id=self.history_provider.source_id,
            summarizer_client=self.client,
            config=DemoCompactionConfig(
                token_budget=self.config.compaction_token_budget,
                keep_last_tool_call_groups=self.config.compaction_keep_last_tool_call_groups,
                summary_target_count=self.config.compaction_summary_target_count,
                summary_threshold=self.config.compaction_summary_threshold,
                keep_last_groups=self.config.compaction_keep_last_groups,
            ),
        ).create_provider()
        self.skills = DemoSkills()
        self.skills_provider = self.skills.build_provider()
        self.toolkit = DemoTools(self.client)
        self.model_settings = load_model_settings(self.config.model)
        self.stream_resolver = StreamResolver(self.config.provider_family)
        self.stream_renderer = self.stream_resolver.resolve()
        self.agent = DemoAgent(
            config=DemoAgentConfig(
                model_name=self.config.model,
                default_options=self.model_settings.default_options,
            ),
            client=self.client,
            history_provider=self.history_provider,
            memory_provider=self.memory_provider,
            skills_provider=self.skills_provider,
            toolkit=self.toolkit,
            compaction_provider=self.compaction_provider,
        ).create()

    def create_session(self):
        return self.agent.create_session()
