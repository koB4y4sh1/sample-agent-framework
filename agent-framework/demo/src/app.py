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
from ui import ProviderFamily, UIResolver
 
 
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
    """デモアプリケーションを構成する依存オブジェクトを組み立てる Factory クラス"""
 
    def __init__(self, config: DemoConfig | None = None) -> None:
        self.config = config or DemoConfig()
        token_provider = get_bearer_token_provider(
            AzureCliCredential(process_timeout=30),
            "https://ai.azure.com/.default",
        )
        # チャット用（ユーザー選択）
        self.chat_client = AnthropicFoundryClient(
            model=self.config.model,
            azure_ad_token_provider=token_provider,
        )
        # 軽量モデル（要約、分析用）
        self.haiku_client = AnthropicFoundryClient(
            model="claude-haiku-4-5",
            azure_ad_token_provider=token_provider,
        )
        # 履歴
        self.store = LocalStore()
        self.history_provider = LocalHistoryProvider(
            store=self.store,
            max_messages=self.config.history_limit,
        )
        # メモリ（ユーザープロファイル）
        self.memory_provider = CustomContextProvider(
            analyser_client=self.haiku_client,
            model=self.config.model,
            history_source_id=self.history_provider.source_id,
        )
        # 圧縮
        self.compaction_provider = DemoCompactionProvider(
            history_source_id=self.history_provider.source_id,
            summarizer_client=self.haiku_client,
            config=DemoCompactionConfig(
                token_budget=self.config.compaction_token_budget,
                keep_last_tool_call_groups=self.config.compaction_keep_last_tool_call_groups,
                summary_target_count=self.config.compaction_summary_target_count,
                summary_threshold=self.config.compaction_summary_threshold,
                keep_last_groups=self.config.compaction_keep_last_groups,
            ),
        ).create_provider()
        # Agent Skills
        self.skills = DemoSkills()
        self.skills_provider = self.skills.build_provider()
        # Tool (MCP含む)
        self.tool = DemoTools(self.chat_client)
        self.model_settings = load_model_settings(self.config.model)
        # Agent
        self.stream_resolver = UIResolver(self.config.provider_family)
        self.stream_renderer = self.stream_resolver.resolve()
        self.agent = DemoAgent(
            config=DemoAgentConfig(
                model_name=self.config.model,
                default_options=self.model_settings.default_options,
            ),
            client=self.chat_client,
            history_provider=self.history_provider,
            memory_provider=self.memory_provider,
            skills_provider=self.skills_provider,
            toolkit=self.tool,
            compaction_provider=self.compaction_provider,
        ).create()
 
    def create_session(self):
        return self.agent.create_session()