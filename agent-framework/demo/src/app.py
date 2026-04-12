from __future__ import annotations

from dataclasses import dataclass

from agent import (
    DemoAgent,
    DemoAgentConfig,
    DemoCompactionConfig,
    DemoCompactionProvider,
    DemoSkills,
    ExecutionContextProvider,
    LocalHistoryProvider,
    LocalStore,
    MessageConversionContextProvider,
    PreferencePolicyProvider,
    CommonMessageConverter,
    ReasoningPolicy,
    ToolRegistry,
    UserProfileContextProvider,
    create_anthropic_chat_client,
    create_gemini_chat_client,
    create_openai_chat_client,
    create_token_provider,
)
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
    reasoning_policy: ReasoningPolicy = "as_text"


class DemoApplication:
    def __init__(self, config: DemoConfig | None = None) -> None:
        self.config = config or DemoConfig()
        self.model_settings = load_model_settings(self.config.model)
        self.provider_family = self.model_settings.provider_family

        token_provider = create_token_provider()
        self.chat_client = self._create_chat_client(
            model=self.config.model,
            token_provider=token_provider,
        )
        self.haiku_client = create_anthropic_chat_client(
            model="claude-haiku-4-5",
            token_provider=token_provider,
        )

        self.store = LocalStore()
        self.history_provider = LocalHistoryProvider(
            store=self.store,
            max_messages=self.config.history_limit,
        )
        self.message_converter = CommonMessageConverter(
            reasoning_policy=self.config.reasoning_policy,
        )
        self.message_conversion_provider = MessageConversionContextProvider(
            history_source_id=self.history_provider.source_id,
            message_converter=self.message_converter,
        )
        self.memory_provider = UserProfileContextProvider(
            analyser_client=self.haiku_client,
            model=self.config.model,
            history_source_id=self.history_provider.source_id,
            message_converter=self.message_converter,
        )
        self.execution_context_provider = ExecutionContextProvider(
            model=self.config.model,
            history_source_id=self.history_provider.source_id,
        )
        self.preference_policy_provider = PreferencePolicyProvider()
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
        self.skills = DemoSkills()
        self.skills_provider = self.skills.build_provider()
        self.tool = ToolRegistry(self.chat_client)

        self.stream_resolver = UIResolver(self.provider_family)
        self.stream_renderer = self.stream_resolver.resolve()
        self.agent = DemoAgent(
            config=DemoAgentConfig(
                model_name=self.config.model,
                default_options=self.model_settings.default_options,
            ),
            client=self.chat_client,
            history_provider=self.history_provider,
            message_conversion_provider=self.message_conversion_provider,
            memory_provider=self.memory_provider,
            extra_context_providers=[
                self.execution_context_provider,
                self.preference_policy_provider,
            ],
            skills_provider=self.skills_provider,
            tool=self.tool,
            compaction_provider=self.compaction_provider,
        ).create()

    def create_session(self, session_id: str | None = None):
        return self.agent.create_session(session_id=session_id)

    def _create_chat_client(self, *, model: str, token_provider):
        if self.provider_family == "anthropic":
            return create_anthropic_chat_client(model=model, token_provider=token_provider)
        if self.provider_family == "openai":
            return create_openai_chat_client(model=model)
        if self.provider_family == "gemini":
            return create_gemini_chat_client(model=model, token_provider=token_provider)
        raise ValueError(f"Unsupported provider family: {self.provider_family}")
