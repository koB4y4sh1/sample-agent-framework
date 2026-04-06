from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework import Agent, BaseChatClient


@dataclass(slots=True)
class DemoAgentConfig:
    model_name: str
    max_tokens: int


class DemoAgentFactory:
    """準備済みの依存オブジェクトから Anthropic デモ用 Agent を構築する。"""

    def __init__(
        self,
        *,
        config: DemoAgentConfig,
        client: BaseChatClient[Any],
        history_provider,
        memory_provider,
        skills_provider,
        toolkit,
    ) -> None:
        self._config = config
        self._client = client
        self._history_provider = history_provider
        self._memory_provider = memory_provider
        self._skills_provider = skills_provider
        self._toolkit = toolkit

    def create(self) -> Agent:
        return Agent(
            client=self._client,
            name="AnthropicDemoAgent",
            instructions=(
                "You are a helpful assistant. Use tools when appropriate. "
                "If multimodal input exists, reference it explicitly."
            ),
            context_providers=[
                self._history_provider,
                self._memory_provider,
                self._skills_provider,
            ],
            tools=[
                *self._toolkit.build_tools(),
            ],
            default_options={
                "max_tokens": self._config.max_tokens,
                "thinking": {"type": "adaptive"},
            },
        )
