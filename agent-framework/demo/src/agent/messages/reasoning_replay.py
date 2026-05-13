from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from agent_framework import Content, Message

from .types import ProviderFamily, ReasoningPolicy

MappingSanitizer: TypeAlias = Callable[[Mapping[str, Any]], dict[str, Any]]


class ReasoningReplaySanitizer:
    """Reasoning content を provider replay 可否に応じて保持・text 化・除外する。"""

    def __init__(
        self,
        *,
        target_provider_family: ProviderFamily | None,
        reasoning_policy: ReasoningPolicy,
        reasoning_label: str,
        sanitize_mapping: MappingSanitizer,
    ) -> None:
        self._target_provider_family = target_provider_family
        self._reasoning_policy = reasoning_policy
        self._reasoning_label = reasoning_label
        self._sanitize_mapping = sanitize_mapping

    def sanitize_content(
        self,
        content: Content,
        *,
        source_provider_family: ProviderFamily | None,
    ) -> Content | None:
        if self._can_replay_reasoning(content, source_provider_family):
            return self._sanitize_native_reasoning(content)
        if self._reasoning_policy == "drop":
            return None
        if not content.text:
            return None
        if self._reasoning_label:
            return Content.from_text(text=f"{self._reasoning_label}\n{content.text}")
        return Content.from_text(text=content.text)

    def source_provider_family(self, message: Message) -> ProviderFamily | None:
        execution = message.additional_properties.get("execution")
        if not isinstance(execution, Mapping):
            return None
        provider_family = execution.get("provider_family")
        if provider_family in {"anthropic", "openai", "google"}:
            return provider_family
        return None

    def _can_replay_reasoning(
        self,
        content: Content,
        source_provider_family: ProviderFamily | None,
    ) -> bool:
        """選択プロバイダー で 利用可能な推論 かの判定

        暗号化された推論本文 は provider を跨いで replay できないため、以下の条件のみ許可する
        - target provider とsource provider が同一
        - 必要な情報が揃っている場合にのみ
            - Anthropic: protected_data
            - OpenAI: reasoning_id と encrypted_content
            - Google: reasoning block は常に replay 可（ただし Google 以外の provider では replay 不可）
        """
        target_provider_family = self._target_provider_family
        if target_provider_family is None:
            return False
        if source_provider_family is None:
            source_provider_family = self._infer_reasoning_provider_family(content)
        if source_provider_family != target_provider_family:
            return False
        if target_provider_family == "anthropic":
            return bool(content.protected_data)
        if target_provider_family == "openai":
            return bool(content.id and content.additional_properties.get("encrypted_content"))
        if target_provider_family == "google":
            return True
        return False

    def _sanitize_native_reasoning(self, content: Content) -> Content:
        data = self._sanitize_mapping(content.to_dict(exclude_none=True))
        data["type"] = content.type
        data.pop("raw_representation", None)
        return Content.from_dict(data)

    def _infer_reasoning_provider_family(self, content: Content) -> ProviderFamily | None:
        if content.protected_data:
            return "anthropic"
        if content.additional_properties.get("encrypted_content"):
            return "openai"
        return None
