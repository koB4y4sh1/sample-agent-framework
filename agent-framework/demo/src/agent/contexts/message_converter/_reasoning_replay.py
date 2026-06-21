from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from agent_framework import Content, Message

from ._types import ProviderFamily, ReasoningPolicy

MappingSanitizer: TypeAlias = Callable[[Mapping[str, Any]], dict[str, Any]]


class ReasoningReplaySanitizer:
    """reasoning content を、再投入できる形に整える helper。

    reasoning は provider ごとに暗号化や署名の扱いが違う。
    そのため、別 provider の reasoning をそのまま使い回すのは危険。

    方針:
    - 同じ provider で、安全に再利用できる情報がある場合だけ native reasoning として残す
    - それ以外は `reasoning_policy` に従って text 化または削除する
    """

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
        """1つの reasoning content を、残す・text 化する・捨てるのどれかに決める。"""
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
        """Message に保存された実行 metadata から、元の provider を読む。"""
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
        """native reasoning として再投入してよいかを判定する。

        最低条件は「元 provider と投入先 provider が同じ」こと。
        さらに provider ごとに必要な保護情報が揃っている場合だけ許可する。
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
        """native reasoning を残す場合も、不要な raw 情報だけは落とす。"""
        data = self._sanitize_mapping(content.to_dict(exclude_none=True))
        data["type"] = content.type
        data.pop("raw_representation", None)
        return Content.from_dict(data)

    def _infer_reasoning_provider_family(self, content: Content) -> ProviderFamily | None:
        """metadata がない場合に、content の形から provider を推測する。"""
        if content.protected_data:
            return "anthropic"
        if content.additional_properties.get("encrypted_content"):
            return "openai"
        return None
