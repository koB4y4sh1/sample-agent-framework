from __future__ import annotations

from agent_framework import Content

from ._types import ProviderFamily
from .base import BaseMessageConverter


class GeminiMessageConverter(BaseMessageConverter):
    def _convert_text_reasoning(
        self,
        content: Content,
        *,
        source_provider_family: ProviderFamily | None,
    ) -> Content | None:
        if source_provider_family == "google":
            return self._build_native_reasoning_content(content)
        return super()._convert_text_reasoning(
            content, source_provider_family=source_provider_family
        )

    def _build_native_reasoning_content(self, content: Content) -> Content:
        data = content.to_dict(exclude_none=True)
        data.pop("raw_representation", None)
        return Content.from_dict(data)


GeminiMessageConverter = GeminiMessageConverter

__all__ = ["GeminiMessageConverter", "GeminiMessageConverter"]
