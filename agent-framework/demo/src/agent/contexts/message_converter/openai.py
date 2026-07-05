from __future__ import annotations

import json
from collections.abc import Mapping

from agent_framework import Content

from ._types import ProviderFamily
from .base import BaseMessageConverter


class OpenAIMessageConverter(BaseMessageConverter):
    """OpenAI に渡す Message へ変換する。

    OpenAI 固有ルール:
    - `text_reasoning` は `id` と `encrypted_content` がある場合だけそのまま残す。
    - `function_call.arguments={"city": "Tokyo"}` は
      `arguments='{"city": "Tokyo"}'` にする。
    """

    def __init__(self, *, reasoning_label: str = "[reasoning]") -> None:
        super().__init__(reasoning_label=reasoning_label)

    def _convert_text_reasoning(
        self,
        content: Content,
        *,
        source_provider_family: ProviderFamily | None,
    ) -> Content | None:
        if source_provider_family == "openai" and content.additional_properties.get(
            "encrypted_content"
        ):
            return self._build_native_reasoning_content(content)
        return super()._convert_text_reasoning(content)

    def _build_native_reasoning_content(self, content: Content) -> Content:
        data = content.to_dict(exclude_none=True)
        data.pop("raw_representation", None)
        return Content.from_dict(data)

    def _convert_tool_call(self, content: Content) -> Content | None:
        """function_call / mcp_server_tool_call の arguments を JSON 文字列にする。"""
        if content.type not in ("function_call", "mcp_server_tool_call"):
            return content
        arguments = getattr(content, "arguments", None)
        if not isinstance(arguments, Mapping):
            return content
        data = content.to_dict(exclude_none=True)
        try:
            data["arguments"] = json.dumps(arguments, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            data["arguments"] = str(arguments)
        return Content.from_dict(data)

    def _convert_approval_request(
        self, content: Content, existing_function_call_ids: set[str]
    ) -> Content | None:
        """approval request に埋め込まれた function_call を取り出す。

        ルール:
        - request 内に `function_call` がなければ request のまま返す。
        - `function_call.additional_properties.server_label` があれば hosted tool 扱いなので request のまま返す。
            ※ hosted tool の場合はそのまま approval request で返すのではなく、mcp_server_tool_call として返したほうがいいかも
        - それ以外は request ではなく `function_call` として返す。
        """
        function_call = content.function_call

        if function_call is None:
            return None

        additional_properties = function_call.additional_properties
        if isinstance(additional_properties, dict) and additional_properties.get(
            "server_label"
        ):
            return content

        if function_call.call_id is None:
            return None
        if function_call.name is None:
            return None

        return Content.from_function_call(
            call_id=function_call.call_id,
            name=function_call.name,
            arguments=function_call.arguments,
            additional_properties=function_call.additional_properties,
            raw_representation=function_call.raw_representation,
        )


__all__ = ["OpenAIMessageConverter"]
