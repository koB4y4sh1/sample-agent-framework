from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from agent_framework import Content, Message

from ._types import ProviderFamily
from .base import BaseMessageConverter


class OpenAIMessageConverter(BaseMessageConverter):
    """OpenAI に渡す Message へ変換する。

    OpenAI 固有ルール:
    - `text_reasoning` は `encrypted_content` がある場合だけそのまま残す。
    - `function_call.arguments={"city": "Tokyo"}` は
      `arguments='{"city": "Tokyo"}'` にする。
    - `function_approval_response` のうち、対応する call が `function_call` に変換済みなら
      `function_result` へ変換する。対応が `function_approval_request` のままなら変換しない。
    """

    def __init__(self, *, reasoning_label: str = "[reasoning]") -> None:
        super().__init__(reasoning_label=reasoning_label)
        self._function_call_ids: set[str] = set()
        self._approval_request_ids: set[str] = set()

    def convert_messages(self, messages: Sequence[Message]) -> list[Message]:
        """事前スキャンで call_id の種別を把握してから変換する。

        `function_approval_response` を `function_result` に変換するかどうかは、
        対応する call が `function_call` か `function_approval_request` かで決まるため、
        変換前に全メッセージをスキャンして call_id を分類する。
        """
        self._function_call_ids, self._approval_request_ids = self._scan_call_ids(
            messages
        )
        return super().convert_messages(messages)

    def _scan_call_ids(
        self, messages: Sequence[Message]
    ) -> tuple[set[str], set[str]]:
        """全メッセージをスキャンし、call_id を function_call 系と approval_request 系に分類する。

        `function_approval_request` のうち:
        - `server_label` なし → `_convert_approval_request` で `function_call` に変換される
          → `function_call_ids` へ
        - `server_label` あり (hosted tool) → そのまま approval_request として残る
          → `approval_request_ids` へ
        """
        function_call_ids: set[str] = set()
        approval_request_ids: set[str] = set()
        for message in messages:
            for content in message.contents:
                if content.type == "function_call":
                    if isinstance(content.call_id, str) and content.call_id:
                        function_call_ids.add(content.call_id)
                elif content.type == "function_approval_request":
                    fc = content.function_call
                    if fc is None or not isinstance(fc.call_id, str) or not fc.call_id:
                        continue
                    ap = fc.additional_properties
                    if isinstance(ap, dict) and ap.get("server_label"):
                        approval_request_ids.add(fc.call_id)
                    else:
                        function_call_ids.add(fc.call_id)
        return function_call_ids, approval_request_ids

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

    def _convert_approval_request(self, content: Content) -> Content | None:
        """approval request に埋め込まれた function_call を取り出す。

        ルール:
        - `function_call` がなければ None（破棄）。
        - `server_label` あり → hosted tool なので approval_request のまま返す。
        - それ以外 → 埋め込みの `function_call` として返す。
        """
        fc = content.function_call
        if fc is None:
            return None

        ap = fc.additional_properties
        if isinstance(ap, dict) and ap.get("server_label"):
            return content

        if not isinstance(fc.call_id, str) or not fc.call_id:
            return None
        if fc.name is None:
            return None

        return Content.from_function_call(
            call_id=fc.call_id,
            name=fc.name,
            arguments=fc.arguments,
            additional_properties=fc.additional_properties,
            raw_representation=fc.raw_representation,
        )

    def _convert_approval_response(self, content: Content) -> Content | None:
        """function_approval_response を必要に応じて function_result へ変換する。

        対応する call が function_call（または approval_request から変換された function_call）
        であれば function_result に変換する。
        対応が approval_request のまま（hosted tool）であればそのまま返す。
        """
        call_id = self._get_approval_call_id(content)
        if call_id and call_id in self._function_call_ids:
            return Content.from_function_result(
                call_id,
                result=getattr(content, "result", None),
                additional_properties=dict(content.additional_properties),
            )
        return content

    def _get_approval_call_id(self, content: Content) -> str | None:
        content_id = getattr(content, "id", None)
        if content_id:
            return str(content_id)
        fc = getattr(content, "function_call", None)
        call_id = getattr(fc, "call_id", None)
        return str(call_id) if call_id else None


__all__ = ["OpenAIMessageConverter"]
