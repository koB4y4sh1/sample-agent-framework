from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

from agent_framework import Content

ContentSanitizer: TypeAlias = Callable[[Content], Content | None]


class ReplayPayloadSanitizer:
    """Content の payload を、provider に再投入しやすい値へ掃除する helper。

    Message の中には、Python object や raw response など、そのまま JSON 化しにくい値が
    入ることがある。provider に渡す前に dict/list/scalar へ寄せる。
    """

    _DROPPED_CONTENT_ADDITIONAL_PROPERTIES = {"fc_id"}
    _NESTED_PAYLOAD_KEYS = ("items", "inputs", "outputs", "output", "result")
    _JSON_ARGUMENT_CONTENT_TYPES = {
        "function_call",
        "mcp_server_tool_call",
    }

    def __init__(self, sanitize_content: ContentSanitizer) -> None:
        self._sanitize_content = sanitize_content

    def sanitize_content_data(self, data: dict[str, Any]) -> None:
        """Content の dict 表現から、再投入に不要または危険な値を落とす。"""
        additional_properties = data.get("additional_properties")
        if isinstance(additional_properties, Mapping):
            cleaned = {
                key: value
                for key, value in self.sanitize_mapping(additional_properties).items()
                if key not in self._DROPPED_CONTENT_ADDITIONAL_PROPERTIES
            }
            if cleaned:
                data["additional_properties"] = cleaned
            else:
                data.pop("additional_properties", None)

        content_type = data.get("type")
        if content_type in self._JSON_ARGUMENT_CONTENT_TYPES:
            data["arguments"] = self.normalize_arguments(data.get("arguments"))

        function_call = data.get("function_call")
        if isinstance(function_call, Mapping):
            nested_call = dict(function_call)
            self.sanitize_content_data(nested_call)
            data["function_call"] = nested_call

        for key in self._NESTED_PAYLOAD_KEYS:
            if key in data:
                data[key] = self.sanitize_nested_payload(data[key])

    def normalize_arguments(self, arguments: Any) -> Any:
        """tool arguments は provider が扱いやすいように JSON 文字列へ寄せる。"""
        if isinstance(arguments, Mapping):
            return self._to_json(arguments)
        return self.sanitize_value(arguments)

    def sanitize_value(self, value: Any) -> Any:
        """任意の値を、JSON に近い素朴な値へ変換する。"""
        if isinstance(value, Content):
            sanitized = self._sanitize_content(value)
            return sanitized.to_dict(exclude_none=True) if sanitized is not None else None
        if isinstance(value, Mapping):
            sanitized = self.sanitize_mapping(value)
            if "type" in sanitized:
                self.sanitize_content_data(sanitized)
            return sanitized
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self.sanitize_value(item) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)

    def sanitize_nested_payload(self, value: Any) -> Any:
        """tool result 内などにネストした payload を再帰的に掃除する。"""
        if isinstance(value, Content):
            sanitized = self._sanitize_content(value)
            if sanitized is None:
                return None
            return self.sanitize_nested_payload(sanitized.to_dict(exclude_none=True))
        if isinstance(value, Mapping):
            sanitized = self.sanitize_mapping(value)
            if "type" in sanitized:
                self.sanitize_content_data(sanitized)
                sanitized.pop("additional_properties", None)
                sanitized.pop("raw_representation", None)
            return sanitized
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self.sanitize_nested_payload(item) for item in value]
        return self.sanitize_value(value)

    def sanitize_mapping(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """mapping の key を文字列化しつつ、value を安全な形へ変換する。"""
        return {str(key): self.sanitize_value(item) for key, item in value.items()}

    def _to_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
