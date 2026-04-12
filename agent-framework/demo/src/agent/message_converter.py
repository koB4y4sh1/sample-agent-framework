from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

from agent_framework import Content, Message


ReasoningPolicy: TypeAlias = Literal["as_text", "drop"]


class CommonMessageConverter:
    """Convert MAF messages into provider-neutral MAF messages.

    This keeps semantic content such as function calls, function results, and
    approval responses intact. Native reasoning blocks are provider-stateful, so
    they are downgraded to assistant text instead of being replayed as reasoning.
    Provider SDK objects and response-scoped identifiers are removed.
    """

    _DIRECT_ROLES = {"system", "user", "assistant"}
    _ASSISTANT_CONTENT_TYPES = {
        "function_call",
        "function_approval_request",
        "mcp_server_tool_call",
        "code_interpreter_tool_call",
        "image_generation_tool_call",
        "shell_tool_call",
    }
    _TOOL_CONTENT_TYPES = {
        "function_result",
        "mcp_server_tool_result",
        "code_interpreter_tool_result",
        "shell_tool_result",
        "shell_command_output",
    }
    _DROP_CONTENT_TYPES = {"usage", "hosted_vector_store"}
    _DROPPED_CONTENT_ADDITIONAL_PROPERTIES = {"fc_id"}

    def __init__(
        self,
        *,
        reasoning_policy: ReasoningPolicy = "as_text",
        reasoning_label: str = "[reasoning]",
    ) -> None:
        self._reasoning_policy = reasoning_policy
        self._reasoning_label = reasoning_label

    def convert_messages(self, messages: Sequence[Message]) -> list[Message]:
        converted: list[Message] = []
        for message in messages:
            converted.extend(self.convert_message(message))
        return converted

    def convert_message(self, message: Message) -> list[Message]:
        converted: list[Message] = []
        current_role: str | None = None
        current_contents: list[Content] = []

        for content in message.contents:
            sanitized = self._sanitize_content(content)
            if sanitized is None:
                continue

            role = self._content_role(message.role, sanitized)
            if role is None:
                continue

            if current_role is not None and role != current_role:
                converted.append(self._build_message(message, current_role, current_contents))
                current_contents = []

            current_role = role
            current_contents.append(sanitized)

        if current_role is not None and current_contents:
            converted.append(self._build_message(message, current_role, current_contents))

        return converted

    def _build_message(self, original: Message, role: str, contents: list[Content]) -> Message:
        return Message(
            role=role,
            contents=contents,
            author_name=original.author_name if role == original.role else None,
            additional_properties=self._sanitize_mapping(original.additional_properties),
        )

    def _content_role(self, original_role: str, content: Content) -> str | None:
        if content.type in self._ASSISTANT_CONTENT_TYPES:
            return "assistant"
        if content.type in self._TOOL_CONTENT_TYPES:
            return "tool"
        if content.type == "function_approval_response":
            return "user"
        if original_role in self._DIRECT_ROLES:
            return original_role
        if original_role == "tool":
            return "tool" if content.type in self._TOOL_CONTENT_TYPES else "user"
        return "user"

    def _sanitize_content(self, content: Content) -> Content | None:
        if content.type in self._DROP_CONTENT_TYPES:
            return None
        if content.type == "text_reasoning":
            return self._sanitize_reasoning(content)

        data = self._sanitize_mapping(content.to_dict(exclude_none=True))
        data["type"] = content.type
        self._sanitize_content_data(data)

        if data.get("type") == "text" and not data.get("text"):
            return None

        return Content.from_dict(data)

    def _sanitize_reasoning(self, content: Content) -> Content | None:
        if self._reasoning_policy == "drop":
            return None
        if not content.text:
            return None
        if self._reasoning_label:
            return Content.from_text(text=f"{self._reasoning_label}\n{content.text}")
        return Content.from_text(text=content.text)

    def _sanitize_content_data(self, data: dict[str, Any]) -> None:
        additional_properties = data.get("additional_properties")
        if isinstance(additional_properties, Mapping):
            cleaned = {
                key: value
                for key, value in self._sanitize_mapping(additional_properties).items()
                if key not in self._DROPPED_CONTENT_ADDITIONAL_PROPERTIES
            }
            if cleaned:
                data["additional_properties"] = cleaned
            else:
                data.pop("additional_properties", None)

        content_type = data.get("type")
        if content_type in {
            "function_call",
            "mcp_server_tool_call",
        }:
            data["arguments"] = self._normalize_arguments(data.get("arguments"))

        function_call = data.get("function_call")
        if isinstance(function_call, Mapping):
            nested_call = dict(function_call)
            self._sanitize_content_data(nested_call)
            data["function_call"] = nested_call

        for key in ("items", "inputs", "outputs", "output", "result"):
            if key in data:
                data[key] = self._sanitize_value(data[key])

    def _normalize_arguments(self, arguments: Any) -> Any:
        if isinstance(arguments, Mapping):
            return self._to_json(arguments)
        return self._sanitize_value(arguments)

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, Content):
            sanitized = self._sanitize_content(value)
            return sanitized.to_dict(exclude_none=True) if sanitized is not None else None
        if isinstance(value, Mapping):
            return self._sanitize_mapping(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)

    def _sanitize_mapping(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return {str(key): self._sanitize_value(item) for key, item in value.items()}

    def _to_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
