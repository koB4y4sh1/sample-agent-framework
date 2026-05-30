from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_framework import Content, Message


class AnthropicReplayConverter:
    """Convert replay content types that Anthropic does not accept directly."""

    _CODE_EXECUTION_TOOL_NAME = "code_execution"
    _SHELL_TOOL_NAME = "bash"

    def convert_messages(self, messages: Sequence[Message]) -> list[Message]:
        return [self.convert_message(message) for message in messages]

    def convert_message(self, message: Message) -> Message:
        return Message(
            role=message.role,
            contents=[self.convert_content(content) for content in message.contents],
            author_name=message.author_name,
            message_id=message.message_id,
            additional_properties=dict(message.additional_properties),
        )

    def convert_content(self, content: Content) -> Content:
        if content.type == "code_interpreter_tool_call":
            return self._convert_code_interpreter_tool_call(content)
        if content.type == "shell_tool_call":
            return self._convert_shell_tool_call(content)
        if content.type == "code_interpreter_tool_result":
            return self._convert_code_interpreter_tool_result(content)
        if content.type == "shell_tool_result":
            return self._convert_shell_tool_result(content)
        return content

    def _convert_code_interpreter_tool_call(self, content: Content) -> Content:
        call_id = self._call_id(content)
        if call_id is None:
            return content
        return Content.from_function_call(
            call_id,
            self._CODE_EXECUTION_TOOL_NAME,
            arguments={"inputs": self._content_list(getattr(content, "inputs", None))},
            additional_properties=dict(content.additional_properties),
            raw_representation=content.raw_representation,
        )

    def _convert_shell_tool_call(self, content: Content) -> Content:
        call_id = self._call_id(content)
        if call_id is None:
            return content
        arguments: dict[str, Any] = {}
        for key in ("commands", "timeout_ms", "max_output_length", "status"):
            value = getattr(content, key, None)
            if value is not None:
                arguments[key] = list(value) if key == "commands" else value
        return Content.from_function_call(
            call_id,
            self._SHELL_TOOL_NAME,
            arguments=arguments,
            additional_properties=dict(content.additional_properties),
            raw_representation=content.raw_representation,
        )

    def _convert_code_interpreter_tool_result(self, content: Content) -> Content:
        call_id = self._call_id(content)
        if call_id is None:
            return content
        return Content.from_function_result(
            call_id,
            result={"outputs": self._content_list(getattr(content, "outputs", None))},
            additional_properties=dict(content.additional_properties),
            raw_representation=content.raw_representation,
        )

    def _convert_shell_tool_result(self, content: Content) -> Content:
        call_id = self._call_id(content)
        if call_id is None:
            return content
        result: dict[str, Any] = {
            "outputs": self._content_list(getattr(content, "outputs", None)),
        }
        max_output_length = getattr(content, "max_output_length", None)
        if max_output_length is not None:
            result["max_output_length"] = max_output_length
        return Content.from_function_result(
            call_id,
            result=result,
            additional_properties=dict(content.additional_properties),
            raw_representation=content.raw_representation,
        )

    def _call_id(self, content: Content) -> str | None:
        call_id = getattr(content, "call_id", None)
        return call_id if isinstance(call_id, str) and call_id else None

    def _content_list(self, contents: Sequence[Any] | None) -> list[Any]:
        return [self._content_payload(content) for content in contents or []]

    def _content_payload(self, content: Any) -> Any:
        if isinstance(content, Content):
            data = content.to_dict(exclude_none=True)
        elif isinstance(content, Mapping):
            data = dict(content)
        else:
            return content

        if not data.get("additional_properties"):
            data.pop("additional_properties", None)
        data.pop("raw_representation", None)
        return data
