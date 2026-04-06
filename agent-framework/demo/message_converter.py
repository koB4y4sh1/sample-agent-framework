from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar, cast

from agent_framework import Content, Message
from anthropic.types import MessageParam

MessageParamT = TypeVar("MessageParamT")


class MessageConverter(ABC, Generic[MessageParamT]):
    @abstractmethod
    def convert_message(self, message: Message) -> MessageParamT:
        """Framework の Message をプロバイダ固有のメッセージ形式へ変換する。"""

    def convert_messages(self, messages: Message | Sequence[Message]) -> list[MessageParamT]:
        if isinstance(messages, Message):
            return [self.convert_message(messages)]
        return [self.convert_message(message) for message in messages]

    @staticmethod
    def extract_base64_data(content: Content) -> str:
        if not content.uri or ";base64," not in content.uri:
            raise ValueError("Base64 data URI is required for data content.")
        return content.uri.split(";base64,", 1)[1]


class AnthropicMessageConverter(MessageConverter[MessageParam]):
    def convert_messages(self, messages: Message | Sequence[Message]) -> list[MessageParam]:
        if isinstance(messages, Message):
            return self._normalize_message_params(self._convert_message_params(messages))

        converted_messages: list[MessageParam] = []
        for message in messages:
            converted_messages.extend(self._convert_message_params(message))
        return self._normalize_message_params(converted_messages)

    def convert_message(self, message: Message) -> MessageParam:
        converted_messages = self._convert_message_params(message)
        if not converted_messages:
            return cast(MessageParam, {"role": "user", "content": []})
        return converted_messages[0]

    def _convert_message_params(self, message: Message) -> list[MessageParam]:
        if message.role != "assistant":
            return [self._convert_message_for_role(message, message.role)]

        message_params: list[MessageParam] = []
        current_role = "assistant"
        current_contents: list[Content] = []

        for content in message.contents:
            target_role = self._resolve_role_for_content(content.type)
            if current_contents and target_role != current_role:
                message_params.append(
                    self._convert_message_for_role(
                        Message(role=current_role, contents=list(current_contents)),
                        current_role,
                    )
                )
                current_contents = []
            current_role = target_role
            current_contents.append(content)

        if current_contents:
            message_params.append(
                self._convert_message_for_role(
                    Message(role=current_role, contents=list(current_contents)),
                    current_role,
                )
            )
        return [message_param for message_param in message_params if message_param["content"]]

    def _normalize_message_params(self, messages: Sequence[MessageParam]) -> list[MessageParam]:
        normalized_messages: list[MessageParam] = []
        previous_tool_use_ids: set[str] = set()

        for message in messages:
            role = message["role"]
            contents = message["content"]
            if not isinstance(contents, list):
                normalized_messages.append(message)
                previous_tool_use_ids = set()
                continue

            normalized_contents: list[dict[str, Any]] = []
            next_tool_use_ids: set[str] = set()

            for content in contents:
                content_type = content.get("type")
                if content_type == "tool_use":
                    tool_use_id = content.get("id")
                    if isinstance(tool_use_id, str):
                        next_tool_use_ids.add(tool_use_id)
                    normalized_contents.append(content)
                    continue
                if content_type == "tool_result":
                    tool_use_id = content.get("tool_use_id")
                    if isinstance(tool_use_id, str) and tool_use_id in previous_tool_use_ids:
                        normalized_contents.append(content)
                    continue
                normalized_contents.append(content)

            if normalized_contents:
                normalized_messages.append(
                    cast(
                        MessageParam,
                        {
                            "role": role,
                            "content": normalized_contents,
                        },
                    )
                )

            if role == "assistant":
                previous_tool_use_ids = next_tool_use_ids
            else:
                previous_tool_use_ids = set()

        return normalized_messages

    def _resolve_role_for_content(self, content_type: str) -> str:
        if content_type in {"function_result", "mcp_server_tool_result"}:
            return "user"
        return "assistant"

    def _convert_message_for_role(self, message: Message, role: str) -> MessageParam:
        anthropic_contents: list[dict[str, Any]] = []

        for content in message.contents:
            match content.type:
                case "text":
                    if content.text:
                        anthropic_contents.append({"type": "text", "text": content.text})
                case "data":
                    if content.has_top_level_media_type("image"):
                        anthropic_contents.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content.media_type,
                                "data": self.extract_base64_data(content),
                            },
                        })
                    elif content.media_type == "application/pdf":
                        anthropic_contents.append({
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": self.extract_base64_data(content),
                            },
                        })
                case "uri":
                    if content.has_top_level_media_type("image"):
                        anthropic_contents.append({
                            "type": "image",
                            "source": {"type": "url", "url": content.uri},
                        })
                    elif content.media_type == "application/pdf":
                        anthropic_contents.append({
                            "type": "document",
                            "source": {"type": "url", "url": content.uri},
                        })
                case "hosted_file":
                    if content.media_type == "application/pdf" and content.file_id:
                        anthropic_contents.append({
                            "type": "document",
                            "source": {"type": "file", "file_id": content.file_id},
                        })
                case "function_call":
                    anthropic_contents.append({
                        "type": "tool_use",
                        "id": content.call_id,
                        "name": content.name,
                        "input": content.parse_arguments() or {},
                    })
                case "function_result":
                    tool_result_content: list[dict[str, Any]] = []
                    for item in content.items or []:
                        if item.type == "text":
                            tool_result_content.append({"type": "text", "text": item.text or ""})
                        elif item.type == "data" and item.has_top_level_media_type("image"):
                            tool_result_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.media_type,
                                    "data": self.extract_base64_data(item),
                                },
                            })
                        elif item.type == "uri" and item.has_top_level_media_type("image"):
                            tool_result_content.append({
                                "type": "image",
                                "source": {"type": "url", "url": item.uri},
                            })
                    anthropic_contents.append({
                        "type": "tool_result",
                        "tool_use_id": content.call_id,
                        "content": tool_result_content or (content.result if content.result is not None else ""),
                        "is_error": content.exception is not None,
                    })
                case "mcp_server_tool_call":
                    anthropic_contents.append({
                        "type": "text",
                        "text": self._format_mcp_tool_call(content),
                    })
                case "mcp_server_tool_result":
                    anthropic_contents.append({
                        "type": "text",
                        "text": self._format_mcp_tool_result(content),
                    })
                case "text_reasoning":
                    if content.text and content.protected_data:
                        anthropic_contents.append({
                            "type": "thinking",
                            "thinking": content.text,
                            "signature": content.protected_data,
                        })

        return cast(
            MessageParam,
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": anthropic_contents,
            },
        )

    def _format_mcp_tool_call(self, content: Content) -> str:
        arguments = content.parse_arguments() or {}
        return (
            f"[MCP Tool Call] server={content.server_name or ''} "
            f"tool={content.tool_name or ''} arguments={arguments}"
        )

    def _format_mcp_tool_result(self, content: Content) -> str:
        output_texts: list[str] = []
        for item in content.output or []:
            if item.type == "text" and item.text:
                output_texts.append(item.text)

        if output_texts:
            output = "\n".join(output_texts)
        else:
            output = str(content.output) if content.output is not None else ""

        return f"[MCP Tool Result] {output}"
