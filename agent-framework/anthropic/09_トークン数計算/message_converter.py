from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeAlias, TypeVar, cast

from agent_framework import Content, Message
from anthropic.types import MessageParam

MessageParamT = TypeVar("MessageParamT")
OpenAIResponseInputItem: TypeAlias = dict[str, Any]
GeminiContent: TypeAlias = dict[str, Any]


class MessageConverter(ABC, Generic[MessageParamT]):
    @abstractmethod
    def convert_message(self, message: Message) -> MessageParamT:
        pass

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
    def convert_message(self, message: Message) -> MessageParam:
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
                        "type": "mcp_tool_use",
                        "id": content.call_id,
                        "name": content.tool_name,
                        "server_name": content.server_name or "",
                        "input": content.parse_arguments() or {},
                    })
                case "mcp_server_tool_result":
                    anthropic_contents.append({
                        "type": "mcp_tool_result",
                        "tool_use_id": content.call_id,
                        "content": content.output if content.output is not None else "",
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
                "role": "assistant" if message.role == "assistant" else "user",
                "content": anthropic_contents,
            },
        )


class OpenAIMessageConverter(MessageConverter[OpenAIResponseInputItem]):
    def convert_message(self, message: Message) -> OpenAIResponseInputItem:
        openai_contents: list[dict[str, Any]] = []

        for content in message.contents:
            match content.type:
                case "text":
                    if content.text:
                        openai_contents.append({"type": "input_text", "text": content.text})
                case "data":
                    if content.has_top_level_media_type("image"):
                        openai_contents.append({
                            "type": "input_image",
                            "image_url": content.uri,
                        })
                    elif content.media_type == "application/pdf":
                        openai_contents.append({
                            "type": "input_file",
                            "filename": "document.pdf",
                            "file_data": content.uri,
                        })
                case "uri":
                    if content.has_top_level_media_type("image"):
                        openai_contents.append({
                            "type": "input_image",
                            "image_url": content.uri,
                        })
                    elif content.media_type == "application/pdf":
                        openai_contents.append({
                            "type": "input_file",
                            "filename": "document.pdf",
                            "file_url": content.uri,
                        })
                case "hosted_file":
                    if content.file_id:
                        openai_contents.append({
                            "type": "input_file",
                            "file_id": content.file_id,
                        })

        return {
            "role": "assistant" if message.role == "assistant" else "user",
            "content": openai_contents or "",
        }


class GeminiMessageConverter(MessageConverter[GeminiContent]):
    def convert_message(self, message: Message) -> GeminiContent:
        gemini_parts: list[dict[str, Any]] = []

        for content in message.contents:
            match content.type:
                case "text":
                    if content.text:
                        gemini_parts.append({"text": content.text})
                case "data":
                    if content.media_type:
                        gemini_parts.append({
                            "inline_data": {
                                "mime_type": content.media_type,
                                "data": self.extract_base64_data(content),
                            }
                        })
                case "uri":
                    if content.media_type and content.uri:
                        gemini_parts.append({
                            "file_data": {
                                "mime_type": content.media_type,
                                "file_uri": content.uri,
                            }
                        })

        return {
            "role": "model" if message.role == "assistant" else "user",
            "parts": gemini_parts,
        }
