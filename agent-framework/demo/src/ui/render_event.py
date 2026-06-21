from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agent_framework import Content
from anthropic.types.beta import (
    BetaWebSearchToolResultBlock,
    BetaWebSearchToolResultError,
)

ProviderFamily = Literal["anthropic", "openai", "google"]
RenderEventKind = Literal[
    "text",
    "reasoning",
    "tool_call",
    "tool_result",
    "mcp_call",
    "mcp_result",
    "usage",
    "data",
    "approval_request",
    "unknown",
]


@dataclass(slots=True)
class RenderEvent:
    kind: RenderEventKind
    text: str
    content_type: str | None = None
    payload: Any | None = None


class BaseRender(ABC):
    """Convert provider stream contents into UI-neutral render events."""

    @abstractmethod
    def render(self, contents: Sequence[Content], text: str | None = None) -> list[RenderEvent]:
        """Return UI-neutral events for the supplied stream update."""

    def render_content(self, content: Content) -> list[RenderEvent]:
        return self.render([content])

    def _common_event(self, content: Content) -> RenderEvent | None:
        content_type = str(content.type)
        if content_type == "data":
            return RenderEvent(
                kind="data",
                text="",
                content_type=content_type,
                payload=content,
            )
        if content_type == "function_approval_request":
            return RenderEvent(
                kind="approval_request",
                text="",
                content_type=content_type,
                payload=content,
            )
        return None

    def _unknown_event(self, content: Content) -> RenderEvent:
        return RenderEvent(
            kind="unknown",
            text=getattr(content, "text", None) or "",
            content_type=str(content.type),
            payload=content,
        )


class AnthropicRender(BaseRender):
    def render(self, contents: Sequence[Content], text: str | None = None) -> list[RenderEvent]:
        events: list[RenderEvent] = []
        for content in contents:
            if common_event := self._common_event(content):
                events.append(common_event)
            elif content.type == "text_reasoning" and (text_value := content.text):
                events.append(
                    RenderEvent(
                        kind="reasoning",
                        text=text_value,
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "function_call":
                events.append(
                    RenderEvent(
                        kind="tool_call",
                        text=f"name: {content.name}, arguments: {content.arguments}",
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "function_result":
                events.extend(self._anthropic_tool_result_events(content))
            elif content.type == "mcp_server_tool_call":
                events.append(
                    RenderEvent(
                        kind="mcp_call",
                        text=f"name: {content.tool_name}, arguments: {content.arguments}",
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "mcp_server_tool_result":
                events.append(
                    RenderEvent(
                        kind="mcp_result",
                        text=str(content.text),
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "usage" and content.usage_details:
                details = content.usage_details
                events.append(
                    RenderEvent(
                        kind="usage",
                        text=(
                            f"input: {details.get('input_token_count')}, "
                            f"output: {details.get('output_token_count')}, "
                            f"input_cache: {details.get('anthropic.cache_creation_input_tokens')}, "
                            f"output_cache: {details.get('anthropic.cache_read_input_tokens')}"
                        ),
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif text_value := getattr(content, "text", None):
                events.append(
                    RenderEvent(
                        kind="text",
                        text=text_value,
                        content_type=content.type,
                        payload=content,
                    )
                )
            else:
                events.append(self._unknown_event(content))

        if text:
            events.append(RenderEvent(kind="text", text=text))
        return events

    def _anthropic_tool_result_events(self, content: Content) -> list[RenderEvent]:
        if isinstance(content.raw_representation, BetaWebSearchToolResultBlock):
            web_search_result = content.raw_representation.content
            if isinstance(web_search_result, BetaWebSearchToolResultError):
                return [
                    RenderEvent(
                        kind="tool_result",
                        text=f"error code: {web_search_result.error_code}",
                        content_type=content.type,
                        payload=content,
                    )
                ]
            return [
                RenderEvent(
                    kind="tool_result",
                    text=f"title: {result.title} url: {result.url}, page_age: {result.page_age}",
                    content_type=content.type,
                    payload=content,
                )
                for result in web_search_result
            ]
        return [
            RenderEvent(
                kind="tool_result",
                text=str(content.result),
                content_type=content.type,
                payload=content,
            )
        ]


class OpenAIRender(BaseRender):
    def render(self, contents: Sequence[Content], text: str | None = None) -> list[RenderEvent]:
        events: list[RenderEvent] = []
        for content in contents:
            if common_event := self._common_event(content):
                events.append(common_event)
            elif content.type in {"reasoning", "text_reasoning"} and (
                text_value := getattr(content, "text", None)
            ):
                events.append(
                    RenderEvent(
                        kind="reasoning",
                        text=text_value,
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "function_call":
                events.append(
                    RenderEvent(
                        kind="tool_call",
                        text=f"{content.name} {content.arguments}",
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "function_result":
                events.append(
                    RenderEvent(
                        kind="tool_result",
                        text=str(content.result),
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "usage" and getattr(content, "usage_details", None):
                events.append(
                    RenderEvent(
                        kind="usage",
                        text=str(content.usage_details),
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif text_value := getattr(content, "text", None):
                events.append(
                    RenderEvent(
                        kind="text",
                        text=text_value,
                        content_type=content.type,
                        payload=content,
                    )
                )
            else:
                events.append(self._unknown_event(content))

        if text:
            events.append(RenderEvent(kind="text", text=text))
        return events


class GeminiRender(BaseRender):
    def render(self, contents: Sequence[Content], text: str | None = None) -> list[RenderEvent]:
        events: list[RenderEvent] = []
        for content in contents:
            if common_event := self._common_event(content):
                events.append(common_event)
            elif content.type in {"thought", "text_reasoning"} and (
                text_value := getattr(content, "text", None)
            ):
                events.append(
                    RenderEvent(
                        kind="reasoning",
                        text=text_value,
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "function_call":
                events.append(
                    RenderEvent(
                        kind="tool_call",
                        text=f"{content.name} {content.arguments}",
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "function_result":
                events.append(
                    RenderEvent(
                        kind="tool_result",
                        text=str(content.result),
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif content.type == "usage" and getattr(content, "usage_details", None):
                events.append(
                    RenderEvent(
                        kind="usage",
                        text=str(content.usage_details),
                        content_type=content.type,
                        payload=content,
                    )
                )
            elif text_value := getattr(content, "text", None):
                events.append(
                    RenderEvent(
                        kind="text",
                        text=text_value,
                        content_type=content.type,
                        payload=content,
                    )
                )
            else:
                events.append(self._unknown_event(content))

        if text:
            events.append(RenderEvent(kind="text", text=text))
        return events


class UIResolver:
    def __init__(self, provider_family: ProviderFamily) -> None:
        self._provider_family = provider_family

    def resolve(self) -> BaseRender:
        if self._provider_family == "anthropic":
            return AnthropicRender()
        if self._provider_family == "openai":
            return OpenAIRender()
        if self._provider_family == "google":
            return GeminiRender()
        raise ValueError(f"Unsupported provider family: {self._provider_family}")
