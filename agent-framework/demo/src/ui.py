from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

from agent_framework import Content
from anthropic.types.beta import BetaWebSearchToolResultBlock
from utils.print import print_color

ProviderFamily = Literal["anthropic", "openai", "gemini"]


class BaseRender(ABC):
    """ストリーミング出力の差分を provider ごとに吸収する抽象基底クラス。"""

    def __init__(self) -> None:
        self._current_output_type: str | None = None
        self._has_output = False

    def start(self) -> None:
        """1 回の assistant 応答表示を開始する。"""
        self._current_output_type = None
        self._has_output = False

    @abstractmethod
    def render(self, contents: Sequence[Content], text: str | None = None) -> None:
        """描画に必要な contents と text を受け取り出力する。"""

    def finish(self) -> None:
        """1 回の assistant 応答表示を終了する。"""
        print()


    def _start_output_block(
        self,
        *,
        next_output_type: str,
        label: str | None = None,
        color_printer: Any | None = None,
    ) -> None:
        if self._has_output and self._current_output_type != next_output_type:
            print()
        if self._current_output_type != next_output_type and label and color_printer is not None:
            color_printer(f"{label} ", end="", flush=True)
        self._current_output_type = next_output_type
        self._has_output = True

    def _print_assistant(self, *values: Any, **kwargs: Any) -> None:
        print_color(*values, color="bright_white", styles=("bold",), **kwargs)

    def _print_reasoning(self, *values: Any, **kwargs: Any) -> None:
        print_color(*values, color="bright_black", **kwargs)

    def _print_tool_call(self, *values: Any, **kwargs: Any) -> None:
        print_color(*values, color="bright_magenta", styles=("bold",), **kwargs)

    def _print_tool_result(self, *values: Any, **kwargs: Any) -> None:
        print_color(*values, color="bright_green", **kwargs)

    def _print_mcp_call(self, *values: Any, **kwargs: Any) -> None:
        print_color(*values, color="bright_cyan", styles=("bold",), **kwargs)

    def _print_mcp_result(self, *values: Any, **kwargs: Any) -> None:
        print_color(*values, color="blue", **kwargs)

    def _print_usage(self, *values: Any, **kwargs: Any) -> None:
        print_color(*values, color="bright_yellow", styles=("bold",), **kwargs)


class AnthropicRender(BaseRender):
    """Anthropic 系クライアント向けのストリーミング描画実装。"""

    def render(self, contents: Sequence[Content], text: str | None = None) -> None:
        for content in contents:
            if content.type == "text_reasoning" and content.text:
                self._start_output_block(
                    next_output_type="text_reasoning",
                    label="[Reasoning]",
                    color_printer=self._print_reasoning,
                )
                self._print_reasoning(content.text, end="", flush=True)
            elif content.type == "function_call":
                self._start_output_block(next_output_type="function_call")
                self._print_tool_call(f"[Tool Call] name: {content.name}, arguments: {content.arguments}")
            elif content.type == "function_result":
                self._start_output_block(next_output_type="function_result")
                if isinstance(content.raw_representation, BetaWebSearchToolResultBlock):
                    # web_search
                    for result in content.raw_representation.content:
                        self._print_tool_result(f"[Tool Result] title: {result.title} url: {result.url}, page_age: {result.page_age}")
                else:
                    self._print_tool_result(f"[Tool Result] {content.result}")
            elif content.type == "mcp_server_tool_call":
                self._start_output_block(next_output_type="mcp_server_tool_call")
                self._print_mcp_call(f"[MCP Call] name: {content.tool_name}, arguments: {content.arguments}")
            elif content.type == "mcp_server_tool_result":
                self._start_output_block(next_output_type="mcp_server_tool_result")
                self._print_mcp_result(f"[MCP Result] {content.text}")
            elif content.type == "usage" and content.usage_details:
                self._start_output_block(next_output_type="usage")
                input_tokens = content.usage_details.get("input_token_count")
                output_tokens = content.usage_details.get("output_token_count")
                input_cache = content.usage_details.get("anthropic.cache_creation_input_tokens")
                output_cache = content.usage_details.get("anthropic.cache_read_input_tokens")
                self._print_usage(f"[Usage] input: {input_tokens}, output: {output_tokens}, input_cache: {input_cache}, output_cache: {output_cache}")

        if text:
            self._start_output_block(
                next_output_type="text",
                label="[Answer]",
                color_printer=self._print_assistant,
            )
            self._print_assistant(text, end="", flush=True)


class OpenAIRender(BaseRender):
    """OpenAI 系クライアント向けの最小ストリーミング描画実装。"""

    def render(self, contents: Sequence[Content], text: str | None = None) -> None:
        for content in contents:
            if content.type in {"reasoning", "text_reasoning"} and getattr(content, "text", None):
                self._start_output_block(
                    next_output_type="text_reasoning",
                    label="[Reasoning]",
                    color_printer=self._print_reasoning,
                )
                self._print_reasoning(content.text, end="", flush=True)
            elif content.type == "function_call":
                self._start_output_block(next_output_type="function_call")
                self._print_tool_call(f"\n[Tool Call] {content.name} {content.arguments}", end="", flush=True)
            elif content.type == "function_result":
                self._start_output_block(next_output_type="function_result")
                self._print_tool_result(f"\n[Tool Result] {content.result}", end="", flush=True)
            elif content.type == "usage" and getattr(content, "usage_details", None):
                self._start_output_block(next_output_type="usage")
                self._print_usage(f"\n[Usage] {content.usage_details}", end="", flush=True)

        if text:
            self._start_output_block(
                next_output_type="text",
                label="[Answer]",
                color_printer=self._print_assistant,
            )
            self._print_assistant(text, end="", flush=True)


class GeminiRender(BaseRender):
    """Gemini 系クライアント向けの最小ストリーミング描画実装。"""

    def render(self, contents: Sequence[Content], text: str | None = None) -> None:
        for content in contents:
            if content.type in {"thought", "text_reasoning"} and getattr(content, "text", None):
                self._start_output_block(
                    next_output_type="text_reasoning",
                    label="[Reasoning]",
                    color_printer=self._print_reasoning,
                )
                self._print_reasoning(content.text, end="", flush=True)
            elif content.type == "function_call":
                self._start_output_block(next_output_type="function_call")
                self._print_tool_call(f"\n[Tool Call] {content.name} {content.arguments}", end="", flush=True)
            elif content.type == "function_result":
                self._start_output_block(next_output_type="function_result")
                self._print_tool_result(f"\n[Tool Result] {content.result}", end="", flush=True)
            elif content.type == "usage" and getattr(content, "usage_details", None):
                self._start_output_block(next_output_type="usage")
                self._print_usage(f"\n[Usage] {content.usage_details}", end="", flush=True)

        if text:
            self._start_output_block(
                next_output_type="text",
                label="[Answer]",
                color_printer=self._print_assistant,
            )
            self._print_assistant(text, end="", flush=True)


class UIResolver:
    """provider 種別に応じた stream renderer を解決する。"""

    def __init__(self, provider_family: ProviderFamily) -> None:
        self._provider_family = provider_family

    def resolve(self) -> BaseRender:
        if self._provider_family == "anthropic":
            return AnthropicRender()
        if self._provider_family == "openai":
            return OpenAIRender()
        if self._provider_family == "gemini":
            return GeminiRender()
        raise ValueError(f"Unsupported provider family: {self._provider_family}")
