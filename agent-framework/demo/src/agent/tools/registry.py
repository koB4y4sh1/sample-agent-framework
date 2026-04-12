from __future__ import annotations

from typing import Any

from agent_framework import (
    BaseChatClient,
    SupportsCodeInterpreterTool,
    SupportsWebSearchTool,
)
from settings import MCPServerSettings

from .builtins import get_weather
from .mcp import MCPToolFactory


class ToolRegistry:
    def __init__(
        self,
        client: BaseChatClient[Any],
        *,
        mcp_settings: list[MCPServerSettings] | None = None,
    ) -> None:
        self._client = client
        self._mcp_tools = MCPToolFactory(client, settings=mcp_settings)

    def build_tools(self) -> list[Any]:
        tools: list[Any] = [get_weather]
        tools.extend(self._build_hosted_tools())
        tools.extend(self._mcp_tools.build_tools())
        return tools

    def _build_hosted_tools(self) -> list[Any]:
        tools: list[Any] = []

        if isinstance(self._client, SupportsWebSearchTool):
            tools.append(self._client.get_web_search_tool())
        if isinstance(self._client, SupportsCodeInterpreterTool):
            tools.append(self._client.get_code_interpreter_tool())
        # if isinstance(self._client, SupportsFileSearchTool):
        #     tools.append(self._client.get_file_search_tool())

        return tools
