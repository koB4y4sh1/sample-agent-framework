from __future__ import annotations

from random import randint
from typing import Annotated, Any

from agent_framework import (
    BaseChatClient,
    SupportsCodeInterpreterTool,
    SupportsFileSearchTool,
    SupportsMCPTool,
    SupportsWebSearchTool,
    tool,
)
from settings import load_mcp_server_settings


@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, "Weather target city such as Tokyo, New York, Paris"],
) -> str:
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"{location} weather is {conditions[randint(0, 3)]}, approx. {randint(10, 30)}C."


class DemoTools:
    def __init__(self, client: BaseChatClient[Any]) -> None:
        self._client = client
        self._mcp_settings = load_mcp_server_settings()

    def build_tools(self) -> list[Any]:
        tools: list[Any] = [get_weather]
        tools.extend(self._build_hosted_tools())
        tools.extend(self._build_hosted_mcp_tools())
        return tools

    def _build_hosted_mcp_tools(self) -> list[Any]:
        if not isinstance(self._client, SupportsMCPTool):
            return []

        return [
            self._client.get_mcp_tool(
                name=settings.name,
                url=settings.url,
            )
            for settings in self._mcp_settings
        ]

    def _build_hosted_tools(self) -> list[Any]:
        tools: list[Any] = []

        if isinstance(self._client, SupportsWebSearchTool):
            tools.append(self._client.get_web_search_tool())
        if isinstance(self._client, SupportsCodeInterpreterTool):
            tools.append(self._client.get_code_interpreter_tool())
        # if isinstance(self._client, SupportsFileSearchTool):
        #     tools.append(self._client.get_file_search_tool())

        return tools
