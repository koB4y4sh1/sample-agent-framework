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
    location: Annotated[str, "場所を表す文字列。例: 'Tokyo', 'New York', 'Paris'"],
) -> str:
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"{location} の天気は {conditions[randint(0, 3)]} で、最高気温は {randint(10, 30)} ℃ です。"


class DemoTools:
    """デモ用のツールキット"""

    def __init__(self, client: BaseChatClient[Any]) -> None:
        self._client = client
        self._mcp_settings = load_mcp_server_settings()

    def build_tools(self) -> list[Any]:
        """利用可能なツール
        
        - 天気情報取得ツール (get_weather)
        - 以下、サポートされている場合に利用可能なツールを
            - Web検索ツール
            - コードインタープリターツール
            - ファイル検索ツール
            - ホストMCP
                - settings/mcp.json でツールの設定を管理
        """
        tools: list[Any] = [get_weather]
        tools.extend(self._build_hosted_tools())
        tools.extend(self._build_hosted_mcp_tools())
        return tools

    def _build_hosted_mcp_tools(self) -> list[Any]:
        """ホストMCPツールの設定"""
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
        """組み込みツールの設定"""
        tools: list[Any] = []

        if isinstance(self._client, SupportsWebSearchTool):
            # Web検索
            tools.append(self._client.get_web_search_tool())
        if isinstance(self._client, SupportsCodeInterpreterTool):
            # コードインタープリター
            tools.append(self._client.get_code_interpreter_tool())
        if isinstance(self._client, SupportsFileSearchTool):
            # ファイル検索
            tools.append(self._client.get_file_search_tool())

        return tools
