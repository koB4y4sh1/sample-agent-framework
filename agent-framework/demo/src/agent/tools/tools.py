"""ツール全体の入口モジュール。

このファイルの役割は、次の3種類のツールの組み立て。

1. function tools: Python関数をLLMから呼べるようにしたツール
2. builtin tools: Web検索やCode Interpreterなど、LLM/providerの組み込みツール
3. MCP tools: mcp.jsonの設定から作る外部MCPツール

Progressive Tool Exposureは、実ツールを最初から全部公開せず、
まずローダーツールだけを公開し、ローダーツール内で
``FunctionInvocationContext.add_tools`` により実ツールを追加する方式。
"""

from __future__ import annotations

from typing import Any

from agent_framework import BaseChatClient
from settings import MCPServerSettings

from .builtin import build_builtin_tools
from .function_tools import build_function_tools, build_progressive_loader_tools
from .mcp import BaseMCPToolFactory, create_mcp_tool_factory


class ToolRegistry:
    """このアプリで使うツール一覧

    呼び出し側は、function tool / builtin tool / MCP tool の違いを意識せず、
    ``build_tools`` または ``build_tools_for_message`` を呼び出す構成。
    """

    def __init__(
        self,
        client: BaseChatClient[Any],
        *,
        mcp_settings: list[MCPServerSettings] | None = None,
    ) -> None:
        """コンストラクタ

        client:
            LLM providerのクライアント
        mcp_settings:
            MCPツールの接続設定
        """

        self._client = client
        self._mcp_tools: BaseMCPToolFactory = create_mcp_tool_factory(
            client,
            settings=mcp_settings,
        )

    def build_tools(self) -> list[Any]:
        """利用可能な全ツール一覧

        利用可能なツールをすべて返す。
        全ツールのツール定義がコンテキストに入るため、トークン増加やツール説明の肥大化になりやすい。
        その場合は、``build_progressive_tools`` を使用して遅延ロードする。
        """

        tools: list[Any] = []

        # 1. 関数ツールの追加
        tools.extend(build_function_tools())

        # 2. 組み込みツールの追加
        tools.extend(build_builtin_tools(self._client))

        # 3. MCPツールの追加
        tools.extend(self._mcp_tools.build_tools())
        return tools

    def build_progressive_tools(self) -> list[Any]:
        """Progressive Tool Exposure 初期公開ツール一覧

        build_progressive_loader_toolsには**ツールを遅延ロードする**ツールが含まれる。
        段階公開することで社内 Tool 増加時のツール選択精度や、権限管理の複雑さを緩和できる。
        """

        tools: list[Any] = []

        # 1. ローダーツールの追加
        tools.extend(build_progressive_loader_tools())

        # 2. 組み込みツールの追加
        tools.extend(build_builtin_tools(self._client))

        # 3. MCPツールの追加
        tools.extend(self._mcp_tools.build_tools())
        return tools
