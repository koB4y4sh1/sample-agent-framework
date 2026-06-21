"""MCPツールの作成モジュール。

MCP(Model Context Protocol)は、外部システムの機能をLLMから呼び出すための接続方式。
このファイルの責務は、mcp.jsonの設定からAgent Framework用MCPツールへの変換。

MCPの主な種類:

1. hosted MCP:
   provider側の ``get_mcp_tool`` を使うMCP接続。
2. local MCP:
   ローカルプロセス(stdio)またはHTTP(streamable_http)によるMCP接続。

providerごとにhosted MCPの引数仕様が違うため、
基底クラスとprovider別クラスに分けた構成。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from inspect import Parameter, signature
from typing import Any

from agent_framework import (
    BaseChatClient,
    MCPStdioTool,
    MCPStreamableHTTPTool,
    SupportsMCPTool,
)
from agent_framework_gemini import GeminiChatClient
from settings import MCPServerSettings, load_mcp_server_settings


def _authorization_bearer(headers: Mapping[str, str] | None) -> str | None:
    """HTTPヘッダーからのAuthorization値の抽出。"""

    if not headers:
        return None
    for key, value in headers.items():
        if key.lower() == "authorization":
            return value
    return None


def _kwargs_accepted_by_get_mcp_tool(
    get_mcp_tool: Callable[..., Any],
    candidates: Mapping[str, Any],
) -> dict[str, Any]:
    """``get_mcp_tool`` が受け取れる引数だけへの絞り込み。"""

    parameters = signature(get_mcp_tool).parameters
    if any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values()):
        return {key: value for key, value in candidates.items() if value is not None}
    explicit_names = {
        name
        for name, parameter in parameters.items()
        if parameter.kind not in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
    }
    return {
        key: value
        for key, value in candidates.items()
        if value is not None and key in explicit_names
    }


class BaseMCPToolFactory(ABC):
    """MCPツールFactoryの基底クラス。

    local MCPの作り方はprovider非依存のため、この基底クラスに実装。
    hosted MCPの作り方はprovider差分があるため、サブクラスの責務。
    """

    def __init__(
        self,
        client: BaseChatClient[Any],
        *,
        settings: list[MCPServerSettings] | None = None,
    ) -> None:
        """MCPツール作成に必要なclientと設定の保持。"""

        self._client = client
        self._settings = (
            settings if settings is not None else load_mcp_server_settings()
        )

    def build_tools(self) -> list[Any]:
        """設定一覧からのMCPツール一覧作成。"""

        return [
            tool
            for settings in self._settings
            if (tool := self._build_tool(settings)) is not None
        ]

    def _build_tool(self, settings: MCPServerSettings) -> Any | None:
        """1件のMCP設定に対するhosted/local作成処理への振り分け。"""

        if settings.mode == "hosted":
            return self._build_hosted_tool(settings)
        if settings.transport == "stdio":
            return self._build_local_stdio_tool(settings)
        if settings.transport == "streamable_http":
            return self._build_local_streamable_http_tool(settings)
        raise ValueError(f"Unsupported local MCP transport: {settings.transport}")

    @abstractmethod
    def _build_hosted_tool(self, settings: MCPServerSettings) -> Any | None:
        """hosted MCPツール作成。"""

        raise NotImplementedError

    def _build_local_stdio_tool(self, settings: MCPServerSettings) -> MCPStdioTool:
        """stdio起動のローカルMCPツール作成。"""

        if not settings.command:
            raise ValueError(
                f"Local stdio MCP setting '{settings.name}' requires command."
            )

        return MCPStdioTool(
            name=settings.name,
            command=settings.command,
            args=settings.args,
            env=settings.env,
            cwd=settings.cwd,
            encoding=settings.encoding,
            description=settings.description,
            approval_mode=settings.approval_mode,
            allowed_tools=settings.allowed_tools,
            tool_name_prefix=settings.tool_name_prefix,
            request_timeout=settings.request_timeout,
            client=self._client,
        )

    def _build_local_streamable_http_tool(
        self, settings: MCPServerSettings
    ) -> MCPStreamableHTTPTool:
        """HTTP接続のローカルMCPツール作成。"""

        if not settings.url:
            raise ValueError(
                f"Local streamable_http MCP setting '{settings.name}' requires url."
            )

        return MCPStreamableHTTPTool(
            name=settings.name,
            url=settings.url,
            description=settings.description,
            approval_mode=settings.approval_mode,
            allowed_tools=settings.allowed_tools,
            tool_name_prefix=settings.tool_name_prefix,
            request_timeout=settings.request_timeout,
            terminate_on_close=settings.terminate_on_close,
            header_provider=self._static_header_provider(settings.headers),
            client=self._client,
        )

    def _static_header_provider(
        self, headers: dict[str, str] | None
    ) -> Callable[[dict[str, Any]], dict[str, str]] | None:
        """固定ヘッダーを返す関数の作成。"""

        if not headers:
            return None
        return lambda _: dict(headers)


class DefaultMCPToolFactory(BaseMCPToolFactory):
    """標準的なhosted MCP作成Factory。"""

    def _build_hosted_tool(self, settings: MCPServerSettings) -> Any | None:
        """clientの ``get_mcp_tool`` に合わせたhosted MCPツール作成。"""

        if not isinstance(self._client, SupportsMCPTool):
            return None

        get_mcp = self._client.get_mcp_tool
        candidates = {
            "name": settings.name,
            "url": settings.url,
            "description": settings.description,
            "approval_mode": settings.approval_mode,
            "allowed_tools": settings.allowed_tools,
            "headers": settings.headers,
            "project_connection_id": settings.project_connection_id,
            "authorization_token": _authorization_bearer(settings.headers),
        }

        return get_mcp(**_kwargs_accepted_by_get_mcp_tool(get_mcp, candidates))


class GeminiMCPToolFactory(BaseMCPToolFactory):
    """Gemini向けhosted MCP Factory。"""

    def _build_hosted_tool(self, settings: MCPServerSettings) -> Any | None:
        """Geminiが受け取れる引数だけによるhosted MCPツール作成。"""

        if not isinstance(self._client, SupportsMCPTool):
            return None
        if not settings.url:
            raise ValueError(
                f"MCP setting '{settings.name}' requires url for Gemini hosted MCP."
            )

        args: dict[str, Any] = {"name": settings.name, "url": settings.url}
        if settings.headers is not None:
            args["headers"] = settings.headers
        if settings.request_timeout is not None:
            args["timeout"] = settings.request_timeout
        if settings.terminate_on_close is not None:
            args["terminate_on_close"] = settings.terminate_on_close
        return self._client.get_mcp_tool(**args)


def create_mcp_tool_factory(
    client: BaseChatClient[Any],
    *,
    settings: list[MCPServerSettings] | None = None,
) -> BaseMCPToolFactory:
    """clientの種類に合うMCP Factoryの選択。"""

    if isinstance(client, GeminiChatClient):
        return GeminiMCPToolFactory(client, settings=settings)
    return DefaultMCPToolFactory(client, settings=settings)
