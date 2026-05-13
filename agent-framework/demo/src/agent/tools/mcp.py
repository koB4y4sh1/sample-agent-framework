from __future__ import annotations

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
    """``get_mcp_tool`` に渡してよいキーワード引数だけを返す。

    hosted MCP 用の ``get_mcp_tool`` はクライアント実装ごとに引数名が違う。
    デモではまず ``candidates`` に「あり得るキー」をすべて入れ、この関数で
    実際のシグネチャに合わせて間引く。

    - シグネチャに ``**kwargs`` がある場合:
      先側が余分なキーを解釈する前提のため、値が ``None`` でないものはすべて残す
      （テスト用モックなど）。
    - そうでない場合:
      名前付きパラメータに存在するキーだけ残す。未知のキーを渡すと
      ``TypeError`` になる実装を避けるため。
    """
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


class MCPToolFactory:
    """mcp.json の設定から MCP 関連ツールを組み立てる。"""

    def __init__(
        self,
        client: BaseChatClient[Any],
        *,
        settings: list[MCPServerSettings] | None = None,
    ) -> None:
        self._client = client
        self._settings = (
            settings if settings is not None else load_mcp_server_settings()
        )

    def build_tools(self) -> list[Any]:
        return [
            tool
            for settings in self._settings
            if (tool := self._build_tool(settings)) is not None
        ]

    def _build_tool(self, settings: MCPServerSettings) -> Any | None:
        if settings.mode == "hosted":
            return self._build_hosted_tool(settings)
        if settings.transport == "stdio":
            return self._build_local_stdio_tool(settings)
        if settings.transport == "streamable_http":
            return self._build_local_streamable_http_tool(settings)
        raise ValueError(f"Unsupported local MCP transport: {settings.transport}")

    def _build_hosted_tool(self, settings: MCPServerSettings) -> Any | None:
        if not isinstance(self._client, SupportsMCPTool):
            return None

        get_mcp = self._client.get_mcp_tool

        if isinstance(self._client, GeminiChatClient):
            # Gemini の get_mcp_tool は、name/url 以外をそのまま HTTP 接続オプションに渡す。
            # approval_mode などを混ぜると、Google 側の型検証で落ちる。
            if not settings.url:
                raise ValueError(
                    f"MCP「{settings.name}」: Gemini の hosted では url が必要です。"
                )
            args: dict[str, Any] = {"name": settings.name, "url": settings.url}
            if settings.headers is not None:
                args["headers"] = settings.headers
            if settings.request_timeout is not None:
                args["timeout"] = settings.request_timeout
            if settings.terminate_on_close is not None:
                args["terminate_on_close"] = settings.terminate_on_close
            return get_mcp(**args)

        # それ以外のクライアント: get_mcp_tool の引数名に合わせて渡す
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

    def _build_local_stdio_tool(self, settings: MCPServerSettings) -> MCPStdioTool:
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
        if not headers:
            return None
        return lambda _: dict(headers)
