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
from settings import MCPServerSettings, load_mcp_server_settings


class MCPToolFactory:
    def __init__(
        self,
        client: BaseChatClient[Any],
        *,
        settings: list[MCPServerSettings] | None = None,
    ) -> None:
        self._client = client
        self._settings = settings if settings is not None else load_mcp_server_settings()

    def build_tools(self) -> list[Any]:
        return [tool for settings in self._settings if (tool := self._build_tool(settings)) is not None]

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

        kwargs = {
            "name": settings.name,
            "url": settings.url,
            "description": settings.description,
            "approval_mode": settings.approval_mode,
            "allowed_tools": settings.allowed_tools,
            "headers": settings.headers,
            "project_connection_id": settings.project_connection_id,
            "authorization_token": self._authorization_token(settings.headers),
        }
        return self._client.get_mcp_tool(**self._filter_supported_kwargs(self._client.get_mcp_tool, kwargs))

    def _build_local_stdio_tool(self, settings: MCPServerSettings) -> MCPStdioTool:
        if not settings.command:
            raise ValueError(f"Local stdio MCP setting '{settings.name}' requires command.")

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

    def _build_local_streamable_http_tool(self, settings: MCPServerSettings) -> MCPStreamableHTTPTool:
        if not settings.url:
            raise ValueError(f"Local streamable_http MCP setting '{settings.name}' requires url.")

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

    def _filter_supported_kwargs(self, func: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        parameters = signature(func).parameters
        if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return {key: value for key, value in kwargs.items() if value is not None}
        return {key: value for key, value in kwargs.items() if key in parameters and value is not None}

    def _authorization_token(self, headers: Mapping[str, str] | None) -> str | None:
        if not headers:
            return None
        for key, value in headers.items():
            if key.lower() == "authorization":
                return value
        return None

    def _static_header_provider(self, headers: dict[str, str] | None) -> Callable[[dict[str, Any]], dict[str, str]] | None:
        if not headers:
            return None
        return lambda _: dict(headers)
