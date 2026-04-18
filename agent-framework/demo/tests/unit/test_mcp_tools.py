from __future__ import annotations

from agent.tools import ToolRegistry
from agent_framework import MCPStdioTool, MCPStreamableHTTPTool
from settings import MCPServerSettings


class FakeHostedMCPClient:
    @staticmethod
    def get_mcp_tool(**kwargs):
        return {"hosted_mcp": kwargs}


class TestMCPTools:
    def test_builds_hosted_mcp_tool_from_client(self) -> None:
        """正常系：hosted MCP設定の場合、client由来のhosted MCP toolが生成されること"""
        tools = ToolRegistry(
            FakeHostedMCPClient(),  # type: ignore[arg-type]
            mcp_settings=[
                MCPServerSettings(
                    name="Hosted",
                    mode="hosted",
                    url="https://example.com/mcp",
                    headers={"Authorization": "Bearer token"},
                )
            ],
        ).build_tools()

        hosted_tool = next(tool for tool in tools if isinstance(tool, dict) and "hosted_mcp" in tool)
        assert hosted_tool["hosted_mcp"]["name"] == "Hosted"
        assert hosted_tool["hosted_mcp"]["url"] == "https://example.com/mcp"
        assert hosted_tool["hosted_mcp"]["headers"] == {"Authorization": "Bearer token"}

    def test_builds_local_stdio_mcp_tool(self) -> None:
        """正常系：local stdio MCP設定の場合、MCPStdioToolが設定値どおり生成されること"""
        tools = ToolRegistry(
            object(),  # type: ignore[arg-type]
            mcp_settings=[
                MCPServerSettings(
                    name="Weather",
                    mode="local",
                    transport="stdio",
                    command="uv",
                    args=["run", "python", "weather.py"],
                    cwd="mcp_server/demo",
                )
            ],
        ).build_tools()

        mcp_tool = next(tool for tool in tools if isinstance(tool, MCPStdioTool))
        assert mcp_tool.name == "Weather"
        assert mcp_tool.command == "uv"
        assert mcp_tool.args == ["run", "python", "weather.py"]
        assert mcp_tool._client_kwargs["cwd"] == "mcp_server/demo"

    def test_builds_local_streamable_http_mcp_tool(self) -> None:
        """正常系：local streamable_http MCP設定の場合、MCPStreamableHTTPToolが設定値どおり生成されること"""
        tools = ToolRegistry(
            object(),  # type: ignore[arg-type]
            mcp_settings=[
                MCPServerSettings(
                    name="Streamable",
                    mode="local",
                    transport="streamable_http",
                    url="http://localhost:8000/mcp",
                    headers={"Authorization": "Bearer token"},
                )
            ],
        ).build_tools()

        mcp_tool = next(tool for tool in tools if isinstance(tool, MCPStreamableHTTPTool))
        assert mcp_tool.name == "Streamable"
        assert mcp_tool.url == "http://localhost:8000/mcp"
