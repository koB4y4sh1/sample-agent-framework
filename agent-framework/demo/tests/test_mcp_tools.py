from __future__ import annotations

import unittest

from agent.tools import ToolRegistry
from agent_framework import MCPStdioTool, MCPStreamableHTTPTool
from settings import MCPServerSettings


class FakeHostedMCPClient:
    @staticmethod
    def get_mcp_tool(**kwargs):
        return {"hosted_mcp": kwargs}


class MCPToolsTests(unittest.TestCase):
    def test_builds_hosted_mcp_tool_from_client(self) -> None:
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
        self.assertEqual(hosted_tool["hosted_mcp"]["name"], "Hosted")
        self.assertEqual(hosted_tool["hosted_mcp"]["url"], "https://example.com/mcp")
        self.assertEqual(hosted_tool["hosted_mcp"]["headers"], {"Authorization": "Bearer token"})

    def test_builds_local_stdio_mcp_tool(self) -> None:
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
        self.assertEqual(mcp_tool.name, "Weather")
        self.assertEqual(mcp_tool.command, "uv")
        self.assertEqual(mcp_tool.args, ["run", "python", "weather.py"])
        self.assertEqual(mcp_tool._client_kwargs["cwd"], "mcp_server/demo")

    def test_builds_local_streamable_http_mcp_tool(self) -> None:
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
        self.assertEqual(mcp_tool.name, "Streamable")
        self.assertEqual(mcp_tool.url, "http://localhost:8000/mcp")


if __name__ == "__main__":
    unittest.main()
