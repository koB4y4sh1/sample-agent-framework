from __future__ import annotations

import asyncio

from agent.tools.function_tools import load_application_tools, load_document_search_tools
from agent.tools import ToolRegistry
from agent_framework import FunctionInvocationContext, MCPStdioTool, MCPStreamableHTTPTool
from settings import MCPServerSettings


def _tool_names(tools):
    return [tool.name for tool in tools]


class FakeHostedMCPClient:
    @staticmethod
    def get_mcp_tool(**kwargs):
        return {"hosted_mcp": kwargs}


class TestMCPTools:
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

        hosted_tool = next(
            tool for tool in tools if isinstance(tool, dict) and "hosted_mcp" in tool
        )
        assert hosted_tool["hosted_mcp"]["name"] == "Hosted"
        assert hosted_tool["hosted_mcp"]["url"] == "https://example.com/mcp"
        assert hosted_tool["hosted_mcp"]["headers"] == {"Authorization": "Bearer token"}

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
        assert mcp_tool.name == "Weather"
        assert mcp_tool.command == "uv"
        assert mcp_tool.args == ["run", "python", "weather.py"]
        assert mcp_tool._client_kwargs["cwd"] == "mcp_server/demo"

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
        assert mcp_tool.name == "Streamable"
        assert mcp_tool.url == "http://localhost:8000/mcp"


class TestProgressiveToolExposure:
    def test_builds_progressive_initial_tools(self) -> None:
        tools = ToolRegistry(object()).build_progressive_tools()

        assert _tool_names(tools) == [
            "load_document_search_tools",
            "load_application_tools",
            "get_weather",
        ]

    def test_document_loader_adds_real_tools_to_invocation_context(self) -> None:
        async def run() -> list[str]:
            tools = [load_document_search_tools]
            context = FunctionInvocationContext(
                function=load_document_search_tools,
                arguments={},
                tools=tools,
            )
            await load_document_search_tools.invoke(arguments={}, context=context)
            return _tool_names(tools)

        assert asyncio.run(run()) == [
            "load_document_search_tools",
            "search_internal_documents",
            "search_faq",
        ]

    def test_application_loader_adds_real_tools_to_invocation_context(self) -> None:
        async def run() -> list[str]:
            tools = [load_application_tools]
            context = FunctionInvocationContext(
                function=load_application_tools,
                arguments={},
                tools=tools,
            )
            await load_application_tools.invoke(arguments={}, context=context)
            return _tool_names(tools)

        assert asyncio.run(run()) == [
            "load_application_tools",
            "search_application_candidates",
            "create_application_draft",
            "request_application_approval",
        ]
