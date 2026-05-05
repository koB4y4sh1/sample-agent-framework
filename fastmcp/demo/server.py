"""MCP Gateway Server"""

from __future__ import annotations

import os
from pathlib import Path

from observability import setup_observability

_demo_dir_str = str(Path(__file__).resolve().parent)
setup_observability()

import uvicorn
from chillax import mcp as chillax_mcp
from documeentor import mcp as documenter_mcp
from weather import mcp as weather_mcp

from fastmcp import FastMCP

mcp = FastMCP("demo-gateway")
mcp.mount(chillax_mcp, namespace="chillax")
mcp.mount(documenter_mcp, namespace="documenter")
mcp.mount(weather_mcp, namespace="weather")

app = mcp.http_app(transport="streamable-http")


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,
        reload_dirs=[_demo_dir_str],
        app_dir=_demo_dir_str,
    )


if __name__ == "__main__":
    main()
