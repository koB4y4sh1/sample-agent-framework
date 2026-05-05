"""chillax / documenter / weather を1つの Streamable HTTP で公開するゲートウェイ。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_demo_dir = Path(__file__).resolve().parent
_demo_dir_str = str(_demo_dir)
if _demo_dir_str not in sys.path:
    sys.path.insert(0, _demo_dir_str)

import uvicorn  # noqa: E402
from fastmcp import FastMCP  # noqa: E402

from chillax import mcp as chillax_mcp  # noqa: E402
from documeentor import mcp as documenter_mcp  # noqa: E402
from weather import mcp as weather_mcp  # noqa: E402

gateway = FastMCP("demo-gateway")
gateway.mount(chillax_mcp, namespace="chillax")
gateway.mount(documenter_mcp, namespace="documenter")
gateway.mount(weather_mcp, namespace="weather")

http_app = gateway.http_app(transport="streamable-http")


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "server:http_app",
        host=host,
        port=port,
        reload=True,
        reload_dirs=[_demo_dir_str],
        app_dir=_demo_dir_str,
    )


if __name__ == "__main__":
    main()
