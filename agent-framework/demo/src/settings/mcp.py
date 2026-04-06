from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MCPServerSettings:
    """MCP サーバー設定。"""

    name: str
    url: str


def load_mcp_server_settings() -> list[MCPServerSettings]:
    """デモで利用する MCP サーバー設定を JSON から返す。"""
    settings_path = Path(__file__).with_suffix(".json")
    with settings_path.open("r", encoding="utf-8") as file:
        settings_data = json.load(file)

    return [
        MCPServerSettings(
            name=item["name"],
            url=item["url"],
        )
        for item in settings_data
    ]
