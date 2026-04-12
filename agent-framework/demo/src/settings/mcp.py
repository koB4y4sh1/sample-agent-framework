from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict


MCPMode: TypeAlias = Literal["hosted", "local"]
LocalMCPTransport: TypeAlias = Literal["stdio", "streamable_http"]


class MCPSpecificApprovalSettings(TypedDict, total=False):
    always_require_approval: Collection[str] | None
    never_require_approval: Collection[str] | None


MCPApprovalMode: TypeAlias = (
    Literal["always_require", "never_require"] | MCPSpecificApprovalSettings
)

SETTINGS_PATH = Path(__file__).with_suffix(".json")
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


@dataclass(slots=True)
class MCPServerSettings:
    """MCP server settings."""

    name: str
    mode: MCPMode = "hosted"
    transport: LocalMCPTransport | None = None
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None
    encoding: str | None = None
    description: str | None = None
    approval_mode: MCPApprovalMode | None = None
    allowed_tools: list[str] | None = None
    headers: dict[str, str] | None = None
    project_connection_id: str | None = None
    tool_name_prefix: str | None = None
    request_timeout: int | None = None
    terminate_on_close: bool | None = None


def load_mcp_server_settings() -> list[MCPServerSettings]:
    """Load MCP server settings used by the demo."""
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        settings_data = json.load(file)

    return [_parse_mcp_server_settings(item) for item in settings_data if item.get("enabled", True)]


def _parse_mcp_server_settings(item: dict[str, Any]) -> MCPServerSettings:
    mode = _normalize_mode(item.get("mode", item.get("type", "hosted")))
    transport = _normalize_transport(item.get("transport")) if mode == "local" else None
    url = item.get("url")
    command = item.get("command")

    if mode == "hosted" and not (url or item.get("project_connection_id")):
        raise ValueError(f"MCP hosted setting '{item.get('name')}' requires url or project_connection_id.")
    if mode == "local" and transport == "stdio" and not command:
        raise ValueError(f"MCP local stdio setting '{item.get('name')}' requires command.")
    if mode == "local" and transport == "streamable_http" and not url:
        raise ValueError(f"MCP local streamable_http setting '{item.get('name')}' requires url.")

    return MCPServerSettings(
        name=item["name"],
        mode=mode,
        transport=transport,
        url=url,
        command=command,
        args=list(item.get("args", [])),
        cwd=_resolve_cwd(item.get("cwd")),
        env=item.get("env"),
        encoding=item.get("encoding"),
        description=item.get("description"),
        approval_mode=_normalize_approval_mode(item.get("approval_mode")),
        allowed_tools=item.get("allowed_tools"),
        headers=item.get("headers"),
        project_connection_id=item.get("project_connection_id"),
        tool_name_prefix=item.get("tool_name_prefix"),
        request_timeout=item.get("request_timeout"),
        terminate_on_close=item.get("terminate_on_close"),
    )


def _normalize_mode(value: Any) -> MCPMode:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"hosted", "hosted_mcp", "hostedmcp"}:
        return "hosted"
    if normalized in {"local", "local_mcp", "localmcp"}:
        return "local"
    raise ValueError(f"Unsupported MCP mode: {value}")


def _normalize_transport(value: Any) -> LocalMCPTransport:
    normalized = str(value or "stdio").strip().lower().replace("-", "_")
    if normalized == "stdio":
        return "stdio"
    if normalized in {"streamable", "streamable_http", "streamablehttp", "http"}:
        return "streamable_http"
    raise ValueError(f"Unsupported local MCP transport: {value}")


def _normalize_approval_mode(value: Any) -> MCPApprovalMode | None:
    if value is None:
        return None
    if value in {"always_require", "never_require"}:
        return value
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported MCP approval_mode: {value}")

    unknown_keys = set(value) - {"always_require_approval", "never_require_approval"}
    if unknown_keys:
        raise ValueError(f"Unsupported MCP approval_mode keys: {sorted(unknown_keys)}")

    approval: MCPSpecificApprovalSettings = {}
    always_require = _normalize_approval_tool_names(value.get("always_require_approval"), "always_require_approval")
    never_require = _normalize_approval_tool_names(value.get("never_require_approval"), "never_require_approval")

    if always_require is not None:
        approval["always_require_approval"] = always_require
    if never_require is not None:
        approval["never_require_approval"] = never_require

    return approval


def _normalize_approval_tool_names(value: Any, key: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"MCP approval_mode.{key} must be a list of strings.")
    return value


def _resolve_cwd(value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return str(path)
