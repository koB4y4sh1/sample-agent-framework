from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ui.render_event import ProviderFamily

from .mcp import (
    LocalMCPTransport,
    MCPApprovalMode,
    MCPMode,
    MCPServerSettings,
    MCPSpecificApprovalSettings,
    load_mcp_server_settings,
)


@dataclass(slots=True)
class ModelSettings:
    provider_family: ProviderFamily
    model_name: str
    default_options: dict[str, Any]


def load_model_settings_list() -> list[ModelSettings]:
    settings_path = Path(__file__).with_name("model.json")
    with settings_path.open("r", encoding="utf-8") as file:
        settings_data = json.load(file)

    return [
        ModelSettings(
            provider_family=item["provider_family"],
            model_name=item["model_name"],
            default_options=item["default_options"],
        )
        for item in settings_data
    ]


def load_model_settings(model_name: str) -> ModelSettings:
    settings_data = load_model_settings_list()

    for item in settings_data:
        if item.model_name == model_name:
            return ModelSettings(
                provider_family=item.provider_family,
                model_name=item.model_name,
                default_options=item.default_options,
            )

    available_models = ", ".join(item.model_name for item in settings_data)
    raise ValueError(f"Model settings not found for '{model_name}'. Available models: {available_models}")


__all__ = [
    "MCPServerSettings",
    "MCPMode",
    "LocalMCPTransport",
    "MCPApprovalMode",
    "MCPSpecificApprovalSettings",
    "ModelSettings",
    "load_model_settings_list",
    "load_mcp_server_settings",
    "load_model_settings",
]
