from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelSettings:
    """モデルごとの設定。"""

    model_name: str
    default_options: dict[str, Any]


def load_model_settings(model_name: str) -> ModelSettings:
    """モデル名に一致する設定を JSON から返す。"""
    settings_path = Path(__file__).with_suffix(".json")
    with settings_path.open("r", encoding="utf-8") as file:
        settings_data = json.load(file)

    for item in settings_data:
        if item["model_name"] == model_name:
            return ModelSettings(
                model_name=item["model_name"],
                default_options=item["default_options"],
            )

    available_models = ", ".join(item["model_name"] for item in settings_data)
    raise ValueError(f"Model settings not found for '{model_name}'. Available models: {available_models}")