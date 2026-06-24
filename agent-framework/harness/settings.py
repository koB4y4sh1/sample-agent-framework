from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarnessSettings(BaseSettings):
    """HA_ prefix の環境変数を読み込む設定クラス。

    `.env` もこのクラスで読むため、呼び出し側で `load_dotenv()` する必要はありません。
    """

    model_config = SettingsConfigDict(
        env_prefix="HA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    foundry_project_endpoint: str = Field(default="")
    foundry_model: str = "gpt-5.4-nano"
    toolbox_name: str = "playwright-browser"
    toolbox_version: str = "2"
    toolbox_api_version: str = "v1"
    goal_file: Path = Path("agent-framework/harness/goal.md")

    @model_validator(mode="after")
    def validate_required_values(self) -> HarnessSettings:
        """pydantic-settings が env を読んだ後に必須値を検証する。"""

        if not self.foundry_project_endpoint.strip():
            raise ValueError("HA_FOUNDRY_PROJECT_ENDPOINT is required")
        return self
