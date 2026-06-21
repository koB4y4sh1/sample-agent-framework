from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nicegui import ui

from settings.mcp import SETTINGS_PATH as MCP_SETTINGS_PATH

MODEL_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings" / "model.json"


def create_admin_ui() -> None:
    @ui.page("/admin")
    def admin() -> None:
        AdminPage()


class AdminPage:
    """MCPとModelを管理するNiceGUI画面。

    想定ユースケース:
    - `settings/mcp.json` の有効化、接続先、承認モードをブラウザから編集したい。
    - `settings/model.json` のモデル一覧とdefault_optionsをブラウザから編集したい。

    既存コードはJSON設定を直接読むため、この画面もJSONを正として編集する。
    保存時にJSONとしてparseできることだけ検証し、ドメイン検証は既存loader側へ委譲する。
    """

    def __init__(self) -> None:
        self._build()

    def _build(self) -> None:
        ui.add_css(
            """
            body { background: #f4f7f5; }
            .admin-shell { max-width: 72rem; margin: 0 auto; padding: 24px; }
            .admin-editor textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
            """
        )
        with ui.column().classes("admin-shell w-full gap-4"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Agent Admin").classes("text-2xl font-semibold")
                with ui.row().classes("gap-3"):
                    ui.link("Chat", "/").classes("text-sm")
                    ui.link("User", "/user").classes("text-sm")
            with ui.tabs().classes("w-full") as tabs:
                mcp_tab = ui.tab("MCP")
                model_tab = ui.tab("Model")
            with ui.tab_panels(tabs, value=mcp_tab).classes("w-full"):
                with ui.tab_panel(mcp_tab):
                    self._build_json_file_editor(
                        title="MCP settings",
                        path=MCP_SETTINGS_PATH,
                        default_value=[],
                        add_template={
                            "enabled": False,
                            "name": "New_MCP",
                            "mode": "hosted",
                            "url": "https://example.com/mcp",
                            "approval_mode": "always_require",
                        },
                    )
                with ui.tab_panel(model_tab):
                    self._build_json_file_editor(
                        title="Model settings",
                        path=MODEL_SETTINGS_PATH,
                        default_value=[],
                        add_template={
                            "provider_family": "openai",
                            "model_name": "gpt-5.4-nano",
                            "default_options": {},
                        },
                    )

    def _build_json_file_editor(
        self,
        *,
        title: str,
        path: Path,
        default_value: Any,
        add_template: dict[str, Any],
    ) -> None:
        with ui.column().classes("w-full gap-3"):
            ui.label(title).classes("text-lg font-semibold")
            ui.label(str(path)).classes("text-xs text-gray-600")
            editor = ui.textarea(value=self._read_json_text(path, default_value)).props(
                "outlined autogrow"
            ).classes("admin-editor w-full")
            status = ui.label("").classes("text-sm text-gray-600")

            with ui.row().classes("gap-2"):
                ui.button(
                    "Reload",
                    icon="refresh",
                    on_click=lambda: self._reload_json_editor(editor, status, path, default_value),
                ).props("outline")
                ui.button(
                    "Add template",
                    icon="add",
                    on_click=lambda: self._append_template(editor, status, add_template),
                ).props("outline")
                ui.button(
                    "Format",
                    icon="format_indent_increase",
                    on_click=lambda: self._format_editor_json(editor, status),
                ).props("outline")
                ui.button(
                    "Save",
                    icon="save",
                    on_click=lambda: self._save_json_editor(editor, status, path),
                )

    def _read_json_text(self, path: Path, default_value: Any) -> str:
        if not path.exists():
            return json.dumps(default_value, ensure_ascii=False, indent=2)
        try:
            return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return path.read_text(encoding="utf-8")

    def _reload_json_editor(self, editor: Any, status: Any, path: Path, default_value: Any) -> None:
        editor.value = self._read_json_text(path, default_value)
        status.text = f"Loaded: {path}"

    def _append_template(self, editor: Any, status: Any, template: dict[str, Any]) -> None:
        try:
            value = json.loads(str(editor.value or "null"))
        except json.JSONDecodeError as error:
            status.text = f"Invalid JSON: {error}"
            return
        if isinstance(value, list):
            value.append(template)
        elif isinstance(value, dict):
            value.setdefault("rules", [])
            if isinstance(value["rules"], list):
                value["rules"].append(template)
            else:
                status.text = "JSON object must contain a list at key 'rules'."
                return
        else:
            status.text = "JSON root must be a list or object."
            return
        editor.value = json.dumps(value, ensure_ascii=False, indent=2)
        status.text = "Template added."

    def _format_editor_json(self, editor: Any, status: Any) -> None:
        try:
            value = json.loads(str(editor.value or "null"))
        except json.JSONDecodeError as error:
            status.text = f"Invalid JSON: {error}"
            return
        editor.value = json.dumps(value, ensure_ascii=False, indent=2)
        status.text = "Formatted."

    def _save_json_editor(self, editor: Any, status: Any, path: Path) -> None:
        try:
            value = json.loads(str(editor.value or "null"))
        except json.JSONDecodeError as error:
            status.text = f"Invalid JSON: {error}"
            ui.notify(status.text, type="negative")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        status.text = f"Saved: {path}"
        ui.notify("Saved.", type="positive")
