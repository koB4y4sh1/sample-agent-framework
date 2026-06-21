from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nicegui import ui

from agent.contexts.user_profile import UserProfile
from agent.middleware import ALLOW_TOOLS_FILE_NAME, DEFAULT_MEMORY_ROOT_DIR


def create_user_ui() -> None:
    @ui.page("/user")
    def user() -> None:
        UserSettingsPage()


class UserSettingsPage:
    """ユーザー単位のProfileとTool自動承認ルールを管理する画面。

    想定ユースケース:
    - `.memory/{user_name}.json` のUserProfileを確認・修正したい。
    - `.memory/{user_name}/allow_tools.json` のTool自動承認ルールを確認・修正・削除したい。
    - Agent全体設定ではなく、ユーザーに紐づく記憶と許可だけを切り出して管理したい。
    """

    def __init__(self) -> None:
        self._build()

    def _build(self) -> None:
        ui.add_css(
            """
            body { background: #f4f7f5; }
            .user-shell { max-width: 72rem; margin: 0 auto; padding: 24px; }
            .user-editor textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
            """
        )
        with ui.column().classes("user-shell w-full gap-4"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("User Settings").classes("text-2xl font-semibold")
                with ui.row().classes("gap-3"):
                    ui.link("Chat", "/").classes("text-sm")
                    ui.link("Admin", "/admin").classes("text-sm")

            user_options = self._user_options()
            user_select = ui.select(user_options, label="User", value=next(iter(user_options), "demouser")).classes(
                "w-80"
            )
            new_user_input = ui.input("New user").classes("w-80")

            with ui.tabs().classes("w-full") as tabs:
                profile_tab = ui.tab("Profile")
                allow_tools_tab = ui.tab("allow_tools")

            with ui.tab_panels(tabs, value=profile_tab).classes("w-full"):
                with ui.tab_panel(profile_tab):
                    profile_editor = self._build_profile_editor(user_select)
                with ui.tab_panel(allow_tools_tab):
                    allow_tools_editor = self._build_allow_tools_editor(user_select)

            def reload_all() -> None:
                profile_editor.reload()
                allow_tools_editor.reload()

            def refresh_users() -> None:
                refreshed = self._user_options()
                user_select.set_options(refreshed)
                if user_select.value not in refreshed:
                    user_select.value = next(iter(refreshed), "demouser")
                reload_all()

            def use_new_user() -> None:
                value = str(new_user_input.value or "").strip()
                if not value:
                    ui.notify("User name is required.", type="warning")
                    return
                options = dict(user_select.options)
                options[value] = value
                user_select.set_options(options)
                user_select.value = value
                reload_all()

            user_select.on_value_change(lambda _: reload_all())
            with ui.row().classes("gap-2 items-end"):
                ui.button("Use new user", icon="person_add", on_click=use_new_user).props("outline")
                ui.button("Refresh users", icon="refresh", on_click=refresh_users).props("outline")

    def _build_profile_editor(self, user_select: Any) -> "_JsonEditor":
        editor = _JsonEditor(
            title="User profile",
            path_provider=lambda: self._profile_path(str(user_select.value or "demouser")),
            default_value=UserProfile().model_dump(),
            template={
                "summary": "User name is demouser",
                "goals": [],
                "preferences": [],
                "constraints": [],
                "working_style": [],
                "communication_preferences": [],
                "recurring_topics": [],
            },
        )
        editor.build()
        return editor

    def _build_allow_tools_editor(self, user_select: Any) -> "_JsonEditor":
        editor = _JsonEditor(
            title="allow_tools",
            path_provider=lambda: self._allow_tools_path(str(user_select.value or "demouser")),
            default_value={"version": 1, "rules": []},
            template={
                "scope": "tool_with_arguments",
                "tool_name": "request_application_approval",
                "server_label": None,
                "arguments": {"draft_id": "\"D-1\""},
            },
        )
        editor.build()
        return editor

    def _user_options(self) -> dict[str, str]:
        root = DEFAULT_MEMORY_ROOT_DIR
        options: dict[str, str] = {}
        if root.exists():
            for path in sorted(root.glob("*.json")):
                options[path.stem] = path.stem
            for path in sorted(root.iterdir()):
                if path.is_dir() and (path / ALLOW_TOOLS_FILE_NAME).exists():
                    options[path.name] = path.name
        if not options:
            options["demouser"] = "demouser"
        return options

    def _profile_path(self, user_name: str) -> Path:
        safe_user_name = _safe_user_name(user_name)
        return DEFAULT_MEMORY_ROOT_DIR / f"{safe_user_name}.json"

    def _allow_tools_path(self, user_name: str) -> Path:
        safe_user_name = _safe_user_name(user_name)
        return DEFAULT_MEMORY_ROOT_DIR / safe_user_name / ALLOW_TOOLS_FILE_NAME


class _JsonEditor:
    def __init__(
        self,
        *,
        title: str,
        path_provider: Any,
        default_value: Any,
        template: dict[str, Any],
    ) -> None:
        self._title = title
        self._path_provider = path_provider
        self._default_value = default_value
        self._template = template
        self._editor: Any | None = None
        self._status: Any | None = None

    def build(self) -> None:
        with ui.column().classes("w-full gap-3"):
            ui.label(self._title).classes("text-lg font-semibold")
            self._path_label = ui.label("").classes("text-xs text-gray-600")
            self._editor = ui.textarea().props("outlined autogrow").classes("user-editor w-full")
            self._status = ui.label("").classes("text-sm text-gray-600")
            with ui.row().classes("gap-2"):
                ui.button("Reload", icon="refresh", on_click=self.reload).props("outline")
                ui.button("Add template", icon="add", on_click=self.add_template).props("outline")
                ui.button("Format", icon="format_indent_increase", on_click=self.format).props("outline")
                ui.button("Save", icon="save", on_click=self.save)
            self.reload()

    def reload(self) -> None:
        path = self._path()
        self._path_label.text = str(path)
        self._editor.value = _read_json_text(path, self._default_value)
        self._status.text = f"Loaded: {path}"

    def add_template(self) -> None:
        value = self._parse()
        if value is None:
            return
        if isinstance(value, list):
            value.append(self._template)
        elif isinstance(value, dict):
            if "rules" in value:
                if not isinstance(value["rules"], list):
                    self._set_status("JSON object must contain a list at key 'rules'.")
                    return
                value["rules"].append(self._template)
            else:
                value.update(self._template)
        else:
            self._set_status("JSON root must be a list or object.")
            return
        self._editor.value = json.dumps(value, ensure_ascii=False, indent=2)
        self._set_status("Template added.")

    def format(self) -> None:
        value = self._parse()
        if value is None:
            return
        self._editor.value = json.dumps(value, ensure_ascii=False, indent=2)
        self._set_status("Formatted.")

    def save(self) -> None:
        value = self._parse()
        if value is None:
            return
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        self._set_status(f"Saved: {path}")
        ui.notify("Saved.", type="positive")

    def _parse(self) -> Any | None:
        try:
            return json.loads(str(self._editor.value or "null"))
        except json.JSONDecodeError as error:
            self._set_status(f"Invalid JSON: {error}")
            ui.notify(str(error), type="negative")
            return None

    def _path(self) -> Path:
        return Path(self._path_provider())

    def _set_status(self, message: str) -> None:
        self._status.text = message


def _read_json_text(path: Path, default_value: Any) -> str:
    if not path.exists():
        return json.dumps(default_value, ensure_ascii=False, indent=2)
    try:
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return path.read_text(encoding="utf-8")


def _safe_user_name(user_name: str) -> str:
    safe = user_name.strip().replace("/", "_").replace("\\", "_")
    return safe or "demouser"
