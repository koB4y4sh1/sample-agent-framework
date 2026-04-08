from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime

from agent.history import MEMORY_ROOT_DIR
from settings import ModelSettings, load_model_settings_list
from utils.print import _enable_windows_ansi, print_color, print_green

if sys.platform == "win32":
    import msvcrt
else:
    msvcrt = None


@dataclass(slots=True)
class BootstrapResult:
    model_settings: ModelSettings
    session_id: str | None


@dataclass(slots=True)
class HistorySession:
    session_id: str
    updated_at: float


class CLIBootstrap:
    """CLI startup bootstrap."""

    def run(self) -> BootstrapResult:
        self._ensure_interactive_terminal()
        model_settings = self._select_model()
        session_id = self._select_session_id()
        result = BootstrapResult(model_settings=model_settings, session_id=session_id)
        self._print_summary(result)
        return result

    def _select_model(self) -> ModelSettings:
        models = load_model_settings_list()
        if not models:
            raise ValueError("No model settings were found in settings/model.json.")

        selected_index = self._select_from_menu(
            title="Available models",
            options=[f"{model.provider_family}: {model.model_name}" for model in models],
            prompt="Use Up/Down and Enter to select a model.",
        )
        return models[selected_index]

    def _select_session_id(self) -> str | None:
        sessions = self._list_history_sessions()
        if not sessions:
            print_green("No saved history was found. A new session will be created.")
            return self._prompt_new_session_id()

        selected_index = self._select_from_menu(
            title="History options",
            options=["Start a new session", "Resume saved session"],
            prompt="Use Up/Down and Enter to choose history handling.",
        )
        if selected_index == 0:
            return self._prompt_new_session_id()
        return self._select_existing_session_id(sessions)

    def _prompt_new_session_id(self) -> str | None:
        session_id = input("New session id (press Enter for auto-generated): ").strip()
        return session_id or None

    def _select_existing_session_id(self, sessions: list[HistorySession]) -> str:
        selected_index = self._select_from_menu(
            title="Saved sessions",
            options=[
                f"{session.session_id} (updated: {datetime.fromtimestamp(session.updated_at).strftime('%Y-%m-%d %H:%M:%S')})"
                for session in sessions
            ],
            prompt="Use Up/Down and Enter to resume a saved session.",
        )
        return sessions[selected_index].session_id

    def _select_from_menu(self, *, title: str, options: list[str], prompt: str) -> int:
        if not options:
            raise ValueError(f"No options available for menu: {title}")

        selected_index = 0

        while True:
            self._clear_screen()

            rendered_lines = [title, prompt]
            for index, option in enumerate(options):
                prefix = ">" if index == selected_index else " "
                rendered_lines.append(f"  {prefix} {option}")

            for index, line in enumerate(rendered_lines):
                if index < 2:
                    print_green(line)
                elif index - 2 == selected_index:
                    print_color(line, color="bright_white", styles=["bold"])
                else:
                    print_color(line, color="bright_black")

            key = self._read_key()
            if key == "UP":
                selected_index = (selected_index - 1) % len(options)
                continue
            if key == "DOWN":
                selected_index = (selected_index + 1) % len(options)
                continue
            if key == "ENTER":
                return selected_index

    def _read_key(self) -> str:
        if msvcrt is None:
            raise RuntimeError("Arrow-key menu is currently supported only on Windows terminals.")

        while True:
            key = msvcrt.getwch()
            if key in {"\r", "\n"}:
                return "ENTER"
            if key in {"\x00", "\xe0"}:
                extended = msvcrt.getwch()
                if extended == "H":
                    return "UP"
                if extended == "P":
                    return "DOWN"
                continue

    def _clear_screen(self) -> None:
        if os.name == "nt":
            os.system("cls")
            return

        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    def _print_summary(self, result: BootstrapResult) -> None:
        print_green("Selected settings:")
        print_color(
            f"  provider_family: {result.model_settings.provider_family}",
            color="bright_white",
            styles=["bold"],
        )
        print_color(
            f"  model: {result.model_settings.model_name}",
            color="bright_white",
            styles=["bold"],
        )
        session_label = result.session_id or "(auto-generated new session)"
        print_color(f"  session_id: {session_label}", color="bright_white", styles=["bold"])
        print()

    def _ensure_interactive_terminal(self) -> None:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise RuntimeError("Interactive terminal is required for CLI bootstrap.")
        _enable_windows_ansi()

    def _list_history_sessions(self) -> list[HistorySession]:
        if not MEMORY_ROOT_DIR.exists():
            return []

        sessions: list[HistorySession] = []
        for path in MEMORY_ROOT_DIR.glob("*.json"):
            sessions.append(
                HistorySession(
                    session_id=path.stem,
                    updated_at=path.stat().st_mtime,
                )
            )

        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions
