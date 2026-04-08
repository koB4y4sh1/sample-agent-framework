from __future__ import annotations

from typing import Any

from agent_framework import Agent, Content, Message
from settings import load_model_settings_list
from ui import BaseRender
from utils.file import AttachmentBuffer
from utils.print import print_color

from color_print import print_green, print_yellow


class DemoChatCLI:
    """Anthropic デモ用 Agent を対話的に操作する CLI ラッパー。"""

    def __init__(
        self,
        agent: Agent,
        session: Any,
        code_interpreter_status: str,
        stream_renderer: BaseRender,
    ) -> None:
        self._agent = agent
        self._session = session
        self._attachments = AttachmentBuffer()
        self._code_interpreter_status = code_interpreter_status
        self._stream_renderer = stream_renderer

    @classmethod
    def select_model(cls) -> str:
        models = load_model_settings_list()
        if not models:
            raise ValueError("No model settings were found in settings/model.json.")

        print_green("Available models:")
        for index, model in enumerate(models, start=1):
            print_green(f"  {index}. {model.model_name}")

        while True:
            try:
                selection = input("Select model number: ").strip()
            except EOFError as error:
                raise RuntimeError("Model selection requires an interactive terminal.") from error

            if not selection.isdigit():
                print_yellow("Please enter a valid number.")
                continue

            selected_index = int(selection)
            if 1 <= selected_index <= len(models):
                return models[selected_index - 1].model_name

            print_yellow("Please choose a number from the list.")

    async def run(self) -> None:
        self._print_help()
        while True:
            user_input = input("[User]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                self._print_status("[End] Session end.", color="bright_black")
                return
            if self._handle_command(user_input):
                continue

            message = self._build_user_message(user_input)
            await self._run_agent(message)

    def _handle_command(self, user_input: str) -> bool:
        if user_input == "/help":
            self._print_help()
            return True
        if user_input == "/clear":
            self._attachments.clear()
            self._print_status("[Info] Pending attachments cleared.", color="bright_black")
            return True
        if user_input == "/code":
            self._print_status(f"[Info] {self._code_interpreter_status}", color="bright_black")
            return True
        if user_input.startswith("/image "):
            self._attachments.add_image(user_input.split(" ", 1)[1])
            self._print_status(
                f"[Info] Added image. Pending attachments: {self._attachments.size}",
                color="bright_black",
            )
            return True
        if user_input.startswith("/file "):
            self._attachments.add_file(user_input.split(" ", 1)[1])
            self._print_status(
                f"[Info] Added file. Pending attachments: {self._attachments.size}",
                color="bright_black",
            )
            return True
        return False

    def _build_user_message(self, user_input: str) -> Message:
        contents = [Content.from_text(text=user_input), *self._attachments.consume()]
        return Message(role="user", contents=contents)

    async def _run_agent(self, message: Message) -> None:
        self._stream_renderer.start()
        async for chunk in self._agent.run(message, session=self._session, stream=True):
            self._stream_renderer.render(chunk.contents, chunk.text)
        self._stream_renderer.finish()

    def _print_help(self) -> None:
        self._print_status("[Start] Anthropic demo chat", color="bright_black")
        self._print_status("Commands:", color="bright_black")
        self._print_status("  /help               Show this help", color="bright_black")
        self._print_status("  /image <path>       Attach one image to the next prompt", color="bright_black")
        self._print_status("  /file <path>        Attach one file to the next prompt", color="bright_black")
        self._print_status("  /clear              Clear pending attachments", color="bright_black")
        self._print_status("  /code               Show code interpreter status", color="bright_black")
        self._print_status("  exit                Quit", color="bright_black")

    def _print_status(self, *values: Any, color: str = "bright_black", **kwargs: Any) -> None:
        print_color(*values, color=color, **kwargs)
