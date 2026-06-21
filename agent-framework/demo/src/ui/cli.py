from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent_framework import Agent, Content, Message
from opentelemetry import trace
from settings import load_model_settings_list
from ui.approval import (
    ToolApprovalContext,
    build_tool_approval_response_message,
    format_tool_approval_prompt,
    pending_tool_approval_context,
)
from ui.render_event import BaseRender, RenderEvent
from utils.file import AttachmentBuffer
from utils.print import print_color

if sys.platform == "win32":
    import msvcrt
else:
    msvcrt = None

tracer = trace.get_tracer(__name__)


@dataclass(slots=True)
class ModelSwitchResult:
    agent: Agent
    session: Any
    stream_renderer: "CLIStreamRenderer"
    tool_provider: Callable[[], list[Any]]
    all_tools_provider: Callable[[], list[Any]]
    model_name: str
    provider_family: str


class CLIStreamRenderer:
    def __init__(self, event_renderer: BaseRender) -> None:
        self._event_renderer = event_renderer
        self._current_output_type: str | None = None
        self._has_output = False

    def start(self) -> None:
        self._current_output_type = None
        self._has_output = False

    def render(self, contents: Sequence[Content], text: str | None = None) -> None:
        for event in self._event_renderer.render(contents, text):
            self._render_event(event)

    def finish(self) -> None:
        print()

    def _render_event(self, event: RenderEvent) -> None:
        if event.kind == "reasoning":
            self._start_output_block(
                next_output_type="reasoning",
                label="[Reasoning]",
                color="bright_black",
            )
            print_color(event.text, color="bright_black", end="", flush=True)
            return
        if event.kind == "text":
            self._start_output_block(
                next_output_type="text",
                label="[Answer]",
                color="bright_white",
                styles=("bold",),
            )
            print_color(
                event.text,
                color="bright_white",
                styles=("bold",),
                end="",
                flush=True,
            )
            return

        label, color, styles = self._event_style(event)
        self._start_output_block(next_output_type=event.kind)
        print_color(f"\n{label} {event.text}", color=color, styles=styles, end="", flush=True)

    def _event_style(self, event: RenderEvent) -> tuple[str, str, tuple[str, ...]]:
        if event.kind == "tool_call":
            return "[Tool Call]", "bright_green", ("bold",)
        if event.kind == "tool_result":
            return "[Tool Result]", "green", ()
        if event.kind == "mcp_call":
            return "[MCP Call]", "bright_blue", ("bold",)
        if event.kind == "mcp_result":
            return "[MCP Result]", "blue", ()
        if event.kind == "usage":
            return "[Usage]", "bright_cyan", ("bold",)
        if event.kind == "approval_request":
            return "[Approval Request]", "bright_yellow", ("bold",)
        return f"[{event.content_type or event.kind}]", "bright_black", ()

    def _start_output_block(
        self,
        *,
        next_output_type: str,
        label: str | None = None,
        color: str | None = None,
        styles: tuple[str, ...] = (),
    ) -> None:
        if self._has_output and self._current_output_type != next_output_type:
            print()
        if self._current_output_type != next_output_type and label and color is not None:
            print_color(label, color=color, styles=styles, end=" ", flush=True)
        self._current_output_type = next_output_type
        self._has_output = True


class DemoChatCLI:
    """Anthropic demo chat CLI."""

    def __init__(
        self,
        agent: Agent,
        session: Any,
        code_interpreter_status: str,
        stream_renderer: "CLIStreamRenderer",
        tool_provider: Callable[[], list[Any]],
        all_tools_provider: Callable[[], list[Any]],
        model_name: str = "",
        provider_family: str = "",
        model_switcher: Callable[[str], Awaitable[ModelSwitchResult]] | None = None,
        pending_tool_approval_context: ToolApprovalContext | None = None,
    ) -> None:
        self._agent = agent
        self._session = session
        self._model_name = model_name
        self._provider_family = provider_family
        self._attachments = AttachmentBuffer()
        self._code_interpreter_status = code_interpreter_status
        self._stream_renderer = stream_renderer
        self._tool_provider = tool_provider
        self._all_tools_provider = all_tools_provider
        self._model_switcher = model_switcher or self._missing_model_switcher
        self._pending_tool_approval_context = (
            pending_tool_approval_context
            or ToolApprovalContext(
                assistant_messages=[],
                requests=[],
            )
        )

    async def _missing_model_switcher(self, model_name: str) -> ModelSwitchResult:
        raise RuntimeError("model_switcher is required to switch models.")

    async def run(self) -> None:
        """Run the interactive CLI loop."""
        self._print_help()
        await self._resume_pending_tool_approvals()
        while True:
            user_input = input("[User]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                self._print_status("[End] Session end.", color="bright_black")
                return
            if await self._handle_command(user_input):
                continue

            message = self._build_user_message(user_input)
            await self._run_agent(message)

    async def _handle_command(self, user_input: str) -> bool:
        if user_input == "/help":
            self._print_help()
            return True
        if user_input == "/clear":
            self._attachments.clear()
            self._print_status(
                "[Info] Pending attachments cleared.", color="bright_black"
            )
            return True
        if user_input == "/skills":
            self._print_status(
                f"[Info] {self._code_interpreter_status}", color="bright_black"
            )
            return True
        if user_input.startswith("/model"):
            model_name = user_input.split(" ", 1)[1].strip() if " " in user_input else ""
            if not model_name:
                selected = self._select_model_interactively()
                if selected is None:
                    self._print_status("[Info] Model switch cancelled.", color="bright_black")
                    return True
                model_name = selected
            await self._switch_model(model_name)
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

    async def _run_agent(
        self,
        message: Message | Sequence[Message],
        *,
        tools: list[Any] | None = None,
    ) -> None:
        current_input: Message | Sequence[Message] = message
        exposed_tools = tools if tools is not None else self._tool_provider()
        with tracer.start_as_current_span("chat_request"):
            while True:
                approval_context = await self._stream_agent_run(
                    current_input,
                    tools=exposed_tools,
                )
                if not approval_context.requests:
                    return
                current_input = self._build_tool_approval_messages(approval_context)

    async def _stream_agent_run(
        self,
        message: Message | Sequence[Message],
        *,
        tools: list[Any],
    ) -> ToolApprovalContext:
        self._stream_renderer.start()
        stream = self._agent.run(
            message,
            session=self._session,
            stream=True,
            tools=tools,
        )
        async for chunk in stream:
            self._stream_renderer.render(chunk.contents, chunk.text)
        self._stream_renderer.finish()

        final_response = await stream.get_final_response()
        return pending_tool_approval_context(final_response.messages)

    async def _resume_pending_tool_approvals(self) -> None:
        if not self._pending_tool_approval_context.requests:
            return

        approval_messages = self._build_tool_approval_messages(
            self._pending_tool_approval_context
        )
        self._pending_tool_approval_context = ToolApprovalContext(
            assistant_messages=[], requests=[]
        )
        await self._run_agent(approval_messages, tools=self._all_tools_provider())

    def _build_tool_approval_messages(
        self, approval_context: ToolApprovalContext
    ) -> Message:
        approvals = [
            self._read_tool_approval(request) for request in approval_context.requests
        ]
        return build_tool_approval_response_message(
            approval_context.requests, approvals
        )

    def _read_tool_approval(self, request: Content) -> bool:
        while True:
            print_color(
                format_tool_approval_prompt(request),
                color="bright_yellow",
                end="",
                flush=True,
            )
            answer = input().strip().lower()
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            self._print_status("[Error] Enter Y or N.", color="red")

    def _print_help(self) -> None:
        self._print_status("[Start] Anthropic demo chat", color="green")
        self._print_status("Commands:", color="green")
        self._print_status("  /help               Show this help", color="green")
        self._print_status(
            "  /model [name]       Switch model (interactive when omitted)", color="green"
        )
        self._print_status(
            "  /image <path>       Attach one image to the next prompt", color="green"
        )
        self._print_status(
            "  /file <path>        Attach one file to the next prompt", color="green"
        )
        self._print_status(
            "  /clear              Clear pending attachments", color="green"
        )
        self._print_status(
            "  /skills             Show Agent skills status", color="green"
        )
        self._print_status("  exit                Quit", color="green")

    def _print_status(self, *values: Any, color: str = "green", **kwargs: Any) -> None:
        print_color(*values, color=color, **kwargs)

    def _select_model_interactively(self) -> str | None:
        models = load_model_settings_list()
        current_index = 0
        for idx, model in enumerate(models):
            if model.model_name == self._model_name:
                current_index = idx
                break

        selected = self._select_from_menu(
            title="Available models",
            options=[f"{model.provider_family}: {model.model_name}" for model in models],
            prompt="Use Up/Down and Enter to switch model. Press Esc to cancel.",
            initial_index=current_index,
        )
        if selected is None:
            return None
        return models[selected].model_name

    def _select_from_menu(
        self,
        *,
        title: str,
        options: list[str],
        prompt: str,
        initial_index: int = 0,
    ) -> int | None:
        if not options:
            raise ValueError(f"No options available for menu: {title}")
        if msvcrt is None:
            raise RuntimeError("Arrow-key menu is currently supported only on Windows terminals.")

        selected_index = max(0, min(initial_index, len(options) - 1))
        while True:
            self._clear_screen()
            self._print_status(title, color="green")
            self._print_status(prompt, color="green")
            for index, option in enumerate(options):
                prefix = ">" if index == selected_index else " "
                color = "bright_white" if index == selected_index else "bright_black"
                self._print_status(f"  {prefix} {option}", color=color)

            key = self._read_menu_key()
            if key == "UP":
                selected_index = (selected_index - 1) % len(options)
                continue
            if key == "DOWN":
                selected_index = (selected_index + 1) % len(options)
                continue
            if key == "ENTER":
                return selected_index
            if key == "ESC":
                return None

    def _read_menu_key(self) -> str:
        if msvcrt is None:
            raise RuntimeError("Arrow-key menu is currently supported only on Windows terminals.")

        while True:
            key = msvcrt.getwch()
            if key in {"\r", "\n"}:
                return "ENTER"
            if key == "\x1b":
                return "ESC"
            if key in {"\x00", "\xe0"}:
                extended = msvcrt.getwch()
                if extended == "H":
                    return "UP"
                if extended == "P":
                    return "DOWN"

    def _clear_screen(self) -> None:
        if os.name == "nt":
            os.system("cls")
            return
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    async def _switch_model(self, model_name: str) -> None:
        if model_name == self._model_name:
            self._print_status(f"[Info] Already using: {model_name}", color="bright_black")
            return
        try:
            result = await self._model_switcher(model_name)
        except ValueError as error:
            self._print_status(f"[Error] {error}", color="red")
            return
        self._agent = result.agent
        self._session = result.session
        self._stream_renderer = result.stream_renderer
        self._tool_provider = result.tool_provider
        self._all_tools_provider = result.all_tools_provider
        self._model_name = result.model_name
        self._provider_family = result.provider_family
        self._print_status(
            f"[Info] Switched model to {self._provider_family}: {self._model_name}",
            color="bright_black",
        )
