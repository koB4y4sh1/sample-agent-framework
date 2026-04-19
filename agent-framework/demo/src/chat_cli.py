from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_framework import Agent, Content, Message
from ui import BaseRender
from utils.file import AttachmentBuffer
from utils.print import print_color

APPROVAL_REQUEST_TYPE = "function_approval_request"
APPROVAL_PROMPT_TEXT = "[Approve] {tool_name} を実行してもよろしいでしょうか？{arguments} Y/N: "


@dataclass(slots=True)
class ToolApprovalContext:
    assistant_messages: list[Message]
    requests: list[Content]


def pending_tool_approval_requests(messages: Sequence[Message]) -> list[Content]:
    return pending_tool_approval_context(messages).requests


def pending_tool_approval_context(messages: Sequence[Message]) -> ToolApprovalContext:
    """Return approval requests only when the latest message is awaiting approval."""
    if not messages:
        return ToolApprovalContext(assistant_messages=[], requests=[])

    last_message = messages[-1]
    requests = [content for content in last_message.contents if content.type == APPROVAL_REQUEST_TYPE]
    if not requests:
        return ToolApprovalContext(assistant_messages=[], requests=[])

    return ToolApprovalContext(assistant_messages=[_copy_message(last_message)], requests=requests)


def build_tool_approval_response_message(requests: Sequence[Content], approvals: Sequence[bool]) -> Message:
    if len(requests) != len(approvals):
        raise ValueError("requests and approvals must have the same length.")

    return Message(
        role="user",
        contents=[
            request.to_function_approval_response(approved)
            for request, approved in zip(requests, approvals, strict=True)
        ],
    )


def format_tool_approval_prompt(request: Content) -> str:
    return APPROVAL_PROMPT_TEXT.format(
        tool_name=tool_approval_name(request),
        arguments=tool_approval_arguments(request),
    )


def tool_approval_name(request: Content) -> str:
    function_call = request.function_call
    if function_call is None:
        return "(unknown)"
    return str(function_call.name or "(unknown)")


def tool_approval_arguments(request: Content) -> str:
    function_call = request.function_call
    if function_call is None:
        return "{}"
    return _format_arguments(function_call.arguments)


def _copy_message(message: Message) -> Message:
    return Message(
        role=message.role,
        contents=list(message.contents),
        author_name=message.author_name,
        message_id=message.message_id,
        additional_properties=dict(message.additional_properties),
    )


def _format_arguments(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        try:
            return json.dumps(json.loads(arguments), ensure_ascii=False, separators=(",", ":"))
        except json.JSONDecodeError:
            return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), default=str)


class DemoChatCLI:
    """Anthropic demo chat CLI."""

    def __init__(
        self,
        agent: Agent,
        session: Any,
        code_interpreter_status: str,
        stream_renderer: BaseRender,
        pending_tool_approval_context: ToolApprovalContext | None = None,
    ) -> None:
        self._agent = agent
        self._session = session
        self._attachments = AttachmentBuffer()
        self._code_interpreter_status = code_interpreter_status
        self._stream_renderer = stream_renderer
        self._pending_tool_approval_context = pending_tool_approval_context or ToolApprovalContext(
            assistant_messages=[],
            requests=[],
        )

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
        if user_input == "/skills":
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

    async def _run_agent(self, message: Message | Sequence[Message]) -> None:
        current_input: Message | Sequence[Message] = message
        while True:
            approval_context = await self._stream_agent_run(current_input)
            if not approval_context.requests:
                return
            current_input = self._build_tool_approval_messages(approval_context)

    async def _stream_agent_run(self, message: Message | Sequence[Message]) -> ToolApprovalContext:
        self._stream_renderer.start()
        stream = self._agent.run(message, session=self._session, stream=True)
        async for chunk in stream:
            self._stream_renderer.render(chunk.contents, chunk.text)
        self._stream_renderer.finish()

        final_response = await stream.get_final_response()
        return pending_tool_approval_context(final_response.messages)

    async def _resume_pending_tool_approvals(self) -> None:
        if not self._pending_tool_approval_context.requests:
            return

        approval_messages = self._build_tool_approval_messages(self._pending_tool_approval_context)
        self._pending_tool_approval_context = ToolApprovalContext(assistant_messages=[], requests=[])
        await self._run_agent(approval_messages)

    def _build_tool_approval_messages(self, approval_context: ToolApprovalContext) -> Message:
        approvals = [self._read_tool_approval(request) for request in approval_context.requests]
        return build_tool_approval_response_message(approval_context.requests, approvals)

    def _read_tool_approval(self, request: Content) -> bool:
        while True:
            print_color(format_tool_approval_prompt(request), color="bright_yellow", end="", flush=True)
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
        self._print_status("  /image <path>       Attach one image to the next prompt", color="green")
        self._print_status("  /file <path>        Attach one file to the next prompt", color="green")
        self._print_status("  /clear              Clear pending attachments", color="green")
        self._print_status("  /skills             Show Agent skills status", color="green")
        self._print_status("  exit                Quit", color="green")

    def _print_status(self, *values: Any, color: str = "green", **kwargs: Any) -> None:
        print_color(*values, color=color, **kwargs)
