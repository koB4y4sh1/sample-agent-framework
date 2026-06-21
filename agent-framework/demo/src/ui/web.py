from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any
from uuid import uuid4

from agent_framework import Content, Message
from nicegui import ui
from nicegui.events import GenericEventArguments, UploadEventArguments

from agent.store import LocalStore
from agent.store.local import MEMORY_ROOT_DIR
from app import DemoSessionRuntime
from ui.approval import (
    ToolApprovalContext,
    build_tool_approval_response_message,
    pending_tool_approval_context,
    tool_approval_arguments,
    tool_approval_name,
)
from ui.render_event import RenderEvent, UIResolver
from settings import load_model_settings_list


@dataclass(slots=True)
class Attachment:
    name: str
    media_type: str
    data: bytes


@dataclass(slots=True)
class HistorySession:
    session_id: str
    updated_at: float


@dataclass(slots=True)
class StreamingTextView:
    element_id: str


@dataclass(slots=True)
class ChatRuntimeState:
    runtime: DemoSessionRuntime
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_approval: ToolApprovalContext = field(
        default_factory=lambda: ToolApprovalContext(
            assistant_messages=[],
            requests=[],
        )
    )


class ChatRuntimeManager:
    def __init__(self) -> None:
        self._states: dict[str, ChatRuntimeState] = {}
        self._states_lock = asyncio.Lock()
        self.store = LocalStore()

    async def get(self, *, session_id: str | None, model_name: str) -> ChatRuntimeState:
        key = session_id or "default"
        async with self._states_lock:
            state = self._states.get(key)
            if state is None:
                runtime = DemoSessionRuntime.create(
                    model_name=model_name,
                    session_id=session_id,
                )
                state = ChatRuntimeState(runtime=runtime)
                state.pending_approval = (
                    await runtime.app.get_pending_tool_approval_context(session_id)
                )
                self._states[key] = state
                return state

            if state.runtime.model_name != model_name:
                state.runtime.switch_model(model_name)
                state.pending_approval = (
                    await state.runtime.app.get_pending_tool_approval_context(session_id)
                )
            return state


class ChatPage:
    def __init__(self, manager: ChatRuntimeManager) -> None:
        self._manager = manager
        self._attachments: list[Attachment] = []
        self._pending_requests: list[Content] = []
        self._pending_approvals: dict[int, bool] = {}
        self._approval_status_labels: dict[int, Any] = {}
        self._is_running = False
        self._models = load_model_settings_list()
        if not self._models:
            raise ValueError("No model settings were found in settings/model.json.")
        self._model_options = {
            item.model_name: f"{item.provider_family}: {item.model_name}"
            for item in self._models
        }
        self._renderers = {
            item.model_name: UIResolver(item.provider_family).resolve()
            for item in self._models
        }

        self._build()

    def _build(self) -> None:
        ui.add_css(
            """
            body { background: #f4f7f5; }
            .chat-shell { height: 100vh; }
            .message-list { background: #f4f7f5; }
            .message-row { display: flex; width: 100%; }
            .message-row.user { justify-content: flex-end; }
            .message-row.assistant,
            .message-row.tool,
            .message-row.system { justify-content: flex-start; }
            .message-stack { max-width: 64rem; width: fit-content; min-width: 18rem; }
            .message-role { color: #5f6f67; font-size: 12px; font-weight: 600; margin: 0 0 4px 2px; }
            .message-bubble { border: 1px solid #cfd8d3; border-radius: 8px; padding: 12px 14px; line-height: 1.55; }
            .message-bubble.assistant { background: #ffffff; }
            .message-bubble.user { background: #e7e7e7; border-color: #d4d4d4; }
            .message-bubble.tool { background: #f8fbff; border-color: #cad9f2; }
            .message-bubble.system { background: #f3f4f6; border-color: #d6d8dc; }
            .attachment-preview { border: 1px solid #cfd8d3; border-radius: 8px; }
            .tool-panel { background: #eef4ff; border: 1px solid #cad9f2; border-radius: 8px; }
            .approval-panel { background: #fff8e8; border: 1px solid #e0b65f; border-radius: 8px; }
            """
        )
        ui.query("body").classes("overflow-hidden")

        with ui.row().classes("chat-shell w-full no-wrap"):
            with ui.column().classes("w-80 h-full bg-[#e9efeb] p-4 gap-3"):
                ui.label("Agent Chat").classes("text-lg font-semibold")
                self._session_select = ui.select(
                    self._session_options(),
                    label="Saved sessions",
                    value="",
                    on_change=self._select_session,
                ).classes("w-full")
                self._session_input = ui.input(
                    "Session",
                    placeholder="default",
                ).classes("w-full")
                self._model_select = ui.select(
                    self._model_options,
                    label="Model",
                    value=self._models[0].model_name,
                ).classes("w-full")
                ui.upload(
                    label="Attachments",
                    multiple=True,
                    auto_upload=True,
                    on_upload=self._on_upload,
                ).classes("w-full")
                self._attachment_list = ui.column().classes("w-full gap-1")
                ui.button(
                    "Reload history",
                    icon="refresh",
                    on_click=self._reload_history,
                ).props("outline").classes("w-full")
                ui.button(
                    "Refresh sessions",
                    icon="manage_search",
                    on_click=self._refresh_sessions,
                ).props("outline").classes("w-full")
                ui.button(
                    "Clear view",
                    icon="delete_sweep",
                    on_click=self._clear_view,
                ).props("outline").classes("w-full")
                self._status = ui.label("Idle").classes("text-sm text-gray-600")

            with ui.column().classes("flex-1 h-full"):
                self._messages = ui.column().classes(
                    "message-list flex-1 w-full overflow-auto p-4 gap-3"
                )
                with ui.row().classes("w-full bg-white border-t p-4 items-end"):
                    self._message_input = ui.textarea(
                        placeholder="Message",
                    ).props("autogrow outlined").classes("flex-1")
                    self._message_input.on(
                        "keydown",
                        self._handle_message_keydown,
                        js_handler=(
                            "(event) => {"
                            " if (event.key === 'Enter' && !event.shiftKey) {"
                            "   event.preventDefault();"
                            "   emit({key: event.key, shiftKey: event.shiftKey});"
                            " }"
                            "}"
                        ),
                    )
                    ui.button(
                        "Send",
                        icon="send",
                        on_click=self._send_message,
                    ).classes("h-12")

        ui.timer(0.1, self._reload_history, once=True)

    async def _select_session(self) -> None:
        selected = str(self._session_select.value or "")
        self._session_input.value = selected
        await self._reload_history()

    def _refresh_sessions(self) -> None:
        self._session_select.set_options(self._session_options())
        current = self._session_id or ""
        if current in self._session_select.options:
            self._session_select.value = current
        ui.notify("Session list refreshed.", type="info")

    def _session_options(self) -> dict[str, str]:
        options = {"": "New session / default"}
        for session in self._list_history_sessions():
            options[session.session_id] = session.session_id
        return options

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

    async def _on_upload(self, event: UploadEventArguments) -> None:
        uploaded = event.file
        name = str(getattr(uploaded, "name", "attachment"))
        media_type = (
            getattr(uploaded, "content_type", None)
            or getattr(uploaded, "type", None)
            or mimetypes.guess_type(name)[0]
            or "application/octet-stream"
        )
        data = uploaded.read()
        if asyncio.iscoroutine(data):
            data = await data
        self._attachments.append(
            Attachment(name=name, media_type=str(media_type), data=bytes(data))
        )
        self._render_attachments()

    def _render_attachments(self) -> None:
        self._attachment_list.clear()
        with self._attachment_list:
            for item in self._attachments:
                ui.label(f"{item.name} ({len(item.data)} bytes)").classes(
                    "text-xs text-gray-600"
                )

    async def _reload_history(self) -> None:
        messages = await self._manager.store.read_messages(self._session_id)
        self._render_history(messages)
        self._restore_pending_approval(messages)

    def _clear_view(self) -> None:
        self._messages.clear()

    async def _send_message(self) -> None:
        if self._is_running:
            return
        if self._pending_requests:
            ui.notify("Approve or deny the pending tool request first.", type="warning")
            return
        text = str(self._message_input.value or "").strip()
        if not text and not self._attachments:
            ui.notify("message or files is required.", type="warning")
            return

        contents = [Content.from_text(text)] if text else []
        contents.extend(
            Content.from_data(
                data=item.data,
                media_type=item.media_type,
                additional_properties={"filename": item.name},
            )
            for item in self._attachments
        )

        self._render_user_input(text, self._attachments)
        self._message_input.value = ""
        self._attachments.clear()
        self._render_attachments()

        await self._run_agent(Message(role="user", contents=contents))

    async def _handle_message_keydown(self, event: GenericEventArguments) -> None:
        args = event.args if isinstance(event.args, dict) else {}
        if args.get("key") == "Enter" and not args.get("shiftKey"):
            await self._send_message()

    async def _set_tool_approval(self, index: int, approved: bool) -> None:
        if not self._pending_requests:
            ui.notify("No pending approval request.", type="warning")
            return
        if not 0 <= index < len(self._pending_requests):
            ui.notify("Invalid approval request.", type="warning")
            return

        self._pending_approvals[index] = approved
        status_label = self._approval_status_labels.get(index)
        if status_label is not None:
            status_label.text = "Approved" if approved else "Denied"
            status_label.classes(
                replace=(
                    "text-sm font-semibold text-green-700"
                    if approved
                    else "text-sm font-semibold text-red-700"
                )
            )

        if len(self._pending_approvals) != len(self._pending_requests):
            remaining = len(self._pending_requests) - len(self._pending_approvals)
            self._status.text = f"Approval required ({remaining} remaining)"
            return

        approvals = [
            self._pending_approvals[index]
            for index in range(len(self._pending_requests))
        ]
        message = build_tool_approval_response_message(
            self._pending_requests,
            approvals,
        )
        self._pending_requests = []
        self._pending_approvals = {}
        self._approval_status_labels = {}
        await self._run_agent(message)

    async def _run_agent(self, message: Message | Sequence[Message]) -> None:
        if self._is_running:
            return
        self._is_running = True
        try:
            state = await self._manager.get(
                session_id=self._session_id,
                model_name=str(self._model_select.value),
            )
            self._status.text = "Running"
            assistant_column = self._add_message_shell("assistant", sent=False)
            text_view: StreamingTextView | None = None

            async with state.lock:
                stream = state.runtime.app.agent.run(
                    message,
                    session=state.runtime.session,
                    stream=True,
                )
                async for chunk in stream:
                    text_view = await self._render_chunk(
                        assistant_column,
                        chunk.contents,
                        chunk.text,
                        text_view,
                    )

                final_response = await stream.get_final_response()
                state.pending_approval = pending_tool_approval_context(
                    final_response.messages
                )

            stored_messages = await state.runtime.app.store.read_messages(
                state.runtime.session_id
            )
            self._render_history(stored_messages or list(final_response.messages))

            if state.pending_approval.requests:
                self._pending_requests = list(state.pending_approval.requests)
                self._pending_approvals = {}
                self._approval_status_labels = {}
                self._render_approval(self._pending_requests)
                self._status.text = (
                    f"Approval required ({len(self._pending_requests)} remaining)"
                )
                self._refresh_sessions()
                return

            self._status.text = "Idle"
            self._refresh_sessions()
        except Exception as error:
            self._status.text = "Error"
            ui.notify(str(error), type="negative", multi_line=True)
            raise
        finally:
            self._is_running = False

    def _render_user_input(self, text: str, attachments: Sequence[Attachment]) -> None:
        column = self._add_message_shell("user", sent=True)
        with column:
            if text:
                ui.markdown(text).classes("whitespace-pre-wrap")
            for item in attachments:
                self._render_attachment(
                    media_type=item.media_type,
                    filename=item.name,
                    data=item.data,
                )

    def _render_history(self, messages: Sequence[Message]) -> None:
        self._messages.clear()
        with self._messages:
            for message in messages:
                column = self._add_message_shell(
                    str(message.role),
                    sent=str(message.role) == "user",
                )
                with column:
                    for content in message.contents:
                        self._render_content(content)

    def _restore_pending_approval(self, messages: Sequence[Message]) -> None:
        approval_context = pending_tool_approval_context(messages)
        self._pending_requests = list(approval_context.requests)
        self._pending_approvals = {}
        self._approval_status_labels = {}
        if not self._pending_requests:
            self._status.text = "Idle"
            return

        self._render_approval(self._pending_requests)
        self._status.text = (
            f"Approval required ({len(self._pending_requests)} remaining)"
        )

    async def _render_chunk(
        self,
        column: ui.column,
        contents: Sequence[Content],
        text: str | None,
        text_view: StreamingTextView | None,
    ) -> StreamingTextView | None:
        with column:
            for event in self._current_renderer.render(contents, text):
                text_view = await self._render_event(event, text_view=text_view)
        return text_view

    async def _render_event(
        self,
        event: RenderEvent,
        *,
        text_view: StreamingTextView | None = None,
    ) -> StreamingTextView | None:
        if event.kind == "text":
            return await self._append_stream_text(event.text, text_view)
        if event.kind == "reasoning":
            ui.markdown(event.text).classes("whitespace-pre-wrap text-gray-600")
            return text_view
        if event.kind == "data" and isinstance(event.payload, Content):
            self._render_data_content(event.payload)
            return text_view
        if event.kind == "approval_request" and isinstance(event.payload, Content):
            with ui.card().classes("approval-panel w-full p-3"):
                ui.label(f"Approval required: {tool_approval_name(event.payload)}")
                ui.code(tool_approval_arguments(event.payload)).classes("w-full")
            return text_view
        if event.kind in {"tool_call", "tool_result", "mcp_call", "mcp_result"}:
            with ui.card().classes("tool-panel w-full p-3"):
                ui.label(self._event_title(event))
                ui.code(event.text).classes("w-full")
            return text_view
        if event.kind == "usage":
            ui.label(f"Usage: {event.text}").classes("text-xs text-gray-500")
            return text_view
        with ui.expansion(event.content_type or event.kind).classes("w-full"):
            ui.code(_format_json(event.payload.to_dict() if isinstance(event.payload, Content) else event.payload)).classes("w-full")
        return text_view

    async def _append_stream_text(
        self,
        text: str,
        text_view: StreamingTextView | None,
    ) -> StreamingTextView:
        if text_view is None:
            text_view = StreamingTextView(element_id=f"stream-{uuid4().hex}")
            ui.html(
                f'<span id="{text_view.element_id}"></span>',
                sanitize=False,
            ).classes("whitespace-pre-wrap break-words leading-relaxed")
        await ui.run_javascript(
            "document.getElementById("
            f"{json.dumps(text_view.element_id)}"
            ")?.appendChild(document.createTextNode("
            f"{json.dumps(text)}"
            "));"
        )
        return text_view

    def _event_title(self, event: RenderEvent) -> str:
        if event.kind == "tool_call":
            return "Tool call"
        if event.kind == "tool_result":
            return "Tool result"
        if event.kind == "mcp_call":
            return "MCP call"
        if event.kind == "mcp_result":
            return "MCP result"
        return event.kind

    def _render_approval(self, requests: Sequence[Content]) -> None:
        with self._messages:
            with ui.card().classes("approval-panel w-full max-w-5xl mx-auto p-3"):
                ui.label("Tool approval").classes("font-semibold")
                ui.label(
                    "Choose Approve or Deny for each request. The run resumes automatically after all requests are decided."
                ).classes("text-sm text-gray-600")
                for index, request in enumerate(requests):
                    with ui.card().classes("w-full p-3 bg-white"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label(
                                f"{index + 1}. {tool_approval_name(request)}"
                            ).classes("font-medium")
                            self._approval_status_labels[index] = ui.label(
                                "Pending"
                            ).classes("text-sm font-semibold text-amber-700")
                        ui.code(tool_approval_arguments(request)).classes("w-full")
                        with ui.row().classes("gap-2"):
                            ui.button(
                                "Approve",
                                icon="check",
                                on_click=partial(self._set_tool_approval, index, True),
                            )
                            ui.button(
                                "Deny",
                                icon="close",
                                color="negative",
                                on_click=partial(self._set_tool_approval, index, False),
                            )

    def _add_message_shell(self, role: str, *, sent: bool) -> ui.column:
        role_class = "user" if sent else role if role in {"assistant", "tool", "system"} else "system"
        with self._messages:
            with ui.element("div").classes(f"message-row {role_class}"):
                with ui.element("div").classes("message-stack"):
                    ui.label(role).classes("message-role")
                    with ui.element("div").classes(f"message-bubble {role_class}"):
                        return ui.column().classes("w-full gap-2")

    def _render_content(self, content: Content) -> None:
        for event in self._current_renderer.render_content(content):
            if event.kind == "text":
                ui.markdown(event.text).classes("whitespace-pre-wrap")
            elif event.kind == "reasoning":
                ui.markdown(event.text).classes("whitespace-pre-wrap text-gray-600")
            elif event.kind == "data" and isinstance(event.payload, Content):
                self._render_data_content(event.payload)
            elif event.kind == "approval_request" and isinstance(event.payload, Content):
                with ui.card().classes("approval-panel w-full p-3"):
                    ui.label(f"Approval required: {tool_approval_name(event.payload)}")
                    ui.code(tool_approval_arguments(event.payload)).classes("w-full")
            elif event.kind in {"tool_call", "tool_result", "mcp_call", "mcp_result"}:
                with ui.card().classes("tool-panel w-full p-3"):
                    ui.label(self._event_title(event))
                    ui.code(event.text).classes("w-full")
            elif event.kind == "usage":
                ui.label(f"Usage: {event.text}").classes("text-xs text-gray-500")
            elif isinstance(event.payload, Content):
                with ui.expansion(event.content_type or event.kind).classes("w-full"):
                    ui.code(_format_json(event.payload.to_dict())).classes("w-full")

    def _render_data_content(self, content: Content) -> None:
        content_type = str(content.type)

        if content_type == "data":
            value = content.to_dict()
            media_type = value.get("media_type") or getattr(
                content, "media_type", "application/octet-stream"
            )
            filename = str(content.additional_properties.get("filename") or "attachment")
            uri = value.get("uri")
            data = _data_uri_bytes(uri)
            self._render_attachment(
                media_type=str(media_type),
                filename=filename,
                data=data,
                uri=uri if isinstance(uri, str) else None,
            )
            return

    @property
    def _current_renderer(self):
        return self._renderers[str(self._model_select.value)]

    def _render_attachment(
        self,
        *,
        media_type: str,
        filename: str,
        data: bytes | None = None,
        uri: str | None = None,
    ) -> None:
        if media_type.startswith("image/") and (uri or data is not None):
            image_uri = uri or _data_uri(media_type, data or b"")
            ui.image(image_uri).classes("attachment-preview max-w-md")
            return
        size = len(data) if data is not None else "unknown"
        ui.label(f"{filename} ({media_type}, {size} bytes)").classes(
            "text-sm text-gray-600"
        )

    @property
    def _session_id(self) -> str | None:
        value = str(self._session_input.value or "").strip()
        return value or None


def _format_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _data_uri(media_type: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _data_uri_bytes(uri: Any) -> bytes | None:
    if not isinstance(uri, str) or ";base64," not in uri:
        return None
    try:
        return base64.b64decode(uri.split(";base64,", 1)[1], validate=False)
    except ValueError:
        return None


_manager = ChatRuntimeManager()
_page_registered = False


def create_chat_ui() -> None:
    global _page_registered
    if _page_registered:
        return
    _page_registered = True

    @ui.page("/")
    def index() -> None:
        ChatPage(_manager)


def run_chat_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 8081,
    reload: bool = False,
) -> None:
    create_chat_ui()
    ui.run(
        host=host,
        port=port,
        title="Demo Agent Chat UI",
        reload=reload,
        show=False,
        language="ja",
    )


if __name__ == "__main__":
    run_chat_ui()
