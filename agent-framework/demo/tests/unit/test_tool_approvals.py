from __future__ import annotations

import importlib.util

from agent_framework import Content, Message
from ui.approval import (
    APPROVAL_ALWAYS_ARGUMENTS,
    APPROVAL_ALWAYS_TOOL,
    APPROVAL_DENY,
    APPROVAL_ONCE,
    ToolApprovalContext,
    build_tool_approval_response_message,
    format_tool_approval_prompt,
    pending_tool_approval_context,
    pending_tool_approval_requests,
)
from ui.cli import DemoChatCLI


class _DummyRenderer:
    def start(self) -> None:
        pass

    def render(self, contents, text) -> None:  # type: ignore[no-untyped-def]
        pass

    def finish(self) -> None:
        pass


def _approval_request(
    *,
    request_id: str = "approval_1",
    call_id: str = "call_1",
    name: str = "get_weather",
    arguments: dict[str, str] | None = None,
) -> Content:
    function_call = Content.from_function_call(
        call_id,
        name,
        arguments=arguments or {"location": "Tokyo"},
    )
    return Content.from_function_approval_request(request_id, function_call)


class TestToolApprovals:
    def test_agent_package_does_not_own_tool_approval_helpers(self) -> None:
        assert importlib.util.find_spec("agent.approvals") is None

    def test_formats_approval_prompt(self) -> None:
        request = _approval_request()

        prompt = format_tool_approval_prompt(request)

        assert prompt.startswith("[Approve] get_weather")
        assert '{"location":"Tokyo"}' in prompt
        assert "0: 拒否" in prompt
        assert "1: このリクエストのみ許可" in prompt
        assert "2: 引数の実行を許可する" in prompt
        assert "3: このツール実行を自動許可する" in prompt

    def test_builds_approval_response_message_from_bool_inputs(self) -> None:
        request = _approval_request()

        message = build_tool_approval_response_message([request], [True])

        assert message.role == "user"
        assert len(message.contents) == 1
        response = message.contents[0]
        assert response.type == "function_approval_response"
        assert response.approved is True
        assert response.id == "approval_1"
        assert response.function_call.name == "get_weather"

    def test_builds_approval_response_for_explicit_decisions(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("ALLOW_TOOLS_MEMORY_ROOT", str(tmp_path))
        monkeypatch.setenv("USERNAME", "alice")
        request = _approval_request()

        message = build_tool_approval_response_message(
            [request, request, request, request],
            [
                APPROVAL_DENY,
                APPROVAL_ONCE,
                APPROVAL_ALWAYS_ARGUMENTS,
                APPROVAL_ALWAYS_TOOL,
            ],
        )

        assert [content.approved for content in message.contents] == [
            False,
            True,
            True,
            True,
        ]
        assert (tmp_path / "alice" / "allow_tools.json").exists()

    def test_returns_no_approval_when_latest_message_is_user_response(self) -> None:
        approved_request = _approval_request(request_id="approval_1", call_id="call_1")
        pending_request = _approval_request(request_id="approval_2", call_id="call_2")

        messages = [
            Message("assistant", [approved_request, pending_request]),
            Message("user", [approved_request.to_function_approval_response(True)]),
        ]

        assert pending_tool_approval_requests(messages) == []

    def test_returns_no_approval_when_latest_message_is_tool_result(self) -> None:
        request = _approval_request(request_id="approval_1", call_id="call_1")

        messages = [
            Message("assistant", [Content.from_text("I need a tool."), request]),
            Message("tool", [Content.from_function_result("call_1", result="Tokyo weather is sunny.")]),
        ]

        assert pending_tool_approval_requests(messages) == []

    def test_returns_unresolved_approval_context_with_full_assistant_message(self) -> None:
        pending_request = _approval_request()
        assistant_message = Message(
            "assistant",
            [
                Content.from_text("I checked the request and need a tool."),
                Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"}),
                pending_request,
            ],
        )

        context = pending_tool_approval_context([Message("user", [Content.from_text("Tokyo weather")]), assistant_message])

        assert context.requests == [pending_request]
        assert len(context.assistant_messages) == 1
        assert [content.type for content in context.assistant_messages[0].contents] == [
            "text",
            "function_call",
            "function_approval_request",
        ]

    def test_returns_only_latest_unresolved_approval_context(self) -> None:
        old_request = _approval_request(request_id="approval_1", call_id="call_1")
        latest_request = _approval_request(request_id="approval_2", call_id="call_2")
        old_assistant_message = Message(
            "assistant",
            [Content.from_text("old approval"), old_request],
        )
        latest_assistant_message = Message(
            "assistant",
            [Content.from_text("latest approval"), latest_request],
        )

        context = pending_tool_approval_context([
            Message("user", [Content.from_text("first")]),
            old_assistant_message,
            Message("user", [Content.from_text("second")]),
            latest_assistant_message,
        ])
        approval_message = build_tool_approval_response_message(context.requests, [True])

        assert context.requests == [latest_request]
        assert len(context.assistant_messages) == 1
        assert context.assistant_messages[0].contents == latest_assistant_message.contents
        assert approval_message.role == "user"
        assert approval_message.contents[0].function_call.call_id == "call_2"

    def test_cli_builds_only_current_approval_message(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        request = _approval_request()
        assistant_message = Message("assistant", [Content.from_text("I need the tool."), request])
        cli = DemoChatCLI(
            agent=object(),  # type: ignore[arg-type]
            session=object(),
            code_interpreter_status="",
            stream_renderer=_DummyRenderer(),  # type: ignore[arg-type]
            tool_provider=lambda: [],
            all_tools_provider=lambda: [],
        )
        monkeypatch.setattr("builtins.input", lambda: "1")

        message = cli._build_tool_approval_messages(ToolApprovalContext([assistant_message], [request]))

        assert isinstance(message, Message)
        assert message.role == "user"
        assert [content.type for content in message.contents] == ["function_approval_response"]
