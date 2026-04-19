from __future__ import annotations

import importlib.util

from chat_cli import (
    DemoChatCLI,
    build_tool_approval_response_message,
    format_tool_approval_prompt,
    pending_tool_approval_context,
    pending_tool_approval_requests,
    ToolApprovalContext,
)
from agent_framework import Content, Message


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
        r"""正常系：CLI用の承認helperをagent.approvalsとして公開しないこと

        概要:
            tool approvalの入力欄表示やY/N応答生成はCLI境界の処理であり、
            agent packageの公開APIやagent.approvals moduleとして持たない。

        入力例:
            importlib.util.find_spec("agent.approvals")

        期待例:
            None
        """

        assert importlib.util.find_spec("agent.approvals") is None

    def test_formats_approval_prompt(self) -> None:
        r"""正常系：承認入力欄に表示するprompt形式を固定すること

        概要:
            ツール承認を求める場面で、ユーザーが何を承認するか判断できるよう、
            tool名・arguments・Y/N入力欄を1つの文字列として表示する。

        入力例:
            {
                "type": "function_approval_request",
                "id": "approval_1",
                "function_call": {
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": {"location": "Tokyo"}
                }
            }

        期待例:
            "[Approve] get_weather を実行してもよろしいでしょうか？{\"location\":\"Tokyo\"} Y/N: "
        """
        request = _approval_request()

        prompt = format_tool_approval_prompt(request)

        assert (
            prompt
            == "[Approve] get_weather \u3092\u5b9f\u884c\u3057\u3066\u3082"
            "\u3088\u308d\u3057\u3044\u3067\u3057\u3087\u3046\u304b\uff1f"
            '{"location":"Tokyo"} Y/N: '
        )

    def test_builds_approval_response_message(self) -> None:
        r"""正常系：Y/Nの結果を今回のuser approval responseへ変換すること

        概要:
            CLIで入力された承認可否を、agent.runへ渡せる
            `user` roleの`function_approval_response` messageにする。

        入力例:
            requests = [
                {
                    "type": "function_approval_request",
                    "id": "approval_1",
                    "function_call": {"name": "get_weather"}
                }
            ]
            approvals = [true]

        期待例:
            {
                "role": "user",
                "contents": [
                    {
                        "type": "function_approval_response",
                        "id": "approval_1",
                        "approved": true,
                        "function_call": {"name": "get_weather"}
                    }
                ]
            }
        """
        request = _approval_request()

        message = build_tool_approval_response_message([request], [True])

        assert message.role == "user"
        assert len(message.contents) == 1
        response = message.contents[0]
        assert response.type == "function_approval_response"
        assert response.approved is True
        assert response.id == "approval_1"
        assert response.function_call.name == "get_weather"

    def test_returns_no_approval_when_latest_message_is_user_response(self) -> None:
        r"""正常系：最後のmessageがuser responseなら承認待ちとして復旧しないこと

        概要:
            再起動時の承認復旧判定は、過去履歴全体ではなく最後のmessageを見る。
            最後が`function_approval_response`なら、すでにユーザー応答済みなので
            approval入力欄を再表示しない。

        入力例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "function_approval_request", "id": "approval_1"},
                        {"type": "function_approval_request", "id": "approval_2"}
                    ]
                },
                {
                    "role": "user",
                    "contents": [{"type": "function_approval_response", "id": "approval_1"}]
                }
            ]

        期待例:
            []
        """
        approved_request = _approval_request(request_id="approval_1", call_id="call_1")
        pending_request = _approval_request(request_id="approval_2", call_id="call_2")

        messages = [
            Message("assistant", [approved_request, pending_request]),
            Message("user", [approved_request.to_function_approval_response(True)]),
        ]

        assert pending_tool_approval_requests(messages) == []

    def test_returns_no_approval_when_latest_message_is_tool_result(self) -> None:
        r"""正常系：最後のmessageがtool resultなら承認待ちとして復旧しないこと

        概要:
            再起動時の承認復旧判定は最後のmessageを見る。
            最後が`function_result`なら、承認後のtool実行まで完了しているため、
            approval入力欄を再表示しない。

        入力例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text", "text": "I need a tool."},
                        {"type": "function_approval_request", "id": "approval_1"}
                    ]
                },
                {
                    "role": "tool",
                    "contents": [{"type": "function_result", "call_id": "call_1"}]
                }
            ]

        期待例:
            []
        """
        request = _approval_request(request_id="approval_1", call_id="call_1")

        messages = [
            Message("assistant", [Content.from_text("I need a tool."), request]),
            Message("tool", [Content.from_function_result("call_1", result="Tokyo weather is sunny.")]),
        ]

        assert pending_tool_approval_requests(messages) == []

    def test_returns_unresolved_approval_context_with_full_assistant_message(self) -> None:
        r"""正常系：最後のassistant block全体を承認復旧contextへ保持すること

        概要:
            最後のmessageに`function_approval_request`がある場合だけ承認待ちとして復旧する。
            その際、approval requestだけを切り出さず、直前assistant block内の
            text/function_call/function_approval_requestをまとめて保持する。

        入力例:
            [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text", "text": "I checked the request and need a tool."},
                        {"type": "function_call", "call_id": "call_1"},
                        {"type": "function_approval_request", "id": "approval_1"}
                    ]
                }
            ]

        期待例:
            {
                "requests": [{"type": "function_approval_request", "id": "approval_1"}],
                "assistant_messages": [
                    {
                        "role": "assistant",
                        "contents": [
                            {"type": "text"},
                            {"type": "function_call"},
                            {"type": "function_approval_request"}
                        ]
                    }
                ]
            }
        """
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
        r"""正常系：過去のapproval requestをapprove_messageへ混ぜないこと

        概要:
            履歴内に古いapproval requestが残っていても、承認対象にするのは
            最後のmessageに含まれるrequestだけにする。
            再起動後に生成する承認応答messageにも、過去のapproval requestを含めない。

        入力例:
            [
                {"role": "user", "contents": [{"type": "text", "text": "first"}]},
                {
                    "role": "assistant",
                    "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "approval_1"}]
                },
                {"role": "user", "contents": [{"type": "text", "text": "second"}]},
                {
                    "role": "assistant",
                    "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "approval_2"}]
                }
            ]

        期待例:
            {
                "requests": [{"type": "function_approval_request", "id": "approval_2"}],
                "approval_message": {
                    "role": "user",
                    "contents": [{"type": "function_approval_response", "id": "approval_2"}]
                }
            }
        """
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
        r"""正常系：CLI承認時はagent.runへ今回のuser responseだけを渡すこと

        概要:
            承認復旧contextには直前assistant blockも保持するが、CLIがagent.runへ渡す
            入力は、今回生成した`user function_approval_response`だけにする。
            これにより、再起動後もagent.runの入力に過去assistant blockを重複投入しない。

        入力例:
            context = {
                "assistant_messages": [
                    {
                        "role": "assistant",
                        "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "approval_1"}]
                    }
                ],
                "requests": [{"type": "function_approval_request", "id": "approval_1"}]
            }
            stdin = "y"

        期待例:
            {
                "role": "user",
                "contents": [
                    {"type": "function_approval_response", "id": "approval_1", "approved": true}
                ]
            }
        """
        request = _approval_request()
        assistant_message = Message("assistant", [Content.from_text("I need the tool."), request])
        cli = DemoChatCLI(
            agent=object(),  # type: ignore[arg-type]
            session=object(),
            code_interpreter_status="",
            stream_renderer=_DummyRenderer(),  # type: ignore[arg-type]
        )
        monkeypatch.setattr("builtins.input", lambda: "y")

        message = cli._build_tool_approval_messages(ToolApprovalContext([assistant_message], [request]))

        assert isinstance(message, Message)
        assert message.role == "user"
        assert [content.type for content in message.contents] == ["function_approval_response"]
