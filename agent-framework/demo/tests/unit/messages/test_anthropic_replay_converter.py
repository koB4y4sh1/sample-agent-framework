from __future__ import annotations

import json

from agent.messages import AnthropicReplayConverter
from agent_framework import Content, Message


def _json_value(value):
    return json.loads(value) if isinstance(value, str) else value


class TestAnthropicReplayConverter:
    def test_converts_unsupported_tool_call_and_result_types(self) -> None:
        converter = AnthropicReplayConverter()
        message = Message(
            "assistant",
            [
                Content.from_code_interpreter_tool_call(
                    call_id="code_1",
                    inputs=[Content.from_text("print('ok')")],
                ),
                Content.from_shell_tool_call(
                    call_id="shell_1",
                    commands=["pwd"],
                    timeout_ms=1000,
                ),
            ],
        )
        result_message = Message(
            "tool",
            [
                Content.from_code_interpreter_tool_result(
                    call_id="code_1",
                    outputs=[Content.from_text("ok")],
                ),
                Content.from_shell_tool_result(
                    call_id="shell_1",
                    outputs=[Content.from_text("C:/Workspace")],
                ),
            ],
        )

        converted = converter.convert_messages([message, result_message])

        code_call = converted[0].contents[0]
        shell_call = converted[0].contents[1]
        code_result = converted[1].contents[0]
        shell_result = converted[1].contents[1]
        assert code_call.type == "function_call"
        assert code_call.name == "code_execution"
        assert _json_value(code_call.arguments) == {"inputs": [{"type": "text", "text": "print('ok')"}]}
        assert shell_call.type == "function_call"
        assert shell_call.name == "bash"
        assert _json_value(shell_call.arguments) == {"commands": ["pwd"], "timeout_ms": 1000}
        assert code_result.type == "function_result"
        assert _json_value(code_result.result) == {"outputs": [{"type": "text", "text": "ok"}]}
        assert shell_result.type == "function_result"
        assert _json_value(shell_result.result) == {"outputs": [{"type": "text", "text": "C:/Workspace"}]}

    def test_keeps_other_content_types_and_missing_call_id_unchanged(self) -> None:
        converter = AnthropicReplayConverter()
        hosted_file = Content.from_hosted_file("file_1")
        missing_call_id = Content.from_shell_tool_call(commands=["pwd"])

        converted = converter.convert_message(Message("assistant", [hosted_file, missing_call_id]))

        assert converted.contents[0] is hosted_file
        assert converted.contents[1] is missing_call_id
