from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from agent.contexts import (
    EXECUTED_INPUT_MESSAGES_METADATA_KEY,
    ExecutionContextProvider,
    MessageConversionContextProvider,
)
from agent.history import LocalHistoryProvider, MessageStore
from agent.messages import AnthropicReplayConverter, CommonMessageConverter, ProviderMessageConverter
from agent_framework import (
    AgentResponse,
    AgentSession,
    Content,
    Message,
    SessionContext,
)


class FakeStore(MessageStore):
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages

    def read_messages(self, session_id: str | None) -> list[Message]:
        return self.messages

    def write_messages(self, session_id: str | None, messages) -> None:  # type: ignore[no-untyped-def]
        self.messages = list(messages)


@dataclass(slots=True)
class CompletedRunContext:
    """after_run用に、framework実行後の公開状態だけを表すテストダブル。"""

    session_id: str | None
    input_messages: list[Message]
    response: AgentResponse | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TestMessageConversionLifecycle:
    """run境界での履歴変換とraw復元を検証する。"""

    def test_history_provider_returns_raw_messages(self) -> None:
        """正常系：履歴取得では保存済みMessageを変換せずrawのまま返すこと"""

        async def run() -> None:
            raw_messages = [Message("assistant", [Content.from_text_reasoning(text="reasoning")])]
            history_provider = LocalHistoryProvider(store=FakeStore(raw_messages))

            messages = await history_provider.get_messages("session")

            assert messages[0] is raw_messages[0]
            assert messages[0].contents[0].type == "text_reasoning"

        asyncio.run(run())

    def test_provider_converts_before_run_and_restores_after_run(self) -> None:
        """正常系：run前にprovider用へ変換し、run後はraw履歴へ戻すこと"""

        async def run() -> None:
            raw_messages = [Message("assistant", [Content.from_text_reasoning(text="reasoning")])]
            context = SessionContext(
                session_id="session",
                input_messages=[],
                context_messages={"history": raw_messages},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(reasoning_policy="as_text"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            converted_messages = context.context_messages["history"]
            assert converted_messages is not raw_messages
            assert converted_messages[0].contents[0].type == "text"

            await provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            assert context.context_messages["history"][0] is raw_messages[0]

        asyncio.run(run())

    def test_provider_replays_code_interpreter_history_with_reasoning(self) -> None:
        async def run() -> None:
            raw_messages = [
                Message("user", [Content.from_text("create a sample Word file")]),
                Message(
                    "assistant",
                    [
                        Content.from_text_reasoning(
                            text="I need to create the file with code.",
                            protected_data="signature",
                        ),
                        Content.from_code_interpreter_tool_call(
                            call_id="srvtoolu_1",
                            inputs=[Content.from_text("{}")],
                        ),
                    ],
                ),
                Message(
                    "tool",
                    [
                        Content.from_code_interpreter_tool_result(
                            call_id="srvtoolu_1",
                            outputs=[
                                Content.from_text("created /tmp/sample_document.docx"),
                                Content.from_hosted_file("file_1", name="sample_document.docx"),
                            ],
                        )
                    ],
                ),
                Message("assistant", [Content.from_hosted_file("file_1"), Content.from_text("Word file created.")]),
            ]
            context = SessionContext(
                session_id="session",
                input_messages=[Message("user", [Content.from_text("continue")])],
                context_messages={"history": raw_messages},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=ProviderMessageConverter(target_provider_family="anthropic"),
                replay_converter=AnthropicReplayConverter(),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            converted_messages = context.context_messages["history"]
            assert [message.role for message in converted_messages] == [
                "user",
                "assistant",
                "tool",
                "assistant",
            ]
            assert [content.type for content in converted_messages[1].contents] == [
                "text_reasoning",
                "function_call",
            ]
            assert converted_messages[1].contents[0].protected_data == "signature"
            assert converted_messages[1].contents[1].name == "code_execution"
            assert converted_messages[2].contents[0].type == "function_result"
            assert converted_messages[2].contents[0].call_id == "srvtoolu_1"
            assert json.loads(converted_messages[2].contents[0].result) == {
                "outputs": [
                    {"type": "text", "text": "created /tmp/sample_document.docx"},
                    {"type": "hosted_file", "file_id": "file_1", "name": "sample_document.docx"},
                ]
            }
            assert [content.type for content in converted_messages[3].contents] == ["hosted_file", "text"]
            assert all(message.contents for message in converted_messages)

        asyncio.run(run())

class TestApprovalReplayIntegration:
    """承認要求を含む履歴をbefore_runでproviderへ渡せる形にすることを検証する。"""

    def test_provider_keeps_unresolved_approval_requests_before_replay(self) -> None:
        r"""正常系：未承認要求はbefore_runでproviderへ渡せるfunction_callとして扱うこと

        入力例:
            context_messages["history"] = [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text", "text": "I need to check the weather."},
                        {"type": "function_call", "call_id": "call_1"},
                        {"type": "function_approval_request", "id": "call_1"}
                    ]
                }
            ]
            input_messages = []

        期待例:
            context_messages["history"] = [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text"},
                        {"type": "function_call", "call_id": "call_1"}
                    ]
                }
            ]
        """
        async def run() -> None:
            function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            approval_request = Content.from_function_approval_request("call_1", function_call)
            assistant_message = Message(
                "assistant",
                [Content.from_text("I need to check the weather."), function_call, approval_request],
            )
            raw_messages = [
                Message("user", [Content.from_text("Tokyo weather")]),
                assistant_message,
            ]
            context = SessionContext(
                session_id="session",
                input_messages=[],
                context_messages={"history": raw_messages},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(reasoning_policy="as_text"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            converted_messages = context.context_messages["history"]
            assert len(converted_messages) == 2
            assert converted_messages[0].role == "user"
            assert converted_messages[0].contents[0].text == "Tokyo weather"
            assert converted_messages[1].role == "assistant"
            assert [content.type for content in converted_messages[1].contents] == [
                "text",
                "function_call",
            ]

        asyncio.run(run())

    def test_provider_omits_approval_requests_supplied_as_current_input_before_replay(self) -> None:
        r"""正常系：今回inputに承認要求と承認応答がある場合、履歴側の同じ承認要求は渡さないこと

        入力例:
            context_messages["history"] = [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]}
            ]
            input_messages = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_call", "call_id": "call_1"}, {"type": "function_approval_request", "id": "call_1"}]},
                {"role": "user", "contents": [{"type": "function_approval_response", "id": "call_1"}]}
            ]

        期待例:
            replay_messages = [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_call", "call_id": "call_1"}]},
                {"role": "user", "contents": [{"type": "function_approval_response", "id": "call_1"}]}
            ]
        """
        async def run() -> None:
            function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            approval_request = Content.from_function_approval_request("call_1", function_call)
            assistant_message = Message(
                "assistant",
                [Content.from_text("I need to check the weather."), function_call, approval_request],
            )
            raw_messages = [
                Message("user", [Content.from_text("Tokyo weather")]),
                assistant_message,
            ]
            context = SessionContext(
                session_id="session",
                input_messages=[
                    assistant_message,
                    Message("user", [approval_request.to_function_approval_response(True)]),
                ],
                context_messages={"history": raw_messages},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(reasoning_policy="as_text"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            converted_messages = context.context_messages["history"]
            execution_messages = [*converted_messages, *context.input_messages]
            assert len(converted_messages) == 1
            assert converted_messages[0].role == "user"
            assert converted_messages[0].contents[0].text == "Tokyo weather"
            assert [message.role for message in execution_messages] == ["user", "assistant", "user"]
            assert [content.type for content in execution_messages[1].contents] == [
                "text",
                "function_call",
            ]
            assert execution_messages[2].contents[0].type == "function_approval_response"

        asyncio.run(run())

    def test_provider_omits_stale_approval_requests_when_current_approval_input_exists(self) -> None:
        r"""正常系：今回inputに別承認がある場合、古い未承認要求は履歴replayから外すこと

        入力例:
            context_messages["history"] = [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]}
            ]
            input_messages = [
                {"role": "assistant", "contents": [{"type": "function_approval_request", "id": "call_2"}]},
                {"role": "user", "contents": [{"type": "function_approval_response", "id": "call_2"}]}
            ]

        期待例:
            context_messages["history"] = [
                {"role": "user"},
                {"role": "assistant", "contents": [{"type": "text"}]}
            ]
        """
        async def run() -> None:
            old_function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            old_approval_request = Content.from_function_approval_request("call_1", old_function_call)
            current_function_call = Content.from_function_call("call_2", "get_weather", arguments={"location": "Osaka"})
            current_approval_request = Content.from_function_approval_request("call_2", current_function_call)
            raw_messages = [
                Message("user", [Content.from_text("Tokyo weather")]),
                Message("assistant", [Content.from_text("I need to check the weather."), old_approval_request]),
            ]
            current_assistant_message = Message(
                "assistant",
                [Content.from_text("I need to check another city."), current_function_call, current_approval_request],
            )
            context = SessionContext(
                session_id="session",
                input_messages=[
                    current_assistant_message,
                    Message("user", [current_approval_request.to_function_approval_response(True)]),
                ],
                context_messages={"history": raw_messages},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(reasoning_policy="as_text"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            converted_messages = context.context_messages["history"]
            assert len(converted_messages) == 2
            assert converted_messages[0].role == "user"
            assert converted_messages[1].role == "assistant"
            assert [content.type for content in converted_messages[1].contents] == ["text"]

        asyncio.run(run())

    def test_provider_keeps_only_matching_request_when_current_input_is_approval_response(self) -> None:
        r"""正常系：今回inputが承認応答だけの場合、対応する直近承認要求だけを履歴replayに残すこと

        入力例:
            context_messages["history"] = [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {"role": "user", "contents": [{"type": "text", "text": "Osaka weather"}]},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_2"}]}
            ]
            input_messages = [
                {"role": "user", "contents": [{"type": "function_approval_response", "id": "call_2"}]}
            ]

        期待例:
            context_messages["history"] = [
                {"role": "user"},
                {"role": "assistant", "contents": [{"type": "text"}]},
                {"role": "user"},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_call", "call_id": "call_2"}]}
            ]
        """
        async def run() -> None:
            old_function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            old_approval_request = Content.from_function_approval_request("call_1", old_function_call)
            current_function_call = Content.from_function_call("call_2", "get_weather", arguments={"location": "Osaka"})
            current_approval_request = Content.from_function_approval_request("call_2", current_function_call)
            raw_messages = [
                Message("user", [Content.from_text("Tokyo weather")]),
                Message("assistant", [Content.from_text("I need to check Tokyo."), old_approval_request]),
                Message("user", [Content.from_text("Osaka weather")]),
                Message("assistant", [Content.from_text("I need to check Osaka."), current_approval_request]),
            ]
            context = SessionContext(
                session_id="session",
                input_messages=[
                    Message("user", [current_approval_request.to_function_approval_response(True)]),
                ],
                context_messages={"history": raw_messages},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(reasoning_policy="as_text"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            converted_messages = context.context_messages["history"]
            assert [message.role for message in converted_messages] == ["user", "assistant", "user", "assistant"]
            assert [content.type for content in converted_messages[1].contents] == ["text"]
            assert [content.type for content in converted_messages[3].contents] == [
                "text",
                "function_call",
            ]
            assert context.input_messages[0].role == "user"
            assert context.input_messages[0].contents[0].type == "function_approval_response"

        asyncio.run(run())

    def test_provider_keeps_approved_approval_request_with_function_result(self) -> None:
        r"""正常系：承認済み要求とtool結果はproviderへ渡せるcall/resultペアとして残すこと

        入力例:
            context_messages["history"] = [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]}
            ]

        期待例:
            context_messages["history"] = [
                {"role": "user"},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_call", "call_id": "call_1"}]},
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]}
            ]
        """
        async def run() -> None:
            function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            approval_request = Content.from_function_approval_request("call_1", function_call)
            function_result = Content.from_function_result("call_1", result="Tokyo weather is sunny.")
            raw_messages = [
                Message("user", [Content.from_text("Tokyo weather")]),
                Message("assistant", [Content.from_text("I need to check the weather."), approval_request]),
                Message("tool", [function_result]),
            ]
            context = SessionContext(
                session_id="session",
                input_messages=[],
                context_messages={"history": raw_messages},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(reasoning_policy="as_text"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            converted_messages = context.context_messages["history"]
            assert [message.role for message in converted_messages] == ["user", "assistant", "tool"]
            assert [content.type for content in converted_messages[1].contents] == ["text", "function_call"]
            assert converted_messages[1].contents[1].arguments == '{"location": "Tokyo"}'
            assert converted_messages[2].contents[0].type == "function_result"
            assert converted_messages[2].contents[0].call_id == "call_1"
            assert converted_messages[2].contents[0].result == "Tokyo weather is sunny."

        asyncio.run(run())

    def test_provider_converts_approval_request_to_function_call_and_keeps_result_text(self) -> None:
        r"""正常系：OpenAI replayでもlocal承認要求はfunction_callにし、拒否結果は元のtextで保持すること

        入力例:
            context_messages["history"] = [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {
                    "role": "tool",
                    "contents": [
                        {
                            "type": "function_result",
                            "call_id": "call_1",
                            "result": "Error: Tool call invocation was rejected by user."
                        }
                    ]
                }
            ]

        期待例:
            context_messages["history"] = [
                {"role": "user"},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_call", "call_id": "call_1"}]},
                {
                    "role": "tool",
                    "contents": [
                        {
                            "type": "function_result",
                            "call_id": "call_1",
                            "result": "Error: Tool call invocation was rejected by user."
                        }
                    ]
                }
            ]
        """
        async def run() -> None:
            function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            approval_request = Content.from_function_approval_request("call_1", function_call)
            function_result = Content.from_function_result(
                "call_1",
                result="Error: Tool call invocation was rejected by user.",
            )
            raw_messages = [
                Message("user", [Content.from_text("Tokyo weather")]),
                Message("assistant", [Content.from_text("I need to check the weather."), approval_request]),
                Message("tool", [function_result]),
            ]
            context = SessionContext(
                session_id="session",
                input_messages=[],
                context_messages={"history": raw_messages},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=ProviderMessageConverter(target_provider_family="openai"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            converted_messages = context.context_messages["history"]
            assert [content.type for content in converted_messages[1].contents] == ["text", "function_call"]
            assert converted_messages[1].contents[1].call_id == "call_1"
            result = converted_messages[2].contents[0]
            assert result.type == "function_result"
            assert result.result == "Error: Tool call invocation was rejected by user."
            assert isinstance(result.items, list)
            assert result.items[0].text == result.result

        asyncio.run(run())

    def test_provider_keeps_stale_approved_approval_request_when_current_approval_input_exists(self) -> None:
        r"""正常系：今回inputに別承認があっても、結果済みの古い承認要求とtool結果はreplayに残すこと

        入力例:
            context_messages["history"] = [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]}
            ]
            input_messages = [
                {"role": "assistant", "contents": [{"type": "function_approval_request", "id": "call_2"}]},
                {"role": "user", "contents": [{"type": "function_approval_response", "id": "call_2"}]}
            ]

        期待例:
            context_messages["history"] = [
                {"role": "user"},
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_call", "call_id": "call_1"}]},
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]}
            ]
        """
        async def run() -> None:
            old_function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            old_approval_request = Content.from_function_approval_request("call_1", old_function_call)
            old_function_result = Content.from_function_result("call_1", result="Tokyo weather is sunny.")
            current_function_call = Content.from_function_call("call_2", "get_weather", arguments={"location": "Osaka"})
            current_approval_request = Content.from_function_approval_request("call_2", current_function_call)
            raw_messages = [
                Message("user", [Content.from_text("Tokyo weather")]),
                Message("assistant", [Content.from_text("I need to check the weather."), old_approval_request]),
                Message("tool", [old_function_result]),
            ]
            current_assistant_message = Message(
                "assistant",
                [Content.from_text("I need to check another city."), current_function_call, current_approval_request],
            )
            context = SessionContext(
                session_id="session",
                input_messages=[
                    current_assistant_message,
                    Message("user", [current_approval_request.to_function_approval_response(True)]),
                ],
                context_messages={"history": raw_messages},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(reasoning_policy="as_text"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            converted_messages = context.context_messages["history"]
            assert [message.role for message in converted_messages] == ["user", "assistant", "tool"]
            assert [content.type for content in converted_messages[1].contents] == ["text", "function_call"]
            assert converted_messages[1].contents[1].call_id == "call_1"
            assert converted_messages[2].contents[0].type == "function_result"
            assert converted_messages[2].contents[0].call_id == "call_1"
            assert converted_messages[2].contents[0].result == "Tokyo weather is sunny."

        asyncio.run(run())

    def test_provider_converts_input_approval_arguments_before_replay(self) -> None:
        r"""正常系：今回input内のfunction_call argumentsは承認応答内も含めJSON文字列にすること

        入力例:
            input_messages = [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "function_call", "call_id": "call_1", "arguments": {"location": "Tokyo"}},
                        {"type": "function_approval_request", "id": "call_1"}
                    ]
                },
                {
                    "role": "user",
                    "contents": [
                        {
                            "type": "function_approval_response",
                            "id": "call_1",
                            "function_call": {"arguments": {"location": "Tokyo"}}
                        }
                    ]
                }
            ]

        期待例:
            input_messages = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_call", "arguments": "{\"location\": \"Tokyo\"}"}]},
                {"role": "user", "contents": [{"type": "function_approval_response", "function_call": {"arguments": "{\"location\": \"Tokyo\"}"}}]}
            ]
        """
        async def run() -> None:
            function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            approval_request = Content.from_function_approval_request("call_1", function_call)
            input_messages = [
                Message(
                    "assistant",
                    [Content.from_text("I need to check the weather."), function_call, approval_request],
                ),
                Message("user", [approval_request.to_function_approval_response(True)]),
            ]
            context = SessionContext(
                session_id="session",
                input_messages=input_messages,
                context_messages={},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(reasoning_policy="as_text"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            assistant_message, approval_message = context.input_messages
            assert assistant_message.contents[1].type == "function_call"
            assert assistant_message.contents[1].arguments == '{"location": "Tokyo"}'
            nested_call = approval_message.contents[0].function_call
            assert nested_call is not None
            assert nested_call.arguments == '{"location": "Tokyo"}'

            await provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            assert context.input_messages == input_messages

        asyncio.run(run())

class TestApprovalHistoryStorageIntegration:
    """承認応答run後の履歴保存用metadataを検証する。"""

    def test_provider_exposes_executed_approval_input_for_history_storage(self) -> None:
        r"""正常系：承認応答run後に得たassistant承認要求とtool結果を履歴保存用metadataへ退避すること

        入力例:
            original input_messages = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_call"}, {"type": "function_approval_request", "id": "call_1"}]},
                {"role": "user", "contents": [{"type": "function_approval_response", "id": "call_1"}]}
            ]
            executed input_messages = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_call"}]},
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]}
            ]

        期待例:
            context.metadata["message_conversion:executed_input_messages"] = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]}
            ]
        """
        async def run() -> None:
            original_function_call = Content.from_function_call(
                "call_1",
                "get_weather",
                arguments={"location": "Tokyo"},
            )
            approval_request = Content.from_function_approval_request("call_1", original_function_call)
            input_messages = [
                Message(
                    "assistant",
                    [Content.from_text("I need to check the weather."), original_function_call, approval_request],
                ),
                Message("user", [approval_request.to_function_approval_response(True)]),
            ]
            context = SessionContext(
                session_id="session",
                input_messages=input_messages,
                context_messages={},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(reasoning_policy="as_text"),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            context.input_messages[0].contents = [
                content for content in context.input_messages[0].contents if content.type != "function_approval_request"
            ]
            context.input_messages[1].role = "tool"
            context.input_messages[1].contents = [
                Content.from_function_result("call_1", result="Tokyo weather is sunny.")
            ]

            await provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            executed_messages = context.metadata[EXECUTED_INPUT_MESSAGES_METADATA_KEY]
            assert [content.type for content in executed_messages[0].contents] == ["text", "function_approval_request"]
            assert executed_messages[1].role == "tool"
            assert executed_messages[1].contents[0].type == "function_result"
            assert context.input_messages == input_messages

        asyncio.run(run())


class TestExecutionMetadataIntegration:
    """実行metadataをcontext/state/messageへ残すことを検証する。"""

    def test_provider_stores_execution_metadata_in_state_and_context(self) -> None:
        """正常系：provider情報を実行contextとしてstateとcontext metadataに保存すること"""

        async def run() -> None:
            context = SessionContext(session_id="session", input_messages=[])
            state = {}
            provider = ExecutionContextProvider(
                model="claude-sonnet-4-6",
                provider_family="anthropic",
                history_source_id="history",
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state=state,
            )

            metadata = state["metadata"]
            assert metadata["model"] == "claude-sonnet-4-6"
            assert metadata["provider_family"] == "anthropic"
            assert metadata["history_source_id"] == "history"
            assert "working_directory" not in metadata
            assert "platform" not in metadata
            assert "session_id" not in metadata
            assert context.metadata["execution_context"] == metadata

        asyncio.run(run())

    def test_history_provider_saves_execution_metadata_to_message_properties(self) -> None:
        """正常系：履歴保存時、inputとresponseのMessageへ実行metadataを付与すること"""

        async def run() -> None:
            store = FakeStore([])
            provider = LocalHistoryProvider(store=store)
            context = CompletedRunContext(
                session_id="session",
                input_messages=[Message("user", ["hello"])],
                response=AgentResponse(messages=[Message("assistant", ["hello"])]),
            )
            context.metadata["execution_context"] = {
                "model": "gpt-5.4",
                "provider_family": "openai",
            }

            await provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            assert store.messages[0].additional_properties["execution"]["model"] == "gpt-5.4"
            assert store.messages[0].additional_properties["execution"]["provider_family"] == "openai"
            assert store.messages[1].additional_properties["execution"]["model"] == "gpt-5.4"

        asyncio.run(run())

class TestHistorySaveNormalization:
    """LocalHistoryProvider保存時のtool call/result順序を検証する。"""

    def test_history_provider_normalizes_mcp_call_result_order_before_save(self) -> None:
        """正常系：履歴保存時、MCP resultを対応するcall順に並べて保存すること"""

        async def run() -> None:
            store = FakeStore([])
            provider = LocalHistoryProvider(store=store)
            context = CompletedRunContext(
                session_id="session",
                input_messages=[Message("user", ["search docs"])],
                response=AgentResponse(
                    messages=[
                        Message(
                            "assistant",
                            [
                                Content.from_mcp_server_tool_call(
                                    "mcp_call_1",
                                    "microsoft_docs_search",
                                    server_name="Microsoft_Learn_MCP",
                                    arguments={},
                                ),
                                Content.from_mcp_server_tool_call(
                                    "mcp_call_2",
                                    "microsoft_docs_fetch",
                                    server_name="Microsoft_Learn_MCP",
                                    arguments={},
                                ),
                            ],
                        ),
                        Message(
                            "assistant",
                            [
                                Content.from_mcp_server_tool_result("mcp_call_2", output=[Content.from_text("fetch")]),
                                Content.from_text("next"),
                                Content.from_mcp_server_tool_result("mcp_call_1", output=[Content.from_text("search")]),
                            ],
                        ),
                    ],
                ),
            )

            await provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            assert [message.role for message in store.messages] == ["user", "assistant"]
            assert [content.type for content in store.messages[1].contents] == [
                "mcp_server_tool_call",
                "mcp_server_tool_call",
                "mcp_server_tool_result",
                "mcp_server_tool_result",
                "text",
            ]
            assert store.messages[1].contents[0].call_id == "mcp_call_1"
            assert store.messages[1].contents[1].call_id == "mcp_call_2"
            assert store.messages[1].contents[2].call_id == "mcp_call_1"
            assert store.messages[1].contents[3].call_id == "mcp_call_2"

        asyncio.run(run())

    def test_history_provider_places_client_tool_result_before_following_text(self) -> None:
        """正常系：履歴保存時、function_resultをfunction_call後のtextより前に保存すること"""

        async def run() -> None:
            store = FakeStore([])
            provider = LocalHistoryProvider(store=store)
            context = CompletedRunContext(
                session_id="session",
                input_messages=[Message("user", ["run tool"])],
                response=AgentResponse(
                    messages=[
                        Message(
                            "assistant",
                            [
                                Content.from_function_call("call_1", "web_search", arguments={}),
                                Content.from_text("final answer"),
                            ],
                        ),
                        Message("tool", [Content.from_function_result("call_1", result="search result")]),
                    ],
                ),
            )

            await provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            assert [message.role for message in store.messages] == ["user", "assistant", "tool", "assistant"]
            assert [content.type for content in store.messages[1].contents] == ["function_call"]
            assert [content.type for content in store.messages[2].contents] == ["function_result"]
            assert [content.type for content in store.messages[3].contents] == ["text"]
            assert store.messages[1].contents[0].call_id == "call_1"
            assert store.messages[2].contents[0].call_id == "call_1"
            assert store.messages[3].contents[0].text == "final answer"

        asyncio.run(run())

class TestApprovalHistorySaveIntegration:
    """承認要求とtool結果を履歴として保存する規則を検証する。"""

    def test_history_provider_stores_executed_approval_input_when_available(self) -> None:
        r"""正常系：実行済み承認inputがある場合、raw承認応答ではなく実行済みmessageを保存すること

        入力例:
            input_messages = [
                {"role": "user", "contents": [{"type": "text", "text": "raw approval response"}]}
            ]
            metadata["message_conversion:executed_input_messages"] = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]}
            ]
            response.messages = [
                {"role": "assistant", "contents": [{"type": "text", "text": "done"}]}
            ]

        期待例:
            stored_messages = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]},
                {"role": "assistant", "contents": [{"type": "text", "text": "done"}]}
            ]
        """
        async def run() -> None:
            store = FakeStore([])
            provider = LocalHistoryProvider(store=store)
            function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            approval_request = Content.from_function_approval_request("call_1", function_call)
            context = CompletedRunContext(
                session_id="session",
                input_messages=[Message("user", [Content.from_text("raw approval response")])],
                response=AgentResponse(messages=[Message("assistant", [Content.from_text("done")])]),
            )
            context.metadata[EXECUTED_INPUT_MESSAGES_METADATA_KEY] = [
                Message(
                    "assistant",
                    [
                        Content.from_text("I need to check the weather."),
                        approval_request,
                    ],
                ),
                Message("tool", [Content.from_function_result("call_1", result="Tokyo weather is sunny.")]),
            ]

            await provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            assert [message.role for message in store.messages] == ["assistant", "tool", "assistant"]
            assert [content.type for content in store.messages[0].contents] == ["text", "function_approval_request"]
            assert store.messages[1].contents[0].type == "function_result"
            assert store.messages[2].contents[0].text == "done"

        asyncio.run(run())

    def test_history_provider_keeps_openai_function_result_text(self) -> None:
        r"""正常系：OpenAI実行履歴でもfunction_result.resultは元のtextで保存すること

        入力例:
            metadata["execution_context"] = {"provider_family": "openai"}
            metadata["message_conversion:executed_input_messages"] = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {
                    "role": "tool",
                    "contents": [
                        {
                            "type": "function_result",
                            "call_id": "call_1",
                            "result": "Error: Tool call invocation was rejected by user."
                        }
                    ]
                }
            ]

        期待例:
            stored_messages = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {
                    "role": "tool",
                    "contents": [
                        {
                            "type": "function_result",
                            "call_id": "call_1",
                            "result": "Error: Tool call invocation was rejected by user."
                        }
                    ]
                }
            ]
        """
        async def run() -> None:
            store = FakeStore([])
            provider = LocalHistoryProvider(store=store)
            function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            approval_request = Content.from_function_approval_request("call_1", function_call)
            context = SessionContext(
                session_id="session",
                input_messages=[Message("user", [Content.from_text("raw approval response")])],
            )
            context.metadata["execution_context"] = {"provider_family": "openai"}
            context.metadata[EXECUTED_INPUT_MESSAGES_METADATA_KEY] = [
                Message(
                    "assistant",
                    [
                        Content.from_text("I need to check the weather."),
                        approval_request,
                    ],
                ),
                Message(
                    "tool",
                    [
                        Content.from_function_result(
                            "call_1",
                            result="Error: Tool call invocation was rejected by user.",
                        )
                    ],
                ),
            ]

            await provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            result = store.messages[1].contents[0]
            assert result.type == "function_result"
            assert result.result == "Error: Tool call invocation was rejected by user."
            assert isinstance(result.items, list)
            assert result.items[0].text == result.result

        asyncio.run(run())

    def test_history_provider_keeps_result_for_existing_approval_request(self) -> None:
        r"""正常系：既存履歴の承認要求に対し、今回inputのtool結果を続きの履歴として保存すること

        入力例:
            existing_messages = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]}
            ]
            input_messages = [
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]}
            ]

        期待例:
            stored_messages = [
                {"role": "assistant", "contents": [{"type": "text"}, {"type": "function_approval_request", "id": "call_1"}]},
                {"role": "tool", "contents": [{"type": "function_result", "call_id": "call_1"}]}
            ]
        """
        async def run() -> None:
            function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            approval_request = Content.from_function_approval_request("call_1", function_call)
            store = FakeStore([
                Message("assistant", [Content.from_text("I need to check the weather."), approval_request]),
            ])
            provider = LocalHistoryProvider(store=store)
            context = SessionContext(
                session_id="session",
                input_messages=[Message("tool", [Content.from_function_result("call_1", result="Tokyo weather.")])],
            )

            await provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            assert [message.role for message in store.messages] == ["assistant", "tool"]
            assert [content.type for content in store.messages[0].contents] == ["text", "function_approval_request"]
            assert store.messages[1].contents[0].type == "function_result"
            assert store.messages[1].contents[0].call_id == "call_1"

        asyncio.run(run())
