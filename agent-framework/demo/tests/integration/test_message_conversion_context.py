from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from agent.contexts import (
    ExecutionContextProvider,
    MessageConversionContextProvider,
)
from agent.history import LocalHistoryProvider, MessageStore
from agent.contexts.message_converter import CommonMessageConverter, ProviderMessageConverter
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

    def test_history_provider_loads_history_and_restores_original_input_before_save(self) -> None:
        async def run() -> None:
            stored_history = [Message("assistant", [Content.from_text("previous")])]
            original_input = [Message("user", [Content.from_text("hello")])]
            store = FakeStore(stored_history)
            history_provider = LocalHistoryProvider(store=store)
            context = SessionContext(session_id="session", input_messages=original_input)

            await history_provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            assert context.context_messages[history_provider.source_id] == stored_history

            completed_context = CompletedRunContext(
                session_id="session",
                input_messages=[Message("user", [Content.from_text("converted hello")])],
                response=AgentResponse(messages=[Message("assistant", [Content.from_text("done")])]),
                metadata=context.metadata,
            )

            await history_provider.after_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=completed_context,  # type: ignore[arg-type]
                state={},
            )

            assert completed_context.input_messages == original_input
            assert [message.role for message in store.messages] == ["assistant", "user", "assistant"]
            assert store.messages[1].contents[0].text == "hello"

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
                message_converter=CommonMessageConverter(),
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

            assert context.context_messages["history"] == converted_messages

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

class TestCombinedReplayConversion:
    def test_provider_converts_history_and_input_as_one_stream(self) -> None:
        async def run() -> None:
            function_call = Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"})
            approval_request = Content.from_function_approval_request("call_1", function_call)
            assistant_message = Message(
                "assistant",
                [Content.from_text("I need to check the weather."), function_call, approval_request],
            )
            context = SessionContext(
                session_id="session",
                input_messages=[
                    assistant_message,
                    Message("user", [approval_request.to_function_approval_response(True)]),
                ],
                context_messages={"history": [Message("user", [Content.from_text("Tokyo weather")])]},
            )
            provider = MessageConversionContextProvider(
                history_source_id="history",
                message_converter=CommonMessageConverter(),
            )

            await provider.before_run(
                agent=object(),  # type: ignore[arg-type]
                session=AgentSession(session_id="session"),
                context=context,
                state={},
            )

            replay_messages = [*context.context_messages["history"], *context.input_messages]
            assert [message.role for message in replay_messages] == ["user", "assistant", "user"]
            assert [content.type for content in replay_messages[1].contents] == ["text", "function_call"]
            assert replay_messages[2].contents[0].type == "function_approval_response"

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
    def test_history_provider_keeps_result_for_existing_approval_request(self) -> None:
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
