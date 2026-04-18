from __future__ import annotations

import asyncio

from agent.contexts import ExecutionContextProvider, MessageConversionContextProvider
from agent.history import LocalHistoryProvider, MessageStore
from agent.messages import CommonMessageConverter
from agent_framework import AgentResponse, AgentSession, Content, Message, SessionContext


class FakeStore(MessageStore):
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages

    def read_messages(self, session_id: str | None) -> list[Message]:
        return self.messages

    def write_messages(self, session_id: str | None, messages) -> None:  # type: ignore[no-untyped-def]
        self.messages = list(messages)


class TestMessageConversionContextProvider:
    def test_history_provider_returns_raw_messages(self) -> None:
        """正常系：履歴取得時の場合、保存済みMessageが変換されずrawのまま返ること"""

        async def run() -> None:
            raw_messages = [Message("assistant", [Content.from_text_reasoning(text="reasoning")])]
            history_provider = LocalHistoryProvider(store=FakeStore(raw_messages))

            messages = await history_provider.get_messages("session")

            assert messages[0] is raw_messages[0]
            assert messages[0].contents[0].type == "text_reasoning"

        asyncio.run(run())

    def test_provider_converts_before_run_and_restores_after_run(self) -> None:
        """正常系：run前後のcontext変換を行う場合、実行前に変換され実行後にrawへ戻ること"""

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


class TestExecutionMetadata:
    def test_provider_stores_execution_metadata_in_state_and_context(self) -> None:
        """正常系：実行contextを初期化する場合、provider情報がstateとcontext metadataに保存されること"""

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
        """正常系：履歴保存時に実行metadataがある場合、inputとresponseのMessage propertiesへ保存されること"""

        async def run() -> None:
            store = FakeStore([])
            provider = LocalHistoryProvider(store=store)
            context = SessionContext(
                session_id="session",
                input_messages=[Message("user", ["hello"])],
            )
            context.metadata["execution_context"] = {
                "model": "gpt-5.4",
                "provider_family": "openai",
            }
            context._response = AgentResponse(messages=[Message("assistant", ["hello"])])  # type: ignore[assignment]

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

    def test_history_provider_normalizes_mcp_call_result_order_before_save(self) -> None:
        """正常系：履歴保存時にMCP result順序が乱れている場合、call順へ正規化されて保存されること"""

        async def run() -> None:
            store = FakeStore([])
            provider = LocalHistoryProvider(store=store)
            context = SessionContext(
                session_id="session",
                input_messages=[Message("user", ["search docs"])],
            )
            context._response = AgentResponse(
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
                ]
            )  # type: ignore[assignment]

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
        """正常系：履歴保存時にfunction_call後のtextがある場合、function_resultがtextより前に保存されること"""

        async def run() -> None:
            store = FakeStore([])
            provider = LocalHistoryProvider(store=store)
            context = SessionContext(
                session_id="session",
                input_messages=[Message("user", ["run tool"])],
            )
            context._response = AgentResponse(
                messages=[
                    Message(
                        "assistant",
                        [
                            Content.from_function_call("call_1", "web_search", arguments={}),
                            Content.from_text("final answer"),
                        ],
                    ),
                    Message("tool", [Content.from_function_result("call_1", result="search result")]),
                ]
            )  # type: ignore[assignment]

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
