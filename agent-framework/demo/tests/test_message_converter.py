from __future__ import annotations

import asyncio

from agent.contexts import ExecutionContextProvider, MessageConversionContextProvider
from agent.history import LocalHistoryProvider, MessageStore
from agent.message_converter import CommonMessageConverter, ProviderMessageConverter
from agent.message_normalizer import MessageHistoryNormalizer
from agent_framework import AgentResponse, AgentSession, Content, Message, SessionContext


class FakeStore(MessageStore):
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages

    def read_messages(self, session_id: str | None) -> list[Message]:
        return self.messages

    def write_messages(self, session_id: str | None, messages) -> None:  # type: ignore[no-untyped-def]
        self.messages = list(messages)


class TestCommonMessageConverter:
    def test_downgrades_reasoning_and_keeps_function_call_on_assistant_role(self) -> None:
        """正常系：assistantのreasoningとfunction_callを再投入する場合、reasoningはtext化されcallはassistantに残ること"""
        message = Message(
            "assistant",
            [
                Content.from_text_reasoning(text="reasoning"),
                Content.from_function_call(
                    "call_1",
                    "get_weather",
                    arguments={"location": "Tokyo"},
                    additional_properties={"fc_id": "fc_response_scoped", "status": "completed"},
                    raw_representation=object(),
                ),
            ],
        )

        converted = CommonMessageConverter().convert_message(message)

        assert len(converted) == 1
        assert converted[0].role == "assistant"
        assert [content.type for content in converted[0].contents] == ["text", "function_call"]
        assert converted[0].contents[0].text == "[reasoning]\nreasoning"
        function_call = converted[0].contents[1]
        assert function_call.arguments == '{"location": "Tokyo"}'
        assert "fc_id" not in function_call.additional_properties
        assert function_call.additional_properties["status"] == "completed"
        assert function_call.raw_representation is None

    def test_drops_reasoning_when_policy_is_drop(self) -> None:
        """正常系：reasoning_policyがdropの場合、reasoningだけが除外されfunction_callは残ること"""
        message = Message(
            "assistant",
            [
                Content.from_text_reasoning(text="reasoning"),
                Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"}),
            ],
        )

        converted = CommonMessageConverter(reasoning_policy="drop").convert_message(message)

        assert len(converted) == 1
        assert converted[0].role == "assistant"
        assert [content.type for content in converted[0].contents] == ["function_call"]

    def test_preserves_anthropic_reasoning_for_anthropic_replay(self) -> None:
        """正常系：Anthropic由来でprotected_dataがある場合、native reasoningとして保持されること"""
        message = Message(
            "assistant",
            [
                Content.from_text_reasoning(
                    text="reasoning",
                    protected_data="signature",
                ),
                Content.from_text("answer"),
            ],
            additional_properties={"execution": {"provider_family": "anthropic"}},
        )

        converted = ProviderMessageConverter(target_provider_family="anthropic").convert_messages([message])

        assert [content.type for content in converted[0].contents] == ["text_reasoning", "text"]
        assert converted[0].contents[0].protected_data == "signature"

    def test_downgrades_anthropic_reasoning_for_openai_replay(self) -> None:
        """正常系：Anthropic reasoningをOpenAIへ再投入する場合、textへ降格されること"""
        message = Message(
            "assistant",
            [
                Content.from_text_reasoning(
                    text="reasoning",
                    protected_data="signature",
                )
            ],
            additional_properties={"execution": {"provider_family": "anthropic"}},
        )

        converted = ProviderMessageConverter(target_provider_family="openai").convert_messages([message])

        assert converted[0].contents[0].type == "text"
        assert converted[0].contents[0].text == "[reasoning]\nreasoning"

    def test_moves_function_result_to_tool_role(self) -> None:
        """正常系：assistant messageにfunction_resultが混在する場合、tool roleのmessageへ分離されること"""
        message = Message(
            "assistant",
            [
                Content.from_text("answer before tool result"),
                Content.from_function_result("call_1", result={"weather": "sunny"}),
            ],
        )

        converted = CommonMessageConverter().convert_messages([message])

        assert len(converted) == 2
        assert converted[0].role == "assistant"
        assert converted[0].contents[0].type == "text"
        assert converted[1].role == "tool"
        assert converted[1].contents[0].type == "function_result"

    def test_keeps_approval_response_as_user_content(self) -> None:
        """正常系：userのapproval responseを再投入する場合、user contentとして保持されること"""
        function_call = Content.from_function_call("call_1", "deploy", arguments={"env": "dev"})
        message = Message(
            "user",
            [Content.from_function_approval_response(True, "call_1", function_call)],
        )

        converted = CommonMessageConverter().convert_messages([message])

        assert len(converted) == 1
        assert converted[0].role == "user"
        assert converted[0].contents[0].type == "function_approval_response"
        nested_call = converted[0].contents[0].function_call
        assert nested_call is not None
        assert nested_call.arguments == '{"env": "dev"}'

    def test_drops_usage_content(self) -> None:
        """正常系：usage contentを再投入する場合、providerへ渡すmessageから除外されること"""
        message = Message("assistant", [Content.from_usage({"total_token_count": 2})])

        converted = CommonMessageConverter().convert_messages([message])

        assert converted == []

    def test_sanitizes_nested_mcp_tool_result_content_for_provider_replay(self) -> None:
        """正常系：MCP tool resultにnested contentがある場合、再投入不要な内部payloadが除去されること"""
        message = Message(
            "assistant",
            [
                Content.from_mcp_server_tool_call(
                    "mcp_call_1",
                    "microsoft_docs_search",
                    server_name="Microsoft_Learn_MCP",
                    arguments={},
                ),
                Content.from_mcp_server_tool_result(
                    "mcp_call_1",
                    output=[
                        Content.from_text(
                            "result",
                            additional_properties={"provider_internal": "drop"},
                            raw_representation=object(),
                        )
                    ],
                    additional_properties={"fc_id": "response_scoped"},
                ),
            ],
        )

        converted = CommonMessageConverter().convert_messages([message])

        assert len(converted) == 1
        assert converted[0].role == "assistant"
        tool_result = converted[0].contents[1]
        assert tool_result.type == "mcp_server_tool_result"
        assert tool_result.additional_properties == {}
        assert tool_result.output[0]["type"] == "text"
        assert "additional_properties" not in tool_result.output[0]
        assert "raw_representation" not in tool_result.output[0]

    def test_moves_delayed_mcp_results_next_to_their_tool_calls(self) -> None:
        """正常系：MCP resultがcallより後続messageにある場合、対応するcall直後へ移動されること"""
        messages = [
            Message(
                "assistant",
                [
                    Content.from_function_call("function_call_1", "load_skill", arguments={}),
                    Content.from_mcp_server_tool_call(
                        "mcp_call_1",
                        "microsoft_docs_search",
                        server_name="Microsoft_Learn_MCP",
                        arguments={},
                    ),
                ],
            ),
            Message("tool", [Content.from_function_result("function_call_1", result="skill")]),
            Message(
                "assistant",
                [
                    Content.from_mcp_server_tool_result(
                        "mcp_call_1",
                        output=[Content.from_text("docs")],
                    ),
                    Content.from_text("next step"),
                ],
            ),
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)
        converted = CommonMessageConverter().convert_messages(normalized)

        assert [message.role for message in converted] == ["assistant", "tool", "assistant"]
        assert [content.type for content in converted[0].contents] == [
            "function_call",
            "mcp_server_tool_call",
            "mcp_server_tool_result",
        ]
        assert [content.type for content in converted[1].contents] == ["function_result"]
        assert converted[0].contents[2].call_id == "mcp_call_1"
        assert converted[1].contents[0].call_id == "function_call_1"
        assert converted[2].contents[0].type == "text"

    def test_drops_orphan_mcp_tool_results_for_claude_replay(self) -> None:
        """異常系：対応するMCP callがないresultの場合、resultだけが除外されtextは残ること"""
        message = Message(
            "assistant",
            [
                Content.from_mcp_server_tool_result(
                    "mcp_call_1",
                    output=[Content.from_text("docs")],
                ),
                Content.from_text("next step"),
            ],
        )

        normalized = MessageHistoryNormalizer().normalize_messages([message])
        converted = CommonMessageConverter().convert_messages(normalized)

        assert len(converted) == 1
        assert [content.type for content in converted[0].contents] == ["text"]

    def test_drops_unresolved_tool_calls_for_provider_replay(self) -> None:
        """異常系：MCP callに対応するresultがない場合、未解決callだけが除外されtextは残ること"""
        message = Message(
            "assistant",
            [
                Content.from_text("before"),
                Content.from_mcp_server_tool_call(
                    "mcp_call_1",
                    "microsoft_docs_fetch",
                    server_name="Microsoft_Learn_MCP",
                    arguments={},
                ),
            ],
        )

        normalized = MessageHistoryNormalizer().normalize_messages([message])
        converted = CommonMessageConverter().convert_messages(normalized)

        assert len(converted) == 1
        assert [content.type for content in converted[0].contents] == ["text"]

    def test_drops_unresolved_function_calls_for_provider_replay(self) -> None:
        """異常系：function_callに対応するresultがない場合、未解決callだけが除外されtextは残ること"""
        message = Message(
            "assistant",
            [
                Content.from_text("before"),
                Content.from_function_call("call_1", "get_weather", arguments={}),
            ],
        )

        normalized = MessageHistoryNormalizer().normalize_messages([message])
        converted = CommonMessageConverter().convert_messages(normalized)

        assert len(converted) == 1
        assert [content.type for content in converted[0].contents] == ["text"]


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
