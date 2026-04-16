from __future__ import annotations

import unittest

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


class CommonMessageConverterTests(unittest.TestCase):
    def test_downgrades_reasoning_and_keeps_function_call_on_assistant_role(self) -> None:
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

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].role, "assistant")
        self.assertEqual([content.type for content in converted[0].contents], ["text", "function_call"])
        self.assertEqual(converted[0].contents[0].text, "[reasoning]\nreasoning")
        function_call = converted[0].contents[1]
        self.assertEqual(function_call.arguments, '{"location": "Tokyo"}')
        self.assertNotIn("fc_id", function_call.additional_properties)
        self.assertEqual(function_call.additional_properties["status"], "completed")
        self.assertIsNone(function_call.raw_representation)

    def test_drops_reasoning_when_policy_is_drop(self) -> None:
        message = Message(
            "assistant",
            [
                Content.from_text_reasoning(text="reasoning"),
                Content.from_function_call("call_1", "get_weather", arguments={"location": "Tokyo"}),
            ],
        )

        converted = CommonMessageConverter(reasoning_policy="drop").convert_message(message)

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].role, "assistant")
        self.assertEqual([content.type for content in converted[0].contents], ["function_call"])

    def test_preserves_anthropic_reasoning_for_anthropic_replay(self) -> None:
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

        self.assertEqual([content.type for content in converted[0].contents], ["text_reasoning", "text"])
        self.assertEqual(converted[0].contents[0].protected_data, "signature")

    def test_downgrades_anthropic_reasoning_for_openai_replay(self) -> None:
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

        self.assertEqual(converted[0].contents[0].type, "text")
        self.assertEqual(converted[0].contents[0].text, "[reasoning]\nreasoning")

    def test_moves_function_result_to_tool_role(self) -> None:
        message = Message(
            "assistant",
            [
                Content.from_text("answer before tool result"),
                Content.from_function_result("call_1", result={"weather": "sunny"}),
            ],
        )

        converted = CommonMessageConverter().convert_messages([message])

        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0].role, "assistant")
        self.assertEqual(converted[0].contents[0].type, "text")
        self.assertEqual(converted[1].role, "tool")
        self.assertEqual(converted[1].contents[0].type, "function_result")

    def test_keeps_approval_response_as_user_content(self) -> None:
        function_call = Content.from_function_call("call_1", "deploy", arguments={"env": "dev"})
        message = Message(
            "user",
            [Content.from_function_approval_response(True, "call_1", function_call)],
        )

        converted = CommonMessageConverter().convert_messages([message])

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].role, "user")
        self.assertEqual(converted[0].contents[0].type, "function_approval_response")
        nested_call = converted[0].contents[0].function_call
        assert nested_call is not None
        self.assertEqual(nested_call.arguments, '{"env": "dev"}')

    def test_drops_usage_content(self) -> None:
        message = Message("assistant", [Content.from_usage({"total_token_count": 2})])

        converted = CommonMessageConverter().convert_messages([message])

        self.assertEqual(converted, [])

    def test_sanitizes_nested_mcp_tool_result_content_for_provider_replay(self) -> None:
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
                )
            ],
        )

        converted = CommonMessageConverter().convert_messages([message])

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].role, "assistant")
        tool_result = converted[0].contents[1]
        self.assertEqual(tool_result.type, "mcp_server_tool_result")
        self.assertEqual(tool_result.additional_properties, {})
        self.assertEqual(tool_result.output[0]["type"], "text")
        self.assertNotIn("additional_properties", tool_result.output[0])
        self.assertNotIn("raw_representation", tool_result.output[0])

    def test_moves_delayed_mcp_results_next_to_their_tool_calls(self) -> None:
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

        self.assertEqual([message.role for message in converted], ["assistant", "tool", "assistant"])
        self.assertEqual(
            [content.type for content in converted[0].contents],
            ["function_call", "mcp_server_tool_call", "mcp_server_tool_result"],
        )
        self.assertEqual([content.type for content in converted[1].contents], ["function_result"])
        self.assertEqual(converted[0].contents[2].call_id, "mcp_call_1")
        self.assertEqual(converted[1].contents[0].call_id, "function_call_1")
        self.assertEqual(converted[2].contents[0].type, "text")

    def test_drops_orphan_mcp_tool_results_for_claude_replay(self) -> None:
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

        self.assertEqual(len(converted), 1)
        self.assertEqual([content.type for content in converted[0].contents], ["text"])

    def test_drops_unresolved_tool_calls_for_provider_replay(self) -> None:
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

        self.assertEqual(len(converted), 1)
        self.assertEqual([content.type for content in converted[0].contents], ["text"])

    def test_drops_unresolved_function_calls_for_provider_replay(self) -> None:
        message = Message(
            "assistant",
            [
                Content.from_text("before"),
                Content.from_function_call("call_1", "get_weather", arguments={}),
            ],
        )

        normalized = MessageHistoryNormalizer().normalize_messages([message])
        converted = CommonMessageConverter().convert_messages(normalized)

        self.assertEqual(len(converted), 1)
        self.assertEqual([content.type for content in converted[0].contents], ["text"])


class MessageConversionContextProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_provider_returns_raw_messages(self) -> None:
        raw_messages = [Message("assistant", [Content.from_text_reasoning(text="reasoning")])]
        history_provider = LocalHistoryProvider(store=FakeStore(raw_messages))

        messages = await history_provider.get_messages("session")

        self.assertIs(messages[0], raw_messages[0])
        self.assertEqual(messages[0].contents[0].type, "text_reasoning")

    async def test_provider_converts_before_run_and_restores_after_run(self) -> None:
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
        self.assertIsNot(converted_messages, raw_messages)
        self.assertEqual(converted_messages[0].contents[0].type, "text")

        await provider.after_run(
            agent=object(),  # type: ignore[arg-type]
            session=AgentSession(session_id="session"),
            context=context,
            state={},
        )

        self.assertIs(context.context_messages["history"][0], raw_messages[0])


class ExecutionMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_stores_execution_metadata_in_state_and_context(self) -> None:
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
        self.assertEqual(metadata["model"], "claude-sonnet-4-6")
        self.assertEqual(metadata["provider_family"], "anthropic")
        self.assertEqual(metadata["history_source_id"], "history")
        self.assertNotIn("working_directory", metadata)
        self.assertNotIn("platform", metadata)
        self.assertNotIn("session_id", metadata)
        self.assertEqual(context.metadata["execution_context"], metadata)

    async def test_history_provider_saves_execution_metadata_to_message_properties(self) -> None:
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

        self.assertEqual(store.messages[0].additional_properties["execution"]["model"], "gpt-5.4")
        self.assertEqual(store.messages[0].additional_properties["execution"]["provider_family"], "openai")
        self.assertEqual(store.messages[1].additional_properties["execution"]["model"], "gpt-5.4")

    async def test_history_provider_normalizes_mcp_call_result_order_before_save(self) -> None:
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

        self.assertEqual([message.role for message in store.messages], ["user", "assistant"])
        self.assertEqual(
            [content.type for content in store.messages[1].contents],
            [
                "mcp_server_tool_call",
                "mcp_server_tool_call",
                "mcp_server_tool_result",
                "mcp_server_tool_result",
                "text",
            ],
        )
        self.assertEqual(store.messages[1].contents[0].call_id, "mcp_call_1")
        self.assertEqual(store.messages[1].contents[1].call_id, "mcp_call_2")
        self.assertEqual(store.messages[1].contents[2].call_id, "mcp_call_1")
        self.assertEqual(store.messages[1].contents[3].call_id, "mcp_call_2")


if __name__ == "__main__":
    unittest.main()
