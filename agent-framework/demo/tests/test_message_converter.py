from __future__ import annotations

import unittest

from agent.contexts import ExecutionContextProvider, MessageConversionContextProvider
from agent.history import LocalHistoryProvider, MessageStore
from agent.message_converter import CommonMessageConverter
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

        converted = CommonMessageConverter().convert_messages([message])

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

        converted = CommonMessageConverter(reasoning_policy="drop").convert_messages([message])

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].role, "assistant")
        self.assertEqual([content.type for content in converted[0].contents], ["function_call"])

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


if __name__ == "__main__":
    unittest.main()
