from __future__ import annotations

from agent.messages import CommonMessageConverter, MessageHistoryNormalizer, ProviderMessageConverter
from agent.messages.replay_payload_sanitizer import ReplayPayloadSanitizer
from agent_framework import Content, Message


class TestProviderMessageConverter:
    def test_preserves_openai_reasoning_when_replay_payload_is_complete(self) -> None:
        """正常系：OpenAI reasoningにidとencrypted_contentがある場合、native reasoningとして保持されること"""
        message = Message(
            "assistant",
            [
                Content.from_text_reasoning(
                    id="rs_1",
                    text="reasoning",
                    additional_properties={"encrypted_content": "ciphertext"},
                )
            ],
            additional_properties={"execution": {"provider_family": "openai"}},
        )

        converted = ProviderMessageConverter(target_provider_family="openai").convert_messages([message])

        assert len(converted) == 1
        reasoning = converted[0].contents[0]
        assert reasoning.type == "text_reasoning"
        assert reasoning.id == "rs_1"
        assert reasoning.text == "reasoning"
        assert reasoning.additional_properties["encrypted_content"] == "ciphertext"

    def test_downgrades_openai_reasoning_without_encrypted_content(self) -> None:
        """異常系：OpenAI reasoningにencrypted_contentがない場合、textへ降格されること"""
        message = Message(
            "assistant",
            [Content.from_text_reasoning(id="rs_1", text="reasoning")],
            additional_properties={"execution": {"provider_family": "openai"}},
        )

        converted = ProviderMessageConverter(target_provider_family="openai").convert_messages([message])

        assert len(converted) == 1
        assert converted[0].contents[0].type == "text"
        assert converted[0].contents[0].text == "[reasoning]\nreasoning"

    def test_drops_empty_reasoning_when_native_replay_is_not_allowed(self) -> None:
        """異常系：native replay不可かつtextが空のreasoningの場合、messageごと除外されること"""
        message = Message(
            "assistant",
            [Content.from_text_reasoning(id="rs_1")],
            additional_properties={"execution": {"provider_family": "openai"}},
        )

        converted = ProviderMessageConverter(target_provider_family="openai").convert_messages([message])

        assert converted == []

    def test_splits_role_changes_and_keeps_author_only_for_original_role(self) -> None:
        """正常系：1message内でroleが変わるcontentが混在する場合、roleごとに分割されauthorが適切に保持されること"""
        message = Message(
            "assistant",
            [
                Content.from_text("before"),
                Content.from_function_result("call_1", result="ok"),
                Content.from_text("after"),
            ],
            author_name="assistant-name",
        )

        converted = CommonMessageConverter().convert_message(message)

        assert [item.role for item in converted] == ["assistant", "tool", "assistant"]
        assert [item.author_name for item in converted] == ["assistant-name", None, "assistant-name"]

    def test_drops_empty_text_and_hosted_vector_store_content(self) -> None:
        """異常系：空textとhosted_vector_storeだけの場合、providerへ渡すmessageから除外されること"""
        message = Message(
            "assistant",
            [
                Content.from_text(""),
                Content.from_hosted_vector_store("vs_1"),
            ],
        )

        converted = CommonMessageConverter().convert_message(message)

        assert converted == []


class TestReplayPayloadSanitizer:
    def test_normalizes_tool_arguments_and_drops_response_scoped_properties(self) -> None:
        """正常系：tool argumentsがdictでresponse scoped propertyがある場合、JSON化され不要propertyが除去されること"""
        sanitizer = ReplayPayloadSanitizer(lambda content: content)
        data = {
            "type": "function_call",
            "arguments": {"location": "Tokyo"},
            "additional_properties": {
                "fc_id": "response-scoped",
                "status": "completed",
                "opaque": object(),
            },
        }

        sanitizer.sanitize_content_data(data)

        assert data["arguments"] == '{"location": "Tokyo"}'
        assert "fc_id" not in data["additional_properties"]
        assert data["additional_properties"]["status"] == "completed"
        assert isinstance(data["additional_properties"]["opaque"], str)

    def test_sanitizes_nested_content_payloads_for_tool_results(self) -> None:
        """正常系：tool result payloadにnested contentがある場合、content内部の不要metadataが除去されること"""
        sanitizer = ReplayPayloadSanitizer(lambda content: content)
        data = {
            "type": "mcp_server_tool_result",
            "output": [
                Content.from_text(
                    "content object",
                    additional_properties={"provider_internal": "drop"},
                    raw_representation=object(),
                ),
                {
                    "type": "text",
                    "text": "content mapping",
                    "additional_properties": {"fc_id": "drop", "keep": "also dropped for nested content"},
                    "raw_representation": object(),
                },
            ],
        }

        sanitizer.sanitize_content_data(data)

        assert data["output"] == [
            {"type": "text", "text": "content object"},
            {"type": "text", "text": "content mapping"},
        ]

    def test_sanitizes_nested_approval_response_function_call(self) -> None:
        """正常系：approval responseにnested function_callがある場合、argumentsがJSON化され不要propertyが除去されること"""
        function_call = Content.from_function_call(
            "call_1",
            "deploy",
            arguments={"environment": "dev"},
            additional_properties={"fc_id": "response-scoped", "status": "completed"},
        )
        message = Message(
            "user",
            [Content.from_function_approval_response(True, "call_1", function_call)],
        )

        converted = CommonMessageConverter().convert_message(message)

        nested_call = converted[0].contents[0].function_call
        assert nested_call is not None
        assert nested_call.arguments == '{"environment": "dev"}'
        assert "fc_id" not in nested_call.additional_properties
        assert nested_call.additional_properties["status"] == "completed"


class TestMessageHistoryNormalizer:
    def test_drops_orphan_client_tool_results(self) -> None:
        """異常系：対応するclient tool callがないresultの場合、orphan resultが除外されること"""
        messages = [
            Message("tool", [Content.from_function_result("call_1", result="orphan")]),
            Message("assistant", [Content.from_text("answer")]),
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)

        assert len(normalized) == 1
        assert normalized[0].role == "assistant"
        assert normalized[0].contents[0].text == "answer"

    def test_places_client_tool_results_in_separate_tool_message(self) -> None:
        """正常系：client tool callとresultが対応する場合、resultが別のtool messageへ配置されること"""
        messages = [
            Message(
                "assistant",
                [
                    Content.from_text("before"),
                    Content.from_function_call("call_1", "load_skill", arguments={}),
                    Content.from_text("after"),
                ],
            ),
            Message("tool", [Content.from_function_result("call_1", result="loaded")]),
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)

        assert [message.role for message in normalized] == ["assistant", "tool"]
        assert [content.type for content in normalized[0].contents] == ["text", "function_call", "text"]
        assert [content.type for content in normalized[1].contents] == ["function_result"]
        assert normalized[1].contents[0].call_id == "call_1"

    def test_drops_client_tool_calls_without_results(self) -> None:
        """異常系：client tool callに対応するresultがない場合、未解決callが除外されること"""
        messages = [
            Message(
                "assistant",
                [
                    Content.from_function_call("call_1", "load_skill", arguments={}),
                    Content.from_text("fallback"),
                ],
            )
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)

        assert len(normalized) == 1
        assert [content.type for content in normalized[0].contents] == ["text"]

    def test_merges_adjacent_assistant_messages_when_mcp_content_is_repaired(self) -> None:
        """正常系：MCP call/result修復後にassistant messageが隣接する場合、1messageへmergeされmetadataが保持されること"""
        messages = [
            Message(
                "assistant",
                [
                    Content.from_mcp_server_tool_call(
                        "mcp_call_1",
                        "search",
                        server_name="docs",
                        arguments={},
                    )
                ],
                author_name="first",
                message_id="message-1",
                additional_properties={"left": True},
            ),
            Message(
                "assistant",
                [
                    Content.from_mcp_server_tool_result("mcp_call_1", output=[Content.from_text("result")]),
                    Content.from_text("done"),
                ],
                author_name="second",
                additional_properties={"right": True},
            ),
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)

        assert len(normalized) == 1
        assert [content.type for content in normalized[0].contents] == [
            "mcp_server_tool_call",
            "mcp_server_tool_result",
            "text",
        ]
        assert normalized[0].author_name == "first"
        assert normalized[0].message_id == "message-1"
        assert normalized[0].additional_properties == {"left": True, "right": True}
