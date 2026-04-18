from __future__ import annotations

from agent.messages import CommonMessageConverter, MessageHistoryNormalizer, ProviderMessageConverter
from agent_framework import Content, Message


class TestProviderMessageConverter:
    def test_preserves_openai_reasoning_when_replay_payload_is_complete(self) -> None:
        r"""正常系：OpenAI reasoningにidとencrypted_contentがある場合、native reasoningとして保持されること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "text_reasoning",
                        "id": "rs_1",
                        "text": "reasoning",
                        "additional_properties": {
                            "encrypted_content": "ciphertext"
                        }
                    }
                ],
                "additional_properties": {
                    "execution": {
                        "provider_family": "openai"
                    }
                }
            }

        期待例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "text_reasoning",
                        "id": "rs_1",
                        "text": "reasoning",
                        "additional_properties": {
                            "encrypted_content": "ciphertext"
                        }
                    }
                ]
            }
        """
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
        r"""異常系：OpenAI reasoningにencrypted_contentがない場合、textへ降格されること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "text_reasoning",
                        "id": "rs_1",
                        "text": "reasoning"
                    }
                ],
                "additional_properties": {
                    "execution": {
                        "provider_family": "openai"
                    }
                }
            }

        期待例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "text",
                        "text": "[reasoning]\nreasoning"
                    }
                ]
            }
        """
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
        r"""異常系：native replay不可かつtextが空のreasoningの場合、messageごと除外されること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "text_reasoning",
                        "id": "rs_1"
                    }
                ],
                "additional_properties": {
                    "execution": {
                        "provider_family": "openai"
                    }
                }
            }

        期待例:
            []
        """
        message = Message(
            "assistant",
            [Content.from_text_reasoning(id="rs_1")],
            additional_properties={"execution": {"provider_family": "openai"}},
        )

        converted = ProviderMessageConverter(target_provider_family="openai").convert_messages([message])

        assert converted == []

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

    def test_splits_role_changes_and_keeps_author_only_for_original_role(self) -> None:
        r"""正常系：1message内でroleが変わるcontentが混在する場合、roleごとに分割されauthorが適切に保持されること

        入力例:
            {
                "role": "assistant",
                "author_name": "assistant-name",
                "contents": [
                    {
                        "type": "text",
                        "text": "before"
                    },
                    {
                        "type": "function_result",
                        "call_id": "call_1",
                        "result": "ok"
                    },
                    {
                        "type": "text",
                        "text": "after"
                    }
                ]
            }

        期待例:
            [
                {
                    "role": "assistant",
                    "author_name": "assistant-name",
                    "contents": [
                        {
                            "type": "text"
                        }
                    ]
                },
                {
                    "role": "tool",
                    "contents": [
                        {
                            "type": "function_result",
                            "call_id": "call_1"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "author_name": "assistant-name",
                    "contents": [
                        {
                            "type": "text"
                        }
                    ]
                }
            ]
        """
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

    def test_drops_empty_text_and_hosted_vector_store_content(self) -> None:
        r"""異常系：空textとhosted_vector_storeだけの場合、providerへ渡すmessageから除外されること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "text",
                        "text": ""
                    },
                    {
                        "type": "hosted_vector_store",
                        "vector_store_id": "vs_1"
                    }
                ]
            }

        期待例:
            []
        """
        message = Message(
            "assistant",
            [
                Content.from_text(""),
                Content.from_hosted_vector_store("vs_1"),
            ],
        )

        converted = CommonMessageConverter().convert_message(message)

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
