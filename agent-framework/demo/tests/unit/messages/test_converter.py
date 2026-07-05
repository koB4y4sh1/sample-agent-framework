from __future__ import annotations

from agent.contexts.message_converter import (
    CommonMessageConverter,
    ProviderMessageConverter,
)
from agent_framework import Content, Message


class TestReasoningConversion:
    """providerごとの有効な reasoning に変換されているかを検証する。"""

    def test_preserves_openai_reasoning_with_id_and_encrypted_content(self) -> None:
        r"""正常系：OpenAI reasoningでidと暗号化データ(encrypted_content)がある場合、text_reasoningとして渡されること

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

        converted = ProviderMessageConverter(
            target_provider_family="openai"
        ).convert_messages([message])

        assert len(converted) == 1
        reasoning = converted[0].contents[0]
        assert reasoning.type == "text_reasoning"
        assert reasoning.id == "rs_1"
        assert reasoning.text == "reasoning"
        assert reasoning.additional_properties["encrypted_content"] == "ciphertext"

    def test_downgrades_openai_reasoning_without_encrypted_content(self) -> None:
        r"""正常系：OpenAI reasoningに暗号化データ(encrypted_content)がない場合、textとして推論サマリが渡されること

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

        converted = ProviderMessageConverter(
            target_provider_family="openai"
        ).convert_messages([message])

        assert len(converted) == 1
        assert converted[0].contents[0].type == "text"
        assert converted[0].contents[0].text == "[reasoning]\nreasoning"

    def test_anthropic_reasoning_for_anthropic(self) -> None:
        """正常系：Anthropic reasoningで暗号化データ(protected_data)がある場合、text_reasoningで渡されること"""
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

        converted = ProviderMessageConverter(
            target_provider_family="anthropic"
        ).convert_messages([message])

        assert [content.type for content in converted[0].contents] == [
            "text_reasoning",
            "text",
        ]
        assert converted[0].contents[0].protected_data == "signature"

    def test_anthropic_reasoning_for_another_provider(self) -> None:
        """正常系：Anthropic reasoningを別providerへ渡す場合、textとして推論サマリが渡されること"""
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

        converted = ProviderMessageConverter(
            target_provider_family="openai"
        ).convert_messages([message])

        assert converted[0].contents[0].type == "text"
        assert converted[0].contents[0].text == "[reasoning]\nreasoning"

    def test_drops_empty_reasoning_when_broken_reasoning(self) -> None:
        r"""異常系：text化もtext_reasoning化もできないreasoningは、messageごと渡さないこと

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

        converted = ProviderMessageConverter(
            target_provider_family="openai"
        ).convert_messages([message])

        assert converted == []


class TestFunctionResultConversion:
    """providerごとの有効なfunction_result に変換されているかを検証する。"""

    def test_keeps_function_result_text_for_openai_provider(self) -> None:
        r"""正常系：OpenAI向けでもfunction_result.resultは元の文字列で保持すること

        入力例:
            {
                "role": "tool",
                "contents": [
                    {
                        "type": "function_result",
                        "call_id": "call_1",
                        "result": "Tokyo weather is sunny."
                    }
                ]
            }

        期待例:
            {
                "role": "tool",
                "contents": [
                    {
                        "type": "function_result",
                        "call_id": "call_1",
                        "result": "Tokyo weather is sunny."
                    }
                ]
            }
        """
        message = Message(
            "tool",
            [Content.from_function_result("call_1", result="Tokyo weather is sunny.")],
        )

        converted = ProviderMessageConverter(
            target_provider_family="openai"
        ).convert_messages([message])

        result = converted[0].contents[0]
        assert result.type == "function_result"
        assert result.result == "Tokyo weather is sunny."
        assert isinstance(result.items, list)
        assert result.items[0].text == result.result

    def test_keeps_rejected_function_result_text_for_openai_provider(self) -> None:
        r"""正常系：拒否されたtool結果もOpenAI向けでは元のエラー文字列で保持すること

        入力例:
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

        期待例:
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
        """
        message = Message(
            "tool",
            [
                Content.from_function_result(
                    "call_1",
                    result="Error: Tool call invocation was rejected by user.",
                )
            ],
        )

        converted = ProviderMessageConverter(
            target_provider_family="openai"
        ).convert_messages([message])

        result = converted[0].contents[0]
        assert result.type == "function_result"
        assert result.result == "Error: Tool call invocation was rejected by user."
        assert isinstance(result.items, list)
        assert result.items[0].text == result.result

    def test_keeps_function_result_text_for_anthropic_provider(self) -> None:
        r"""正常系：Anthropic向けでもfunction_result.resultは元の文字列で保持すること

        入力例:
            {
                "role": "tool",
                "contents": [
                    {
                        "type": "function_result",
                        "call_id": "call_1",
                        "result": "Tokyo weather is sunny."
                    }
                ]
            }

        期待例:
            {
                "role": "tool",
                "contents": [
                    {
                        "type": "function_result",
                        "call_id": "call_1",
                        "result": "Tokyo weather is sunny."
                    }
                ]
            }
        """
        message = Message(
            "tool",
            [Content.from_function_result("call_1", result="Tokyo weather is sunny.")],
        )

        converted = ProviderMessageConverter(
            target_provider_family="anthropic"
        ).convert_messages([message])

        assert converted[0].contents[0].result == "Tokyo weather is sunny."


class TestFunctionApprovalConversion:
    """providerごとの有効な承認要求(function_approval) に変換されているかを検証する。"""

    def test_converts_local_approval_request_to_function_call_for_provider(
        self,
    ) -> None:
        r"""正常系：承認要求(function_approval_request) は、function_call として渡されること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {"type": "text", "text": "I need a tool."},
                    {
                        "type": "function_approval_request",
                        "id": "call_1",
                        "function_call": {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo"}
                        }
                    }
                ]
            }

        期待例:
            {
                "role": "assistant",
                "contents": [
                    {"type": "text", "text": "I need a tool."},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": "{\"location\": \"Tokyo\"}"
                    }
                ]
            }
        """
        function_call = Content.from_function_call(
            "call_1", "get_weather", arguments={"location": "Tokyo"}
        )
        approval_request = Content.from_function_approval_request(
            "call_1", function_call
        )
        message = Message(
            "assistant", [Content.from_text("I need a tool."), approval_request]
        )

        converted = ProviderMessageConverter(
            target_provider_family="anthropic"
        ).convert_messages([message])

        assert [content.type for content in converted[0].contents] == [
            "text",
            "function_call",
        ]
        converted_call = converted[0].contents[1]
        assert converted_call.call_id == "call_1"
        assert converted_call.name == "get_weather"
        assert converted_call.arguments == {"location": "Tokyo"}

    def test_keeps_mcp_approval_request_for_provider(self) -> None:
        r"""正常系：MCP承認要求(server_labelが存在する要求)は function_approval_request として渡されること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "function_approval_request",
                        "id": "call_1",
                        "function_call": {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "search",
                            "additional_properties": {"server_label": "docs"}
                        }
                    }
                ]
            }

        期待例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "function_approval_request",
                        "id": "call_1",
                        "function_call": {
                            "additional_properties": {"server_label": "docs"}
                        }
                    }
                ]
            }
        """
        function_call = Content.from_function_call(
            "call_1",
            "search",
            arguments={"query": "docs"},
            additional_properties={"server_label": "docs"},
        )
        approval_request = Content.from_function_approval_request(
            "call_1", function_call
        )
        message = Message("assistant", [approval_request])

        converted = ProviderMessageConverter(
            target_provider_family="anthropic"
        ).convert_messages([message])

        assert converted[0].contents[0].type == "function_approval_request"
        assert isinstance(converted[0].contents[0].function_call, Content)
        assert (
            converted[0].contents[0].function_call.additional_properties["server_label"]
            == "docs"
        )

    def test_drops_duplicate_local_approval_request_when_function_call_exists(
        self,
    ) -> None:
        r"""異常系：同じassistant message内にfunction_callが既にある場合、重複するcall_idのfunction_approval_requestは除外すること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {"type": "text", "text": "I need a tool."},
                    {"type": "function_call", "call_id": "call_1"},
                    {"type": "function_approval_request", "id": "call_1"}
                ]
            }

        期待例:
            {
                "role": "assistant",
                "contents": [
                    {"type": "text", "text": "I need a tool."},
                    {"type": "function_call", "call_id": "call_1"}
                ]
            }
        """
        function_call = Content.from_function_call(
            "call_1", "get_weather", arguments={"location": "Tokyo"}
        )
        approval_request = Content.from_function_approval_request(
            "call_1", function_call
        )
        message = Message(
            "assistant",
            [Content.from_text("I need a tool."), function_call, approval_request],
        )

        converted = ProviderMessageConverter(
            target_provider_family="anthropic"
        ).convert_messages([message])

        assert [content.type for content in converted[0].contents] == [
            "text",
            "function_call",
        ]
        assert converted[0].contents[1].call_id == "call_1"

    def test_keeps_approval_response_as_user_content(self) -> None:
        r"""正常系：userのapproval responseはuser contentとして保持すること

        入力例:
            {
                "role": "user",
                "contents": [
                    {
                        "type": "function_approval_response",
                        "approved": true,
                        "id": "call_1",
                        "function_call": {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "deploy",
                            "arguments": {"env": "dev"}
                        }
                    }
                ]
            }

        期待例:
            {
                "role": "user",
                "contents": [
                    {
                        "type": "function_approval_response",
                        "id": "call_1",
                        "function_call": {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "deploy",
                            "arguments": "{\"env\": \"dev\"}"
                        }
                    }
                ]
            }
        """
        function_call = Content.from_function_call(
            "call_1", "deploy", arguments={"env": "dev"}
        )
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
        assert nested_call.arguments == {"env": "dev"}


class TestToolConversion:
    """provider ごとの有効な tool(function) の call/result に変換されているかを検証する。"""

    def test_sanitizes_function_call_payload_for_provider(self) -> None:
        r"""正常系：function_callはproviderへ渡せるpayloadへ整形されること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                        "additional_properties": {
                            "fc_id": "fc_response_scoped",
                            "status": "completed"
                        },
                        "raw_representation": "<object>"
                    }
                ]
            }

        期待例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": "{\"location\": \"Tokyo\"}",
                        "additional_properties": {
                            "status": "completed"
                        }
                    }
                ]
            }
        """
        message = Message(
            "assistant",
            [
                Content.from_function_call(
                    "call_1",
                    "get_weather",
                    arguments={"location": "Tokyo"},
                    additional_properties={
                        "fc_id": "fc_response_scoped",
                        "status": "completed",
                    },
                    raw_representation=object(),
                )
            ],
        )

        converted = CommonMessageConverter().convert_message(message)

        assert len(converted) == 1
        assert converted[0].role == "assistant"
        assert [content.type for content in converted[0].contents] == ["function_call"]
        function_call = converted[0].contents[0]
        assert function_call.arguments == {"location": "Tokyo"}
        assert function_call.additional_properties["fc_id"] == "fc_response_scoped"
        assert function_call.additional_properties["status"] == "completed"
        assert function_call.raw_representation is None

    def test_openai_stringifies_function_call_arguments(self) -> None:
        message = Message(
            "assistant",
            [
                Content.from_function_call(
                    "call_1",
                    "get_weather",
                    arguments={"location": "Tokyo"},
                    additional_properties={"fc_id": "fc_response_scoped"},
                )
            ],
        )

        converted = ProviderMessageConverter(
            target_provider_family="openai"
        ).convert_message(message)

        function_call = converted[0].contents[0]
        assert function_call.arguments == '{"location": "Tokyo"}'
        assert function_call.additional_properties["fc_id"] == "fc_response_scoped"


class TestMcpToolConversion:
    """providerごとの有効なMCP tool call/result に変換されているか検証する。"""

    def test_sanitizes_nested_mcp_server_tool_result_content_for_provider(
        self,
    ) -> None:
        r"""正常系：MCP tool result の output内 から不要なプロパティを削除すること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "mcp_server_tool_call",
                        "call_id": "mcp_call_1",
                        "name": "microsoft_docs_search",
                        "server_name": "Microsoft_Learn_MCP",
                        "arguments": {}
                    },
                    {
                        "type": "mcp_server_tool_result",
                        "call_id": "mcp_call_1",
                        "output": [
                            {
                                "type": "text",
                                "text": "result",
                                "additional_properties": {
                                    "provider_internal": "drop"
                                },
                                "raw_representation": "<object>"
                            }
                        ],
                        "additional_properties": {
                            "fc_id": "response_scoped"
                        }
                    }
                ]
            }

        期待例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "mcp_server_tool_call",
                        "call_id": "mcp_call_1",
                        "name": "microsoft_docs_search",
                        "server_name": "Microsoft_Learn_MCP",
                        "arguments": "{}"
                    },
                    {
                        "type": "mcp_server_tool_result",
                        "call_id": "mcp_call_1",
                        "output": [
                            {
                                "type": "text",
                                "text": "result"
                            }
                        ]
                    }
                ]
            }
        """
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
        assert tool_result.additional_properties == {"fc_id": "response_scoped"}
        assert tool_result.output[0]["type"] == "text"
        assert "additional_properties" not in tool_result.output[0]
        assert "raw_representation" not in tool_result.output[0]


class TestBasicConversion:
    """providerへ渡すmessageのrole/content分類を検証する。"""

    def test_splits_role_changes_and_keeps_author_only_for_original_role(self) -> None:
        r"""正常系：1つのmessage内でassistant contentとtool contentが混在する場合、roleごとのmessageへ分けること

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
        assert [item.author_name for item in converted] == [
            "assistant-name",
            None,
            "assistant-name",
        ]

    def test_drops_usage_content(self) -> None:
        """正常系：usage contentはproviderへ渡すmessageから除外すること"""
        message = Message("assistant", [Content.from_usage({"total_token_count": 2})])

        converted = CommonMessageConverter().convert_messages([message])

        assert converted == []

    def test_drops_empty_text_and_hosted_vector_store_content(self) -> None:
        r"""正常系：空textとhosted_vector_storeだけのmessageはproviderへ渡さないこと

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
