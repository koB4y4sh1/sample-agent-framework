from __future__ import annotations

from agent.messages import CommonMessageConverter
from agent.messages.replay_payload_sanitizer import ReplayPayloadSanitizer
from agent_framework import Content, Message


class TestReplayPayloadSanitizer:
    def test_normalizes_tool_arguments_and_drops_response_scoped_properties(self) -> None:
        r"""正常系：tool argumentsがdictでresponse scoped propertyがある場合、JSON化され不要propertyが除去されること

        入力例:
            {
                "type": "function_call",
                "arguments": {
                    "location": "Tokyo"
                },
                "additional_properties": {
                    "fc_id": "response-scoped",
                    "status": "completed",
                    "opaque": "<object>"
                }
            }

        期待例:
            {
                "type": "function_call",
                "arguments": "{\"location\": \"Tokyo\"}",
                "additional_properties": {
                    "status": "completed",
                    "opaque": "<str>"
                }
            }
        """
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
        r"""正常系：tool result payloadにnested contentがある場合、content内部の不要metadataが除去されること

        入力例:
            {
                "type": "mcp_server_tool_result",
                "output": [
                    {
                        "type": "text",
                        "text": "content object",
                        "additional_properties": {
                            "provider_internal": "drop"
                        },
                        "raw_representation": "<object>"
                    },
                    {
                        "type": "text",
                        "text": "content mapping",
                        "additional_properties": {
                            "fc_id": "drop",
                            "keep": "also dropped for nested content"
                        },
                        "raw_representation": "<object>"
                    }
                ]
            }

        期待例:
            {
                "type": "mcp_server_tool_result",
                "output": [
                    {
                        "type": "text",
                        "text": "content object"
                    },
                    {
                        "type": "text",
                        "text": "content mapping"
                    }
                ]
            }
        """
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
        r"""正常系：approval responseにnested function_callがある場合、argumentsがJSON化され不要propertyが除去されること

        入力例:
            {
                "role": "user",
                "contents": [
                    {
                        "type": "function_approval_response",
                        "function_call": {
                            "type": "function_call",
                            "call_id": "call_1",
                            "arguments": {
                                "environment": "dev"
                            },
                            "additional_properties": {
                                "fc_id": "response-scoped",
                                "status": "completed"
                            }
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
                        "function_call": {
                            "type": "function_call",
                            "call_id": "call_1",
                            "arguments": "{\"environment\": \"dev\"}",
                            "additional_properties": {
                                "status": "completed"
                            }
                        }
                    }
                ]
            }
        """
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
