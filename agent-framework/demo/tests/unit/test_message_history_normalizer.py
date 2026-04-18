from __future__ import annotations

from agent.messages import MessageHistoryNormalizer
from agent_framework import Content, Message


class TestMessageHistoryNormalizer:
    def test_drops_orphan_client_tool_results(self) -> None:
        r"""異常系：対応するclient tool callがないresultの場合、orphan resultが除外されること

        入力例:
            [
                {
                    "role": "tool",
                    "contents": [
                        {
                            "type": "function_result",
                            "call_id": "call_1",
                            "result": "orphan"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "contents": [
                        {
                            "type": "text",
                            "text": "answer"
                        }
                    ]
                }
            ]

        期待例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {
                            "type": "text",
                            "text": "answer"
                        }
                    ]
                }
            ]
        """
        messages = [
            Message("tool", [Content.from_function_result("call_1", result="orphan")]),
            Message("assistant", [Content.from_text("answer")]),
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)

        assert len(normalized) == 1
        assert normalized[0].role == "assistant"
        assert normalized[0].contents[0].text == "answer"

    def test_places_client_tool_results_in_separate_tool_message(self) -> None:
        r"""正常系：client tool callの後にtextが続く場合、resultがtextより前のtool messageへ配置されること

        入力例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {
                            "type": "text",
                            "text": "before"
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_1"
                        },
                        {
                            "type": "text",
                            "text": "after"
                        }
                    ]
                },
                {
                    "role": "tool",
                    "contents": [
                        {
                            "type": "function_result",
                            "call_id": "call_1",
                            "result": "loaded"
                        }
                    ]
                }
            ]

        期待例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {
                            "type": "text",
                            "text": "before"
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_1"
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
                    "contents": [
                        {
                            "type": "text",
                            "text": "after"
                        }
                    ]
                }
            ]
        """
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

        assert [message.role for message in normalized] == ["assistant", "tool", "assistant"]
        assert [content.type for content in normalized[0].contents] == ["text", "function_call"]
        assert [content.type for content in normalized[1].contents] == ["function_result"]
        assert normalized[1].contents[0].call_id == "call_1"
        assert [content.type for content in normalized[2].contents] == ["text"]
        assert normalized[2].contents[0].text == "after"

    def test_drops_client_tool_calls_without_results(self) -> None:
        r"""異常系：client tool callに対応するresultがない場合、未解決callが除外されること

        入力例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {
                            "type": "function_call",
                            "call_id": "call_1"
                        },
                        {
                            "type": "text",
                            "text": "fallback"
                        }
                    ]
                }
            ]

        期待例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {
                            "type": "text",
                            "text": "fallback"
                        }
                    ]
                }
            ]
        """
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
        r"""正常系：MCP call/result修復後にassistant messageが隣接する場合、1messageへmergeされmetadataが保持されること

        入力例:
            [
                {
                    "role": "assistant",
                    "author_name": "first",
                    "message_id": "message-1",
                    "contents": [
                        {
                            "type": "mcp_server_tool_call",
                            "call_id": "mcp_call_1"
                        }
                    ],
                    "additional_properties": {
                        "left": true
                    }
                },
                {
                    "role": "assistant",
                    "author_name": "second",
                    "contents": [
                        {
                            "type": "mcp_server_tool_result",
                            "call_id": "mcp_call_1"
                        },
                        {
                            "type": "text",
                            "text": "done"
                        }
                    ],
                    "additional_properties": {
                        "right": true
                    }
                }
            ]

        期待例:
            [
                {
                    "role": "assistant",
                    "author_name": "first",
                    "message_id": "message-1",
                    "contents": [
                        {
                            "type": "mcp_server_tool_call",
                            "call_id": "mcp_call_1"
                        },
                        {
                            "type": "mcp_server_tool_result",
                            "call_id": "mcp_call_1"
                        },
                        {
                            "type": "text",
                            "text": "done"
                        }
                    ],
                    "additional_properties": {
                        "left": true,
                        "right": true
                    }
                }
            ]
        """
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
