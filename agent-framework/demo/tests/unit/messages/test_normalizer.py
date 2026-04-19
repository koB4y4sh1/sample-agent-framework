from __future__ import annotations

from agent.messages import MessageHistoryNormalizer
from agent_framework import Content, Message


class TestToolHistoryNormalization:
    """local tool call/resultの履歴正規化を検証する。"""

    def test_drops_orphan_tool_results(self) -> None:
        r"""異常系：対応するfunction_callがないtool resultは履歴から除外すること

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

    def test_drops_client_tool_calls_without_results(self) -> None:
        r"""異常系：対応するfunction_resultがないfunction_callは履歴から除外すること

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

    def test_places_tool_results_in_separate_tool_message(self) -> None:
        r"""正常系：function_call後にtextが続く場合、function_resultをtextより前のtool messageに置くこと

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

        assert [message.role for message in normalized] == [
            "assistant",
            "tool",
            "assistant",
        ]
        assert [content.type for content in normalized[0].contents] == [
            "text",
            "function_call",
        ]
        assert [content.type for content in normalized[1].contents] == [
            "function_result"
        ]
        assert normalized[1].contents[0].call_id == "call_1"
        assert [content.type for content in normalized[2].contents] == ["text"]
        assert normalized[2].contents[0].text == "after"


class TestMcpToolHistoryNormalization:
    """MCP tool call/resultの履歴正規化を検証する。"""

    def test_merges_adjacent_assistant_messages_when_mcp_content_is_repaired(
        self,
    ) -> None:
        r"""正常系：MCP call/resultを同じassistant messageにまとめ、metadataも保持すること

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
                    Content.from_mcp_server_tool_result(
                        "mcp_call_1", output=[Content.from_text("result")]
                    ),
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

    def test_moves_delayed_mcp_results_next_to_their_tool_calls(self) -> None:
        r"""正常系：後続messageにあるMCP resultは対応するMCP callの直後へ寄せること

        入力例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "function_call", "call_id": "function_call_1"},
                        {"type": "mcp_server_tool_call", "call_id": "mcp_call_1"}
                    ]
                },
                {
                    "role": "tool",
                    "contents": [
                        {"type": "function_result", "call_id": "function_call_1"}
                    ]
                },
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "mcp_server_tool_result", "call_id": "mcp_call_1"},
                        {"type": "text", "text": "next step"}
                    ]
                }
            ]

        期待例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "function_call", "call_id": "function_call_1"},
                        {"type": "mcp_server_tool_call", "call_id": "mcp_call_1"},
                        {"type": "mcp_server_tool_result", "call_id": "mcp_call_1"}
                    ]
                },
                {
                    "role": "tool",
                    "contents": [
                        {"type": "function_result", "call_id": "function_call_1"}
                    ]
                },
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text", "text": "next step"}
                    ]
                }
            ]
        """
        messages = [
            Message(
                "assistant",
                [
                    Content.from_function_call(
                        "function_call_1", "load_skill", arguments={}
                    ),
                    Content.from_mcp_server_tool_call(
                        "mcp_call_1",
                        "microsoft_docs_search",
                        server_name="Microsoft_Learn_MCP",
                        arguments={},
                    ),
                ],
            ),
            Message(
                "tool",
                [Content.from_function_result("function_call_1", result="skill")],
            ),
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

        assert [message.role for message in normalized] == [
            "assistant",
            "tool",
            "assistant",
        ]
        assert [content.type for content in normalized[0].contents] == [
            "function_call",
            "mcp_server_tool_call",
            "mcp_server_tool_result",
        ]
        assert [content.type for content in normalized[1].contents] == [
            "function_result"
        ]
        assert normalized[0].contents[2].call_id == "mcp_call_1"
        assert normalized[1].contents[0].call_id == "function_call_1"
        assert normalized[2].contents[0].type == "text"

    def test_drops_orphan_mcp_tool_results(self) -> None:
        r"""異常系：対応するMCP callがないMCP tool resultは履歴から除外すること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {
                        "type": "mcp_server_tool_result",
                        "call_id": "mcp_call_1",
                        "output": [
                            {"type": "text", "text": "docs"}
                        ]
                    },
                    {"type": "text", "text": "next step"}
                ]
            }

        期待例:
            {
                "role": "assistant",
                "contents": [
                    {"type": "text", "text": "next step"}
                ]
            }
        """
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

        assert len(normalized) == 1
        assert [content.type for content in normalized[0].contents] == ["text"]

    def test_drops_unresolved_mcp_tool_calls(self) -> None:
        r"""異常系：対応するresultがないMCP callは履歴から除外すること

        入力例:
            {
                "role": "assistant",
                "contents": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "mcp_server_tool_call",
                        "call_id": "mcp_call_1",
                        "name": "microsoft_docs_fetch",
                        "server_name": "Microsoft_Learn_MCP",
                        "arguments": {}
                    }
                ]
            }

        期待例:
            {
                "role": "assistant",
                "contents": [
                    {"type": "text", "text": "before"}
                ]
            }
        """
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

        assert len(normalized) == 1
        assert [content.type for content in normalized[0].contents] == ["text"]


class TestApprovalHistoryNormalization:
    """tool approval request/response/resultの履歴構造を検証する。"""

    def test_preserves_approval_request_and_function_result(
        self,
    ) -> None:
        r"""正常系：承認要求と対応するtool結果を履歴上の対応pairとして残すこと

        入力例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text"},
                        {"type": "function_approval_request", "id": "call_1"}
                    ]
                },
                {
                    "role": "tool",
                    "contents": [{"type": "function_result", "call_id": "call_1"}]
                }
            ]

        期待例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text"},
                        {"type": "function_approval_request", "id": "call_1"}
                    ]
                },
                {
                    "role": "tool",
                    "contents": [{"type": "function_result", "call_id": "call_1"}]
                }
            ]
        """
        function_call = Content.from_function_call(
            "call_1", "get_weather", arguments={"location": "Tokyo"}
        )
        approval_request = Content.from_function_approval_request(
            "call_1", function_call
        )
        function_result = Content.from_function_result(
            "call_1", result="Tokyo weather is sunny."
        )
        messages = [
            Message(
                "assistant",
                [Content.from_text("I need to check the weather."), approval_request],
            ),
            Message("tool", [function_result]),
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)

        assert [message.role for message in normalized] == ["assistant", "tool"]
        assert [content.type for content in normalized[0].contents] == [
            "text",
            "function_approval_request",
        ]
        assert normalized[0].contents[1] == approval_request
        assert normalized[1].contents == [function_result]

    def test_preserves_unresolved_approval_request_and_drops_duplicate_function_call(
        self,
    ) -> None:
        r"""正常系：未解決の承認要求は残し、同じcall_idのfunction_callは除外すること

        入力例:
            [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text", "text": "I need to check the weather."},
                        {"type": "function_call", "call_id": "call_1"},
                        {"type": "function_approval_request", "id": "call_1"}
                    ]
                }
            ]

        期待例:
            [
                {"role": "user", "contents": [{"type": "text", "text": "Tokyo weather"}]},
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text"},
                        {"type": "function_approval_request", "id": "call_1"}
                    ]
                }
            ]
        """
        function_call = Content.from_function_call(
            "call_1", "get_weather", arguments={"location": "Tokyo"}
        )
        approval_request = Content.from_function_approval_request(
            "call_1", function_call
        )
        messages = [
            Message("user", [Content.from_text("Tokyo weather")]),
            Message(
                "assistant",
                [
                    Content.from_text("I need to check the weather."),
                    function_call,
                    approval_request,
                ],
            ),
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)

        assert len(normalized) == 2
        assert normalized[0].role == "user"
        assert normalized[0].contents[0].text == "Tokyo weather"
        assert [content.type for content in normalized[1].contents] == [
            "text",
            "function_approval_request",
        ]
        assert normalized[1].contents[1] == approval_request

    def test_drops_duplicate_function_call_when_approval_request_has_result(
        self,
    ) -> None:
        r"""異常系：承認要求にtool結果がある場合、同じcall_idのfunction_callは重複として除外すること

        入力例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text"},
                        {"type": "function_call", "call_id": "call_1"},
                        {"type": "function_approval_request", "id": "call_1"}
                    ]
                },
                {
                    "role": "tool",
                    "contents": [{"type": "function_result", "call_id": "call_1"}]
                }
            ]

        期待例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "text"},
                        {"type": "function_approval_request", "id": "call_1"}
                    ]
                },
                {
                    "role": "tool",
                    "contents": [{"type": "function_result", "call_id": "call_1"}]
                }
            ]
        """
        function_call = Content.from_function_call(
            "call_1", "get_weather", arguments={"location": "Tokyo"}
        )
        approval_request = Content.from_function_approval_request(
            "call_1", function_call
        )
        function_result = Content.from_function_result(
            "call_1", result="Tokyo weather is sunny."
        )
        messages = [
            Message(
                "assistant",
                [
                    Content.from_text("I need to check the weather."),
                    function_call,
                    approval_request,
                ],
            ),
            Message("tool", [function_result]),
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)

        assert [message.role for message in normalized] == ["assistant", "tool"]
        assert [content.type for content in normalized[0].contents] == [
            "text",
            "function_approval_request",
        ]
        assert normalized[0].contents[1] == approval_request
        assert normalized[1].contents == [function_result]

    def test_places_approval_response_next_to_approval_request(self) -> None:
        r"""正常系：承認要求と承認応答を隣接させ、後続textは別assistant messageへ分けること

        入力例:
            [
                {
                    "role": "assistant",
                    "contents": [
                        {"type": "function_approval_request", "id": "call_1"},
                        {"type": "text", "text": "after"}
                    ]
                },
                {"role": "user", "contents": [{"type": "function_approval_response", "id": "call_1"}]}
            ]

        期待例:
            [
                {"role": "assistant", "contents": [{"type": "function_approval_request", "id": "call_1"}]},
                {"role": "user", "contents": [{"type": "function_approval_response", "id": "call_1"}]},
                {"role": "assistant", "contents": [{"type": "text", "text": "after"}]}
            ]
        """
        function_call = Content.from_function_call(
            "call_1", "get_weather", arguments={"location": "Tokyo"}
        )
        approval_request = Content.from_function_approval_request(
            "call_1", function_call
        )
        approval_response = approval_request.to_function_approval_response(True)
        messages = [
            Message("assistant", [approval_request, Content.from_text("after")]),
            Message("user", [approval_response]),
        ]

        normalized = MessageHistoryNormalizer().normalize_messages(messages)

        assert [message.role for message in normalized] == [
            "assistant",
            "user",
            "assistant",
        ]
        assert normalized[0].contents == [approval_request]
        assert normalized[1].contents == [approval_response]
        assert normalized[2].contents[0].text == "after"

    def test_preserves_unresolved_approval_request_without_response_or_result(
        self,
    ) -> None:
        r"""正常系：承認応答やtool結果がまだない承認要求も履歴に残すこと

        入力例:
            [
                {"role": "assistant", "contents": [{"type": "function_approval_request", "id": "call_1"}]}
            ]

        期待例:
            [
                {"role": "assistant", "contents": [{"type": "function_approval_request", "id": "call_1"}]}
            ]
        """
        function_call = Content.from_function_call(
            "call_1", "get_weather", arguments={"location": "Tokyo"}
        )
        approval_request = Content.from_function_approval_request(
            "call_1", function_call
        )

        normalized = MessageHistoryNormalizer().normalize_messages(
            [Message("assistant", [approval_request])]
        )

        assert len(normalized) == 1
        assert normalized[0].contents == [approval_request]
