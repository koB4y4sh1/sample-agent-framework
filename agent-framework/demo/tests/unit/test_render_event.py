from __future__ import annotations

from agent_framework import Content

from ui.render_event import AnthropicRender, GeminiRender, OpenAIRender


class TestStreamTextRendering:
    def test_openai_render_does_not_duplicate_chunk_text(self) -> None:
        events = OpenAIRender().render([Content.from_text("hello")], "hello")

        assert [(event.kind, event.text) for event in events] == [("text", "hello")]

    def test_anthropic_render_does_not_duplicate_chunk_text(self) -> None:
        events = AnthropicRender().render([Content.from_text("hello")], "hello")

        assert [(event.kind, event.text) for event in events] == [("text", "hello")]

    def test_gemini_render_does_not_duplicate_chunk_text(self) -> None:
        events = GeminiRender().render([Content.from_text("hello")], "hello")

        assert [(event.kind, event.text) for event in events] == [("text", "hello")]

    def test_render_keeps_distinct_chunk_text(self) -> None:
        events = OpenAIRender().render([Content.from_function_call("call_1", "tool", arguments={})], "done")

        assert [(event.kind, event.text) for event in events] == [
            ("tool_call", "tool {}"),
            ("text", "done"),
        ]
