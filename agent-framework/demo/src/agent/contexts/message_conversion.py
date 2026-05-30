from __future__ import annotations

from typing import Any, Protocol

from agent_framework import (
    AgentSession,
    ContextProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
)

from agent.messages import MessageHistoryNormalizer, ProviderMessageConverter

EXECUTED_INPUT_MESSAGES_METADATA_KEY = "message_conversion:executed_input_messages"


class ReplayMessageConverter(Protocol):
    def convert_messages(self, messages: list[Message]) -> list[Message]:
        ...


class MessageConversionContextProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "message_conversion"
    _METADATA_KEY_PREFIX = "message_conversion_original:"

    def __init__(
        self,
        *,
        history_source_id: str,
        message_converter: ProviderMessageConverter,
        message_normalizer: MessageHistoryNormalizer | None = None,
        replay_converter: ReplayMessageConverter | None = None,
        source_id: str | None = None,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._history_source_id = history_source_id
        self._message_converter = message_converter
        self._message_normalizer = message_normalizer or MessageHistoryNormalizer(normalize_approval_exchanges=True)
        self._replay_converter = replay_converter
        self._metadata_key = f"{self._METADATA_KEY_PREFIX}{self.source_id}:{history_source_id}"

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        messages = context.context_messages.get(self._history_source_id)
        original_input_messages = list(context.input_messages)
        current_input_approval_request_ids = self._approval_ids(
            original_input_messages,
            content_type="function_approval_request",
        )
        current_input_approval_response_ids = self._approval_ids(
            original_input_messages,
            content_type="function_approval_response",
        )
        context.metadata[f"{self._metadata_key}:input"] = original_input_messages
        context.input_messages = self._convert_for_replay(original_input_messages)

        if not messages:
            return

        context.metadata[self._metadata_key] = list(messages)
        normalized_messages = self._message_normalizer.normalize_messages(
            messages,
            current_input_approval_request_ids=current_input_approval_request_ids,
            current_input_approval_response_ids=current_input_approval_response_ids,
        )
        context.context_messages[self._history_source_id] = self._convert_for_replay(normalized_messages)

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        original_input_messages = context.metadata.get(f"{self._metadata_key}:input")
        approval_storage_messages = self._approval_storage_messages(
            original_messages=original_input_messages,
            executed_messages=context.input_messages,
        )
        if approval_storage_messages:
            context.metadata[EXECUTED_INPUT_MESSAGES_METADATA_KEY] = approval_storage_messages
        elif self._has_function_results(context.input_messages):
            context.metadata[EXECUTED_INPUT_MESSAGES_METADATA_KEY] = list(context.input_messages)

        original_messages = context.metadata.pop(self._metadata_key, None)
        if isinstance(original_messages, list) and all(isinstance(item, Message) for item in original_messages):
            context.context_messages[self._history_source_id] = original_messages
        original_input_messages = context.metadata.pop(f"{self._metadata_key}:input", None)
        if isinstance(original_input_messages, list) and all(isinstance(item, Message) for item in original_input_messages):
            context.input_messages = original_input_messages

    def _approval_ids(self, messages: list[Message], *, content_type: str) -> set[str]:
        approval_ids: set[str] = set()
        for message in messages:
            for content in message.contents:
                if content.type != content_type:
                    continue
                content_id = getattr(content, "id", None)
                if content_id:
                    approval_ids.add(str(content_id))
                    continue
                function_call = getattr(content, "function_call", None)
                call_id = getattr(function_call, "call_id", None)
                if call_id:
                    approval_ids.add(str(call_id))
        return approval_ids

    def _has_function_results(self, messages: list[Message]) -> bool:
        return any(
            content.type in {"function_result", "code_interpreter_tool_result", "shell_tool_result"}
            for message in messages
            for content in message.contents
        )

    def _approval_storage_messages(
        self,
        *,
        original_messages: Any,
        executed_messages: list[Message],
    ) -> list[Message]:
        if not isinstance(original_messages, list) or not all(isinstance(item, Message) for item in original_messages):
            return []

        approval_messages = [
            self._copy_approval_storage_message(message)
            for message in original_messages
            if self._has_approval_request(message) and not self._has_approval_response(message)
        ]
        result_messages = [
            self._copy_message(message)
            for message in executed_messages
            if any(content.type == "function_result" for content in message.contents)
        ]
        if not approval_messages or not result_messages:
            return []
        return [*approval_messages, *result_messages]

    def _has_approval_request(self, message: Message) -> bool:
        return any(content.type == "function_approval_request" for content in message.contents)

    def _has_approval_response(self, message: Message) -> bool:
        return any(content.type == "function_approval_response" for content in message.contents)

    def _convert_for_replay(self, messages: list[Message]) -> list[Message]:
        converted = self._message_converter.convert_messages(messages)
        if self._replay_converter is None:
            return converted
        return self._replay_converter.convert_messages(converted)

    def _copy_message(self, message: Message) -> Message:
        return Message(
            role=message.role,
            contents=list(message.contents),
            author_name=message.author_name,
            message_id=message.message_id,
            additional_properties=dict(message.additional_properties),
        )

    def _copy_approval_storage_message(self, message: Message) -> Message:
        approval_ids = {
            call_id
            for content in message.contents
            if content.type == "function_approval_request"
            and (function_call := getattr(content, "function_call", None)) is not None
            and (call_id := getattr(function_call, "call_id", None)) is not None
        }
        contents = [
            content
            for content in message.contents
            if content.type != "function_call" or content.call_id is None or content.call_id not in approval_ids
        ]
        return Message(
            role=message.role,
            contents=contents,
            author_name=message.author_name,
            message_id=message.message_id,
            additional_properties=dict(message.additional_properties),
        )
