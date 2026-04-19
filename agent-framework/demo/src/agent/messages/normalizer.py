from __future__ import annotations

from collections.abc import Sequence

from agent_framework import Content, Message


class MessageHistoryNormalizer:
    """MAF Message 履歴を provider に再投入できる構造へ正規化する。

    MAF の実行結果では、tool call と tool result が別メッセージに分かれたり、
    MCP result が call より後の assistant メッセージに出ることがある。
    そのまま会話履歴として provider に渡すと、Anthropic のように tool use/result
    の対応関係を厳密に検証する API で 400 エラーになる。

    この normalizer は保存時・実行前の履歴に対して、以下の provider replay 用の
    構造へ揃える。

    - hosted MCP:
        `mcp_server_tool_call` と `mcp_server_tool_result` を同一 assistant メッセージ内に置く。
        複数 call がある場合は call 群の後に call 順の result 群を並べる。
    - client/local tool:
        `function_call` などの call は assistant、対応する result は直後の toolメッセージに置く。
    - 対応する result がない call、または対応する call がない result は replay 不能な中途半端な履歴なので除外する。
    """

    _CLIENT_TOOL_CALL_TYPES = {
        "function_call",
        "code_interpreter_tool_call",
        "shell_tool_call",
    }
    _CLIENT_TOOL_RESULT_TYPES = {
        "function_result",
        "code_interpreter_tool_result",
        "shell_tool_result",
        "shell_command_output",
    }
    _APPROVAL_REQUEST_TYPE = "function_approval_request"
    _APPROVAL_RESPONSE_TYPE = "function_approval_response"

    def __init__(self, *, normalize_approval_exchanges: bool = True) -> None:
        self._normalize_approval_exchanges = normalize_approval_exchanges

    def normalize_messages(
        self,
        messages: Sequence[Message],
        *,
        current_input_approval_request_ids: set[str] | None = None,
        current_input_approval_response_ids: set[str] | None = None,
    ) -> list[Message]:
        """履歴メッセージを provider replay 用の call/result 構造へ正規化する。"""
        return self._normalize_tool_results(
            messages,
            current_input_approval_request_ids=current_input_approval_request_ids or set(),
            current_input_approval_response_ids=current_input_approval_response_ids or set(),
        )

    def _normalize_tool_results(
        self,
        messages: Sequence[Message],
        *,
        current_input_approval_request_ids: set[str],
        current_input_approval_response_ids: set[str],
    ) -> list[Message]:
        mcp_results_by_call_id = self._collect_mcp_results(messages)
        client_results_by_call_id = self._collect_client_results(messages)
        approval_responses_by_id = (
            self._collect_approval_responses(messages) if self._normalize_approval_exchanges else {}
        )
        normalized: list[Message] = []

        for message in messages:
            if self._is_current_input_approval_message(message, current_input_approval_request_ids):
                continue

            contents = self._without_tool_results(message.contents)
            if not contents:
                continue

            contents_with_results: list[Content] = []
            pending_client_result_contents: list[Content] = []
            pending_approval_response_contents: list[Content] = []
            pending_mcp_call_ids: list[str] = []
            approval_request_ids = {
                request_id
                for content in contents
                if content.type == self._APPROVAL_REQUEST_TYPE
                and (request_id := self._approval_id(content)) is not None
            }

            def append_current_message() -> None:
                nonlocal contents_with_results
                if contents_with_results:
                    normalized.append(self._build_message(message, contents_with_results))
                    contents_with_results = []

            def flush_mcp_results() -> None:
                for call_id in pending_mcp_call_ids:
                    contents_with_results.extend(mcp_results_by_call_id.pop(call_id, []))
                pending_mcp_call_ids.clear()

            def flush_client_results() -> None:
                nonlocal pending_client_result_contents
                if pending_client_result_contents:
                    normalized.append(Message(role="tool", contents=pending_client_result_contents))
                    pending_client_result_contents = []

            def flush_approval_responses() -> None:
                nonlocal pending_approval_response_contents
                if pending_approval_response_contents:
                    normalized.append(Message(role="user", contents=pending_approval_response_contents))
                    pending_approval_response_contents = []

            for content in contents:
                if content.type == "mcp_server_tool_call" and isinstance(content.call_id, str):
                    if content.call_id in mcp_results_by_call_id:
                        contents_with_results.append(content)
                        pending_mcp_call_ids.append(content.call_id)
                    continue

                if content.type in self._CLIENT_TOOL_CALL_TYPES and isinstance(content.call_id, str):
                    if content.call_id in approval_request_ids:
                        continue
                    flush_mcp_results()
                    client_results = client_results_by_call_id.pop(content.call_id, [])
                    if client_results:
                        contents_with_results.append(content)
                        pending_client_result_contents.extend(client_results)
                    continue

                if content.type == self._APPROVAL_REQUEST_TYPE:
                    flush_mcp_results()
                    request_id = self._approval_id(content)
                    client_results = client_results_by_call_id.pop(request_id, []) if request_id else []
                    if client_results:
                        contents_with_results.append(content)
                        pending_client_result_contents.extend(client_results)
                        continue

                    if not self._normalize_approval_exchanges:
                        contents_with_results.append(content)
                        continue

                    approval_responses = approval_responses_by_id.pop(request_id, []) if request_id else []
                    if (
                        current_input_approval_request_ids
                        and request_id not in current_input_approval_request_ids
                        and not approval_responses
                    ) or (
                        current_input_approval_response_ids
                        and request_id not in current_input_approval_response_ids
                        and not approval_responses
                    ):
                        continue
                    contents_with_results.append(content)
                    pending_approval_response_contents.extend(approval_responses)
                    continue

                flush_mcp_results()
                if pending_client_result_contents or pending_approval_response_contents:
                    append_current_message()
                    flush_client_results()
                    flush_approval_responses()
                contents_with_results.append(content)

            flush_mcp_results()
            append_current_message()
            flush_client_results()
            flush_approval_responses()

        return self._merge_adjacent_mcp_assistant_messages(normalized)

    def _collect_mcp_results(self, messages: Sequence[Message]) -> dict[str, list[Content]]:
        results: dict[str, list[Content]] = {}
        for message in messages:
            for content in message.contents:
                if content.type == "mcp_server_tool_result" and isinstance(content.call_id, str):
                    results.setdefault(content.call_id, []).append(content)
        return results

    def _collect_client_results(self, messages: Sequence[Message]) -> dict[str, list[Content]]:
        results: dict[str, list[Content]] = {}
        for message in messages:
            for content in message.contents:
                if content.type in self._CLIENT_TOOL_RESULT_TYPES and isinstance(content.call_id, str):
                    results.setdefault(content.call_id, []).append(content)
        return results

    def _collect_approval_responses(self, messages: Sequence[Message]) -> dict[str, list[Content]]:
        responses: dict[str, list[Content]] = {}
        request_ids = {
            request_id
            for message in messages
            for content in message.contents
            if content.type == self._APPROVAL_REQUEST_TYPE and (request_id := self._approval_id(content)) is not None
        }
        for message in messages:
            for content in message.contents:
                if content.type != self._APPROVAL_RESPONSE_TYPE:
                    continue
                response_id = self._approval_id(content)
                if response_id in request_ids:
                    responses.setdefault(response_id, []).append(content)
        return responses

    def _without_tool_results(self, contents: Sequence[Content]) -> list[Content]:
        return [
            content
            for content in contents
            if content.type != "mcp_server_tool_result"
            and content.type not in self._CLIENT_TOOL_RESULT_TYPES
            and (not self._normalize_approval_exchanges or content.type != self._APPROVAL_RESPONSE_TYPE)
        ]

    def _approval_id(self, content: Content) -> str | None:
        content_id = getattr(content, "id", None)
        if content_id:
            return str(content_id)
        function_call = getattr(content, "function_call", None)
        call_id = getattr(function_call, "call_id", None)
        return str(call_id) if call_id else None

    def _is_current_input_approval_message(self, message: Message, approval_ids: set[str]) -> bool:
        if not self._normalize_approval_exchanges or not approval_ids:
            return False
        return any(
            content.type == self._APPROVAL_REQUEST_TYPE and self._approval_id(content) in approval_ids
            for content in message.contents
        )

    def _merge_adjacent_mcp_assistant_messages(self, messages: Sequence[Message]) -> list[Message]:
        merged: list[Message] = []
        for message in messages:
            if merged and self._should_merge_assistant_messages(merged[-1], message):
                merged[-1] = self._merge_messages(merged[-1], message)
                continue
            merged.append(message)
        return merged

    def _should_merge_assistant_messages(self, left: Message, right: Message) -> bool:
        return (
            left.role == "assistant"
            and right.role == "assistant"
            and (self._has_mcp_content(left) or self._has_mcp_content(right))
        )

    def _has_mcp_content(self, message: Message) -> bool:
        return any(
            content.type in {"mcp_server_tool_call", "mcp_server_tool_result"}
            for content in message.contents
        )

    def _merge_messages(self, left: Message, right: Message) -> Message:
        additional_properties = dict(left.additional_properties)
        additional_properties.update(right.additional_properties)
        return Message(
            role=left.role,
            contents=[*left.contents, *right.contents],
            author_name=left.author_name or right.author_name,
            message_id=left.message_id,
            additional_properties=additional_properties,
        )

    def _build_message(self, original: Message, contents: list[Content]) -> Message:
        return Message(
            role=original.role,
            contents=contents,
            author_name=original.author_name,
            message_id=original.message_id,
            additional_properties=dict(original.additional_properties),
        )
