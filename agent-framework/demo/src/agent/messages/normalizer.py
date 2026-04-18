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

    def normalize_messages(self, messages: Sequence[Message]) -> list[Message]:
        """履歴メッセージを provider replay 用の call/result 構造へ正規化する。"""
        return self._normalize_tool_results(messages)

    def _normalize_tool_results(self, messages: Sequence[Message]) -> list[Message]:
        mcp_results_by_call_id = self._collect_mcp_results(messages)
        client_results_by_call_id = self._collect_client_results(messages)
        normalized: list[Message] = []

        for message in messages:
            contents = self._without_tool_results(message.contents)
            if not contents:
                continue

            contents_with_results: list[Content] = []
            client_result_contents: list[Content] = []
            pending_mcp_call_ids: list[str] = []

            def flush_mcp_results() -> None:
                for call_id in pending_mcp_call_ids:
                    contents_with_results.extend(mcp_results_by_call_id.pop(call_id, []))
                pending_mcp_call_ids.clear()

            for content in contents:
                if content.type == "mcp_server_tool_call" and isinstance(content.call_id, str):
                    if content.call_id in mcp_results_by_call_id:
                        contents_with_results.append(content)
                        pending_mcp_call_ids.append(content.call_id)
                    continue

                if content.type in self._CLIENT_TOOL_CALL_TYPES and isinstance(content.call_id, str):
                    flush_mcp_results()
                    client_results = client_results_by_call_id.pop(content.call_id, [])
                    if client_results:
                        contents_with_results.append(content)
                        client_result_contents.extend(client_results)
                    continue

                flush_mcp_results()
                contents_with_results.append(content)

            flush_mcp_results()
            if contents_with_results:
                normalized.append(self._build_message(message, contents_with_results))
            if client_result_contents:
                normalized.append(Message(role="tool", contents=client_result_contents))

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

    def _without_tool_results(self, contents: Sequence[Content]) -> list[Content]:
        return [
            content
            for content in contents
            if content.type != "mcp_server_tool_result" and content.type not in self._CLIENT_TOOL_RESULT_TYPES
        ]

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
