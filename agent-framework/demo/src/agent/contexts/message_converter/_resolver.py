from __future__ import annotations

from collections.abc import Sequence

from agent_framework import Content, Message


class ToolExchangeResolver:
    """tool call / tool result / approval response を call_id で対応付けて解決する。

    保存済み履歴では、call と result が別 Message に分かれていたり、
    result が call から離れた位置に保存されることがある。
    このクラスは `call_id` または approval request の `id` を使い、
    LLM に渡す前に以下の形へ直す。

    具体ルール:
    - assistant の `function_call(call_id="call_1")` と
      tool の `function_result(call_id="call_1")` があれば、
      出力は assistant Message の直後に tool Message を置く。
    - assistant の `mcp_server_tool_call(call_id="mcp_1")` と
      `mcp_server_tool_result(call_id="mcp_1")` があれば、
      同じ assistant Message 内で call の後ろに result を置く。
    - `function_result(call_id="x")` だけがあり、対応する call がなければ捨てる。
    - `function_call(call_id="x")` だけがあり、対応する result がなければ捨てる。
    - `function_approval_request(id="call_1")` と
      `function_approval_response(id="call_1")` があれば、
      request の直後に user Message として response を置く。
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
    _MCP_CONTENT_TYPES = {"mcp_server_tool_call", "mcp_server_tool_result"}
    _APPROVAL_REQUEST_TYPE = "function_approval_request"
    _APPROVAL_RESPONSE_TYPE = "function_approval_response"

    def __init__(self, *, pair_approval_exchanges: bool = True) -> None:
        self._pair_approval_exchanges = pair_approval_exchanges

    def resolve_messages(
        self,
        messages: Sequence[Message],
        *,
        current_input_approval_request_ids: set[str] | None = None,
        current_input_approval_response_ids: set[str] | None = None,
    ) -> list[Message]:
        """Message 配列内の tool/approval の対応関係を call_id/id で解決する。"""
        return self._resolve_tool_exchanges(
            messages,
            current_input_approval_request_ids=current_input_approval_request_ids
            or set(),
            current_input_approval_response_ids=current_input_approval_response_ids
            or set(),
        )

    def _resolve_tool_exchanges(
        self,
        messages: Sequence[Message],
        *,
        current_input_approval_request_ids: set[str],
        current_input_approval_response_ids: set[str],
    ) -> list[Message]:
        # 先に result/approval response を ID 別に集める。
        # 後で対応する call/request を見つけた場所へ移動する。
        mcp_results_by_call_id = self._collect_mcp_results(messages)
        client_results_by_call_id = self._collect_client_results(messages)
        approval_responses_by_id = (
            self._collect_approval_responses(messages)
            if self._pair_approval_exchanges
            else {}
        )
        resolved: list[Message] = []

        for message in messages:
            if self._is_current_input_approval_message(
                message, current_input_approval_request_ids
            ):
                continue

            # result/approval response は元の位置には残さない。
            # 対応する call/request の位置へ後で差し込む。
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
                if not contents_with_results:
                    return
                new_msg = self._build_message(message, contents_with_results)
                contents_with_results = []
                if (
                    resolved
                    and resolved[-1].role == "assistant"
                    and new_msg.role == "assistant"
                    and any(
                        c.type in self._MCP_CONTENT_TYPES
                        for c in (*resolved[-1].contents, *new_msg.contents)
                    )
                ):
                    resolved[-1] = self._merge_messages(resolved[-1], new_msg)
                else:
                    resolved.append(new_msg)

            def flush_mcp_results() -> None:
                for call_id in pending_mcp_call_ids:
                    contents_with_results.extend(
                        mcp_results_by_call_id.pop(call_id, [])
                    )
                pending_mcp_call_ids.clear()

            def flush_client_results() -> None:
                nonlocal pending_client_result_contents
                if pending_client_result_contents:
                    resolved.append(
                        Message(role="tool", contents=pending_client_result_contents)
                    )
                    pending_client_result_contents = []

            def flush_approval_responses() -> None:
                nonlocal pending_approval_response_contents
                if pending_approval_response_contents:
                    resolved.append(
                        Message(
                            role="user", contents=pending_approval_response_contents
                        )
                    )
                    pending_approval_response_contents = []

            for content in contents:
                match content.type:
                    case "mcp_server_tool_call":
                        # call と result はどちらも同じ assistant Message の contents に置く。
                        if content.call_id in mcp_results_by_call_id:
                            contents_with_results.append(content)
                            pending_mcp_call_ids.append(content.call_id)
                    case (
                        "function_call"
                        | "code_interpreter_tool_call"
                        | "shell_tool_call"
                    ):
                        # call は assistant Message、result は直後の tool Message に置く。

                        # Guard: call_id がない
                        if content.call_id is None:
                            continue
                        # Guard: ツール承認で, 既に登録
                        if content.call_id in approval_request_ids:
                            continue

                        flush_mcp_results()

                        client_results = client_results_by_call_id.pop(
                            content.call_id, []
                        )
                        if client_results:
                            contents_with_results.append(content)
                            pending_client_result_contents.extend(client_results)
                    case "function_approval_request":
                        # request に対応する response または実行済み result を request の直後へ置く。
                        flush_mcp_results()
                        request_id = self._approval_id(content)
                        # Guard: request_id がない
                        if not request_id:
                            continue
                        client_results = client_results_by_call_id.pop(request_id, [])
                        if client_results:
                            contents_with_results.append(content)
                            pending_client_result_contents.extend(client_results)
                            continue

                        if not self._pair_approval_exchanges:
                            contents_with_results.append(content)
                            continue

                        approval_responses = (
                            approval_responses_by_id.pop(request_id, [])
                            if request_id
                            else []
                        )
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
                    case _:
                        flush_mcp_results()
                        if (
                            pending_client_result_contents
                            or pending_approval_response_contents
                        ):
                            append_current_message()
                            flush_client_results()
                            flush_approval_responses()
                        contents_with_results.append(content)

            flush_mcp_results()
            append_current_message()
            flush_client_results()
            flush_approval_responses()

        return resolved

    def _collect_mcp_results(
        self, messages: Sequence[Message]
    ) -> dict[str, list[Content]]:
        """`mcp_server_tool_result.call_id` ごとに result を集める。"""
        results: dict[str, list[Content]] = {}
        for message in messages:
            for content in message.contents:
                if content.type == "mcp_server_tool_result" and isinstance(
                    content.call_id, str
                ):
                    results.setdefault(content.call_id, []).append(content)
        return results

    def _collect_client_results(
        self, messages: Sequence[Message]
    ) -> dict[str, list[Content]]:
        """`function_result` などの client/local tool result を call_id ごとに集める。"""
        results: dict[str, list[Content]] = {}
        for message in messages:
            for content in message.contents:
                if content.type in self._CLIENT_TOOL_RESULT_TYPES and isinstance(
                    content.call_id, str
                ):
                    results.setdefault(content.call_id, []).append(content)
        return results

    def _collect_approval_responses(
        self, messages: Sequence[Message]
    ) -> dict[str, list[Content]]:
        """`function_approval_response` を対応する request ID ごとに集める。"""
        responses: dict[str, list[Content]] = {}
        request_ids = {
            request_id
            for message in messages
            for content in message.contents
            if content.type == self._APPROVAL_REQUEST_TYPE
            and (request_id := self._approval_id(content)) is not None
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
        """元の位置から result/approval response を取り除く。

        例:
        `tool[function_result(call_id="call_1")]` はここでは空になる。
        対応する `function_call(call_id="call_1")` を見つけた場所で追加する。
        """
        return [
            content
            for content in contents
            if content.type != "mcp_server_tool_result"
            and content.type not in self._CLIENT_TOOL_RESULT_TYPES
            and (
                not self._pair_approval_exchanges
                or content.type != self._APPROVAL_RESPONSE_TYPE
            )
        ]

    def _approval_id(self, content: Content) -> str | None:
        """approval request/response の対応 ID を取り出す。

        request は `content.id`、response は `content.id` または
        `content.function_call.call_id` を使う。
        """
        content_id = getattr(content, "id", None)
        if content_id:
            return str(content_id)
        function_call = getattr(content, "function_call", None)
        call_id = getattr(function_call, "call_id", None)
        return str(call_id) if call_id else None

    def _is_current_input_approval_message(
        self, message: Message, approval_ids: set[str]
    ) -> bool:
        """今回入力側に同じ approval request がある場合、履歴側の同じ request を飛ばす。"""
        if not self._pair_approval_exchanges or not approval_ids:
            return False
        return any(
            content.type == self._APPROVAL_REQUEST_TYPE
            and self._approval_id(content) in approval_ids
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


__all__ = ["ToolExchangeResolver"]
