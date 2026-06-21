from __future__ import annotations

from collections.abc import Sequence

from agent_framework import Content, Message

from ._normalizer import MessageHistoryNormalizer
from ._reasoning_replay import ReasoningReplaySanitizer
from ._replay_payload_sanitizer import ReplayPayloadSanitizer
from ._types import ProviderFamily, ReasoningPolicy


class BaseProviderMessageConverter:
    """保存済みの Message を、もう一度 LLM に渡せる形へ変換する基底クラス。

    Agent Framework が保存する Message には、実行時の内部情報や provider 固有の
    payload が混ざる。これをそのまま別の LLM 呼び出しに再投入すると、role の不整合や
    provider が受け付けないフィールドで失敗することがある。

    このクラスの責務:
    - content の種類に応じて role を決め直す
      例: `function_call` は assistant、`function_result` は tool
    - 再投入に不要な内部情報を落とす
      例: `raw_representation`、一部の `additional_properties`
    - reasoning を再利用できる場合は残し、できない場合は text 化または削除する

    注意:
    tool call と tool result の順序補正はここでは行わない。
    それは `MessageHistoryNormalizer` の責務。
    """

    _DIRECT_ROLES = {"system", "user", "assistant"}
    _ASSISTANT_CONTENT_TYPES = {
        "function_call",
        "function_approval_request",
        "mcp_server_tool_call",
        "mcp_server_tool_result",
        "code_interpreter_tool_call",
        "image_generation_tool_call",
        "shell_tool_call",
    }
    _TOOL_CONTENT_TYPES = {
        "function_result",
        "code_interpreter_tool_result",
        "shell_tool_result",
        "shell_command_output",
    }
    _DROP_CONTENT_TYPES = {"usage", "hosted_vector_store"}

    def __init__(
        self,
        *,
        target_provider_family: ProviderFamily | None = None,
        reasoning_policy: ReasoningPolicy = "as_text",
        reasoning_label: str = "[reasoning]",
    ) -> None:
        self._target_provider_family = target_provider_family
        self._payload_sanitizer = ReplayPayloadSanitizer(
            lambda content: self._sanitize_content(content, source_provider_family=None)
        )
        self._reasoning_sanitizer = ReasoningReplaySanitizer(
            target_provider_family=target_provider_family,
            reasoning_policy=reasoning_policy,
            reasoning_label=reasoning_label,
            sanitize_mapping=self._payload_sanitizer.sanitize_mapping,
        )

    def convert_messages(self, messages: Sequence[Message]) -> list[Message]:
        """複数の Message を、provider に再投入しやすい Message リストへ変換する。

        1つの Message が複数の Message に分かれることがある。
        例: assistant の Message に `function_result` が混ざっている場合、
        assistant Message と tool Message に分割する。
        """
        converted: list[Message] = []
        for message in messages:
            converted.extend(self.convert_message(message))
        return converted

    def convert_message(self, message: Message) -> list[Message]:
        """1つの Message を、role ごとに整った Message リストへ変換する。

        content を1つずつ見て、不要な情報を落とし、正しい role を決める。
        変換後に有効な content が残らなければ空リストを返す。
        """
        converted: list[Message] = []
        current_role: str | None = None
        current_contents: list[Content] = []
        source_provider_family = self._reasoning_sanitizer.source_provider_family(message)
        # 承認要求の中には元の function_call が埋め込まれている。
        # 同じ call_id の function_call がすでにある場合は、重複して再投入しない。
        function_call_ids = {
            content.call_id
            for content in message.contents
            if content.type == "function_call" and isinstance(content.call_id, str)
        }

        for content in message.contents:
            sanitized = self._sanitize_content(
                content,
                source_provider_family=source_provider_family,
                existing_function_call_ids=function_call_ids,
            )
            if sanitized is None:
                continue

            role = self._content_role(message.role, sanitized)
            if role is None:
                continue

            if current_role is not None and role != current_role:
                converted.append(self._build_message(message, current_role, current_contents))
                current_contents = []

            current_role = role
            current_contents.append(sanitized)

        if current_role is not None and current_contents:
            converted.append(self._build_message(message, current_role, current_contents))

        return converted

    def _build_message(self, original: Message, role: str, contents: list[Content]) -> Message:
        return Message(
            role=role,
            contents=contents,
            author_name=original.author_name if role == original.role else None,
            additional_properties=self._payload_sanitizer.sanitize_mapping(original.additional_properties),
        )

    def _content_role(self, original_role: str, content: Content) -> str | None:
        """content の種類から、再投入時に使う role を決める。"""
        if content.type in self._ASSISTANT_CONTENT_TYPES:
            return "assistant"
        if content.type in self._TOOL_CONTENT_TYPES:
            return "tool"
        if content.type == "function_approval_response":
            return "user"
        if original_role in self._DIRECT_ROLES:
            return original_role
        if original_role == "tool":
            return "tool" if content.type in self._TOOL_CONTENT_TYPES else "user"
        return "user"

    def _sanitize_content(
        self,
        content: Content,
        *,
        source_provider_family: ProviderFamily | None,
        existing_function_call_ids: set[str] | None = None,
    ) -> Content | None:
        """provider に再投入してよい content だけを残し、危ない payload を掃除する。"""
        if content.type in self._DROP_CONTENT_TYPES:
            return None
        if content.type == "text_reasoning":
            return self._reasoning_sanitizer.sanitize_content(
                content,
                source_provider_family=source_provider_family,
            )

        data = self._payload_sanitizer.sanitize_mapping(content.to_dict(exclude_none=True))
        data["type"] = content.type
        self._payload_sanitizer.sanitize_content_data(data)

        # 空文字の text は provider によってエラー原因になるため捨てる。
        if data.get("type") == "text" and not data.get("text"):
            return None

        if data.get("type") == "function_approval_request":
            data = self._normalize_function_approval_request(data, existing_function_call_ids or set())
            if not data:
                return None

        return Content.from_dict(data)

    def _normalize_function_approval_request(self, data: dict, existing_function_call_ids: set[str]) -> dict:
        """承認要求を、provider が扱いやすい function_call 形へ寄せる。"""
        function_call = data.get("function_call")
        if not isinstance(function_call, dict):
            return data

        additional_properties = function_call.get("additional_properties")
        if isinstance(additional_properties, dict) and additional_properties.get("server_label"):
            return data

        call_id = function_call.get("call_id")
        if isinstance(call_id, str) and call_id in existing_function_call_ids:
            return {}

        normalized = dict(function_call)
        normalized["type"] = "function_call"
        return normalized


CommonMessageConverter = BaseProviderMessageConverter

__all__ = [
    "BaseProviderMessageConverter",
    "CommonMessageConverter",
    "MessageHistoryNormalizer",
    "ProviderFamily",
    "ReasoningPolicy",
]
