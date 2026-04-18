from __future__ import annotations

from collections.abc import Sequence

from agent_framework import Content, Message

from .reasoning_replay import ReasoningReplaySanitizer
from .replay_payload_sanitizer import ReplayPayloadSanitizer
from .types import ProviderFamily, ReasoningPolicy


class ProviderMessageConverter:
    """履歴 Message の再投入用変換

    Agent Framework が保持する Message を、LLM へ再投入しやすいrole / content 構成へ変換

    - content 種別に応じた role の再分類。
        例: `function_call` は assistant、`function_result` は tool、
        `function_approval_response` は user として扱う
    - 再利用不要な内部情報の除去。
        例: raw object、`fc_id` など
    - tool result 内のネストした content に対する payload の整理
        例: `additional_properties`、`raw_representation` の除去
    - provider 差分を伴う reasoning の再投入可否判定と整形
        再利用可能な場合のみ保持し、それ以外は `reasoning_policy` に従って、text 化または除外

    role の考え方自体は provider 非依存。
    provider ごとの差分は主に reasoning と一部 payload の扱い。

    対象外はメッセージ順序の補正。
    tool call と tool result の対応付けや順序調整は、
    `MessageHistoryNormalizer` の責務。
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
        """複数の MAF Message を provider replay 用の MAF Message リストへ変換する。

        1つの入力 Message に異なる role として渡すべき content が混在する場合があるため、
        出力件数は入力件数と一致しない。例えば assistant message 内に `function_result` が
        混ざっている場合、assistant message と tool message に分割される。
        """
        converted: list[Message] = []
        for message in messages:
            converted.extend(self.convert_message(message))
        return converted

    def convert_message(self, message: Message) -> list[Message]:
        """単一の MAF Message を role ごとの provider replay 用 Message に変換する。

        content ごとに不要な provider 内部情報を除去し、target provider で再利用できない
        reasoning を `reasoning_policy` に従って処理する。変換後に有効な content が
        残らない場合は空リストを返す。
        """
        converted: list[Message] = []
        current_role: str | None = None
        current_contents: list[Content] = []
        source_provider_family = self._reasoning_sanitizer.source_provider_family(message)

        for content in message.contents:
            sanitized = self._sanitize_content(content, source_provider_family=source_provider_family)
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
    ) -> Content | None:
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

        # ignore no text content, which may cause issues for some providers like Anthropic
        if data.get("type") == "text" and not data.get("text"):
            return None

        return Content.from_dict(data)


CommonMessageConverter = ProviderMessageConverter
