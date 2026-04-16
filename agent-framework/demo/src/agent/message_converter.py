from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

from agent_framework import Content, Message


ProviderFamily: TypeAlias = Literal["anthropic", "openai", "gemini"]
ReasoningPolicy: TypeAlias = Literal["as_text", "drop"]


class ProviderMessageConverter:
    """MAF Message 履歴を target provider へ再投入するために変換する。

    この converter は、保存済みの MAF Message を `agent.run` の context message として
    再利用する直前に、provider が受け付けやすい MAF Message へ整えるための層である。
    OpenAI / Anthropic / Gemini のネイティブ API 形式へ直接変換するのではなく、
    Agent Framework に渡す MAF Message としての role / content を補正する。

    主な責務は以下。

    - content の種類に応じて role を分離する。
      例: `function_call` は assistant、`function_result` は tool、
      `function_approval_response` は user として渡す。
    - provider SDK の raw object や、特定 response 内でしか意味を持たない `fc_id` などの
      内部プロパティを取り除く。
    - tool result の nested content から、Anthropic などが拒否する
      `additional_properties` / `raw_representation` を取り除く。
    - reasoning は provider 固有の署名・暗号化情報が必要なため、同一 provider に
      replay できる場合だけ reasoning block として保持する。provider を跨ぐ場合や
      必要な署名が欠ける場合は `reasoning_policy` に従って text 化または除外する。

    tool call/result の順序補正はこの class の責務ではない。MCP result を assistant 内に
    戻したり、function result を tool メッセージへ対応付けたりする履歴構造の補正は
    `MessageHistoryNormalizer` で先に実施する。
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
    _CLIENT_TOOL_CALL_TYPES = {
        "function_call",
        "code_interpreter_tool_call",
        "shell_tool_call",
    }
    _TOOL_CONTENT_TYPES = {
        "function_result",
        "code_interpreter_tool_result",
        "shell_tool_result",
        "shell_command_output",
    }
    _DROP_CONTENT_TYPES = {"usage", "hosted_vector_store"}
    _DROPPED_CONTENT_ADDITIONAL_PROPERTIES = {"fc_id"}

    def __init__(
        self,
        *,
        target_provider_family: ProviderFamily | None = None,
        reasoning_policy: ReasoningPolicy = "as_text",
        reasoning_label: str = "[reasoning]",
    ) -> None:
        self._target_provider_family = target_provider_family
        self._reasoning_policy = reasoning_policy
        self._reasoning_label = reasoning_label

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
        source_provider_family = self._source_provider_family(message)

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
            additional_properties=self._sanitize_mapping(original.additional_properties),
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
            return self._sanitize_reasoning(content, source_provider_family=source_provider_family)

        data = self._sanitize_mapping(content.to_dict(exclude_none=True))
        data["type"] = content.type
        self._sanitize_content_data(data)

        if data.get("type") == "text" and not data.get("text"):
            return None

        return Content.from_dict(data)

    def _sanitize_reasoning(
        self,
        content: Content,
        *,
        source_provider_family: ProviderFamily | None,
    ) -> Content | None:
        if self._can_replay_reasoning(content, source_provider_family):
            return self._sanitize_native_reasoning(content)
        if self._reasoning_policy == "drop":
            return None
        if not content.text:
            return None
        if self._reasoning_label:
            return Content.from_text(text=f"{self._reasoning_label}\n{content.text}")
        return Content.from_text(text=content.text)

    def _can_replay_reasoning(
        self,
        content: Content,
        source_provider_family: ProviderFamily | None,
    ) -> bool:
        target_provider_family = self._target_provider_family
        if target_provider_family is None:
            return False
        if source_provider_family is None:
            source_provider_family = self._infer_reasoning_provider_family(content)
        if source_provider_family != target_provider_family:
            return False
        if target_provider_family == "anthropic":
            return bool(content.protected_data)
        if target_provider_family == "openai":
            return bool(content.id and content.additional_properties.get("encrypted_content"))
        if target_provider_family == "gemini":
            return True
        return False

    def _sanitize_native_reasoning(self, content: Content) -> Content:
        data = self._sanitize_mapping(content.to_dict(exclude_none=True))
        data["type"] = content.type
        data.pop("raw_representation", None)
        return Content.from_dict(data)

    def _source_provider_family(self, message: Message) -> ProviderFamily | None:
        execution = message.additional_properties.get("execution")
        if not isinstance(execution, Mapping):
            return None
        provider_family = execution.get("provider_family")
        if provider_family in {"anthropic", "openai", "gemini"}:
            return provider_family
        return None

    def _infer_reasoning_provider_family(self, content: Content) -> ProviderFamily | None:
        if content.protected_data:
            return "anthropic"
        if content.additional_properties.get("encrypted_content"):
            return "openai"
        return None

    def _sanitize_content_data(self, data: dict[str, Any]) -> None:
        additional_properties = data.get("additional_properties")
        if isinstance(additional_properties, Mapping):
            cleaned = {
                key: value
                for key, value in self._sanitize_mapping(additional_properties).items()
                if key not in self._DROPPED_CONTENT_ADDITIONAL_PROPERTIES
            }
            if cleaned:
                data["additional_properties"] = cleaned
            else:
                data.pop("additional_properties", None)

        content_type = data.get("type")
        if content_type in {
            "function_call",
            "mcp_server_tool_call",
        }:
            data["arguments"] = self._normalize_arguments(data.get("arguments"))

        function_call = data.get("function_call")
        if isinstance(function_call, Mapping):
            nested_call = dict(function_call)
            self._sanitize_content_data(nested_call)
            data["function_call"] = nested_call

        for key in ("items", "inputs", "outputs", "output", "result"):
            if key in data:
                data[key] = self._sanitize_nested_payload(data[key])

    def _normalize_arguments(self, arguments: Any) -> Any:
        if isinstance(arguments, Mapping):
            return self._to_json(arguments)
        return self._sanitize_value(arguments)

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, Content):
            sanitized = self._sanitize_content(value, source_provider_family=None)
            return sanitized.to_dict(exclude_none=True) if sanitized is not None else None
        if isinstance(value, Mapping):
            sanitized = self._sanitize_mapping(value)
            if "type" in sanitized:
                self._sanitize_content_data(sanitized)
            return sanitized
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)

    def _sanitize_nested_payload(self, value: Any) -> Any:
        if isinstance(value, Content):
            sanitized = self._sanitize_content(value, source_provider_family=None)
            if sanitized is None:
                return None
            return self._sanitize_nested_payload(sanitized.to_dict(exclude_none=True))
        if isinstance(value, Mapping):
            sanitized = self._sanitize_mapping(value)
            if "type" in sanitized:
                self._sanitize_content_data(sanitized)
                sanitized.pop("additional_properties", None)
                sanitized.pop("raw_representation", None)
            return sanitized
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._sanitize_nested_payload(item) for item in value]
        return self._sanitize_value(value)

    def _sanitize_mapping(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return {str(key): self._sanitize_value(item) for key, item in value.items()}

    def _to_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)


CommonMessageConverter = ProviderMessageConverter
