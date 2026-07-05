from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_framework import Content, Message

from ._types import ProviderFamily

# ================================
# region Converter
# ================================
class BaseMessageConverter:
    """Content の種類ごとに role を決め、必要なら Message を分割する。

    変換ルール:
    - type = `text` / `data`
        - 元の Message の role を使う
    - type =  `text_reasoning`
        - `role="assistant"` で `type="text"` に変換
        - 推論本文の先頭に [reasoning] ラベルを付ける
        - 推論本文が空なら削除
    - type = `function_call` / `function_approval_request` / tool call 系
        - `role="assistant"` の Message に置く
    - type = `function_result` / tool result 系
        - `role="tool"` の Message に置く
    - 上記以外の type 削除
    """

    def __init__(
        self,
        *,
        reasoning_label: str = "[reasoning]",
    ) -> None:
        self._reasoning_label = reasoning_label

    def convert_messages(self, messages: Sequence[Message]) -> list[Message]:
        """各 Message に `convert_message` を適用し、結果を1つの list にする。"""
        converted: list[Message] = []
        for message in messages:
            converted.extend(self.convert_message(message))
        return converted

    def convert_message(self, message: Message) -> list[Message]:
        """1つの Message を、Content の種類に合う role の Message へ変換する。

        例:
        assistant Message に `[text_reasoning, function_call]` がある場合:
        - OpenAI 向けなら `text_reasoning` と `function_call` を assistant Message に残す。
        - Common/Anthropic 向けで `text_reasoning.text == ""` なら reasoning を削除し、
        `function_call` だけの assistant Message にする。

        例:
        assistant Message に `[text, function_result]` が混ざっている場合:
        - text は assistant Message
        - function_result は tool Message
        に分割する。
        """
        converted: list[Message] = []
        current_role: str | None = None
        current_contents: list[Content] = []

        # メッセージがどのproviderによって生成されたかの情報を取得する。
        source_provider_family = self._source_provider_family(message)

        for content in message.contents:
            # 1. LLM に送らない Content は削除する。
            prepared = self._convert_content(
                content,
                source_provider_family=source_provider_family,
            )
            if prepared is None:
                continue

            # 2. Content の種類に応じた、role を決める。
            # tool call は assistant、tool result は tool、approval response は user など。
            # 3. role が変わった = Message を分割する。
            if current_role is not None and message.role != current_role:
                converted.append(
                    self._build_message(message, current_role, current_contents)
                )
                current_contents = []

            current_role = message.role
            current_contents.append(prepared)

        # 残った最後の Message を追加する。
        if current_role is not None and current_contents:
            converted.append(
                self._build_message(message, current_role, current_contents)
            )

        return converted

    def _build_message(
        self, original: Message, role: str, contents: list[Content]
    ) -> Message:
        return Message(
            role=role,
            contents=contents,
            author_name=original.author_name if role == original.role else None,
            additional_properties=original.additional_properties,
        )

    def _convert_content(
        self,
        content: Content,
        *,
        source_provider_family: ProviderFamily | None = None,
    ) -> Content | None:
        """1つの Content を送信用 Content に変換する。不要なら None を返す。

        - LLM に送らない Content は None を返す。
        - `text` は `_convert_text` に委譲する。
        - `text_reasoning` は `_build_reasoning_content` に委譲する。
        - `function_approval_request` は必要なら `function_call` に変換する。
        - tool call / tool result 系は `_convert_tool_call` / `_convert_tool_result` に委譲する。
        - その他は 返却 しない
        """
        match content.type:
            case "text":
                return self._convert_text(content)
            case "data" | "uri":
                return self._convert_data(content)
            case "text_reasoning":
                return self._convert_text_reasoning(
                    content, source_provider_family=source_provider_family
                )
            case (
                "function_call"  # ローカルツール呼び出
                | "mcp_server_tool_call"  # host MCP ツール呼び出し
                | "code_interpreter_tool_call"  # Code Interpreter ツール呼び出し
                | "shell_tool_call"  # Shell ツール呼び出し
                | "image_generation_tool_call"  # 画像生成ツール呼び出し
            ):
                return self._convert_tool_call(content)
            case (
                "function_result"  # ローカルツール実行結果
                | "mcp_server_tool_result"  # host MCP ツール実行結果
                | "code_interpreter_tool_result"  # Code Interpreter ツール実行結果
                | "shell_tool_result"  # Shell ツール実行結果
                | "shell_command_output"
            ):
                return self._convert_tool_result(content)
            case "function_approval_request":
                return self._convert_approval_request(content)
            case "function_approval_response":
                return self._convert_approval_response(content)
            case _:
                return None

    def _convert_text(self, content: Content) -> Content | None:
        """text Content を送信用に変換する。空文字なら None を返す。"""
        if not content.text:
            return None
        return content

    def _convert_tool_call(self, content: Content) -> Content | None:
        return content

    def _convert_tool_result(self, content: Content) -> Content | None:
        return content

    def _convert_data(self, content: Content) -> Content | None:
        return content

    def _convert_approval_request(self, content: Content) -> Content | None:
        return content

    def _convert_approval_response(self, content: Content) -> Content | None:
        return content

    def _convert_text_reasoning(
        self,
        content: Content,
        *,
        source_provider_family: ProviderFamily | None = None,
    ) -> Content | None:
        """`text_reasoning` を送信用 Content に変換する。

        base では native reasoning を復元せず、常に `text` Content に落とす。
        - `text == ""` なら削除する（None を返す）。
        - それ以外は `reasoning_label` を付けた `text` Content にする。

        provider 固有の native reasoning 復元（OpenAI の `id`+`encrypted_content`、
        Anthropic の `protected_data` など）は、サブクラスがこのメソッドを override し、
        復元できる場合だけ `_convert_reasoning_content` を呼び、
        それ以外は `super()._convert_reasoning_content(...)` に委譲して実装する。
        """
        if not content.text:
            return None
        if self._reasoning_label:
            return Content.from_text(text=f"{self._reasoning_label}\n{content.text}")
        return Content.from_text(text=content.text)

    def _source_provider_family(self, message: Message) -> ProviderFamily | None:
        execution = message.additional_properties.get("execution")
        if not isinstance(execution, Mapping):
            return None
        provider_family = execution.get("provider_family")
        if provider_family in {"anthropic", "openai", "google"}:
            return provider_family
        return None


__all__ = [
    "BaseMessageConverter",
    "ProviderFamily",
]
