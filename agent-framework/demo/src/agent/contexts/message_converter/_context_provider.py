from __future__ import annotations

from typing import Any

from agent_framework import (
    AgentSession,
    ContextProvider,
    Message,
    SessionContext,
    SupportsAgentRun,
)

from ._resolver import ToolExchangeResolver
from ._types import MessageConverter, ProviderFamily
from .anthropic import AnthropicMessageConverter
from .base import BaseMessageConverter
from .gemini import GeminiMessageConverter
from .openai import OpenAIMessageConverter

_MESSAGE_PART_KEY = "message_conversion:part"
_HISTORY_PART = "history"
_INPUT_PART = "input"


class MessageConversionContextProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "message_conversion"

    def __init__(
        self,
        *,
        history_source_id: str,
        target_provider_family: ProviderFamily | None = None,
        tool_resolver: ToolExchangeResolver | None = None,
        source_id: str | None = None,
    ) -> None:
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)
        self._history_source_id = history_source_id
        self._message_converter = self._create_converter(target_provider_family)
        self._tool_resolver = tool_resolver or ToolExchangeResolver(
            pair_approval_exchanges=False
        )

    @staticmethod
    def _create_converter(
        target_provider_family: ProviderFamily | None,
    ) -> MessageConverter:
        if target_provider_family == "anthropic":
            return AnthropicMessageConverter()
        if target_provider_family == "openai":
            return OpenAIMessageConverter()
        if target_provider_family == "google":
            return GeminiMessageConverter()
        return BaseMessageConverter()

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """履歴と今回入力をまとめて、LLM に渡せる Message へ変換する。

        履歴の最後と今回入力の先頭は、LLM から見ると連続した会話である。
        そのため 過去会話 と 今回入力 を先に結合してから
        順序整理と provider 向け変換を行い、変換後に元の格納先へ分け直す。
        """
        # 1. 会話履歴と今回入力をコピーして取得する。
        history_messages = self._copy_messages_with_part(
            list(context.context_messages.get(self._history_source_id) or []),
            _HISTORY_PART,
        )
        input_messages = self._copy_messages_with_part(
            list(context.input_messages),
            _INPUT_PART,
        )

        # 2. 履歴と今回入力を結合し、会話履歴をproviderに対応したcontentへ変換する。
        converted_messages = self._message_converter.convert_messages(
            [*history_messages, *input_messages]
        )
        # 3. tool call/result の ペアリングとプルーニングを行う。
        # ペアリング — call_id で tool call と tool result を対応付けて隣接させる
        # プルーニング — 対応する result のない call、対応する call のない result を捨てる
        arranged_messages = self._tool_resolver.resolve_messages(converted_messages)

        # 4. 一時ラベルを使って、変換後の会話履歴と今回入力を反映させる。
        history_for_replay, input_for_replay = self._split_messages_by_part(
            arranged_messages
        )
        context.context_messages[self._history_source_id] = history_for_replay
        context.input_messages = input_for_replay

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        return None

    def _copy_messages_with_part(
        self, messages: list[Message], part: str
    ) -> list[Message]:
        copied: list[Message] = []
        for message in messages:
            additional_properties = dict(message.additional_properties)
            additional_properties[_MESSAGE_PART_KEY] = part
            copied.append(
                Message(
                    role=message.role,
                    contents=list(message.contents),
                    author_name=message.author_name,
                    message_id=message.message_id,
                    additional_properties=additional_properties,
                )
            )
        return copied

    def _split_messages_by_part(
        self, messages: list[Message]
    ) -> tuple[list[Message], list[Message]]:
        history_messages: list[Message] = []
        input_messages: list[Message] = []
        for message in messages:
            part = message.additional_properties.pop(_MESSAGE_PART_KEY, _HISTORY_PART)
            if part == _INPUT_PART:
                input_messages.append(message)
            else:
                history_messages.append(message)
        return history_messages, input_messages
