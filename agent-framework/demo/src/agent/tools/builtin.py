"""LLM/providerが持つ組み込みツールの作成モジュール。

ここで扱うbuiltin toolは、Python関数として自作するツールではなく、
Web検索、Code Interpreter、画像生成などprovider側が用意している機能。

providerごとに対応状況や引数が変わる可能性があるため、
基底クラスとprovider別クラスに分割した構成。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_framework import (
    BaseChatClient,
    SupportsCodeInterpreterTool,
    SupportsImageGenerationTool,
    SupportsWebSearchTool,
)
from agent_framework.foundry import FoundryChatClient
from agent_framework_anthropic import AnthropicFoundryClient
from agent_framework_gemini import GeminiChatClient


class BaseBuiltinToolFactory(ABC):
    """builtin tool Factoryの基底クラス。

    provider別Factoryは、このクラスを継承して ``build_tools`` を実装する前提。
    """

    def __init__(self, client: BaseChatClient[Any]) -> None:
        """providerのチャットクライアント保持。"""

        self._client = client

    @abstractmethod
    def build_tools(self) -> list[Any]:
        """このproviderで使えるbuiltin tool一覧。"""

        raise NotImplementedError


class DefaultBuiltinToolFactory(BaseBuiltinToolFactory):
    """特別なprovider固有処理がない場合の標準Factory。"""

    def build_tools(self) -> list[Any]:
        """clientが対応しているbuiltin toolのみの一覧。"""

        return self._build_supported_builtin_tools()

    def _build_supported_builtin_tools(self) -> list[Any]:
        """Supports* interfaceを見たbuiltin toolの追加処理。"""

        tools: list[Any] = []

        # Web検索対応client向けのWeb検索ツール追加。
        if isinstance(self._client, SupportsWebSearchTool):
            tools.append(self._client.get_web_search_tool())

        # Code Interpreter対応client向けのコード実行ツール追加。
        if isinstance(self._client, SupportsCodeInterpreterTool):
            tools.append(self._client.get_code_interpreter_tool())

        # 画像生成対応client向けの画像生成ツール追加。
        if isinstance(self._client, SupportsImageGenerationTool):
            tools.append(self._client.get_image_generation_tool())

        # File Search対応が必要になった場合の追加位置。
        # if isinstance(self._client, SupportsFileSearchTool):
        #     tools.append(self._client.get_file_search_tool())

        return tools


class OpenAIBuiltinToolFactory(DefaultBuiltinToolFactory):
    """OpenAI/Foundry系client向けbuiltin tool Factory。

    現時点では標準Factoryと同じ処理。
    provider固有差分が出た場合の変更対象はこのクラスのみ。
    """


class AnthropicBuiltinToolFactory(DefaultBuiltinToolFactory):
    """Anthropic系client向けbuiltin tool Factory。

    現時点では標準Factoryと同じ処理。
    provider固有差分が出た場合の変更対象はこのクラスのみ。
    """


class GeminiBuiltinToolFactory(DefaultBuiltinToolFactory):
    """Gemini系client向けbuiltin tool Factory。

    現時点では標準Factoryと同じ処理。
    provider固有差分が出た場合の変更対象はこのクラスのみ。
    """


def create_builtin_tool_factory(
    client: BaseChatClient[Any],
) -> BaseBuiltinToolFactory:
    """clientの種類に合うbuiltin tool Factoryの選択。

    呼び出し側がprovider名を直接判定しないための集約関数。
    """

    if isinstance(client, AnthropicFoundryClient):
        return AnthropicBuiltinToolFactory(client)
    if isinstance(client, GeminiChatClient):
        return GeminiBuiltinToolFactory(client)
    if isinstance(client, FoundryChatClient):
        return OpenAIBuiltinToolFactory(client)
    return DefaultBuiltinToolFactory(client)


def build_builtin_tools(client: BaseChatClient[Any]) -> list[Any]:
    """clientに合うFactoryを使ったbuiltin tool一覧の取得。"""

    return create_builtin_tool_factory(client).build_tools()
