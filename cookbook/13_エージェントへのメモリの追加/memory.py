from typing import  Any, MutableSequence, Sequence

from pydantic import BaseModel
from agent_framework import ChatMessage, ChatOptions, ContextProvider, Context, ChatClientProtocol

# 1. モデル クラスの定義
class UserInfo(BaseModel):
    name: str | None = None
    age: int | None = None

# 2. ContextProvider を作成する
class UserInfoMemory(ContextProvider):
    def __init__(self, chat_client: ChatClientProtocol, user_info: UserInfo | None = None, **kwargs: Any):
        """メモリの作成.

        スレッドの再開時にシリアル化されたデータから状態を復元できるようにする必要がある.
        """

        self._chat_client = chat_client
        if user_info:
            self.user_info = user_info
        elif kwargs:
            self.user_info = UserInfo.model_validate(kwargs)
        else:
            self.user_info = UserInfo()

    async def invoking(
        self, 
        messages: ChatMessage | MutableSequence[ChatMessage], 
        **kwargs: Any
    ) -> Context:
        """エージェントが基になる推論サービスを呼び出す前に呼び出される.
        
        エージェントに追加のコンテキストを提供できる.
        """
        instructions: list[str] = []

        if self.user_info.name is None:
            # 年齢情報がないときは、ユーザーに不足情報を要求する指示文を追加する
            instructions.append(
                "Ask the user for their name and politely decline to answer any questions until they provide it."
            )
        else:
            # 現在のユーザー名を示す、指示文の作成 
            instructions.append(f"The user's name is {self.user_info.name}.")

        if self.user_info.age is None:
            # 年齢情報がないときは、ユーザーに不足情報を要求する指示文を追加する
            instructions.append(
                "Ask the user for their age and politely decline to answer any questions until they provide it."
            )
        else:
            # 現在のユーザーの年齢を示す、指示文の作成
            instructions.append(f"The user's age is {self.user_info.age}.")

        # コンテキストの追加
        return Context(instructions=" ".join(instructions))
    
    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """基になる推論サービスからエージェントが応答を受信した後に呼び出される.
        
        要求メッセージ、応答メッセージの検査、コンテキストの状態を更新できる.
        """
        # user メッセージかチェック
        user_messages = [msg for msg in request_messages if hasattr(msg, "role") and msg.role.value == "user"]

        if (self.user_info.name is None or self.user_info.age is None) and user_messages:
            try:
                # ユーザーメッセージからメモリ（ユーザー名、年齢）を抽出する
                result = await self._chat_client.get_response(
                    messages=request_messages,
                    chat_options=ChatOptions(
                        instructions="Extract the user's name and age from the message if present. If not present return nulls.",
                        response_format=UserInfo,
                    ),
                )

                # ユーザー情報を更新する
                if result.value:
                    if self.user_info.name is None and result.value.name:
                        self.user_info.name = result.value.name
                    if self.user_info.age is None and result.value.age:
                        self.user_info.age = result.value.age

            except Exception:
                pass  # Failed to extract, continue without updating


    def serialize(self) -> str:
        """ユーザー情報のシリアル化.

        スレッドの永続化をするのに必要.
        """
        return self.user_info.model_dump_json()
