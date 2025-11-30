import asyncio

from agent_framework import  ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

from memory import UserInfoMemory

# 複数の関数ツールを使用してクラスを作成する
chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

# ContextProvider インスタンスを生成
# エージェントの呼び出し間で永続化する必要がある独自情報を保存するメモリ
memory_provider = UserInfoMemory(chat_client)

async def main():

    # Create the agent with memory
    async with ChatAgent(
        chat_client=chat_client,
        instructions="You are a friendly assistant. Always address the user by their name.",
        context_providers=memory_provider,
    ) as agent:
        # 会話履歴を保持するスレッドの作成（ContextProvider インスタンスがアタッチされている）
        thread = agent.get_new_thread()

        print(await agent.run("Hello, what is the square root of 9?", thread=thread))
        print(await agent.run("My name is Ruaidhrí", thread=thread))
        print(await agent.run("I am 20 years old", thread=thread))

        # スレッドから CotextProvider の メモリにアクセスする
        user_info_memory = thread.context_provider.providers[0]
        if user_info_memory:
            print()
            print(f"MEMORY - User Name: {user_info_memory.user_info.name}")
            print(f"MEMORY - User Age: {user_info_memory.user_info.age}")


if __name__ == "__main__":
    asyncio.run(main())