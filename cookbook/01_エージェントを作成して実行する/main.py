import asyncio
from agent_framework import ChatMessage, TextContent, UriContent, Role
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

# 0. 事前準備
    # Azure CLIのインストール・az login
    # 環境変数(.env)の設定・AZURE_OPENAI_ENDPOINT,AZURE_OPENAI_CHAT_DEPLOYMENT_NAME

# 1. エージェントを作成する
agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
    instructions="You are good at telling jokes.",
    name="Joker"
)


# 2. エージェントを実行する
async def main():
    result = await agent.run("Tell me a joke about a pirate.")
    print("エージェントの結果:",result.text)

asyncio.run(main())


# 3. ストリーミングを使用したエージェントの実行
async def stream(): 
    print("↓ エージェントの結果（ストリーム）:")
    async for update in agent.run_stream("Tell me a joke about a pirate."):
        if update.text:
            print(update.text, end="", flush=True)
    print()

asyncio.run(stream())


# 4. ChatMessage を使用したエージェントの実行
message = ChatMessage(
    role=Role.USER,
    contents=[
        TextContent(text="Tell me a joke about this image?"),
        UriContent(uri="https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F231426%2F1f1c3a3b-73e6-6990-7bbe-b1b2ad5c59d4.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&s=de30107daa4742fa3871883a96e7de95", media_type="image/jpeg")
    ]
)
async def chat_message(): 
    print("↓ エージェントの結果（ChatMessage）:")
    result = await agent.run(message)
    print(result.text)

asyncio.run(chat_message())