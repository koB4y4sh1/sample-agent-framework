import asyncio

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

from RedisStorage import RedisChatMessageStore

agent = ChatAgent(
    chat_client=AzureOpenAIChatClient(
        endpoint="https://<myresource>.openai.azure.com",
        credential=AzureCliCredential(),
        ai_model_id="gpt-4o-mini"
    ),
    name="Joker",
    instructions="You are good at telling jokes.",
    # 外部の永続ストレージからチャット履歴を取得する
    chat_message_store_factory=lambda: RedisChatMessageStore(
        redis_url="redis://localhost:6379"
    )
)

async def main():
    # Use the agent with persistent chat history
    thread = agent.get_new_thread()
    response = await agent.run("Tell me a joke about pirates", thread=thread)
    print(response.text)

asyncio.run(main())