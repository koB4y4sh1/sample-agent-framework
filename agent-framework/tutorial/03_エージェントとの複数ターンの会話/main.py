import asyncio
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

# 1. エージェントを作成する
agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
    instructions="You are good at telling jokes.",
    name="Joker"
)

# 2. 会話状態オブジェクト(短期記憶)を作成する
thread = agent.get_new_thread()

# 3. 複数ターンの会話を使用してエージェントを実行する
async def main():
    result1 = await agent.run("Tell me a joke about a pirate.", thread=thread)
    print("Answer1: ",result1.text)

    result2 = await agent.run("Now add some emojis to the joke and tell it in the voice of a pirate's parrot.", thread=thread)
    print("Answer1: ",result2.text)

asyncio.run(main())

# 複数の会話を同時に扱う単一のエージェント
async def multi_thread():
    thread1 = agent.get_new_thread()
    thread2 = agent.get_new_thread()

    result1 = await agent.run("Tell me a joke about a pirate.", thread=thread1)
    print("Thread1: ",result1.text)

    result2 = await agent.run("Tell me a joke about a robot.", thread=thread2)
    print("Thread2: ",result2.text)

    result3 = await agent.run("Now add some emojis to the joke and tell it in the voice of a pirate's parrot.", thread=thread1)
    print("Thread1: ",result3.text)

    result4 = await agent.run("Now add some emojis to the joke and tell it in the voice of a robot.", thread=thread2)
    print("Thread2: ",result4.text)

asyncio.run(multi_thread())