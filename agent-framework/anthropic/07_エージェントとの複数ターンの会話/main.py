import asyncio

from agent_framework import Agent
from agent_framework.foundry import AnthropicFoundryClient
from azure.identity import AzureCliCredential, get_bearer_token_provider
from providers import HistoryManager, HistoryManeger, create_message_store

from color_print import print_green, print_yellow


async def main():
    # 1. エージェントの作成
    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://ai.azure.com/.default",
    )
    store = create_message_store("local_file")
    history_provider = HistoryManeger(store=store, max_messages=100)
    memory_provider = HistoryManager(
        max_messages=10,
    )
    agent = Agent(
        client=AnthropicFoundryClient(
            model="claude-haiku-4-5",
            azure_ad_token_provider=token_provider,
        ),
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
        context_providers=[
            history_provider,
            memory_provider,
        ],
    )

    # 2. セッションの作成
    session = agent.create_session()

    # 3. エージェントとの対話
    print_yellow("[Start] Session start. Type a message and press Enter. Type 'exit' to stop.")
    print_yellow("Type a message and press Enter. Type 'exit' to stop.")
    while True:
        user_input = input("[User]: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print_yellow("[End] Session end.")
            break

        result = await agent.run(user_input, session=session)
        print_green("[Agent]:", result.text)
    
asyncio.run(main())
