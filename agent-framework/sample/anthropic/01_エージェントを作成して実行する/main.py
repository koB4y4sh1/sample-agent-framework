import asyncio
from random import randint
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.foundry import AnthropicFoundryClient
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()


@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


token_provider = get_bearer_token_provider( 
    AzureCliCredential(),
    "https://ai.azure.com/.default",
)


async def non_streaming() -> None:
    print("=== [Example] Non-streaming ===")

    agent = Agent(
        client=AnthropicFoundryClient(model="claude-haiku-4-5", azure_ad_token_provider=token_provider),
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    query = "今日の東京の天気は?"
    print(f"[User] question: {query}")
    result = await agent.run(query)
    print(f"[Assistant] answer: {result}\n")



async def streaming() -> None:
    print("=== [Example] Streaming ===")

    agent = Agent(
        client=AnthropicFoundryClient(model="claude-opus-4-6", azure_ad_token_provider=token_provider),
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    query = "今日のパリとドイツの天気は?"
    print(f"[User] question: {query}")
    print("[Assistant]: ", end="", flush=True)
    async for chunk in agent.run(query, stream=True):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")




if __name__ == "__main__":
    asyncio.run(non_streaming())
    asyncio.run(streaming())