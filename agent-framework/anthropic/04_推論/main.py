import asyncio
from random import randint
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.foundry import AnthropicFoundryClient
from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv

from color_print import print_blue, print_green, print_yellow

load_dotenv()


@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def stream_reasoning() -> None:
    print("=== [Example] Streaming with reasoning ===")

    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://ai.azure.com/.default",
    )
    client = AnthropicFoundryClient(model="claude-sonnet-4-6", azure_ad_token_provider=token_provider)
    
    # Create MCP tool configuration using instance method
    mcp_tool = client.get_mcp_tool(
        name="Microsoft_Learn_MCP",
        url="https://learn.microsoft.com/api/mcp",
    )

    # Create web search tool configuration using instance method
    web_search_tool = client.get_web_search_tool()

    agent = Agent(
        client=client,
        name="DocsAgent",
        instructions="You are a helpful agent for both Microsoft docs questions and general questions.",
        tools=[mcp_tool, web_search_tool],
        default_options={
            # Anthropic は max_tokens パラメータが必須
            "max_tokens": 20000,
            "thinking": {"type": "adaptive"}, 
            # claude-opus-4-6 と claude-sonnet-4-6 は enableとbudget_tokens は 廃止
            # 出典： https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
            # "thinking": {"type": "enabled", "budget_tokens": 10000}, 
        },
    )

    query = "Python から Rust を学ぶためのステップを教えてください。"
    print(f"[User] question: {query}")
    print_green("[Assistant] answer: ", end="", flush=True)

    async for chunk in agent.run(query, stream=True):
        for content in chunk.contents:
            if content.type == "text_reasoning":
                print_yellow(content.text, end="", flush=True)
            if content.type == "usage":
                print("\n") # escape
                print_blue(f"[Usage so far: {content.usage_details}]", end="", flush=True)
        if chunk.text:
            print_green("\n") # escape
            print_green(chunk.text, end="", flush=True)




if __name__ == "__main__":
    asyncio.run(stream_reasoning())