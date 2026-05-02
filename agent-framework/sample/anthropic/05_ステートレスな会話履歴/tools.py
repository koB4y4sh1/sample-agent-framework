import asyncio

from agent_framework import Agent, Content, Message
from agent_framework.foundry import AnthropicFoundryClient, FoundryChatClient
from agent_framework_openai import OpenAIChatClient
from azure.identity import AzureCliCredential, get_bearer_token_provider

# 1. ツール呼び出しと結果を含むメッセージの例
messages = [
        Message(
            role="user",
            contents=[
                Content.from_text(text="東京の天気を教えて"),
            ]
        ),
        Message(
            role="assistant",
            contents=[
                Content.from_text(text="わかりました。天気を調べます。"),
                # NOTE: ツール呼び出し結果は role="assistant" に呼び出しを、role="tool" に結果を含める。
                # function_calling や local MCP が該当する。
                Content.from_function_call(
                    call_id="unique_id_001",
                    name="get_weather",
                    arguments="{\"location\": \"Tokyo\"}",
                ),
                Content.from_function_call(
                    call_id="unique_id_002",
                    name="get_weather",
                    arguments="{\"location\": \"Osaka\"}",
                ),
            ]
        ),
        Message(
            role="tool", # role="user" でも問題ない。ツールの出力であることがわかりやすいように role="tool" と明記
            contents=[
                Content.from_function_result(
                    call_id="unique_id_001",
                    result="晴れ、25度",
                ),
                Content.from_function_result(
                    call_id="unique_id_002",
                    result="曇り、22度",
                ),
            ]
        ),
        Message(
            role="assistant",
            contents=[
                Content.from_text(text="結果が不足しているため、他の方法で調べます。"),
                # NOTE: MCP実行結果は role="assistant" のメッセージに含めることもできる。
                # host MCP では LLM 側でツールの呼び出しと結果を紐づけが行われるため
                Content.from_mcp_server_tool_call(
                    call_id="unique_id_003",
                    tool_name="web_search",
                    arguments="{\"query\": \"東京の天気\"}",
                ),
                Content.from_mcp_server_tool_call(
                    call_id="unique_id_004",
                    tool_name="web_search",
                    arguments="{\"query\": \"大阪東の天気\"}",
                ),
                Content.from_mcp_server_tool_result(
                    call_id="unique_id_003",
                    output="東京は晴れで25度です。",
                ),
                Content.from_mcp_server_tool_result(
                    call_id="unique_id_004",
                    output="大阪は曇りで22度です。",
                ),
                Content.from_text(text="東京は晴れで25度、大阪は曇りで22度です。"),
            ]
        ),
        Message(
            role="user",
            contents=[
                Content.from_text(text="ありがとうございます！"),
            ]
        )
    ]


# 2. 各クライアントでの実行例
async def run_anthropic():
    agent = Agent(
        client=AnthropicFoundryClient(
            model="claude-haiku-4-5", 
            azure_ad_token_provider=get_bearer_token_provider( 
                AzureCliCredential(),
                "https://ai.azure.com/.default",
            )
        ),
        name="AnthropicAgent",
        instructions="You are a helpful weather agent.",
    )
    result = await agent.run(messages)
    print(f"[Anthropic]\n{result.text}")

async def run_foundry():
    agent = Agent(
        client=FoundryChatClient(model="gpt-5.4-nano", credential=AzureCliCredential()),
        name="FoundryAgent",
        instructions="You are a helpful weather agent.",
    )
    
    result = await agent.run(messages)
    print(f"[Foundry]\n{result.text}")

async def run_openai():
    agent = Agent(
        client=OpenAIChatClient(model="gpt-5.4-nano", credential=AzureCliCredential()),
        name="OpenAIAgent",
        instructions="You are a helpful weather agent.",
    )
    
    result = await agent.run(messages)
    print(f"[OpenAI]\n{result.text}")

async def main():
    await run_anthropic()
    await run_foundry()
    await run_openai()

asyncio.run(main())