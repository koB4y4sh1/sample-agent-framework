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
                Content.from_text_reasoning(
                    id="rs_045fb12aa0dfc2ea0069170733badc8197969fd65da2ca2501",
                    text="**Formulating a short introduction**\n\nI need to introduce myself in under 8 words. First, I tried \"ChatAPI: Helpful AI offering concise answers,\" which totals 6 words. Then I revised it to \"ChatAPI: Your helpful, concise AI assistant,\" which is also 6 words. Finally, I thought of \"I'm ChatAPI: concise, insightful, helpful AI,\" which is 7 words. I could also go with \"I'm ChatAPI: your helpful, insightful AI\" for a total of 7 words.",
                    protected_data="gAAAAABpFwc8gea3wdYML_Vmb6fRPBc5jJ708Y0ReNfxqLUQ2zFNWmbyBstUDp_0nFTsLZRmp15oScu1_BdyeqKz-h-o-xlAHPM_U4AcjVwigbUzkErTzPfXjV3i3hiC3FtoEWYr4N7RSRY9B7Z-j4PjAb1bSQP6uRIwOhSCFZ2OWt0nw0lRoT5NA-qe8yti0HmpKh4rJJHvkFOVeyhlNGMApO_1VOOUO-igBDKhmraCdDhZXc_LppOXcFhtD9Er-DIi3wslv9UAm61t6dR2yUXMQywQsqAo0Xg-8GJgtF6pVO9YUC-UgfW2uV_ofx_hzyVzrYLYQxNVYHtWMraFSKhjdQdUpcWdPirziQ1i5z2yLgzFoR5_4mllrTNQ21qnAOmZjfptBWf4heF_vmyU_aw_xjvKCzbM_w==",
                    # additional_properties={"encrypted_content":"gAAAAABpFwc8gea3wdYML_Vmb6fRPBc5jJ708Y0ReNfxqLUQ2zFNWmbyBstUDp_0nFTsLZRmp15oScu1_BdyeqKz-h-o-xlAHPM_U4AcjVwigbUzkErTzPfXjV3i3hiC3FtoEWYr4N7RSRY9B7Z-j4PjAb1bSQP6uRIwOhSCFZ2OWt0nw0lRoT5NA-qe8yti0HmpKh4rJJHvkFOVeyhlNGMApO_1VOOUO-igBDKhmraCdDhZXc_LppOXcFhtD9Er-DIi3wslv9UAm61t6dR2yUXMQywQsqAo0Xg-8GJgtF6pVO9YUC-UgfW2uV_ofx_hzyVzrYLYQxNVYHtWMraFSKhjdQdUpcWdPirziQ1i5z2yLgzFoR5_4mllrTNQ21qnAOmZjfptBWf4heF_vmyU_aw_xjvKCzbM_w=="}
                ),
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