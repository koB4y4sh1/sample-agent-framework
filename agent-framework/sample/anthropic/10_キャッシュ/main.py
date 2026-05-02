import asyncio
from pathlib import Path

from agent_framework import Agent, Content, Message
from agent_framework.foundry import AnthropicFoundryClient
from azure.identity import AzureCliCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider( 
    AzureCliCredential(),
    "https://ai.azure.com/.default",
)

# 4. メッセージを含むエージェントを実行
async def main():
    # クライアント
    client = AnthropicFoundryClient(model="claude-sonnet-4-6", azure_ad_token_provider=token_provider)
    
    # Files API
    # word_file = await client.anthropic_client.beta.files.upload(
    #     file=("sample.docx", word_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    # )
    # excel_file = await client.anthropic_client.beta.files.upload(
    #     file=("sample.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    # )
    # powerpoint_file = await client.anthropic_client.beta.files.upload(
    #     file=("sample.pptx", powerpoint_bytes, "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    # )
    # print(word_file)
    # print(excel_file)
    # print(powerpoint_file)

    # Agent
    agent = Agent(
        client=client,
        name="AnthropicAgent",
        instructions="You are an helpful assistant",
        default_options={"cache_control": {"type": "ephemeral"}}, # NOTE: cache_control は未対応、実行時にエラーが発生する。 
    )

    # Message API With Code Execution
    response = agent.run(
        stream=True,
        messages=[
            Message(
                role="user",
                contents=[
                    Content.from_text(text="こんにちは！"),
                ]
            ),
            Message(
                role="assistant",
                contents=[
                    Content.from_text(text="こんにちは！😊 お元気ですか？何かお手伝いできることはありますか？"),
                ]
            ),
            Message(
                role="user",
                contents=[
                    Content.from_text(text="Hi"),
                ]
            )
        ]
    )

    async for update in response:
        for content in update.contents:
            if content.type == "text":
                print(content.text, end="", flush=True)
            elif content.type == "usage":
                print(f"Token usage: {content.usage_details}") # TODO: cache_controlで自動キャッシュを適用しても、キャッシュヒットされない


asyncio.run(main())