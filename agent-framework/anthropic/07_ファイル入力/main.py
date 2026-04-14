import asyncio
from pathlib import Path

from agent_framework import Agent, Content, Message
from agent_framework.foundry import AnthropicFoundryClient
from azure.identity import AzureCliCredential, get_bearer_token_provider

    # テキストと画像を含む user メッセージを作成する
base_dir = Path(__file__).parent
image_bytes = (base_dir / "image.png").read_bytes()
pdf_bytes = (base_dir / "sample.pdf").read_bytes()
word_bytes = (base_dir / "sample.docx").read_bytes()
excel_bytes = (base_dir / "sample.xlsx").read_bytes()
powerpoint_bytes = (base_dir / "sample.pptx").read_bytes()


token_provider = get_bearer_token_provider( 
    AzureCliCredential(),
    "https://ai.azure.com/.default",
)

# 4. メッセージを含むエージェントを実行
async def main():
    # クライアント
    client = AnthropicFoundryClient(model="claude-haiku-4-5", azure_ad_token_provider=token_provider)
    
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
        instructions="You are a helpful assistant that analyzes files.",
        tools=[client.get_code_interpreter_tool()],
        default_options={"betas": ["files-api-2025-04-14"]}, # NOTE: betas は未対応、additional_beta_flagsは実行時にエラーが発生する。 
    )

    # Message API With Code Execution
    response = agent.run(
        stream=True,
        messages=[
            Message(
                role="user",
                contents=[
                    Content.from_text(text="添付したword, excel, powerpointのファイル内容についてCodeInterpreterを用いて分析した結果を"),
                    # NOTE: container_uploadはサポート外、contentsとして追加しても内部返還時に除外される
                    # .venv\Lib\site-packages\agent_framework_anthropic\_chat_client.py _prepare_message_for_anthropic 
                    {"type": "container_upload", "file_id": "file_011Ca3wfvM7hToo3KGuvmPC1"},
                    {"type": "container_upload", "file_id": "file_011Ca3wg23ADRP64XhKeHSjb"},
                    {"type": "container_upload", "file_id": "file_011Ca3wg6aGMzfg2vQuRvrmu"},
                ]
            )
        ]
    )

    async for update in response:
        for content in update.contents:
            if content.type == "text":
                print(content.text)
            elif content.type == "code_interpreter_tool_call":
                print(f"Code Interpreter call: {content.arguments}")
            elif content.type == "code_interpreter_tool_result": # codeinterpreter の結果はfunction_resultとして返される
                print(f"Code Interpreter result: {content.result}")
            elif content.type == "hosted_file":
                print(f"Create File with ID: {content.file_id}")


asyncio.run(main())