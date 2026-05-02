import asyncio
from pathlib import Path

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
    word_file = await client.anthropic_client.beta.files.upload(
        file=("sample.docx", word_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    )
    excel_file = await client.anthropic_client.beta.files.upload(
        file=("sample.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    )
    powerpoint_file = await client.anthropic_client.beta.files.upload(
        file=("sample.pptx", powerpoint_bytes, "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    )
    print(word_file)
    print(excel_file)
    print(powerpoint_file)

    # Message API With Code Execution
    response = await client.anthropic_client.beta.messages.create(
        model="claude-opus-4-6",
        betas=["files-api-2025-04-14"],
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "添付したword, excel, powerpointのファイル内容についてCodeInterpreterを用いて分析してください。"},
                    {"type": "container_upload", "file_id": word_file.id},
                    {"type": "container_upload", "file_id": excel_file.id},
                    {"type": "container_upload", "file_id": powerpoint_file.id},
                ],
            }
        ],
        tools=[client.get_code_interpreter_tool()],
    )

    for content in response.content:
        if getattr(content, "text", None):
            print(content.text)


asyncio.run(main())