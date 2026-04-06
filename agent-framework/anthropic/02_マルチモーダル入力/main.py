import asyncio
from pathlib import Path

from agent_framework import Agent, Content, Message
from agent_framework.foundry import AnthropicFoundryClient

# ローカル ファイル システムからイメージを読み込む
base_dir = Path(__file__).parent
image_bytes = (base_dir / "image.png").read_bytes()
pdf_bytes = (base_dir / "sample.pdf").read_bytes()
word_bytes = (base_dir / "sample.docx").read_bytes()
excel_bytes = (base_dir / "sample.xlsx").read_bytes()
powerpoint_bytes = (base_dir / "sample.pptx").read_bytes()


# 4. メッセージを含むエージェントを実行
async def main():
    agent = Agent(
        client=AnthropicFoundryClient(model="claude-haiku-4-5",),
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
    )
    
    # 2. テキスト プロンプトと画像 URL の両方を含む ChatMessage を作成する
    message = Message(
        role="user",
        contents=[
            Content.from_text(text="入力画像の内容を教えてください。"),
            Content.from_data( # Image
                data=image_bytes,
                media_type="image/png"
            ),
            # 以下は入力することはできるが、現状のモデルでは対応していないため、コメントアウトしている
            # Content.from_data( # PDF
            #     data=pdf_bytes,
            #     media_type="application/pdf"
            # ),
            # Content.from_data( # Word
            #     data=word_bytes,
            #     media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            # ),
            # Content.from_data( # Excel
            #     data=excel_bytes,
            #     media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            # ),
            # Content.from_data( # PowerPoint
            #     data=powerpoint_bytes,
            #     media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            # )
        ]
    )
    result = await agent.run(message)
    print(result.text)

asyncio.run(main())