import asyncio
from pathlib import Path

from agent_framework import Content, Message
from agent_framework.openai import OpenAIChatClient
from azure.identity import AzureCliCredential

agent = OpenAIChatClient(credential=AzureCliCredential()).as_agent(
    name="VisionAgent",
    instructions="You are a helpful agent that can analyze images and documents.",
)

# ローカル ファイル システムからイメージを読み込む
base_dir = Path(__file__).parent
image_bytes = (base_dir / "image.png").read_bytes()
pdf_bytes = (base_dir / "sample.pdf").read_bytes()
word_bytes = (base_dir / "sample.docx").read_bytes()
excel_bytes = (base_dir / "sample.xlsx").read_bytes()
powerpoint_bytes = (base_dir / "sample.pptx").read_bytes()


async def main() -> None:
    # 2. テキスト プロンプトと画像 URL の両方を含む ChatMessage を作成する  

        
    message = Message(
        role="user",
        contents=[
            Content.from_text(text="What do you see in this image?"),
            Content.from_data(
                data=image_bytes,
                media_type="image/png",
            ),
            Content.from_data( # PDF
                data=pdf_bytes,
                media_type="application/pdf",
                additional_properties={"filename": "sample.pdf"}
            ),
            # Content.from_data( # Word
            #     data=word_bytes,
            #     media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            #     additional_properties={"filename": "sample.docx"}
            # ),
            # Content.from_data( # Excel
            #     data=excel_bytes,
            #     media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            #     additional_properties={"filename": "sample.xlsx"}
            # ),
            # Content.from_data( # PowerPoint
            #     data=powerpoint_bytes,
            #     media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            #     additional_properties={"filename": "sample.pptx"}
            # )
        ],
    )

    result = await agent.run(message)
    print(result.text)


asyncio.run(main())
