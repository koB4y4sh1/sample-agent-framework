import asyncio
from agent_framework import ChatMessage, DataContent, TextContent, UriContent, Role
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential


# 1. 画像を分析できるエージェントを作成する
agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
    name="VisionAgent",
    instructions="You are a helpful agent that can analyze images"
)


# 2. テキスト プロンプトと画像 URL の両方を含む ChatMessage を作成する
# ローカル ファイル システムからイメージを読み込む
with open("./image.png", "rb") as f:
    image_bytes = f.read()

message = ChatMessage(
    role=Role.USER,
    contents=[
        TextContent(text="What do you see in this image?"),
        UriContent( # Web URL
            uri="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            media_type="image/jpeg"
        ),
        DataContent( # Base64
            data=image_bytes,
            media_type="image/jpeg"
        ) 
    ]
)



# 4. メッセージを含むエージェントを実行
async def main():
    result = await agent.run(message)
    print(result.text)

asyncio.run(main())