import asyncio
from pathlib import Path

from agent_framework import Content, Message
from agent_framework.foundry import AnthropicFoundryClient
from azure.identity import AzureCliCredential, get_bearer_token_provider
from color_print import print_blue
from message_converter import AnthropicMessageConverter

base_dir = Path(__file__).parent
image_bytes = (base_dir / "image.png").read_bytes()
reasoning="""
ユーザーが入力画像の内容について説明を求めています。この画像は、Microsoftの Agent Framework とその関連するエコシステムについての図のようです。

画像の内容を説明します：

1. **中央部分**：Microsoft Agent Framework（紫色の立体的なロゴ）が中心に配置されている

2. **上部**：M365 Agents SDKが表示されており、矢印でMicrosoft Agent Frameworkに接続されている

3. **上部のアイコン群**：複数のMicrosoft関連サービスのアイコン
   - 何かのツール
   - メモ帳のような機能
   - Slackのようなアイコン
   - チャットアイコン
   - Teamsのようなアイコン
   - Photoのようなアイコン
   - Messengerのようなアイコン

4. **左側**：Copilot Studio（ターコイズ/青色のロゴ）とMicrosoft Agent Frameworkが双方向矢印で接続されている

5. **右側**：Azure AI Foundry（青と紫のロゴ）とMicrosoft Agent Frameworkが双方向矢印で接続されている

6. **下部のアイコン群**：複数のデータベース/AI関連サービス
   - 惑星のようなアイコン
   - ストレージのようなアイコン
   - MongoDB
   - Snowflake
   - Gemini
   - AI関連のアイコン
   - 回路のようなアイコン

このダイアグラムは、Microsoft Agent Frameworkを中心としたエコシステム全体を示しています。
"""
answer = """
# 入力画像の内容について

この画像は、**Microsoft Agent Framework（マイクロソフト エージェント フレームワーク）**を中心とした技術アーキテクチャ図です。      

## 主な構成要素

### 中央
- **Microsoft Agent Framework** - 紫色の球状ロゴで示されるコア技術

### 上部
- **M365 Agents SDK** - フレームワークの上に配置
- 複数のMicrosoftアプリケーションアイコン（Slack、Slack、Slack、Microsoft Teams、Microsoft Copilot、Facebookメッセンジャーなど）   

### 左側
- **Copilot Studio** - ビルディングブロック型のアイコン
- フレームワークとの双方向接続

### 右側
- **Azure AI Foundry** - 青とピンク色のロゴ
- フレームワークとの双方向接続

### 下部
- 各種統合サービス：Cosmos DB、Pinecone、MongoDB、K8s、Gemini、AI、その他のAIツール

## 概要
このダイアグラムは、Microsoftのエージェント開発プラットフォームが、Copilot Studioと Azure AI Foundryを通じて、複数のクラウドサービ スやAIモデルと連携して動作する統合的なエコシステムを表現しています。
"""

async def main() -> None:

    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://ai.azure.com/.default",
    )
    client = AnthropicFoundryClient(model="claude-haiku-4-5", azure_ad_token_provider=token_provider)
    message_converter = AnthropicMessageConverter()

    # 1. メッセージの作成
    messages = [
        # Turn 1: User
        Message(
            role="user",
            contents=[
                Content.from_text(text="入力画像の内容を教えてください。"),
                Content.from_data(data=image_bytes, media_type="image/png"),
            ],
        ), 
        # Turn 2: Assistant
        Message(
            role="assistant",
            contents=[
                Content.from_text_reasoning(
                    text=reasoning,
                    protected_data="実際のsignature", # ストリーム中では、type:signature_delta で、実際の署名データを取得できる
                ),
                Content.from_text(text=answer),
            ],
        ),
        # Turn 3: User
        Message(
            role="user",
            contents=[
                Content.from_text(text="Microsoft Agent Framework について、もう少し詳しく教えてください。"),
            ],
        )
        
    ]
    
    if not client.model:
        raise ValueError("Model is required for token counting.")
    
    
    # 2. トークン数を計算する
    token = await client.anthropic_client.messages.count_tokens(
        model=client.model,
        messages=message_converter.convert_messages(messages),
    )
    print_blue(f"[Token] {token.input_tokens}")


asyncio.run(main())
