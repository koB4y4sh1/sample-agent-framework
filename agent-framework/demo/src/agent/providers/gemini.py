from __future__ import annotations

from agent_framework_gemini import GeminiChatClient


def create_gemini_chat_client(
    *,
    model: str,
    token_provider=None,
) -> GeminiChatClient:
    return GeminiChatClient(
        model=model,
        # api_key= "GEMINI_API_KEY",            ※必須 環境変数から API キーを取得
        # vertexai="GOOGLE_GENAI_USE_VERTEXAI", ※必須 接続先を Google AI Studio (Gemini API) から Vertex AI に切り替えるためのフラグ
        # project="GOOGLE_GENAI_PROJECT",       ※必須 Google Claude の Project ID
        # location="GOOGLE_CLOUD_LOCATION",     ※必須 Vertex AI の Location
        # credentials="gcloud auth login",      gcloud 資格情報（Azure環境では使用不可）
    )
