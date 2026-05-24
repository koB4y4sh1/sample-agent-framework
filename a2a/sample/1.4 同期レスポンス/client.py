import argparse
import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_artifact_text
from a2a.types import Message, Part, Role, SendMessageRequest


async def main() -> None:
    """リクエスト送信後、同じ処理の中で返ってきた結果を表示します。"""
    parser = argparse.ArgumentParser(description="Sync response client")
    parser.add_argument("--url", default="http://127.0.0.1:41244")
    parser.add_argument("--text", default="hello")
    args = parser.parse_args()

    async with httpx.AsyncClient() as httpx_client:
        card = await A2ACardResolver(httpx_client, args.url).get_agent_card()

    client = await create_client(card)
    message = Message(
        role=Role.ROLE_USER,
        message_id=str(uuid.uuid4()),
        context_id=str(uuid.uuid4()),
        parts=[Part(text=args.text)],
    )

    # このサンプルではサーバー側で待ち時間を入れないため、すぐ結果が返ります。
    async for event in client.send_message(SendMessageRequest(message=message)):
        if event.HasField("task"):
            for artifact in event.task.artifacts:
                print(get_artifact_text(artifact, delimiter=" "))
        if event.HasField("artifact_update"):
            print(get_artifact_text(event.artifact_update.artifact, delimiter=" "))

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
