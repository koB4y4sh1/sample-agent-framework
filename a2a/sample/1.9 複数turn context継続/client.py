import argparse
import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_artifact_text
from a2a.types import Message, Part, Role, SendMessageRequest


async def send_one_turn(client, context_id: str, text: str) -> None:
    """同じ context_id を使って 1 turn 分のメッセージを送ります。"""
    message = Message(
        role=Role.ROLE_USER,
        message_id=str(uuid.uuid4()),
        context_id=context_id,
        parts=[Part(text=text)],
    )
    async for event in client.send_message(SendMessageRequest(message=message)):
        if event.HasField("task"):
            for artifact in event.task.artifacts:
                print(get_artifact_text(artifact, delimiter=" "))
        if event.HasField("artifact_update"):
            print(get_artifact_text(event.artifact_update.artifact, delimiter=" "))


async def main() -> None:
    """2 回の送信で同じ context_id を使い、会話継続を確認します。"""
    parser = argparse.ArgumentParser(description="Multi turn context client")
    parser.add_argument("--url", default="http://127.0.0.1:41249")
    args = parser.parse_args()

    async with httpx.AsyncClient() as httpx_client:
        card = await A2ACardResolver(httpx_client, args.url).get_agent_card()

    client = await create_client(card)
    context_id = str(uuid.uuid4())
    print(f"context_id: {context_id}")

    # task_id は毎回変わりますが、context_id を同じにすると同じ会話として扱えます。
    await send_one_turn(client, context_id, "first")
    await send_one_turn(client, context_id, "second")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
