import argparse
import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_artifact_text
from a2a.types import Message, Part, Role, SendMessageRequest


async def main() -> None:
    """artifact_update を受け取るたびにチャンクを表示します。"""
    parser = argparse.ArgumentParser(description="Streaming task client")
    parser.add_argument("--url", default="http://127.0.0.1:41245")
    parser.add_argument("--text", default="start")
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

    async for event in client.send_message(SendMessageRequest(message=message)):
        if event.HasField("artifact_update"):
            artifact = event.artifact_update.artifact
            print(get_artifact_text(artifact, delimiter=""), end="", flush=True)
    print()

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
