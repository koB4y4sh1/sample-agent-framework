import argparse
import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_artifact_text
from a2a.types import Message, Part, Role, SendMessageRequest, TaskState


async def main() -> None:
    """1 回だけ Task を送信し、サーバーから返るイベントを表示します。"""
    parser = argparse.ArgumentParser(description="Task send client")
    parser.add_argument("--url", default="http://127.0.0.1:41243")
    parser.add_argument("--text", default="ping")
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

    # SendMessageRequest が A2A の Task 送信リクエストです。
    async for event in client.send_message(SendMessageRequest(message=message)):
        if event.HasField("task"):
            print(f"task_id: {event.task.id}")
            print(f"state: {TaskState.Name(event.task.status.state)}")
            for artifact in event.task.artifacts:
                print(f"artifact: {get_artifact_text(artifact, delimiter=' ')}")
        if event.HasField("artifact_update"):
            artifact = event.artifact_update.artifact
            print(f"artifact: {get_artifact_text(artifact, delimiter=' ')}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
