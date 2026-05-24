import argparse
import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_artifact_text, get_message_text
from a2a.types import Message, Part, Role, SendMessageRequest, TaskState


async def main() -> None:
    """長期実行タスクの status_update と最終 artifact を表示します。"""
    parser = argparse.ArgumentParser(description="Long running task client")
    parser.add_argument("--url", default="http://127.0.0.1:41246")
    parser.add_argument("--text", default="run")
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
        if event.HasField("task"):
            print(f"task_id: {event.task.id}")
        if event.HasField("status_update"):
            status = event.status_update.status
            text = ""
            if status.HasField("message"):
                text = " " + get_message_text(status.message, delimiter=" ")
            print(f"state: {TaskState.Name(status.state)}{text}")
        if event.HasField("artifact_update"):
            print(get_artifact_text(event.artifact_update.artifact, delimiter=" "))

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
