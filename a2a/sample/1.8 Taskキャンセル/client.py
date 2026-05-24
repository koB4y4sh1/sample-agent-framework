import argparse
import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_message_text
from a2a.types import CancelTaskRequest, Message, Part, Role, SendMessageRequest, TaskState


async def main() -> None:
    """Task を開始し、途中で cancel_task を送ってキャンセル状態を確認します。"""
    parser = argparse.ArgumentParser(description="Cancel task client")
    parser.add_argument("--url", default="http://127.0.0.1:41248")
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

    stream = client.send_message(SendMessageRequest(message=message))
    first_event = await anext(stream)
    task_id = first_event.task.id
    print(f"started task_id: {task_id}")

    await asyncio.sleep(1.5)
    canceled_task = await client.cancel_task(CancelTaskRequest(id=task_id))
    print(f"cancel_task state: {TaskState.Name(canceled_task.status.state)}")

    async for event in stream:
        if event.HasField("status_update"):
            status = event.status_update.status
            text = ""
            if status.HasField("message"):
                text = " " + get_message_text(status.message, delimiter=" ")
            print(f"stream state: {TaskState.Name(status.state)}{text}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
