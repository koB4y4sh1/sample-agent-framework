import argparse
import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_artifact_text
from a2a.types import GetTaskRequest, Message, Part, Role, SendMessageRequest, TaskState


async def main() -> None:
    """送信直後の task_id を使って get_task で現在状態を取得します。"""
    parser = argparse.ArgumentParser(description="Get task client")
    parser.add_argument("--url", default="http://127.0.0.1:41247")
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

    # get_task は task_id を指定して、サーバー側に保存された現在状態を取得します。
    await asyncio.sleep(0.5)
    task = await client.get_task(GetTaskRequest(id=task_id))
    print(f"get_task state: {TaskState.Name(task.status.state)}")

    async for event in stream:
        if event.HasField("artifact_update"):
            print(get_artifact_text(event.artifact_update.artifact, delimiter=" "))
        if event.HasField("status_update"):
            state = TaskState.Name(event.status_update.status.state)
            print(f"stream state: {state}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
