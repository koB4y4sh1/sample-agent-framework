from __future__ import annotations

import argparse
import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, create_client
from a2a.helpers import get_artifact_text, get_message_text
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)


async def main() -> None:
    """A2A サーバへレビュー依頼を送り、task の状態と artifact を表示します。"""
    parser = argparse.ArgumentParser(description="Code Review Desk A2A クライアント")
    parser.add_argument("--url", default="http://127.0.0.1:41250")
    parser.add_argument("--text", required=True)
    parser.add_argument("--context-id", default=None)
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=None,
        help="指定秒数後に cancel_task を送ります。長いレビューを途中で止めたいときに使います。",
    )
    args = parser.parse_args()

    async with httpx.AsyncClient() as httpx_client:
        card = await A2ACardResolver(httpx_client, args.url).get_agent_card()

    client = await create_client(card)
    message = Message(
        role=Role.ROLE_USER,
        message_id=str(uuid.uuid4()),
        context_id=args.context_id or str(uuid.uuid4()),
        parts=[Part(text=args.text)],
    )

    stream = client.send_message(SendMessageRequest(message=message))
    first_event = await anext(stream)
    task_id = first_event.task.id
    print(f"started task_id: {task_id}")

    task = await client.get_task(GetTaskRequest(id=task_id))
    print(f"get_task state: {TaskState.Name(task.status.state)}")
    current_artifact_name = None

    cancel_task = None
    if args.cancel_after is not None:
        cancel_task = asyncio.create_task(
            _cancel_after(client, task_id, args.cancel_after)
        )

    async for event in stream:
        if event.HasField("status_update"):
            status = event.status_update.status
            message_text = ""
            if status.HasField("message"):
                message_text = " " + get_message_text(status.message, delimiter=" ")
            print(f"state: {TaskState.Name(status.state)}{message_text}")
        if event.HasField("artifact_update"):
            artifact = event.artifact_update.artifact
            artifact_name = getattr(artifact, "name", "artifact")
            if artifact_name != current_artifact_name:
                if current_artifact_name is not None:
                    print()
                print(f"\n[{artifact_name}]")
                current_artifact_name = artifact_name
            print(get_artifact_text(artifact, delimiter=""), end="")
        if event.HasField("task"):
            print(f"state: {TaskState.Name(event.task.status.state)}")

    if cancel_task is not None:
        await cancel_task
    print()
    await client.close()


async def _cancel_after(client, task_id: str, seconds: float) -> None:
    """指定秒数待ってから cancel_task を送信します。"""
    await asyncio.sleep(seconds)
    canceled_task = await client.cancel_task(CancelTaskRequest(id=task_id))
    print(f"cancel_task state: {TaskState.Name(canceled_task.status.state)}")


if __name__ == "__main__":
    asyncio.run(main())
