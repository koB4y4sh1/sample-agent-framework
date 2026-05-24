import argparse
import asyncio
import os
import signal
import uuid
from typing import Any

import grpc
import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text
from a2a.helpers.agent_card import display_agent_card
from a2a.types import Message, Part, Role, SendMessageRequest, TaskState


async def _handle_stream(
    stream: Any, current_task_id: str | None
) -> str | None:
    # サーバーから届くイベントを順番に読み取って表示します。
    async for event in stream:
        if event.HasField('message'):
            print('メッセージ:', get_message_text(event.message, delimiter=' '))
            return None

        if not current_task_id:
            if event.HasField('task'):
                current_task_id = event.task.id
                print('--- タスク開始 ---')
                print(f'タスク [state={TaskState.Name(event.task.status.state)}]')
            else:
                raise ValueError(f'最初のイベントが想定外です: {event}')

        if event.HasField('status_update'):
            state_name = TaskState.Name(event.status_update.status.state)
            message_text = (
                ': '
                + get_message_text(
                    event.status_update.status.message, delimiter=' '
                )
                if event.status_update.status.HasField('message')
                else ''
            )
            print(f'タスク状態更新 [state={state_name}]{message_text}')
            if state_name in (
                'TASK_STATE_COMPLETED',
                'TASK_STATE_FAILED',
                'TASK_STATE_CANCELED',
                'TASK_STATE_REJECTED',
            ):
                current_task_id = None
                print('--- タスク終了 ---')
        elif event.HasField('artifact_update'):
            print(
                f'タスク成果物更新 [name={event.artifact_update.artifact.name}]:',
                get_artifact_text(
                    event.artifact_update.artifact, delimiter=' '
                ),
            )
    return current_task_id


async def main() -> None:
    """A2A のターミナルクライアントを起動します。"""
    parser = argparse.ArgumentParser(description='A2A ターミナルクライアント')
    parser.add_argument(
        '--url', default='http://127.0.0.1:41241', help='エージェントの基準 URL'
    )
    parser.add_argument(
        '--transport',
        default=None,
        help='優先するトランスポート (JSONRPC, HTTP+JSON, GRPC)',
    )
    args = parser.parse_args()

    # 接続時に使うクライアント設定を用意します。
    config = ClientConfig(
        grpc_channel_factory=grpc.aio.insecure_channel,
    )
    if args.transport:
        config.supported_protocol_bindings = [args.transport]

    # どの URL に、どのトランスポート優先度でつなぐかを表示します。
    print(
        f'{args.url} に接続します (優先トランスポート: {args.transport or "Any"})'
    )

    async with httpx.AsyncClient() as httpx_client:
        # Agent Card を取得して、対応機能を確認します。
        resolver = A2ACardResolver(httpx_client, args.url)
        card = await resolver.get_agent_card()
        print('\n✓ Agent Card を取得しました:')
        display_agent_card(card)

    # Agent Card をもとに実際のクライアントを作成します。
    client = await create_client(card, client_config=config)

    actual_transport = getattr(client, '_transport', client)
    print(f'  選択されたトランスポート: {actual_transport.__class__.__name__}')

    print('\n接続しました。メッセージを送るか、/quit で終了してください。')

    current_task_id = None
    # 1 回の会話で使う context_id を固定します。
    current_context_id = str(uuid.uuid4())

    while True:
        try:
            # input() はブロッキングなので、別スレッドで読み取ります。
            loop = asyncio.get_running_loop()
            user_input = await loop.run_in_executor(None, input, 'あなた: ')
        except KeyboardInterrupt:
            break

        if user_input.lower() in ('/quit', '/exit'):
            break
        if not user_input.strip():
            continue

        # ユーザー発話を A2A の Message に変換します。
        message = Message(
            role=Role.ROLE_USER,
            message_id=str(uuid.uuid4()),
            parts=[Part(text=user_input)],
            task_id=current_task_id,
            context_id=current_context_id,
        )

        # 送信用のリクエストを作ります。
        request = SendMessageRequest(message=message)

        try:
            # ストリームで返ってくるイベントを順番に処理します。
            stream = client.send_message(request)
            current_task_id = await _handle_stream(stream, current_task_id)
        except (httpx.RequestError, grpc.RpcError) as e:
            print(f'エージェントとの通信でエラーが発生しました: {e}')

    # 終了時にクライアント接続を閉じます。
    await client.close()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))
    asyncio.run(main())