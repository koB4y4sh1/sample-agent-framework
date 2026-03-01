import json
import tempfile
import os

import asyncio
from agent_framework import  ChatAgent
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential


# 1. エージェントの作成
agent = ChatAgent(
    chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential(),),
    name="Assistant",
    instructions="You are a helpful assistant."
)

# 2. 会話状態を保持する新しいスレッドを取得
thread = agent.get_new_thread()

async def main():
    # 3. エージェント実行時に thread を渡すことで、やり取りが thread に保持される
    response = await agent.run("Tell me a short pirate joke.", thread=thread)
    print(f"Agent 1: {response.text}")

    # 4. thread の永続化
    # 4-1. オブジェクト(dict) へシリアライズ
    serialized_thread = await thread.serialize()
    serialized_json = json.dumps(serialized_thread)
    print(serialized_json)

    # 4-2. JSONに変換することで、DB、ファイル保存ができる
    temp_dir = tempfile.gettempdir() # Temp フォルダ
    file_path = os.path.join(temp_dir, "agent_thread.json")
    with open(file_path, "w") as f:
        f.write(serialized_json)

    # 5. thread の再開
    # 5-1. 永続化された会話履歴を JSON として読み取る
    with open(file_path, "r") as f:
        loaded_json = f.read()
    reloaded_data = json.loads(loaded_json)
    
    # 5-2. 逆シリアル化し thread を再作成する
    resumed_thread = await agent.deserialize_thread(reloaded_data)

    # 5-3. 再開されたスレッドを使用して会話を続行する
    response = await agent.run("Now tell that joke in the voice of a pirate.", thread=resumed_thread)
    print(f"Agent 2: {response.text}")

asyncio.run(main())

