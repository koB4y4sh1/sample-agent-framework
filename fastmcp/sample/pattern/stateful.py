import json
from datetime import datetime


class StatefulMCPServer:
    def __init__(self):
        self.client_sessions = {}

    async def execute_tool(self ,name,args):
        return "dummy"
    
    async def handle_connection(self, client_id: str):
        # クライアント固有のセッションを作成
        self.client_sessions[client_id] = {
            "history": [],
            "preferences": {},
            "cache": {}
        }

    async def handle_tool_call(self, client_id: str, tool_name: str, args: dict):
        session = self.client_sessions[client_id]

        # 履歴を更新
        session["history"].append({
            "tool": tool_name,
            "args": args,
            "timestamp": datetime.now()
        })

        # キャッシュを確認
        cache_key = f"{tool_name}:{json.dumps(args)}"
        if cache_key in session["cache"]:
            return session["cache"][cache_key]

        # ツールを実行
        result = await self.execute_tool(tool_name, args)

        # 結果をキャッシュ
        session["cache"][cache_key] = result

        return result
