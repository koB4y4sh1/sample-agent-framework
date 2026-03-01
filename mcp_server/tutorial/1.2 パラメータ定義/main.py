
from typing import List, Literal, Optional
from mcp.server.fastmcp import FastMCP

# MCPサーバーを初期化
mcp = FastMCP("current-time-server")


async def create_reminder(message, time, priority, recipients):
    return "11000"

@mcp.tool()
async def schedule_reminder(
    message: str,
    time: str,
    priority: Literal["low", "medium", "high"] = "medium",
    recipients: Optional[List[str]] = None
) -> str:
    """リマインダーをスケジュールします。

    Args:
        message: リマインダーのメッセージ
        time: 時刻（ISO 8601形式）
        priority: 優先度（low/medium/high）
        recipients: 通知先のリスト
    Returns:
        作成結果のメッセージ
    """
    # 実際の処理（擬似的な実装）
    reminder_id = await create_reminder(message, time, priority, recipients)
    return f"リマインダーを作成しました（ID: {reminder_id}）"




if __name__ == "__main__":
    # サーバーを起動
    mcp.run(transport="sse")