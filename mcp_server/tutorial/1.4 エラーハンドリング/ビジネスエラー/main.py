from datetime import datetime, timezone
from typing import Optional, List, Literal
from mcp.server.fastmcp import FastMCP


async def create_reminder(message, time, priority, recipients):
    return "11000"

class InvalidReminderTimeError(Exception):
    """リマインダー時刻が無効な場合のエラー"""
    # 基本的なカスタム例外として、追加の実装は不要
    pass

class TooManyRecipientsError(Exception):
    """受信者が多すぎる場合のエラー"""
    # 基本的なカスタム例外として、追加の実装は不要
    pass

mcp = FastMCP("reminder-server")

@mcp.tool()
async def schedule_reminder(
    message: str,
    time: str,
    priority: Literal["low", "medium", "high"] = "medium",
    recipients: Optional[List[str]] = None
) -> str:
    """リマインダーをスケジュールします。"""

    # 時刻の検証
    try:
        reminder_time = datetime.fromisoformat(time)
        if reminder_time < datetime.now(timezone.utc):
            raise InvalidReminderTimeError(
                "リマインダー時刻は現在時刻より後である必要があります。"
                f"指定された時刻: {time}"
            )
    except ValueError:
        raise InvalidReminderTimeError(
            f"時刻の形式が無効です。ISO 8601形式で指定してください。入力値: {time}"
        )

    # 受信者数の検証
    if recipients and len(recipients) > 50:
        raise TooManyRecipientsError(
            f"受信者は最大50人までです。現在の受信者数: {len(recipients)}"
        )

    # 実際の処理（擬似的な実装）
    reminder_id = await create_reminder(message, time, priority, recipients)
    return f"リマインダーを作成しました（ID: {reminder_id}）"


if __name__ == "__main__":
    # サーバーを起動
    mcp.run(transport="sse")