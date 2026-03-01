import asyncio
from datetime import datetime, timezone
from typing import Callable, List, Literal, Optional, TypeVar

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

T = TypeVar('T')

mcp = FastMCP("product-search-server")


class RetryStrategy:
    """リトライ戦略"""

    def __init__(self, max_attempts: int = 3, initial_delay: float = 1.0):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay

    async def execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
            """リトライ付きで関数を実行"""
            last_error = None
            delay = self.initial_delay

            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if not self.is_retryable(e) or attempt == self.max_attempts - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay *= 2  # 指数関数的バックオフ

            raise last_error

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """リトライ可能なエラーかを判定"""
        return isinstance(error, (ConnectionError, TimeoutError, DatabaseConnectionError))


class LLMFriendlyErrorFormatter:
    """LLMが理解しやすいエラーフォーマッター"""

    @staticmethod
    def format_error(error: Exception, context: dict = None) -> dict:
        """エラーをLLMフレンドリーな形式に変換"""

        error_response = {
            "error_type": type(error).__name__,
            "message": str(error),
            "is_retryable": isinstance(error, (ConnectionError, TimeoutError)),
            "suggested_actions": [],
            "additional_info": {}
        }

        # エラータイプに応じた具体的な提案
        if isinstance(error, InvalidReminderTimeError):
            error_response["suggested_actions"] = [
                "現在時刻を確認し、未来の時刻を指定してください",
                "タイムゾーンを考慮した時刻指定を行ってください"
            ]
        elif isinstance(error, TooManyRecipientsError):
            error_response["suggested_actions"] = [
                "受信者を50人以下に減らしてください",
                "グループを分けて複数のリマインダーを作成してください"
            ]
        elif isinstance(error, SearchTimeoutError):
            error_response["suggested_actions"] = [
                "検索クエリをより具体的にしてください",
                "検索フィルターを追加して結果を絞り込んでください"
            ]

        return error_response


class DatabaseConnectionError(Exception):
    """データベース接続エラー"""
    # 基本的なカスタム例外として、追加の実装は不要
    pass

class SearchTimeoutError(Exception):
    """検索タイムアウトエラー"""
    # 基本的なカスタム例外として、追加の実装は不要
    pass

class InvalidReminderTimeError(Exception):
    """リマインダー時刻が無効な場合のエラー"""
    # 基本的なカスタム例外として、追加の実装は不要
    pass

class TooManyRecipientsError(Exception):
    """受信者が多すぎる場合のエラー"""
    # 基本的なカスタム例外として、追加の実装は不要
    pass


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


async def create_reminder(message, time, priority, recipients):
    return "11000"

class SearchFilters(BaseModel):
    """検索フィルターの定義"""
    category: Optional[str] = Field(None, description="商品カテゴリー")
    min_price: Optional[float] = Field(None, ge=0, description="最低価格")
    max_price: Optional[float] = Field(None, ge=0, description="最高価格")
    in_stock: Optional[bool] = Field(None, description="在庫ありのみ")

@mcp.tool()
async def search_products(
    query: str,
    filters: Optional[SearchFilters] = None,
    limit: int = Field(10, ge=1, le=100)
) -> str:
    """データベースから商品を検索します。"""

    try:
        # タイムアウト付きで検索実行
        results = await asyncio.wait_for(
            product_database(query, filters, limit),
            timeout=30.0
        )
        return format_results(results)

    except asyncio.TimeoutError:
        raise SearchTimeoutError(
            "検索がタイムアウトしました（30秒）。"
            "クエリを簡潔にするか、フィルターを調整してください。"
        )

    except ConnectionError as e:
        raise DatabaseConnectionError(
            "データベースに接続できません。"
            "しばらく待ってから再試行してください。"
            f"詳細: {str(e)}"
        )


async def product_database(query, filters, limit):
    return "11000"

def format_results(results):
    return "finish"

if __name__ == "__main__":
    # サーバーを起動
    mcp.run(transport="sse")