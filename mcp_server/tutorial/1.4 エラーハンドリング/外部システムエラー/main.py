import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("product-search-server")


async def product_database(query, filters, limit):
    return "11000"

def format_results(results):
    return "finish"


class SearchFilters(BaseModel):
    """検索フィルターの定義"""
    category: Optional[str] = Field(None, description="商品カテゴリー")
    min_price: Optional[float] = Field(None, ge=0, description="最低価格")
    max_price: Optional[float] = Field(None, ge=0, description="最高価格")
    in_stock: Optional[bool] = Field(None, description="在庫ありのみ")

class DatabaseConnectionError(Exception):
    """データベース接続エラー"""
    # 基本的なカスタム例外として、追加の実装は不要
    pass

class SearchTimeoutError(Exception):
    """検索タイムアウトエラー"""
    # 基本的なカスタム例外として、追加の実装は不要
    pass


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


if __name__ == "__main__":
    # サーバーを起動
    mcp.run(transport="sse")