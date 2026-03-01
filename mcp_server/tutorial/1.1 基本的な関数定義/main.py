from datetime import datetime

from mcp.server.fastmcp import FastMCP

# MCPサーバーを初期化
mcp = FastMCP("current-time-server")


@mcp.tool()
async def get_current_time() -> str:
    """現在時刻を取得します。
    
    Returns:
        現在の日時を文字列形式で返します
    """
    current_time = datetime.now().strftime("%Y-%m-&d %H:%M:%S")
    return f"現在時刻:{current_time}"


if __name__ == "__main__":
    # サーバーを起動
    mcp.run(transport="sse")