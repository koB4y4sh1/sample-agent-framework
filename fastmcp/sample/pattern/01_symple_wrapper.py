from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("symple_wrapper_pattern")


@mcp.tool()
async def get_data(self, query: str) -> str:
    """外部APIからデータを取得してLLM向けに整形"""
    try:
        response = await self.client.get(f"{self.api_url}/search?q={query}")
        data = response.json()

        # JSONを自然言語に変換
        results = []
        for item in data["results"]:
            results.append(f"- {item['title']}: {item['description']}")
        return "\n".join(results) if results else "データが見つかりませんでした"

    except Exception as e:
        return f"データ取得中にエラーが発生しました: {str(e)}"
