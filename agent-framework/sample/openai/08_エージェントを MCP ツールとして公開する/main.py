import anyio
from typing import Annotated
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from mcp.server.stdio import stdio_server


# 1-1. Tool の定義
def get_specials() -> Annotated[str, "Returns the specials from the menu."]:
    return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

# 1-2. Tool の定義
def get_item_price(
    menu_item: Annotated[str, "The name of the menu item."],
) -> Annotated[str, "Returns the price of the menu item."]:
    return "$9.99"

# 2. 公開するエージェントを作成
agent = AzureOpenAIResponsesClient(credential=AzureCliCredential()).create_agent(
    name="RestaurantAgent",
    description="Answer questions about the menu.",
    tools=[get_specials, get_item_price],
)

# 3. エージェントを MCP サーバーに変換
server = agent.as_mcp_server()

# MCP サーバーを起動
async def run():
    async def handle_stdin():
        # 標準の入力/出力を介して受信要求をリッスンするように MCP サーバーを設定
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    await handle_stdin()

if __name__ == "__main__":
    anyio.run(run)