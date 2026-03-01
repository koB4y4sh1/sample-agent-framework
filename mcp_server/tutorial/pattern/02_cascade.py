import asyncio
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from .api import event_api, weather_api

# Initialize FastMCP server
mcp = FastMCP("cascade_pattern")


@mcp.tool()
async def decide_plan(self, location: str, date: datetime) -> str:
    """外部APIからデータを取得してLLM向けに整形"""
    try:

        # 独立した情報を並列取得
        weather_task = asyncio.create_task(weather_api.get_forecast(location))
        events_task = asyncio.create_task(event_api.search(location, date))
        weather, events = await asyncio.gather(weather_task, events_task)

        # 天気に基づいて次の処理を決定
        if weather["condition"] == "rainy":
            venues = filter_indoor_venues(events)
        else:
            venues = events  # 屋外イベントも含める

        # 統合された提案を生成
        return generate_plan(weather, venues)

    except Exception as e:
        return f"データ取得中にエラーが発生しました: {str(e)}" # Bad practice
        # 一部APIがダウンしていても可能な限り有用な結果を返すことを目指す
    
def filter_indoor_venues(events):
    indoor = [event for event in events if events["outside"] is True]
    return indoor

def generate_plan(weather, venues):
    return "I think play game tomorrow"