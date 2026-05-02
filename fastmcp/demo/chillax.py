"""
Chillax MCP Server - 天気に基づいた過ごし方提案サーバー

天気情報を取得し、その結果に基づいて最適なYouTube動画を提案する
カスケード処理の実例としてのMCPサーバー実装
"""

import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("chillax-mcp-server")

# .envファイルから環境変数を読み込む
load_dotenv()

# API設定
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"
YOUTUBE_BASE_URL = "https://www.googleapis.com/youtube/v3/search"

# 定数
class WeatherCondition(Enum):
    PERFECT = "perfect"  # 晴れ/曇り、快適な気温
    HOT = "hot"  # 猛暑
    COLD = "cold"  # 極寒
    RAINY = "rainy"  # 雨/雪
    STORMY = "stormy"  # 嵐/警報級

class Language(Enum):
    JA = "ja"
    EN = "en"
    KO = "ko"
    ZH = "zh"

# 都市名から言語を推測するマッピング
CITY_LANGUAGE_MAP = {
    # 日本
    "tokyo": Language.JA,
    "osaka": Language.JA,
    "kyoto": Language.JA,
    "yokohama": Language.JA,
    "nagoya": Language.JA,
    "東京": Language.JA,
    "大阪": Language.JA,
    "京都": Language.JA,
    # 英語圏
    "london": Language.EN,
    "new york": Language.EN,
    "los angeles": Language.EN,
    "chicago": Language.EN,
    "toronto": Language.EN,
    # 韓国
    "seoul": Language.KO,
    "busan": Language.KO,
    "서울": Language.KO,
    # 中国
    "beijing": Language.ZH,
    "shanghai": Language.ZH,
    "北京": Language.ZH,
    "上海": Language.ZH,
}

# 天気条件に応じた検索クエリテンプレート
SEARCH_QUERIES = {
    WeatherCondition.PERFECT: {
        Language.JA: ["アウトドア vlog", "公園 散歩", "ピクニック", "観光地 おすすめ", "サイクリング"],
        Language.EN: ["outdoor activities", "park walking", "picnic ideas", "travel destinations", "cycling vlog"],
    },
    WeatherCondition.HOT: {
        Language.JA: ["涼しい部屋 過ごし方", "夏 室内", "アイス レシピ", "エアコン 快適", "避暑地"],
        Language.EN: ["indoor summer activities", "cool room ideas", "ice cream recipes", "beat the heat", "air conditioning tips"],
    },
    WeatherCondition.COLD: {
        Language.JA: ["冬 室内 過ごし方", "温かい飲み物 レシピ", "こたつ", "暖房 快適", "冬の読書"],
        Language.EN: ["cozy winter activities", "hot beverage recipes", "warm indoor ideas", "winter reading", "fireplace ambience"],
    },
    WeatherCondition.RAINY: {
        Language.JA: ["雨の日 過ごし方", "ジャズ BGM", "読書 おすすめ", "室内 趣味", "料理 レシピ"],
        Language.EN: ["rainy day activities", "jazz music", "book recommendations", "indoor hobbies", "cooking recipes"],
    },
    WeatherCondition.STORMY: {
        Language.JA: ["台風 備え", "防災", "安全な過ごし方", "リラックス 音楽", "瞑想"],
        Language.EN: ["storm preparation", "safety tips", "relaxation music", "meditation", "calming videos"],
    },
}


def detect_language(city: str) -> Language:
    """都市名から言語を推測"""
    city_lower = city.lower().strip()
    return CITY_LANGUAGE_MAP.get(city_lower, Language.EN)


def categorize_weather(weather_data: Dict[str, Any]) -> WeatherCondition:
    """天気データから天気条件を分類"""
    # メインの天気状態を取得
    main_weather = weather_data.get("weather", [{}])[0].get("main", "").lower()
    temp = weather_data.get("main", {}).get("temp", 20)
    temp_max = weather_data.get("main", {}).get("temp_max", temp)
    temp_min = weather_data.get("main", {}).get("temp_min", temp)
    
    # 嵐や極端な天気をチェック
    if main_weather in ["thunderstorm", "tornado"]:
        return WeatherCondition.STORMY
    
    # 雨や雪をチェック
    if main_weather in ["rain", "snow", "drizzle"]:
        return WeatherCondition.RAINY
    
    # 気温をチェック
    if temp_max >= 30:
        return WeatherCondition.HOT
    if temp_min <= 10:
        return WeatherCondition.COLD
    
    # それ以外は快適な天気
    return WeatherCondition.PERFECT


async def get_weather_forecast(city: str, days_ahead: int = 0) -> Dict[str, Any]:
    """
    指定した都市の天気予報を取得
    
    Args:
        city: 都市名（例: "Tokyo", "London"）
        days_ahead: 何日後の予報を取得するか（0=今日、1=明日、最大5）
    
    Returns:
        天気情報と分析結果
    """
    if not OPENWEATHER_API_KEY:
        return {"error": "OpenWeatherMap API key not configured"}
    
    if days_ahead < 0 or days_ahead > 5:
        return {"error": "days_ahead must be between 0 and 5"}
    
    # 言語を検出
    language = detect_language(city)
    
    # OpenWeatherMap APIを呼び出し
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": language.value,
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(OPENWEATHER_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            return {"error": f"Failed to fetch weather data: {str(e)}"}
    
    # 指定日時の予報を探す
    target_date = datetime.now() + timedelta(days=days_ahead)
    target_date_str = target_date.strftime("%Y-%m-%d")
    
    # 該当日の予報を抽出（12時頃のデータを優先）
    forecast_list = data.get("list", [])
    target_forecast = None
    
    for forecast in forecast_list:
        forecast_time = datetime.fromtimestamp(forecast["dt"])
        if forecast_time.strftime("%Y-%m-%d") == target_date_str:
            # 12時に最も近い予報を選択
            if forecast_time.hour >= 11 and forecast_time.hour <= 13:
                target_forecast = forecast
                break
            elif target_forecast is None:
                target_forecast = forecast
    
    if not target_forecast:
        return {"error": f"No forecast available for {days_ahead} days ahead"}
    
    # 天気を分類
    weather_condition = categorize_weather(target_forecast)
    
    # 結果を整形
    result = {
        "city": city,
        "date": target_date_str,
        "weather": {
            "main": target_forecast["weather"][0]["main"],
            "description": target_forecast["weather"][0]["description"],
            "temperature": target_forecast["main"]["temp"],
            "temp_min": target_forecast["main"]["temp_min"],
            "temp_max": target_forecast["main"]["temp_max"],
            "humidity": target_forecast["main"]["humidity"],
        },
        "condition": weather_condition.value,
        "language": language.value,
    }
    
    return result

async def suggest_videos(weather_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    天気情報に基づいてYouTube動画を提案
    
    Args:
        weather_info: get_weather_forecastの返り値
    
    Returns:
        おすすめ動画のリスト
    """
    if not YOUTUBE_API_KEY:
        return {"error": "YouTube API key not configured"}
    
    if "error" in weather_info:
        return {"error": f"Invalid weather info: {weather_info['error']}"}
    
    # 天気条件と言語を取得
    condition = WeatherCondition(weather_info.get("condition", "perfect"))
    language = Language(weather_info.get("language", "en"))
    
    # 適切な検索クエリを選択
    queries = SEARCH_QUERIES.get(condition, {}).get(language, SEARCH_QUERIES[condition][Language.EN])
    
    # YouTube APIで動画を検索
    videos = []
    async with httpx.AsyncClient() as client:
        for query in queries[:2]:  # 最初の2つのクエリを使用
            params = {
                "part": "snippet",
                "q": query,
                "key": YOUTUBE_API_KEY,
                "type": "video",
                "maxResults": 3,
                "regionCode": language.value[:2].upper() if language != Language.EN else "US",
            }
            
            try:
                response = await client.get(YOUTUBE_BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                for item in data.get("items", []):
                    video = {
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"]["channelTitle"],
                        "description": item["snippet"]["description"][:200] + "...",
                        "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                        "search_query": query,
                    }
                    videos.append(video)
                    
                    if len(videos) >= 5:  # 5本集まったら終了
                        break
            except httpx.HTTPError :
                continue
            
            if len(videos) >= 5:
                break
    
    # 結果にコンテキストを追加
    result = {
        "weather_summary": f"{weather_info['city']} - {weather_info['date']}: {weather_info['weather']['description']} ({weather_info['weather']['temperature']}°C)",
        "suggestion_reason": _get_suggestion_reason(condition, language),
        "videos": videos[:5],  # 最大5本
    }
    
    return result


def _get_suggestion_reason(condition: WeatherCondition, language: Language) -> str:
    """提案理由を生成"""
    reasons = {
        WeatherCondition.PERFECT: {
            Language.JA: "素晴らしい天気です！外出を楽しむのに最適な日です。",
            Language.EN: "Perfect weather! It's a great day to enjoy outdoor activities.",
        },
        WeatherCondition.HOT: {
            Language.JA: "とても暑い日になりそうです。涼しい室内で快適に過ごしましょう。",
            Language.EN: "It's going to be very hot. Stay cool and comfortable indoors.",
        },
        WeatherCondition.COLD: {
            Language.JA: "寒い日になりそうです。温かい室内で心地よく過ごしましょう。",
            Language.EN: "It's going to be cold. Stay warm and cozy indoors.",
        },
        WeatherCondition.RAINY: {
            Language.JA: "雨の日は室内でゆったりと。読書や音楽を楽しむのに最適です。",
            Language.EN: "A rainy day is perfect for indoor activities. Enjoy some music or a good book.",
        },
        WeatherCondition.STORMY: {
            Language.JA: "荒天が予想されます。安全を第一に、リラックスして過ごしましょう。",
            Language.EN: "Stormy weather expected. Stay safe and relaxed indoors.",
        },
    }
    
    return reasons.get(condition, {}).get(language, reasons[condition][Language.EN])


@mcp.tool()
async def get_activity_suggestion(city: str, days_ahead: int = 0) -> Dict[str, Any]:
    """
    都市と日付を指定して、天気に基づいた過ごし方を提案
    
    Args:
        city: 都市名
        days_ahead: 何日後か（0-5）
    
    Returns:
        天気情報と動画提案を含む完全な結果
    """
    # Step 1: 天気情報を取得
    weather_info = await get_weather_forecast(city, days_ahead)
    
    if "error" in weather_info:
        return weather_info
    
    # Step 2: 動画を提案
    video_suggestions = await suggest_videos(weather_info)
    
    # 結果を統合
    return {
        "weather": weather_info,
        "suggestions": video_suggestions,
    }


if __name__ == "__main__":
    # サーバーを起動
    mcp.run(transport="stdio")