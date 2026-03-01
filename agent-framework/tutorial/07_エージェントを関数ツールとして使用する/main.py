import anyio
from typing import Annotated
from pydantic import Field
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

# 1-1. 関数の作成
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    return f"The weather in {location} is cloudy with a high of 15°C."

# 1-2. 関数ツールとして使用する サブエージェント の作成
weather_agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
    name="WeatherAgent",
    description="An agent that answers questions about the weather.",
    instructions="You answer questions about the weather.",
    tools=get_weather
)

# 2. メインエージェントを作成
main_agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
    instructions="You are a helpful assistant who responds in French.",
    tools=weather_agent.as_tool( # 関数ツールに変換
        name="WeatherLookup", # ツール名のカスタマイズ
        description="Look up weather information for any location",
        arg_name="query",  # 引数名のカスタマイズ
        arg_description="The weather query or location" # 
    ) 
)

# 3. エージェント実行
async def main():
    result = await main_agent.run("What is the weather like in Amsterdam?")
    print(result.text)
    

if __name__ == "__main__":
    anyio.run(main)