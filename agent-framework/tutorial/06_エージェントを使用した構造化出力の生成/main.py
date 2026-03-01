import asyncio
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from agent_framework import AgentRunResponse
from pydantic import BaseModel


# 1. 出力の構造を表す Pydantic モデル
class PersonInfo(BaseModel):
    """Information about a person."""
    name: str | None = None
    age: int | None = None
    occupation: str | None = None

# 2. エージェントに作成する
agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
    name="HelpfulAssistant",
    instructions="You are a helpful assistant that extracts person information from text."
)

async def main():
    # 3. response_format で構造化出力形式を指定し、エージェントを実行する
    response = await agent.run(
        "Please provide information about John Smith, who is a 35-year-old software engineer.",
        response_format=PersonInfo
    )
    if response.value:
        person_info = response.value
        print(f"Name: {person_info.name}, Age: {person_info.age}, Occupation: {person_info.occupation}")
    else:
        print("No structured data found in response")


async def stream():
    final_response = await AgentRunResponse.from_agent_response_generator(
        agent.run_stream(
            "Please provide information about John Smith, who is a 35-year-old software engineer.",
            response_format=PersonInfo
        ),
        output_format_type=PersonInfo,
    )

    if final_response.value:
        person_info = final_response.value
        print(f"Name: {person_info.name}, Age: {person_info.age}, Occupation: {person_info.occupation}")

asyncio.run(main())
asyncio.run(stream())