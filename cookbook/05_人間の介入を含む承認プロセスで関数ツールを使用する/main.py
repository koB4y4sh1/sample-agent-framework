import asyncio
from random import randrange
from typing import Annotated

from agent_framework import ChatAgent, ChatMessage, Role, ai_function
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
conditions = ["sunny", "cloudy", "raining", "snowing", "clear"]

# 関数ツールの作成
# 1-1. 承認を必要としない単純な関数ツールを作成する
@ai_function
def get_weather(location: Annotated[str, "The city and state, e.g. San Francisco, CA"]) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is cloudy with a high of 15°C."

# 1-2. 承認を必要とする関数を作成する
# agent-framework==1.0.0b251001 の場合はapproval_modeに対応していなく、エラーになるので注意
@ai_function(approval_mode="always_require")
def get_weather_detail(location: Annotated[str, "The city and state, e.g. San Francisco, CA"]) -> str:
    """Get the current weather for a given location."""
    # Simulate weather data
    return (
        f"The weather in {location} is {conditions[randrange(0, len(conditions))]} and {randrange(-10, 30)}°C, "
        "with a humidity of 88%. "
        f"Tomorrow will be {conditions[randrange(0, len(conditions))]} with a high of {randrange(-10, 30)}°C."
    )


    # 2. 関数ツールをエージェントに提供する
agent = ChatAgent(
        chat_client=AzureOpenAIResponsesClient(credential=AzureCliCredential()),
        name="WeatherAgent",
        instructions="You are a helpful weather assistant.",
        tools=[get_weather,get_weather_detail],
    ) 


async def main():
        # 3. エージェントを実行する
        result = await agent.run("What is the detailed weather like in Amsterdam?")
        print(f"Agent: {result.text}")

        # 4. 承認ステップ
        # 4-1. エージェントからユーザーへの承認要求を確認する
        if result.user_input_requests:
            for user_input_needed in result.user_input_requests:
                print(f"Function: {user_input_needed.function_call.name}")
                print(f"Arguments: {user_input_needed.function_call.arguments}")

            # 4-2. 実行を承認する
            user_approval = True  # or False to reject
            print("User: Approve")

            # 4-3. 承認応答を ChatMessage でエージェントに渡す
            approval_message = ChatMessage(
                role=Role.USER, 
                contents=[user_input_needed.create_response(user_approval)]
            )
            final_result = await agent.run([
                "What is the detailed weather like in Amsterdam?",
                ChatMessage(role=Role.ASSISTANT, contents=[user_input_needed]),
                approval_message
            ])

            # 4-4. 承認した結果をエージェントから取得する
            print(f"Agent: {final_result.text}")


async def loop_approve():
    async def handle_approvals(query: str) -> str:
        """Handle function call approvals in a loop."""
        current_input = query

        while True:
            # 3. エージェントを実行する
            result = await agent.run(current_input)

            # 4.承認要求の確認
            if not result.user_input_requests:
                # 承認が必要ない場合は最終結果を返却
                return result.text

            # Build new input with all context
            new_inputs = [query]

            # 5. 承認ステップ
            for user_input_needed in result.user_input_requests:
                print(f"Approval needed for: {user_input_needed.function_call.name}")
                print(f"Arguments: {user_input_needed.function_call.arguments}")

                # 5-1. エージェントの承認要求を Chatmessage に登録
                new_inputs.append(ChatMessage(role=Role.ASSISTANT, contents=[user_input_needed]))

                # 5-2. 実行を承認する
                user_approval = True  # Replace with actual user input

                # 5-3. ユーザーの承認を Chatmessage に登録
                new_inputs.append(
                    ChatMessage(role=Role.USER, contents=[user_input_needed.create_response(user_approval)])
                )

            # ChatMessage に追加した情報を含めて再度エージェント実行
            current_input = new_inputs

    # Usage
    result_text = await handle_approvals("Get detailed weather for Seattle and Portland")
    print(f"Agent: {result_text}")


asyncio.run(main())
asyncio.run(loop_approve())
