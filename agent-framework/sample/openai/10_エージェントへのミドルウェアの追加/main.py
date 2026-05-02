import asyncio
from typing import  Awaitable, Callable

from agent_framework import AgentRunContext
from agent_framework import FunctionInvocationContext
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential


# 1. 関数の作成する
def get_time():
    """Get the current time."""
    from datetime import datetime
    time = datetime.now().strftime("%H:%M:%S")
    print(f"Function Execute: {time}")
    return time

# 2. ミドルウェアの作成
# 1. エージェントミドルウェアの作成
async def logging_agent_middleware(
    context: AgentRunContext,
    next: Callable[[AgentRunContext], Awaitable[None]],
) -> None:
    """Simple middleware that logs agent execution."""
    # Pre-processing
    print("Agent starting...")

    await next(context) # execute agent

    # Post-processing
    print("Agent finished!")


# 3. 関数ミドルウェアの作成
async def logging_function_middleware(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    """Middleware that logs function calls."""
    print(f"Calling function: {context.function.name}")

    await next(context)

    print(f"Function result: {context.result}")


async def main():
    # 4. エージェントにミドルウェアを追加する
    async with AzureOpenAIResponsesClient(credential=AzureCliCredential()).create_agent(
        name="TimeAgent",
        instructions="You can tell the current time.",
        tools=[get_time],
        middleware=[logging_agent_middleware, logging_function_middleware],  # ミドルウェアを登録
    ) as agent:
        result = await agent.run("What time is it?")
        print(result.text)

asyncio.run(main())

