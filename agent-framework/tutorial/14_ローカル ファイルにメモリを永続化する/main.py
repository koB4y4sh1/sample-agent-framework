import asyncio
from pathlib import Path

from agent_framework import Agent
from agent_framework.foundry import AnthropicFoundryClient

from providers import LocalFileConversationMemoryProvider, LocalFileHistoryProvider

MEMORY_DIR = Path(__file__).parent / ".memory"


async def run_with_context_provider() -> None:
    agent = Agent(
        client=AnthropicFoundryClient(model="claude-haiku-4-5"),
        name="ContextMemoryAgent",
        instructions="You are a helpful assistant.",
        context_providers=[
            LocalFileConversationMemoryProvider(
                root_dir=MEMORY_DIR,
                max_messages=10,
            )
        ],
    )

    session = agent.create_session(session_id="local-context-demo")
    response = await agent.run("前回の会話を踏まえて短く要約してください。", session=session)
    print(response.text)


async def run_with_history_provider() -> None:
    agent = Agent(
        client=AnthropicFoundryClient(model="claude-haiku-4-5"),
        name="HistoryMemoryAgent",
        instructions="You are a helpful assistant.",
        context_providers=[
            LocalFileHistoryProvider(
                root_dir=MEMORY_DIR,
                max_messages=10,
            )
        ],
    )

    session = agent.create_session(session_id="local-history-demo")
    response = await agent.run("保存済みの会話履歴を考慮して回答してください。", session=session)
    print(response.text)


async def main() -> None:
    await run_with_context_provider()
    await run_with_history_provider()


if __name__ == "__main__":
    asyncio.run(main())
