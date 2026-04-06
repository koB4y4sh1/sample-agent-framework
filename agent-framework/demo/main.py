import asyncio

from app import DemoApplication
from chat_cli import DemoChatCLI


async def main() -> None:
    app = DemoApplication()
    session = app.create_session()
    cli = DemoChatCLI(
        agent=app.agent,
        session=session,
        code_interpreter_status=app.skills.describe(),
        stream_renderer=app.stream_renderer,
    )
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
