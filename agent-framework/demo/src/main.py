import asyncio

from app import DemoApplication, DemoConfig
from bootstrap import CLIBootstrap
from chat_cli import DemoChatCLI


async def main() -> None:
    bootstrap_result = CLIBootstrap().run()
    app = DemoApplication(config=DemoConfig(model=bootstrap_result.model))
    session = app.create_session(session_id=bootstrap_result.session_id)
    cli = DemoChatCLI(
        agent=app.agent,
        session=session,
        code_interpreter_status=app.skills.describe(),
        stream_renderer=app.stream_renderer,
    )
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
