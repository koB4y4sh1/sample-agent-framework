import asyncio

from app import DemoApplication, DemoConfig
from chat_cli import DemoChatCLI


async def main() -> None:
    selected_model = DemoChatCLI.select_model()
    app = DemoApplication(config=DemoConfig(model=selected_model))
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
