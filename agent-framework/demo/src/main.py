import asyncio

from app import DemoApplication, DemoConfig
from bootstrap import CLIBootstrap
from chat_cli import DemoChatCLI
from observability import setup_observability


async def main() -> None:
    bootstrap_result = CLIBootstrap().run()
    setup_observability()
    app = DemoApplication(
        config=DemoConfig(
            provider_family=bootstrap_result.model_settings.provider_family,
            model=bootstrap_result.model_settings.model_name,
        )
    )
    session = app.create_session(session_id=bootstrap_result.session_id)
    cli = DemoChatCLI(
        agent=app.agent,
        session=session,
        code_interpreter_status=app.skills.describe(),
        stream_renderer=app.stream_renderer,
        pending_tool_approval_context=(
            await app.get_pending_tool_approval_context(bootstrap_result.session_id)
            if bootstrap_result.resumed_history
            else None
        ),
    )
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
