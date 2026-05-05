import asyncio

from app import DemoSessionRuntime
from bootstrap import CLIBootstrap
from chat_cli import DemoChatCLI, ModelSwitchResult
from observability import setup_observability


async def main() -> None:
    setup_observability()
    bootstrap_result = CLIBootstrap().run()
    runtime = DemoSessionRuntime.create(
        model_name=bootstrap_result.model_settings.model_name,
        session_id=bootstrap_result.session_id,
    )

    async def switch_model(model_name: str) -> ModelSwitchResult:
        switched = runtime.switch_model(model_name)
        return ModelSwitchResult(
            agent=switched.app.agent,
            session=switched.session,
            stream_renderer=switched.app.stream_renderer,
            model_name=switched.model_name,
            provider_family=switched.provider_family,
        )

    cli = DemoChatCLI(
        agent=runtime.app.agent,
        session=runtime.session,
        model_name=runtime.model_name,
        provider_family=runtime.provider_family,
        code_interpreter_status=runtime.app.skills.describe(),
        stream_renderer=runtime.app.stream_renderer,
        model_switcher=switch_model,
        pending_tool_approval_context=(
            await runtime.app.get_pending_tool_approval_context(runtime.session_id)
            if bootstrap_result.resumed_history
            else None
        ),
    )
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
