import asyncio
import sys
from collections.abc import Sequence

from app import DemoSessionRuntime
from bootstrap import CLIBootstrap
from observability import setup_observability
from ui.cli import DemoChatCLI, ModelSwitchResult
from ui.web import run_chat_ui


async def run_cli() -> None:
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
            tool_provider=switched.app.progressive_tools,
            all_tools_provider=switched.app.all_tools,
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
        tool_provider=runtime.app.progressive_tools,
        all_tools_provider=runtime.app.all_tools,
        model_switcher=switch_model,
        pending_tool_approval_context=(
            await runtime.app.get_pending_tool_approval_context(runtime.session_id)
            if bootstrap_result.resumed_history
            else None
        ),
    )
    await cli.run()


def main(argv: Sequence[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    mode = args[0] if args else "cli"
    if mode == "cli":
        asyncio.run(run_cli())
        return
    if mode == "web":
        run_chat_ui()
        return
    raise SystemExit(f"Unsupported mode: {mode}. Use 'cli' or 'web'.")


if __name__ == "__main__":
    main()
