import asyncio
from types import SimpleNamespace
from typing import Any, cast

from agent.contexts.user_profile import (
    UserProfile,
    UserProfileContextProvider,
    UserProfileUpdateDecision,
)
from agent_framework import Message, SessionContext


class _AnalyserClient:
    def __init__(self, decision: UserProfileUpdateDecision) -> None:
        self.decision = decision
        self.messages: list[Message] | None = None

    async def get_response(
        self,
        messages: list[Message],
        *,
        options: dict[str, Any],
    ) -> SimpleNamespace:
        self.messages = messages
        assert options["response_format"] is UserProfileUpdateDecision
        return SimpleNamespace(value=self.decision)


def _provider(
    tmp_path: Any,
    decision: UserProfileUpdateDecision,
) -> tuple[UserProfileContextProvider, _AnalyserClient]:
    analyser = _AnalyserClient(decision)
    UserProfileContextProvider.PROFILE_ROOT_DIR = tmp_path
    provider = UserProfileContextProvider(analyser_client=cast(Any, analyser))
    return provider, analyser


def test_after_run_uses_only_current_profile_and_user_question(tmp_path: Any) -> None:
    updated = UserProfile(
        summary="User is a software engineer",
        communication_preferences=["Concise technical answers"],
    )
    provider, analyser = _provider(
        tmp_path,
        UserProfileUpdateDecision(should_update=True, profile=updated),
    )
    current = UserProfile(summary="User is a software engineer")
    state: dict[str, Any] = {provider.PROFILE_STATE_KEY: current}
    context = SessionContext(
        input_messages=[
            Message("system", ["internal input"]),
            Message("user", ["Please use concise technical answers."]),
        ],
        context_messages={
            "history": [
                Message("user", ["This conversation history must not be sent."])
            ]
        },
    )

    asyncio.run(
        provider.after_run(
            agent=cast(Any, None),
            session=cast(Any, None),
            context=context,
            state=state,
        )
    )

    assert state[provider.PROFILE_STATE_KEY] == updated
    assert analyser.messages is not None
    request = analyser.messages[-1].text
    assert current.model_dump_json(indent=2) in request
    assert "Please use concise technical answers." in request
    assert "conversation history must not be sent" not in request


def test_after_run_keeps_profile_when_update_is_not_required(tmp_path: Any) -> None:
    provider, _ = _provider(
        tmp_path,
        UserProfileUpdateDecision(should_update=False),
    )
    current = UserProfile(summary="Existing profile")
    state: dict[str, Any] = {provider.PROFILE_STATE_KEY: current}
    context = SessionContext(input_messages=[Message("user", ["What time is it?"])])

    asyncio.run(
        provider.after_run(
            agent=cast(Any, None),
            session=cast(Any, None),
            context=context,
            state=state,
        )
    )

    assert state[provider.PROFILE_STATE_KEY] is current


def test_after_run_skips_analysis_without_user_text(tmp_path: Any) -> None:
    provider, analyser = _provider(
        tmp_path,
        UserProfileUpdateDecision(should_update=True, profile=UserProfile()),
    )
    state: dict[str, Any] = {provider.PROFILE_STATE_KEY: UserProfile()}
    context = SessionContext(input_messages=[Message("assistant", ["not a question"])])

    asyncio.run(
        provider.after_run(
            agent=cast(Any, None),
            session=cast(Any, None),
            context=context,
            state=state,
        )
    )

    assert analyser.messages is None
