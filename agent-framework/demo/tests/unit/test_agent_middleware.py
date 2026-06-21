from __future__ import annotations

from agent.middleware import (
    AllowToolsStore,
    AutoToolApprovalConfig,
    build_auto_tool_approval_rule,
)
from agent_framework import Content


def test_auto_tool_approval_allows_persisted_tool_name(tmp_path) -> None:
    config = AutoToolApprovalConfig(user_name="alice", memory_root_dir=tmp_path)
    store = AllowToolsStore(config)
    function_call = Content.from_function_call(
        "call_1",
        "request_application_approval",
        arguments={"draft_id": "D-1"},
    )
    store.add_function_call(function_call, "tool")
    rule = build_auto_tool_approval_rule(config)

    assert rule(function_call) is True


def test_auto_tool_approval_requires_same_arguments_for_argument_scope(tmp_path) -> None:
    config = AutoToolApprovalConfig(user_name="alice", memory_root_dir=tmp_path)
    store = AllowToolsStore(config)
    allowed_call = Content.from_function_call(
        "call_1",
        "request_application_approval",
        arguments={"draft_id": "D-1"},
    )
    different_call = Content.from_function_call(
        "call_2",
        "request_application_approval",
        arguments={"draft_id": "D-2"},
    )
    store.add_function_call(allowed_call, "tool_with_arguments")
    rule = build_auto_tool_approval_rule(config)

    assert rule(allowed_call) is True
    assert rule(different_call) is False


def test_auto_tool_approval_keeps_unpersisted_tool_for_human_review(tmp_path) -> None:
    rule = build_auto_tool_approval_rule(
        AutoToolApprovalConfig(user_name="alice", memory_root_dir=tmp_path)
    )
    function_call = Content.from_function_call(
        "call_1",
        "request_application_approval",
        arguments={"query": "policy"},
    )

    assert rule(function_call) is False
