from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agent.middleware import allow_tool_request
from agent_framework import (
    Content,
    Message,
    create_always_approve_tool_response,
    create_always_approve_tool_with_arguments_response,
)

APPROVAL_REQUEST_TYPE = "function_approval_request"
ApprovalDecision = Literal["deny", "once", "always_arguments", "always_tool"]
ApprovalInput = ApprovalDecision | bool

APPROVAL_DENY: ApprovalDecision = "deny"
APPROVAL_ONCE: ApprovalDecision = "once"
APPROVAL_ALWAYS_ARGUMENTS: ApprovalDecision = "always_arguments"
APPROVAL_ALWAYS_TOOL: ApprovalDecision = "always_tool"
APPROVAL_PROMPT_TEXT = (
    "[Approve] {tool_name} を実行してもよろしいでしょうか？"
    "{arguments}\n"
    "  0: 拒否\n"
    "  1: このリクエストのみ許可\n"
    "  2: 引数の実行を許可する\n"
    "  3: このツール実行を自動許可する\n"
    "選択 [0-3]: "
)
APPROVAL_ALLOW_OPTIONS: dict[str, str] = {
    APPROVAL_ONCE: "このリクエストのみ許可",
    APPROVAL_ALWAYS_ARGUMENTS: "引数の実行を許可する",
    APPROVAL_ALWAYS_TOOL: "このツール実行を自動許可する",
}

@dataclass(slots=True)
class ToolApprovalContext:
    assistant_messages: list[Message]
    requests: list[Content]


def pending_tool_approval_requests(messages: Sequence[Message]) -> list[Content]:
    return pending_tool_approval_context(messages).requests


def pending_tool_approval_context(messages: Sequence[Message]) -> ToolApprovalContext:
    """Return approval requests only when the latest message is awaiting approval."""
    if not messages:
        return ToolApprovalContext(assistant_messages=[], requests=[])

    last_message = messages[-1]
    requests = [
        content
        for content in last_message.contents
        if content.type == APPROVAL_REQUEST_TYPE
    ]
    if not requests:
        return ToolApprovalContext(assistant_messages=[], requests=[])

    return ToolApprovalContext(
        assistant_messages=[_copy_message(last_message)], requests=requests
    )


def build_tool_approval_response_message(
    requests: Sequence[Content], approvals: Sequence[ApprovalInput]
) -> Message:
    if len(requests) != len(approvals):
        raise ValueError("requests and approvals must have the same length.")

    return Message(
        role="user",
        contents=[
            build_tool_approval_response(request, approval)
            for request, approval in zip(requests, approvals, strict=True)
        ],
    )


def build_tool_approval_response(request: Content, approval: ApprovalInput) -> Content:
    decision = normalize_approval_decision(approval)
    if decision == APPROVAL_DENY:
        return request.to_function_approval_response(False)
    if decision == APPROVAL_ONCE:
        return request.to_function_approval_response(True)
    if decision == APPROVAL_ALWAYS_ARGUMENTS:
        allow_tool_request(request, "tool_with_arguments")
        return create_always_approve_tool_with_arguments_response(request)
    if decision == APPROVAL_ALWAYS_TOOL:
        allow_tool_request(request, "tool")
        return create_always_approve_tool_response(request)
    raise ValueError(f"Unsupported approval decision: {decision}")


def normalize_approval_decision(value: ApprovalInput) -> ApprovalDecision:
    if value is True:
        return APPROVAL_ONCE
    if value is False:
        return APPROVAL_DENY
    if value in {
        APPROVAL_DENY,
        APPROVAL_ONCE,
        APPROVAL_ALWAYS_ARGUMENTS,
        APPROVAL_ALWAYS_TOOL,
    }:
        return value
    raise ValueError(f"Unsupported approval decision: {value}")


def format_tool_approval_prompt(request: Content) -> str:
    return APPROVAL_PROMPT_TEXT.format(
        tool_name=tool_approval_name(request),
        arguments=tool_approval_arguments(request),
    )


def tool_approval_name(request: Content) -> str:
    function_call = request.function_call
    if function_call is None:
        return "(unknown)"
    return str(function_call.name or "(unknown)")


def tool_approval_arguments(request: Content) -> str:
    function_call = request.function_call
    if function_call is None:
        return "{}"
    return _format_arguments(function_call.arguments)


def _copy_message(message: Message) -> Message:
    return Message(
        role=message.role,
        contents=list(message.contents),
        author_name=message.author_name,
        message_id=message.message_id,
        additional_properties=dict(message.additional_properties),
    )


def _format_arguments(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        try:
            return json.dumps(
                json.loads(arguments), ensure_ascii=False, separators=(",", ":")
            )
        except json.JSONDecodeError:
            return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), default=str)
