from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_framework import Content, Message

APPROVAL_REQUEST_TYPE = "function_approval_request"
APPROVAL_PROMPT_TEXT = (
    "[Approve] {tool_name} \u3092\u5b9f\u884c\u3057\u3066\u3082"
    "\u3088\u308d\u3057\u3044\u3067\u3057\u3087\u3046\u304b\uff1f"
    "{arguments} Y/N: "
)


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
    requests: Sequence[Content], approvals: Sequence[bool]
) -> Message:
    if len(requests) != len(approvals):
        raise ValueError("requests and approvals must have the same length.")

    return Message(
        role="user",
        contents=[
            request.to_function_approval_response(approved)
            for request, approved in zip(requests, approvals, strict=True)
        ],
    )


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
