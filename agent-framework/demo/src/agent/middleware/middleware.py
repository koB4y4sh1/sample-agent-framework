from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework import ToolApprovalMiddleware

from .auto_tool_approval import (
    AutoToolApprovalConfig,
    build_auto_tool_approval_rule,
)
from .task_completion_loop import TaskCompletionLoopConfig, build_task_completion_loop


@dataclass(frozen=True, slots=True)
class AgentMiddlewareConfig:
    """Agentで使うTool承認と自己改善ループのmiddleware設定。

    想定ユースケース:
    - 許可済みTool実行は `.memory/{user_name}/allow_tools.json` に基づいて自動承認したい。
    - 1回の応答で不足がある場合だけ、judgeによりAgentを追加実行したい。
    """

    auto_tool_approval: AutoToolApprovalConfig = AutoToolApprovalConfig()
    task_completion_loop: TaskCompletionLoopConfig = TaskCompletionLoopConfig()


def build_middleware(
    *,
    judge_client: Any,
    config: AgentMiddlewareConfig | None = None,
) -> list[Any]:
    """MAFのmiddleware配列を構築する。

    順序は ToolApprovalMiddleware、TaskCompletionLoop の順にする。
    Tool実行可否を先に確定し、その後に必要な場合だけ自己改善ループを回すため。
    """

    resolved = config or AgentMiddlewareConfig()
    middleware: list[Any] = [
        ToolApprovalMiddleware(
            auto_approval_rules=[
                build_auto_tool_approval_rule(resolved.auto_tool_approval)
            ]
        ),
    ]
    if resolved.task_completion_loop.enabled:
        middleware.append(build_task_completion_loop(judge_client, resolved.task_completion_loop))
    return middleware
