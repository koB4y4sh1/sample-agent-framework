from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework import AgentLoopMiddleware


@dataclass(frozen=True, slots=True)
class TaskCompletionLoopConfig:
    """AgentLoopMiddleware による自己改善ループの設定。

    想定ユースケース:
    - 調査、設計、要約、複数ステップの作業で、1回の回答では情報不足や未完了が残りやすい。
    - 軽量judge modelで「ユーザー要求に答え切れているか」を判定し、不十分な場合だけ再実行したい。
    - 反復上限を必ず設け、品質改善とコスト/レイテンシの上限を両立したい。

    単純な雑談や短いQ&Aへ常時適用すると遅くなるため、設定で有効/無効を切り替える。
    """

    enabled: bool = True
    max_iterations: int = 2
    criteria: tuple[str, ...] = (
        "The answer directly satisfies the user's latest request.",
        "The answer does not leave unresolved TODOs or ask the user to do work the agent can do.",
        "Tool results and gathered facts are reflected in the final answer.",
    )


def build_task_completion_loop(
    judge_client: Any,
    config: TaskCompletionLoopConfig | None = None,
) -> AgentLoopMiddleware:
    """軽量judgeで回答完了性を評価し、不十分な場合だけAgentを再実行する。

    想定ユースケース:
    - 「調べてまとめて」「設計して」「複数観点で確認して」のような、1回の生成で抜け漏れが出やすい依頼。
    - 回答品質は上げたいが、無制限ループや重いjudgeは避けたい。

    `AgentLoopMiddleware.with_judge` を使うため、ループ制御自体はMAFに委譲する。
    """

    resolved = config or TaskCompletionLoopConfig()
    return AgentLoopMiddleware.with_judge(
        judge_client,
        criteria=resolved.criteria,
        max_iterations=resolved.max_iterations,
        fresh_context=False,
    )
