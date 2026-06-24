from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from harness.agent import HarnessAgentConfig
from harness.run import run_browser_harness
from harness.runtime import LoopConfig, RetryPolicy


def parse_args() -> argparse.Namespace:
    """CLI 引数を定義して読み取る。

    このファイルは「設定を受け取る入口」です。
    実際の Agent 作成は agent.py、ループ実行は run.py に分けています。
    """

    parser = argparse.ArgumentParser(
        description="Run a long-running Harness browser agent with Foundry Toolbox MCP."
    )
    parser.add_argument(
        "task",
        nargs="+",
        help="Browser task. Include the approved target domain in the instruction.",
    )
    parser.add_argument(
        "--allowed-domain",
        action="append",
        default=[],
        help="Approved domain. Repeat this option for multiple domains.",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=30)
    parser.add_argument("--cycle-timeout-seconds", type=int, default=900)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-initial-delay-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-delay-seconds", type=float, default=30.0)
    parser.add_argument("--retry-backoff-multiplier", type=float, default=2.0)
    parser.add_argument("--retry-jitter-seconds", type=float, default=1.0)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--stall-limit", type=int, default=3)
    parser.add_argument("--consecutive-error-limit", type=int, default=3)
    parser.add_argument(
        "--unattended",
        action="store_true",
        help="Keep recovering from BLOCKED/stalled cycles until limits are reached.",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=None,
        help="Maximum wall-clock runtime. Omit for no runtime cap.",
    )
    parser.add_argument("--max-context-window-tokens", type=int, default=128_000)
    parser.add_argument("--max-output-tokens", type=int, default=8_000)
    parser.add_argument(
        "--history-dir", type=Path, default=Path("agent-framework/harness/runs/history")
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("agent-framework/harness/runs")
    )
    parser.add_argument("--enable-judge-loop", action="store_true")
    parser.add_argument("--judge-iterations", type=int, default=2)
    parser.add_argument(
        "--require-tool-approval",
        action="store_true",
        help="Require Harness tool approval middleware. This can pause high-frequency browser loops.",
    )
    return parser.parse_args()


def normalize_domains(domains: list[str]) -> tuple[str, ...]:
    """許可ドメインを比較しやすい形に正規化する。

    例:
    - https://example.com/ -> example.com
    - HTTP://Example.COM -> example.com
    """

    normalized = tuple(
        sorted(
            {
                domain.strip()
                .lower()
                .removeprefix("https://")
                .removeprefix("http://")
                .strip("/")
                for domain in domains
                if domain.strip()
            }
        )
    )
    if not normalized:
        raise ValueError("At least one --allowed-domain is required.")
    return normalized


def build_configs(
    args: argparse.Namespace,
) -> tuple[str, HarnessAgentConfig, LoopConfig]:
    """CLI 引数から Agent 設定と Loop 設定を組み立てる。"""

    task = " ".join(args.task)
    allowed_domains = normalize_domains(args.allowed_domain)
    session_id = args.session_id or datetime.now(UTC).strftime("browser-%Y%m%dT%H%M%SZ")
    run_prefix = args.run_dir / session_id

    agent_config = HarnessAgentConfig.from_env(
        allowed_domains=allowed_domains,
        history_dir=args.history_dir,
        max_context_window_tokens=args.max_context_window_tokens,
        max_output_tokens=args.max_output_tokens,
        enable_judge_loop=args.enable_judge_loop,
        judge_iterations=args.judge_iterations,
        require_tool_approval=args.require_tool_approval,
        unattended=args.unattended,
    )
    loop_config = LoopConfig(
        max_cycles=args.max_cycles,
        cycle_timeout_seconds=args.cycle_timeout_seconds,
        retry_policy=RetryPolicy(
            max_attempts=args.retry_attempts,
            initial_delay_seconds=args.retry_initial_delay_seconds,
            max_delay_seconds=args.retry_max_delay_seconds,
            backoff_multiplier=args.retry_backoff_multiplier,
            jitter_seconds=args.retry_jitter_seconds,
        ),
        sleep_seconds=args.sleep_seconds,
        stall_limit=args.stall_limit,
        consecutive_error_limit=args.consecutive_error_limit,
        log_file=run_prefix.with_suffix(".jsonl"),
        checkpoint_file=run_prefix.with_suffix(".checkpoint.json"),
        heartbeat_file=run_prefix.with_suffix(".heartbeat.json"),
        session_id=session_id,
        resume=args.resume,
        unattended=args.unattended,
        max_runtime_seconds=(
            int(args.max_runtime_hours * 3600)
            if args.max_runtime_hours is not None
            else None
        ),
    )
    return task, agent_config, loop_config


async def async_main() -> int:
    """非同期メイン処理。

    設定を作り、長時間実行ループへ処理を渡します。
    `.env` は pydantic-settings の HarnessSettings が読み込みます。
    """

    args = parse_args()
    task, agent_config, loop_config = build_configs(args)
    return await run_browser_harness(
        task=task,
        agent_config=agent_config,
        loop_config=loop_config,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(async_main()))
    except ValueError as exc:
        # CLI 引数やタスク範囲の設定ミスです。
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(64) from None
    except RuntimeError as exc:
        # 必須環境変数不足など、起動環境の設定ミスです。
        print(f"Runtime configuration error: {exc}", file=sys.stderr)
        raise SystemExit(78) from None
    except KeyboardInterrupt:
        # Ctrl+C で止めた場合は一般的な終了コード 130 を返します。
        raise SystemExit(130) from None
