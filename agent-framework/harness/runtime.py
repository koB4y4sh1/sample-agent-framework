from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from harness.agent import HarnessAgentConfig


LoopStatus = Literal["CONTINUE", "COMPLETE", "NEEDS_USER", "BLOCKED"]

# Agent に毎サイクル渡すテンプレートです。
# 1回の agent.run() に全部任せず、「短い作業を何度も回す」ための指示です。
LOOP_PROMPT = """
Original task:
{task}

Allowed domains:
{allowed_domains}

Cycle {cycle} of {max_cycles}.

Continue from the current session state and the previous checkpoint summary.
Do only the next bounded unit of browser work needed to make progress.
Use the browser tool when page state must be inspected or changed.
If the previous checkpoint was BLOCKED or made no progress, diagnose the cause,
try one different recovery strategy, and continue unless user input is required.

Previous checkpoint:
{checkpoint_summary}

At the end of your response, include exactly one status block:

LOOP_STATUS: one of CONTINUE, COMPLETE, NEEDS_USER, BLOCKED
LOOP_SUMMARY: concise summary of what changed or what was learned in this cycle
LOOP_NEXT: the next concrete action, or NONE
LOOP_EVIDENCE: source URL, page title, selector, visible text, or error that supports the status

Status rules:
- COMPLETE only when the original task is fully satisfied.
- CONTINUE when more autonomous browser work remains.
- NEEDS_USER when login, MFA, explicit approval, or missing user input is required.
- BLOCKED when the browser tool, environment, or target site prevents progress.
""".strip()

STATUS_PATTERN = re.compile(
    r"^LOOP_STATUS:\s*(CONTINUE|COMPLETE|NEEDS_USER|BLOCKED)\s*$",
    re.MULTILINE,
)
FIELD_PATTERN = re.compile(
    r"^(LOOP_SUMMARY|LOOP_NEXT|LOOP_EVIDENCE):\s*(.*)$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """一時失敗を何回・何秒間隔で再試行するかの設定。"""

    max_attempts: int
    initial_delay_seconds: float
    max_delay_seconds: float
    backoff_multiplier: float
    jitter_seconds: float


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """長時間実行ループ全体の設定。"""

    max_cycles: int
    cycle_timeout_seconds: int
    retry_policy: RetryPolicy
    sleep_seconds: float
    stall_limit: int
    consecutive_error_limit: int
    log_file: Path
    checkpoint_file: Path
    heartbeat_file: Path
    session_id: str
    resume: bool
    unattended: bool
    max_runtime_seconds: int | None


@dataclass(frozen=True, slots=True)
class CycleResult:
    """1サイクルの実行結果。"""

    cycle: int
    attempt: int
    status: LoopStatus
    text: str
    summary: str
    next_action: str
    evidence: str


def validate_loop_config(config: LoopConfig) -> None:
    """明らかに危険な設定を起動前に弾く。"""

    if config.max_cycles <= 0:
        raise ValueError("--max-cycles must be greater than 0")
    if config.cycle_timeout_seconds <= 0:
        raise ValueError("--cycle-timeout-seconds must be greater than 0")
    if config.retry_policy.max_attempts <= 0:
        raise ValueError("--retry-attempts must be greater than 0")
    if config.stall_limit <= 0:
        raise ValueError("--stall-limit must be greater than 0")
    if config.consecutive_error_limit <= 0:
        raise ValueError("--consecutive-error-limit must be greater than 0")


def load_checkpoint(path: Path) -> dict[str, object]:
    """前回実行の最新チェックポイントを読む。"""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as checkpoint:
        data = json.load(checkpoint)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return data


def checkpoint_summary(checkpoint: dict[str, object]) -> str:
    """チェックポイントを Agent に渡しやすい短い JSON 文字列にする。"""

    if not checkpoint:
        return "NONE"
    values = {
        "last_cycle": checkpoint.get("cycle"),
        "last_status": checkpoint.get("status"),
        "last_summary": checkpoint.get("summary"),
        "last_next": checkpoint.get("next_action"),
        "last_evidence": checkpoint.get("evidence"),
    }
    return json.dumps(values, ensure_ascii=False)


def checkpoint_cycle(checkpoint: dict[str, object]) -> int:
    """チェックポイントから最後に完了した cycle 番号を安全に取り出す。"""

    value = checkpoint.get("cycle", 0)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0


def parse_cycle_result(*, cycle: int, attempt: int, text: str) -> CycleResult:
    """Agent の回答から LOOP_STATUS などの運用用フィールドを取り出す。"""

    status_match = STATUS_PATTERN.search(text)
    fields = dict(FIELD_PATTERN.findall(text))
    if status_match is None:
        return CycleResult(
            cycle=cycle,
            attempt=attempt,
            status="BLOCKED",
            text=text,
            summary="Missing LOOP_STATUS block.",
            next_action="NONE",
            evidence="The agent response did not include the required status block.",
        )
    return CycleResult(
        cycle=cycle,
        attempt=attempt,
        status=status_match.group(1),  # type: ignore[arg-type]
        text=text,
        summary=fields.get("LOOP_SUMMARY", "").strip() or "No summary provided.",
        next_action=fields.get("LOOP_NEXT", "").strip() or "NONE",
        evidence=fields.get("LOOP_EVIDENCE", "").strip() or "No evidence provided.",
    )


def error_result(
    cycle: int, attempt: int, summary: str, exc: BaseException
) -> CycleResult:
    """例外を CycleResult に変換する。"""

    return CycleResult(
        cycle=cycle,
        attempt=attempt,
        status="BLOCKED",
        text=(
            "LOOP_STATUS: BLOCKED\n"
            f"LOOP_SUMMARY: {summary}\n"
            "LOOP_NEXT: NONE\n"
            f"LOOP_EVIDENCE: {type(exc).__name__}: {exc}"
        ),
        summary=summary,
        next_action="NONE",
        evidence=f"{type(exc).__name__}: {exc}",
    )


def response_text(result: object) -> str:
    """AgentResponse から text を取り出す。なければ文字列化する。"""

    text = getattr(result, "text", None)
    return text if isinstance(text, str) else str(result)


def retry_delay(policy: RetryPolicy, attempt: int) -> float:
    """再試行までの待ち時間を計算する。"""

    base = min(
        policy.max_delay_seconds,
        policy.initial_delay_seconds
        * (policy.backoff_multiplier ** max(0, attempt - 1)),
    )
    return base + random.uniform(0, policy.jitter_seconds)


def runtime_exceeded(started_at: float, max_runtime_seconds: int | None) -> bool:
    """最大実行時間を超えたか判定する。None の場合は時間上限なし。"""

    return (
        max_runtime_seconds is not None
        and time.monotonic() - started_at >= max_runtime_seconds
    )


def write_checkpoint(path: Path, result: CycleResult) -> dict[str, object]:
    """最新サイクルの状態を checkpoint JSON として保存する。"""

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "cycle": result.cycle,
        "attempt": result.attempt,
        "status": result.status,
        "summary": result.summary,
        "next_action": result.next_action,
        "evidence": result.evidence,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return payload


def write_heartbeat(path: Path, result: CycleResult, session_id: str) -> None:
    """外部監視用の heartbeat ファイルを書く。"""

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "cycle": result.cycle,
        "status": result.status,
        "summary": result.summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(path: Path, event: dict[str, object]) -> None:
    """JSONL 形式で監査ログを追記する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(UTC).isoformat(), **event}
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(payload, ensure_ascii=False) + "\n")


def safe_agent_config(config: HarnessAgentConfig) -> dict[str, object]:
    """ログに出してよい形に Agent 設定を変換する。秘密値は隠す。"""

    return {
        **asdict(config),
        "history_dir": str(config.history_dir),
    }


def safe_loop_config(config: LoopConfig) -> dict[str, object]:
    """ログに出しやすい形に Loop 設定を変換する。Path は文字列化する。"""

    return {
        **asdict(config),
        "log_file": str(config.log_file),
        "checkpoint_file": str(config.checkpoint_file),
        "heartbeat_file": str(config.heartbeat_file),
    }


def print_cycle(result: CycleResult, *, stalled_cycles: int) -> None:
    """人間がコンソールで見るためのサイクル結果を表示する。"""

    print(
        f"\n=== Cycle {result.cycle} attempt {result.attempt}: {result.status} ===\n"
        f"Summary: {result.summary}\n"
        f"Next: {result.next_action}\n"
        f"Evidence: {result.evidence}\n"
        f"Stalled cycles: {stalled_cycles}\n",
        flush=True,
    )
