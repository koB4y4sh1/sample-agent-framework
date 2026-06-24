from __future__ import annotations

import asyncio
import time

from harness.agent import HarnessAgentConfig, build_harness_agent
from harness.runtime import (
    LOOP_PROMPT,
    CycleResult,
    LoopConfig,
    append_log,
    checkpoint_cycle,
    checkpoint_summary,
    error_result,
    load_checkpoint,
    parse_cycle_result,
    print_cycle,
    response_text,
    retry_delay,
    runtime_exceeded,
    safe_agent_config,
    safe_loop_config,
    validate_loop_config,
    write_checkpoint,
    write_heartbeat,
)


async def run_browser_harness(
    *,
    task: str,
    agent_config: HarnessAgentConfig,
    loop_config: LoopConfig,
) -> int:
    """ブラウザ Harness Agent を長時間ループで実行する本体。"""

    validate_loop_config(loop_config)
    runtime = await build_harness_agent(agent_config)
    try:
        agent = runtime.agent
        session = agent.create_session(session_id=loop_config.session_id)
        checkpoint = (
            load_checkpoint(loop_config.checkpoint_file) if loop_config.resume else {}
        )

        append_log(
            loop_config.log_file,
            {
                "type": "start",
                "task": task,
                "session_id": loop_config.session_id,
                "resume": loop_config.resume,
                "agent_config": safe_agent_config(agent_config),
                "loop_config": safe_loop_config(loop_config),
                "available_tools": list(runtime.available_tools),
                "source_docs": [
                    "https://learn.microsoft.com/ja-jp/agent-framework/agents/?pivots=programming-language-python"
                ],
            },
        )

        last_progress_key = ""
        stalled_cycles = 0
        consecutive_errors = 0
        started_at = time.monotonic()

        for cycle in range(
            checkpoint_cycle(checkpoint) + 1, loop_config.max_cycles + 1
        ):
            if runtime_exceeded(started_at, loop_config.max_runtime_seconds):
                append_log(loop_config.log_file, {"type": "max_runtime", "cycle": cycle})
                print("Stopped: max runtime reached.")
                return 7

            prompt = LOOP_PROMPT.format(
                task=task,
                allowed_domains="\n".join(
                    f"- {domain}" for domain in agent_config.allowed_domains
                ),
                cycle=cycle,
                max_cycles=loop_config.max_cycles,
                checkpoint_summary=checkpoint_summary(checkpoint),
            )

            result = await run_cycle_with_retry(
                agent=agent,
                session=session,
                prompt=prompt,
                cycle=cycle,
                loop_config=loop_config,
            )

            progress_key = f"{result.summary}\n{result.next_action}\n{result.evidence}"
            stalled_cycles = (
                stalled_cycles + 1 if progress_key == last_progress_key else 0
            )
            last_progress_key = progress_key
            consecutive_errors = (
                consecutive_errors + 1 if result.status == "BLOCKED" else 0
            )

            checkpoint = write_checkpoint(loop_config.checkpoint_file, result)
            write_heartbeat(loop_config.heartbeat_file, result, loop_config.session_id)
            append_log(
                loop_config.log_file,
                {
                    "type": "cycle",
                    "cycle": cycle,
                    "attempt": result.attempt,
                    "status": result.status,
                    "stalled_cycles": stalled_cycles,
                    "consecutive_errors": consecutive_errors,
                    "summary": result.summary,
                    "next_action": result.next_action,
                    "evidence": result.evidence,
                    "text": result.text,
                },
            )

            print_cycle(result, stalled_cycles=stalled_cycles)

            if result.status == "COMPLETE":
                append_log(loop_config.log_file, {"type": "complete", "cycle": cycle})
                return 0
            if result.status == "NEEDS_USER":
                append_log(loop_config.log_file, {"type": "needs_user", "cycle": cycle})
                return 2
            if consecutive_errors >= loop_config.consecutive_error_limit:
                append_log(
                    loop_config.log_file, {"type": "circuit_open", "cycle": cycle}
                )
                print(
                    f"Stopped: consecutive errors reached {loop_config.consecutive_error_limit}."
                )
                return 5
            if result.status == "BLOCKED":
                append_log(loop_config.log_file, {"type": "blocked", "cycle": cycle})
                if not loop_config.unattended:
                    return 3
                await asyncio.sleep(loop_config.sleep_seconds)
                continue
            if stalled_cycles >= loop_config.stall_limit:
                append_log(loop_config.log_file, {"type": "stalled", "cycle": cycle})
                if not loop_config.unattended:
                    print(
                        f"Stopped: no progress for {loop_config.stall_limit} cycles."
                    )
                    return 4
                stalled_cycles = 0

            await asyncio.sleep(loop_config.sleep_seconds)

        append_log(
            loop_config.log_file,
            {"type": "max_cycles", "cycles": loop_config.max_cycles},
        )
        print(f"Stopped: reached max_cycles={loop_config.max_cycles}.")
        return 6
    finally:
        await runtime.close()


async def run_cycle_with_retry(
    *,
    agent,
    session,
    prompt: str,
    cycle: int,
    loop_config: LoopConfig,
) -> CycleResult:
    """1サイクルを再試行付きで実行する。"""

    policy = loop_config.retry_policy
    last_result: CycleResult | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            text = await run_cycle(
                agent=agent,
                session=session,
                prompt=prompt,
                timeout_seconds=loop_config.cycle_timeout_seconds,
            )
            result = parse_cycle_result(cycle=cycle, attempt=attempt, text=text)
            if result.status != "BLOCKED" or attempt == policy.max_attempts:
                return result
            last_result = result
        except TimeoutError as exc:
            last_result = error_result(cycle, attempt, "Cycle timed out.", exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_result = error_result(cycle, attempt, "Cycle failed.", exc)

        append_log(
            loop_config.log_file,
            {
                "type": "retry",
                "cycle": cycle,
                "attempt": attempt,
                "status": last_result.status,
                "summary": last_result.summary,
                "evidence": last_result.evidence,
            },
        )
        if attempt < policy.max_attempts:
            await asyncio.sleep(retry_delay(policy, attempt))

    if last_result is None:
        raise RuntimeError("Retry loop ended without a result")
    return last_result


async def run_cycle(*, agent, session, prompt: str, timeout_seconds: int) -> str:
    """Agent を1回だけ実行し、タイムアウトをかけて文字列結果を返す。"""

    result = await asyncio.wait_for(
        agent.run(prompt, session=session),
        timeout=timeout_seconds,
    )
    return response_text(result)
