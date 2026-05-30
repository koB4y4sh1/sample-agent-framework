from __future__ import annotations

import logging
from dataclasses import dataclass

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from .agent import REVIEW_ARTIFACTS, CodeReviewAgent

logger = logging.getLogger(__name__)

REVIEW_DRAFT_ARTIFACT_ID = "review-draft"
REVIEW_DRAFT_ARTIFACT_NAME = "Review Draft"


@dataclass(slots=True)
class CodeReviewRequest:
    """1 回の task 実行で使う値をまとめた入れ物です。"""

    task_id: str
    context_id: str
    user_input: str
    updater: TaskUpdater


async def prepare_review_request(
    context: RequestContext,
    event_queue: EventQueue,
) -> CodeReviewRequest | None:
    """A2A の入力を、このデモで扱いやすい形へ整えます。"""
    task_id = context.task_id or ""
    context_id = context.context_id or ""
    if not task_id or not context_id:
        return None

    await event_queue.enqueue_event(
        Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            history=[context.message] if context.message else [],
        )
    )
    user_input = context.get_user_input().strip()
    return CodeReviewRequest(
        task_id=task_id,
        context_id=context_id,
        user_input=user_input,
        updater=TaskUpdater(event_queue, task_id, context_id),
    )


def build_a2a_message(updater: TaskUpdater, text: str):
    """定型の A2A メッセージを短く書けるようにします。"""
    return updater.new_agent_message(parts=[Part(text=text)])


async def reject_if_empty(request: CodeReviewRequest) -> bool:
    """レビュー対象が空なら、ここで処理を終了します。"""
    if request.user_input:
        return False

    await request.updater.reject(
        message=build_a2a_message(
            request.updater,
            "レビュー対象が空です。PR 概要または差分を送ってください。",
        )
    )
    return True


class CodeReviewExecutor(AgentExecutor):
    """A2A の task を受け取り、レビュー処理へ流す実行クラスです。"""

    def __init__(self, agent: CodeReviewAgent) -> None:
        self._agent = agent
        # cancel_task で止められるように、実行中の task_id を記録します。
        self._running_task_ids: set[str] = set()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """レビュー依頼を受け取り、状態更新と artifact streaming で結果を返します。"""
        request = await prepare_review_request(context, event_queue)
        if request is None:
            return

        if await reject_if_empty(request):
            return

        await self._run_review(request)

    async def _run_review(self, request: CodeReviewRequest) -> None:
        """通常のレビュー依頼を LLM に渡して結果を返します。"""
        self._running_task_ids.add(request.task_id)
        try:
            has_review_history = self._agent.has_review_history(request.context_id)
            start_message = "前回の指摘を踏まえて再レビューしています。"
            if not has_review_history:
                start_message = "レビュー依頼を受け付けました。リスクを確認しています。"

            await request.updater.update_status(
                TaskState.TASK_STATE_WORKING,
                message=build_a2a_message(
                    request.updater,
                    start_message,
                ),
            )

            if request.task_id not in self._running_task_ids:
                return

            await request.updater.update_status(
                TaskState.TASK_STATE_WORKING,
                message=build_a2a_message(
                    request.updater,
                    "レビューの下書きを streaming しています。",
                ),
            )
            review_text = await self._stream_review_text(request)
            if request.task_id not in self._running_task_ids:
                return

            await request.updater.update_status(
                TaskState.TASK_STATE_WORKING,
                message=build_a2a_message(
                    request.updater,
                    "レビュー結果を成果物ごとに整理しています。",
                ),
            )
            artifacts = self._split_review_artifacts(review_text)
            for artifact_id, artifact_name in REVIEW_ARTIFACTS:
                artifact_text = artifacts.get(artifact_id, "")
                if not artifact_text:
                    continue
                await self._add_artifact_chunk(
                    updater=request.updater,
                    artifact_id=artifact_id,
                    name=artifact_name,
                    text=artifact_text,
                    append=False,
                    last_chunk=True,
                )
            await request.updater.complete()
        except Exception as error:
            logger.exception("OpenAI SDK によるコードレビュー実行に失敗しました")
            await request.updater.failed(
                message=build_a2a_message(
                    request.updater,
                    f"コードレビュー実行に失敗しました: {error}",
                )
            )
        finally:
            self._running_task_ids.discard(request.task_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """cancel_task が呼ばれたときに、実行中レビューをキャンセル扱いにします。"""
        task_id = context.task_id or ""
        self._running_task_ids.discard(task_id)
        await TaskUpdater(event_queue, task_id, context.context_id or "").cancel()

    async def _stream_review_text(self, request: CodeReviewRequest) -> str:
        """OpenAI の差分出力を、A2A で扱いやすいまとまりにして流します。

        OpenAI の delta は数文字単位で届くことがあるため、そのまま送ると
        artifact_update が細かくなりすぎます。ここでは少しだけためてから送り、
        A2A 側では見やすい粒度のストリーミングになるようにしています。
        """
        full_text_parts: list[str] = []
        buffer = ""
        pending_chunk: str | None = None
        append = False

        async for delta in self._agent.stream(
            user_input=request.user_input,
            session_key=request.context_id,
        ):
            if request.task_id not in self._running_task_ids:
                return ""

            # delta を少しだけためて、A2A へはまとまった塊で流します。
            full_text_parts.append(delta)
            buffer += delta
            while len(buffer) >= 120:
                next_chunk = buffer[:120]
                buffer = buffer[120:]
                if pending_chunk is not None:
                    # 次の塊が来た時点で、前の塊は最後ではないと確定します。
                    await self._add_artifact_chunk(
                        updater=request.updater,
                        artifact_id=REVIEW_DRAFT_ARTIFACT_ID,
                        name=REVIEW_DRAFT_ARTIFACT_NAME,
                        text=pending_chunk,
                        append=append,
                        last_chunk=False,
                    )
                    append = True
                pending_chunk = next_chunk

        if pending_chunk is None and not buffer:
            await self._add_artifact_chunk(
                updater=request.updater,
                artifact_id=REVIEW_DRAFT_ARTIFACT_ID,
                name=REVIEW_DRAFT_ARTIFACT_NAME,
                text="",
                append=False,
                last_chunk=True,
            )
            return ""

        if pending_chunk is None:
            await self._add_artifact_chunk(
                updater=request.updater,
                artifact_id=REVIEW_DRAFT_ARTIFACT_ID,
                name=REVIEW_DRAFT_ARTIFACT_NAME,
                text=buffer,
                append=False,
                last_chunk=True,
            )
            return "".join(full_text_parts).strip()

        if not buffer:
            await self._add_artifact_chunk(
                updater=request.updater,
                artifact_id=REVIEW_DRAFT_ARTIFACT_ID,
                name=REVIEW_DRAFT_ARTIFACT_NAME,
                text=pending_chunk,
                append=append,
                last_chunk=True,
            )
            return "".join(full_text_parts).strip()

        await self._add_artifact_chunk(
            updater=request.updater,
            artifact_id=REVIEW_DRAFT_ARTIFACT_ID,
            name=REVIEW_DRAFT_ARTIFACT_NAME,
            text=pending_chunk,
            append=append,
            last_chunk=False,
        )
        await self._add_artifact_chunk(
            updater=request.updater,
            artifact_id=REVIEW_DRAFT_ARTIFACT_ID,
            name=REVIEW_DRAFT_ARTIFACT_NAME,
            text=buffer,
            append=True,
            last_chunk=True,
        )
        return "".join(full_text_parts).strip()

    def _split_review_artifacts(self, review_text: str) -> dict[str, str]:
        """見出し付きレビュー本文を artifact ごとの本文へ分解します。"""
        headings = {
            f"## {title}": artifact_id for artifact_id, title in REVIEW_ARTIFACTS
        }
        sections = {artifact_id: "" for artifact_id, _ in REVIEW_ARTIFACTS}
        current_artifact_id: str | None = None
        current_lines: list[str] = []

        for line in review_text.splitlines():
            artifact_id = headings.get(line.strip())
            if artifact_id is not None:
                if current_artifact_id is not None:
                    sections[current_artifact_id] = "\n".join(current_lines).strip()
                current_artifact_id = artifact_id
                current_lines = []
                continue

            if current_artifact_id is not None:
                current_lines.append(line)

        if current_artifact_id is not None:
            sections[current_artifact_id] = "\n".join(current_lines).strip()

        # 見出しが崩れた場合でも、最低限 Summary で全文を返せるようにします。
        if not any(sections.values()):
            sections["summary"] = review_text.strip()

        return sections

    async def _add_artifact_chunk(
        self,
        *,
        updater: TaskUpdater,
        artifact_id: str,
        name: str,
        text: str,
        append: bool,
        last_chunk: bool,
    ) -> None:
        """1 つの artifact_update を送ります。"""
        await updater.add_artifact(
            parts=[Part(text=text)],
            artifact_id=artifact_id,
            name=name,
            append=append,
            last_chunk=last_chunk,
        )
