from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework import CompactionProvider, Message
from agent_framework._compaction import (
    CharacterEstimatorTokenizer,
    SlidingWindowStrategy,
    SummarizationStrategy,
    TokenBudgetComposedStrategy,
    ToolResultCompactionStrategy,
    TruncationStrategy,
    apply_compaction,
)
from utils.print import print_gray


@dataclass(slots=True)
class DemoCompactionConfig:
    token_budget: int = 16_000
    keep_last_tool_call_groups: int = 1
    summary_target_count: int = 4
    summary_threshold: int = 2
    keep_last_groups: int = 20
    ad_hoc_max_n: int = 8_000
    ad_hoc_compact_to: int = 4_000


class DemoCompactionProvider:
    """app.py から利用する compaction 部品を組み立てる Factory。

    戦略の順序は Microsoft Learn の推奨構成に合わせる。
    1. ToolResultCompactionStrategy
       まず古いツール結果を圧縮する。情報欠落が最も少ないため、最初に適用する。
    2. SummarizationStrategy
       要約用クライアントがある場合、古い会話を要約に置き換える。
    3. SlidingWindowStrategy
       それでも長い場合、最新の会話グループだけを残す。
    4. TruncationStrategy
       ad-hoc compaction 用の最終手段として使う。
    """

    def __init__(
        self,
        *,
        history_source_id: str,
        summarizer_client: Any | None = None,
        config: DemoCompactionConfig | None = None,
    ) -> None:
        self._history_source_id = history_source_id
        self._summarizer_client = summarizer_client
        self._config = config or DemoCompactionConfig()
        self._tokenizer = CharacterEstimatorTokenizer()

    @property
    def tokenizer(self) -> CharacterEstimatorTokenizer:
        return self._tokenizer

    def _debug(self, message: str) -> None:
        print_gray(f"[compaction] {message}")

    def create_before_strategy(self) -> TokenBudgetComposedStrategy:
        """各モデル呼び出しの前に使う compaction パイプラインを生成する。

        まずツール結果を圧縮し、次に必要なら会話要約を行い、
        最後にスライディングウィンドウで履歴サイズを制限する。
        """
        strategies: list[Any] = [
            ToolResultCompactionStrategy(
                keep_last_tool_call_groups=self._config.keep_last_tool_call_groups,
            )
        ]
        self._debug(
            "before_strategy: ToolResultCompactionStrategy を追加 "
            f"(keep_last_tool_call_groups={self._config.keep_last_tool_call_groups})"
        )
        if self._summarizer_client is not None:
            strategies.append(
                SummarizationStrategy(
                    client=self._summarizer_client,
                    target_count=self._config.summary_target_count,
                    threshold=self._config.summary_threshold,
                )
            )
            self._debug(
                "before_strategy: SummarizationStrategy を追加 "
                f"(target_count={self._config.summary_target_count}, "
                f"threshold={self._config.summary_threshold})"
            )
        else:
            self._debug("before_strategy: SummarizationStrategy はスキップ (summarizer_client が未設定)")
        strategies.append(
            SlidingWindowStrategy(keep_last_groups=self._config.keep_last_groups)
        )
        self._debug(
            "before_strategy: SlidingWindowStrategy を追加 "
            f"(keep_last_groups={self._config.keep_last_groups})"
        )
        self._debug(
            "before_strategy: TokenBudgetComposedStrategy を生成 "
            f"(token_budget={self._config.token_budget}, strategies={len(strategies)})"
        )

        return TokenBudgetComposedStrategy(
            token_budget=self._config.token_budget,
            tokenizer=self._tokenizer,
            strategies=strategies,
        )

    def create_after_strategy(self) -> ToolResultCompactionStrategy:
        """永続化済み履歴に対して実行後に使う戦略を生成する。

        次回ターン開始時の履歴を軽くするため、古いツール結果だけを圧縮する。
        """
        self._debug(
            "after_strategy: ToolResultCompactionStrategy を生成 "
            f"(keep_last_tool_call_groups={self._config.keep_last_tool_call_groups})"
        )
        return ToolResultCompactionStrategy(
            keep_last_tool_call_groups=self._config.keep_last_tool_call_groups,
        )

    def create_provider(self) -> CompactionProvider:
        """demo エージェントに注入する CompactionProvider を生成する。"""
        self._debug(
            "provider: CompactionProvider を生成 "
            f"(history_source_id={self._history_source_id})"
        )
        return CompactionProvider(
            before_strategy=self.create_before_strategy(),
            after_strategy=self.create_after_strategy(),
            history_source_id=self._history_source_id,
            tokenizer=self._tokenizer,
        )

    async def compact_messages(self, messages: list[Message]) -> list[Message]:
        """任意のメッセージ配列に対して アドホック compaction を実行する。

        ここでは最終手段として TruncationStrategy を使い、
        サイズ条件を満たすまで古いグループを削る。
        """
        self._debug(
            "ad_hoc: TruncationStrategy を実行 "
            f"(input_messages={len(messages)}, max_n={self._config.ad_hoc_max_n}, "
            f"compact_to={self._config.ad_hoc_compact_to})"
        )
        return await apply_compaction(
            messages,
            strategy=TruncationStrategy(
                max_n=self._config.ad_hoc_max_n,
                compact_to=self._config.ad_hoc_compact_to,
                tokenizer=self._tokenizer,
            ),
            tokenizer=self._tokenizer,
        )
