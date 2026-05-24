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
    """demo 用 compaction の調整値。

    compaction は「モデルに渡す会話履歴を小さくする処理」です。
    会話が長くなると、毎回のモデル呼び出しに含める履歴も大きくなり、
    トークン上限、コスト、応答時間の問題が出ます。この設定は、その問題を
    どの程度まで許容し、どこから履歴を小さくするかを決めます。

    Attributes:
        token_budget: before_strategy が目標にする推定トークン数。
            この値以下になったら、後続の圧縮戦略は原則として早期終了します。
        keep_last_tool_call_groups: 圧縮せずに残す最新のツール呼び出しグループ数。
            古いツール結果は短い要約メッセージへ折りたたまれます。
        summary_target_count: 要約後もそのまま残す最新メッセージ数。
        summary_threshold: 要約を始めるための余裕値。
            対象メッセージ数が summary_target_count + summary_threshold を超えると、
            古い部分が要約候補になります。
        keep_last_groups: 最後に残す最新の非 system メッセージグループ数。
            要約しても長い場合の強めの制限です。
        ad_hoc_max_n: compact_messages で TruncationStrategy を開始する推定トークン数。
        ad_hoc_compact_to: compact_messages が削減先として目標にする推定トークン数。
    """

    token_budget: int = 16_000
    keep_last_tool_call_groups: int = 1
    summary_target_count: int = 4
    summary_threshold: int = 2
    keep_last_groups: int = 20
    ad_hoc_max_n: int = 8_000
    ad_hoc_compact_to: int = 4_000


class DemoCompactionProvider:
    """demo エージェント用の CompactionProvider を組み立てる Factory。

    Agent Framework の compaction は、履歴そのものの意味をできるだけ残しながら、
    モデルに渡すメッセージ量を減らす仕組みです。このクラスでは、Microsoft Learn の
    Python 向け構成に合わせて、次の順序で戦略を組み立てます。

    1. ToolResultCompactionStrategy:
        古いツール呼び出し結果を短い要約メッセージへ折りたたみます。
        ユーザー発話や通常のアシスタント応答を消さないため、最初に実行します。
    2. SummarizationStrategy:
        summarizer_client がある場合だけ、古い会話を LLM で要約します。
        直近の会話はそのまま残し、古い文脈を短く保持します。
    3. SlidingWindowStrategy:
        それでも長い場合に、最新のメッセージグループだけを残します。
        文脈保持よりもサイズ上限を優先する段階です。
    4. TruncationStrategy:
        compact_messages で使う ad-hoc 用の最終手段です。
        条件を満たすまで古いグループを削除します。
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
        """モデル呼び出し前に実行する compaction パイプラインを生成する。

        before_strategy は、今回の LLM 呼び出しに渡す直前の履歴を小さくします。
        この demo では TokenBudgetComposedStrategy を使い、推定トークン数が
        token_budget 以下になった時点で後続戦略を止めます。

        実行順:
            1. 古いツール結果を短くする。
            2. summarizer_client があれば、古い会話を要約する。
            3. 最新 keep_last_groups グループだけに制限する。
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
        """モデル呼び出し後に、保存済み履歴へ適用する戦略を生成する。

        after_strategy は、今回の応答が終わって履歴へ保存された後に動きます。
        ここでは会話本文は保ったまま、古いツール結果だけを折りたたみます。
        これにより、次のターン開始時点の履歴サイズを小さくできます。
        """
        self._debug(
            "after_strategy: ToolResultCompactionStrategy を生成 "
            f"(keep_last_tool_call_groups={self._config.keep_last_tool_call_groups})"
        )
        return ToolResultCompactionStrategy(
            keep_last_tool_call_groups=self._config.keep_last_tool_call_groups,
        )

    def create_provider(self) -> CompactionProvider:
        """demo エージェントの context_providers に渡す CompactionProvider を生成する。

        history_source_id は、どの履歴プロバイダーのメッセージを圧縮対象にするかを
        Agent Framework に伝える識別子です。この demo では LocalHistoryProvider の
        source_id を渡します。
        """
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
        """任意のメッセージ配列に対して ad-hoc compaction を実行する。

        Agent の通常実行とは別に、手元の messages を直接小さくしたい場合に使います。
        ここでは TruncationStrategy を使い、推定トークン数が ad_hoc_max_n を超えたら
        古いグループから削除し、ad_hoc_compact_to 付近まで減らします。

        注意:
            TruncationStrategy は情報を要約せず削除します。会話の意味を残したい
            通常ルートでは、create_before_strategy の段階的な圧縮を優先します。
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
