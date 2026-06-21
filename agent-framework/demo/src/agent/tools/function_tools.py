"""Python関数として実装するツールの定義モジュール。

ここにある関数は ``@tool`` により、LLMが必要に応じて呼び出せる
function toolとして公開可能。

Progressive Tool Exposureでは、最初から実ツールを全部見せるのではなく、
まずローダーツールだけを公開。
ローダーツールが ``FunctionInvocationContext.add_tools`` で実ツールを追加する構成。

重要:
    実運用では、このファイル内の処理を直接業務システムへ接続する想定。
    このサンプルでは動作確認しやすい固定文の返却のみ。
"""

from __future__ import annotations

from random import randint
from typing import Annotated, Any

from agent_framework import FunctionInvocationContext, tool


# =======================================
# region Loader
# =======================================
@tool(approval_mode="never_require")
def load_document_search_tools(ctx: FunctionInvocationContext) -> str:
    """文書検索系の実ツールを現在のrunへ追加するローダーツール。

    追加される実ツール:
    - search_internal_documents
    - search_faq
    """

    ctx.add_tools([search_internal_documents, search_faq])
    return "社内文書検索とFAQ検索を利用可能にしました。"


@tool(approval_mode="never_require")
def load_application_tools(ctx: FunctionInvocationContext) -> str:
    """申請系の実ツールを現在のrunへ追加するローダーツール。

    追加される実ツール:
    - search_application_candidates
    - create_application_draft
    - request_application_approval
    """

    ctx.add_tools(
        [
            search_application_candidates,
            create_application_draft,
            request_application_approval,
        ]
    )
    return "申請候補検索、申請下書き作成、承認依頼を利用可能にしました。"


# =======================================
# region Sample
# =======================================


@tool
def search_internal_documents(
    query: Annotated[str, "Query for internal policies, rules, and company documents"],
) -> str:
    """社内文書や規程を検索するサンプルツール。"""

    return f"社内文書検索結果: '{query}' に関連する規程候補を返しました。"


@tool
def search_faq(
    query: Annotated[str, "Query for frequently asked questions"],
) -> str:
    """FAQを検索するサンプルツール。"""

    return f"FAQ検索結果: '{query}' に関連するFAQ候補を返しました。"


@tool
def search_application_candidates(
    query: Annotated[str, "Query for application forms or request types"],
) -> str:
    """ユーザー入力に合う申請種別を探すサンプルツール。"""

    return f"申請候補検索結果: '{query}' に関連する申請種別候補を返しました。"


@tool(approval_mode="always_require")
def create_application_draft(
    application_type: Annotated[str, "Application type to draft"],
    purpose: Annotated[str, "Purpose or reason for the application"],
) -> str:
    """申請の下書きを作成するサンプルツール。

    業務データを作る操作のため、ユーザー承認を必須化。
    """

    return f"申請下書きを作成しました: 種別={application_type}, 目的={purpose}"


@tool(approval_mode="always_require")
def request_application_approval(
    draft_id: Annotated[str, "Draft application identifier"],
    approver: Annotated[str, "Approver name or group"],
) -> str:
    """作成済み下書きの承認依頼を送るサンプルツール。

    外部へ通知する操作のため、ユーザー承認を必須化。
    """

    return f"承認依頼を送信しました: draft_id={draft_id}, approver={approver}"


# =======================================
# region Weather
# =======================================


@tool(approval_mode="always_require")
def get_weather(
    location: Annotated[str, "Weather target city such as Tokyo, New York, Paris"],
) -> str:
    """指定された都市の天気を返すサンプルツール。

    ``approval_mode="always_require"`` による、実行前ユーザー承認の必須化。
    """

    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"{location} weather is {conditions[randint(0, 3)]}, approx. {randint(10, 30)}C."


# =======================================
# region Build Tools
# =======================================


def build_function_tools() -> list[Any]:
    """このアプリで使う全function tool一覧。

    新しい関数ツールを追加した場合の追加先。
    DevUIなど、run中にツールを動的追加しない経路向けの一覧。
    """

    return [
        get_weather,
        search_internal_documents,
        search_faq,
        search_application_candidates,
        create_application_draft,
        request_application_approval,
    ]


def build_progressive_loader_tools() -> list[Any]:
    """Progressive Tool Exposure用の初期公開ツール一覧。

    各ツールはローダーツールが実行されることで段階的に公開される。
    """

    return [
        load_document_search_tools,
        load_application_tools,
        get_weather,
    ]
