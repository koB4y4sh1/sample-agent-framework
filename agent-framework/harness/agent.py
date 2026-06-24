from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from agent_framework import (
    AgentLoopMiddleware,
    AgentModeProvider,
    FileHistoryProvider,
    MCPStreamableHTTPTool,
    TodoProvider,
    TodoSessionStore,
    create_harness_agent,
)
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from harness.browser_tools import CdpBrowserController, build_browser_function_tools
from harness.settings import HarnessSettings


# Agent に常に渡す基本ルールです。
# ここは「ブラウザで何をしてよいか / 何をしてはいけないか」を決める安全柵です。
BASE_INSTRUCTIONS = """
You are a browser operations agent for approved business and test sites.

Operational rules:
- Operate only on the allowed domains listed below.
- Do not automate exams, quizzes, CAPTCHA, or access-control bypasses.
- Before submit, delete, purchase, publish, permission change, external sharing,
  or any irreversible action, stop with LOOP_STATUS: NEEDS_USER.
- If login or MFA is required, stop with LOOP_STATUS: NEEDS_USER and describe
  exactly what the user must complete manually.
- Use the todo tools to plan and track multi-step work.
- Summarize extracted facts with the source URL, page title, selector, visible
  text, or browser error that supports the finding.
- Never mark work COMPLETE unless the original task is fully satisfied.
""".strip()


@dataclass(frozen=True, slots=True)
class HarnessAgentConfig:
    """Harness Agent を作るために必要な設定をまとめた入れ物。

    ここに集約しておくと、CLI やテストから Agent の作り方を変えやすくなります。
    Toolbox の接続情報は `HarnessSettings` で検証します。
    """

    project_endpoint: str
    model: str
    toolbox_name: str
    toolbox_version: str
    toolbox_api_version: str
    allowed_domains: tuple[str, ...]
    history_dir: Path
    max_context_window_tokens: int
    max_output_tokens: int
    enable_judge_loop: bool
    judge_iterations: int
    require_tool_approval: bool
    unattended: bool

    @classmethod
    def from_env(
        cls,
        *,
        allowed_domains: tuple[str, ...],
        history_dir: Path,
        max_context_window_tokens: int,
        max_output_tokens: int,
        enable_judge_loop: bool,
        judge_iterations: int,
        require_tool_approval: bool,
        unattended: bool,
    ) -> HarnessAgentConfig:
        """環境変数と CLI 引数から Agent 設定を作る。"""

        settings = HarnessSettings()
        return cls(
            project_endpoint=settings.foundry_project_endpoint,
            model=settings.foundry_model,
            toolbox_name=settings.toolbox_name,
            toolbox_version=settings.toolbox_version,
            toolbox_api_version=settings.toolbox_api_version,
            allowed_domains=allowed_domains,
            history_dir=history_dir,
            max_context_window_tokens=max_context_window_tokens,
            max_output_tokens=max_output_tokens,
            enable_judge_loop=enable_judge_loop,
            judge_iterations=judge_iterations,
            require_tool_approval=require_tool_approval,
            unattended=unattended,
        )


class _ToolboxAuth(httpx.Auth):
    """Foundry Toolbox MCP への各リクエストに Bearer token を付ける認証クラス。"""

    def __init__(self, token_provider: Callable[[], str]) -> None:
        self._get_token = token_provider

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {self._get_token()}"
        yield request


@dataclass(slots=True)
class HarnessAgentRuntime:
    """Agent と、終了時に閉じる必要がある tool 接続をまとめた実行用オブジェクト。"""

    agent: Any
    close: Callable[[], Awaitable[None]]
    available_tools: tuple[str, ...]


async def _noop_close() -> None:
    """閉じるものがない構成で使う空の close 処理。"""


async def _close_tool(tool: Any) -> None:
    """tool が close を持っていれば、同期/非同期どちらでも安全に閉じる。"""

    close = getattr(tool, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _domain_instructions(allowed_domains: tuple[str, ...]) -> str:
    """許可ドメイン一覧を Agent の指示文に埋め込む文字列へ変換する。"""

    domains = "\n".join(f"- {domain}" for domain in allowed_domains)
    return f"Allowed domains:\n{domains}"


def _goal_instructions(settings: HarnessSettings) -> str:
    """goal.md を読み取り、Agent に渡す追加指示を作る。

    CLI 引数には反映しません。Agent が「ユーザーのゴール」として認識するためだけに使います。
    goal.md にはログイン情報が含まれる前提なので、ログ出力や最終回答で資格情報を表示しない指示も付けます。
    """

    goal_file = settings.goal_file
    if not goal_file.exists():
        return ""
    goal_text = goal_file.read_text(encoding="utf-8").strip()
    if not goal_text:
        return ""
    return f"""
## User Goal From goal.md

The following markdown is the user's goal and login context.
Use it as task context. Do not copy credentials into summaries, logs, or final answers.
Use the username and password only for login to the authorized target site.
If MFA, CAPTCHA, or human approval is required, stop with LOOP_STATUS: NEEDS_USER.

```markdown
{goal_text}
```
""".strip()


def _toolbox_url(config: HarnessAgentConfig) -> str:
    """Foundry project endpoint から Toolbox MCP endpoint URL を組み立てる。"""

    return (
        f"{config.project_endpoint.rstrip('/')}"
        f"/toolboxes/{config.toolbox_name}"
        f"/versions/{config.toolbox_version}"
        f"/mcp?api-version={config.toolbox_api_version}"
    )


def _build_browser_tool(config: HarnessAgentConfig, credential: DefaultAzureCredential):
    """Foundry Toolbox の MCP endpoint をブラウザ操作 tool として作る。"""

    token_provider = get_bearer_token_provider(
        credential,
        "https://ai.azure.com/.default",
    )
    http_client = httpx.AsyncClient(
        auth=_ToolboxAuth(token_provider),
        headers={"Foundry-Features": "Toolboxes=V1Preview"},
        timeout=120.0,
    )
    return MCPStreamableHTTPTool(
        name=config.toolbox_name,
        url=_toolbox_url(config),
        http_client=http_client,
        load_prompts=False,
    )


def _unattended_instructions(enabled: bool) -> str:
    """無人運転モード用の追加指示を返す。

    無人運転でも、ログイン/MFA/承認/危険操作は自動化しません。
    一時的なブラウザエラーだけを自律回復の対象にします。
    """

    if not enabled:
        return "Unattended mode: disabled. Stop with BLOCKED when progress cannot be made."
    return """
Unattended mode: enabled.
- Continue autonomously while the next action is safe and within the allowed domains.
- Treat transient browser errors, navigation failures, timeouts, missing elements, stale page state,
  and unclear page layout as recoverable. Try a different strategy before returning BLOCKED.
- Use LOOP_STATUS: NEEDS_USER only for login, MFA, explicit approval, missing user input,
  or irreversible/high-impact actions.
- Use LOOP_STATUS: BLOCKED only after recovery attempts are exhausted in the current cycle.
""".strip()


def _browser_tool_instructions() -> str:
    """Agent に Python 側ブラウザツールの使い方を教える。"""

    return """
Browser tool usage:
- Use browser_start once before browser_goto if no browser session exists.
- Use browser_goto to open an allowed-domain URL.
- Use browser_snapshot after navigation or actions to inspect visible text, links, buttons, and inputs.
- Use browser_fill for text inputs and browser_click for buttons/links.
- Use browser_press for simple keyboard actions such as Enter or Tab.
- Do not call Toolbox meta tools directly. Do not use tool_search or call_tool yourself.
""".strip()


async def build_harness_agent(config: HarnessAgentConfig) -> HarnessAgentRuntime:
    """Foundry Browser Automation tool 付きの Harness Agent を作る。

    この関数がやっていること:
    1. Azure 認証を作る
    2. FoundryChatClient を作る
    3. Browser Automation tool を作る
    4. 履歴/Todo/Mode/Judge などを束ねて create_harness_agent() に渡す
    """

    # DefaultAzureCredential は Azure CLI、環境変数、Managed Identity などを順に使います。
    settings = HarnessSettings()
    credential = DefaultAzureCredential()

    # LLM と通信するクライアントです。Browser tool もこのクライアント種別に紐づきます。
    client = FoundryChatClient(
        project_endpoint=config.project_endpoint,
        model=config.model,
        credential=credential,
        allow_preview=True,
    )

    # 実際にブラウザを操作する「手足」です。toolbox mode では MCP tool になります。
    browser_tool = _build_browser_tool(config, credential)
    await browser_tool.connect()
    browser_controller = CdpBrowserController(
        toolbox=browser_tool,
        allowed_domains=config.allowed_domains,
    )
    browser_function_tools = build_browser_function_tools(browser_controller)
    available_tools = tuple(getattr(tool, "name", "") for tool in browser_function_tools)

    middleware = []
    if config.enable_judge_loop:
        # 各サイクルの応答が雑な場合に、モデル自身に再確認させる品質チェックです。
        # コストと時間は増えますが、長時間実行では完了判定ミスの抑制に効きます。
        middleware.append(
            AgentLoopMiddleware.with_judge(
                client,
                criteria=[
                    "The response includes exactly one LOOP_STATUS block.",
                    "The status is supported by browser evidence or an explicit blocker.",
                    "COMPLETE is used only when all requested browser work is done.",
                    "NEEDS_USER is used before login, MFA, submission, deletion, purchase, publication, or permission changes.",
                ],
                max_iterations=config.judge_iterations,
                fresh_context=False,
            )
        )

    agent = create_harness_agent(
        client=client,
        name="browser-harness-agent",
        description="Harness agent that controls a Foundry Browser Automation tool.",
        agent_instructions=(
            f"{BASE_INSTRUCTIONS}\n\n"
            f"{_domain_instructions(config.allowed_domains)}\n\n"
            f"{_unattended_instructions(config.unattended)}\n\n"
            f"{_browser_tool_instructions()}\n\n"
            f"{_goal_instructions(settings)}"
        ),
        tools=browser_function_tools,
        # 会話履歴をファイルに残します。プロセス再起動時の resume 材料になります。
        history_provider=FileHistoryProvider(config.history_dir),
        # 長い作業で「残タスク」を Agent 側に管理させます。
        todo_provider=TodoProvider(store=TodoSessionStore()),
        # Harness の mode 管理です。このサンプルでは最初から execute にします。
        mode_provider=AgentModeProvider(default_mode="execute"),
        # 勝手な Web 検索を避け、許可ドメインのブラウザ操作に集中させます。
        disable_web_search=True,
        # ブラウザ操作は高頻度 tool call なので既定では承認 middleware を無効化します。
        # 送信/削除などの意味的な承認はプロンプトで NEEDS_USER に止めます。
        disable_tool_auto_approval=not config.require_tool_approval,
        max_context_window_tokens=config.max_context_window_tokens,
        max_output_tokens=config.max_output_tokens,
        middleware=middleware or None,
    )

    async def close_browser_tool() -> None:
        """この Agent が作った browser tool 接続を閉じる。"""

        await browser_controller.close()
        await _close_tool(browser_tool)

    return HarnessAgentRuntime(
        agent=agent,
        close=close_browser_tool if hasattr(browser_tool, "close") else _noop_close,
        available_tools=available_tools,
    )
