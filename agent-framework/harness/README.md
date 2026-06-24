# Harness Browser Agent

`create_harness_agent()` を使い、Foundry Toolbox MCP と Python Playwright を組み合わせたブラウザ操作エージェントのサンプルです。

## ファイル構成

- `agent.py`: Foundry client、Toolbox MCP、Playwright操作ツール、`create_harness_agent()`、履歴/Todo/Mode/Judge middleware の構築
- `browser_tools.py`: Toolbox で CDP URL を取得し、Python Playwright で接続して `browser_goto` / `browser_snapshot` / `browser_click` / `browser_fill` などを提供
- `settings.py`: `pydantic-settings` による `HA_` 環境変数と `.env` の読み込み・検証
- `run.py`: 長時間実行ループ、再試行、タイムアウト、停滞検知、チェックポイント、ハートビート、JSONLログ
- `main.py`: CLI、環境変数読み込み、設定検証、終了コード

## 前提

- Azure AI Foundry project が利用可能
- Foundry Toolbox `playwright-browser` が作成済み
- Azure Playwright Workspace にセッション作成できる権限がある
- `az login` 済み

`.env` に以下を設定します。`settings.py` の `HarnessSettings` が `HA_` prefix の環境変数を読み込みます。

```env
HA_FOUNDRY_PROJECT_ENDPOINT=https://aif-gptapp-dev-swedencentral.services.ai.azure.com/api/projects/proj-gptapp-dev-swedencentral
HA_FOUNDRY_MODEL=gpt-5.4-nano
HA_TOOLBOX_NAME=playwright-browser
HA_TOOLBOX_VERSION=2
HA_TOOLBOX_API_VERSION=v1
HA_GOAL_FILE=agent-framework/harness/goal.md
```

`goal.md` は CLI 引数には変換されません。Agent の instructions に追加され、ユーザーのゴールとログイン情報として参照されます。

## 実行

```powershell
uv run python -m harness.main `
  --allowed-domain example.com `
  "example.com を開いてページタイトルを確認してください"
```

長時間タスクではサイクル数、1サイクルのタイムアウト、リトライ回数を明示します。

```powershell
uv run python -m harness.main `
  --allowed-domain example.com `
  --session-id example-run-001 `
  --max-cycles 30 `
  --cycle-timeout-seconds 900 `
  --retry-attempts 3 `
  --consecutive-error-limit 3 `
  "example.com だけを対象に、検索結果を確認して要点をまとめてください"
```

寝ている間に自律継続させる場合は `--unattended` を付けます。`NEEDS_USER`、最大実行時間、連続エラー上限に達した場合だけ停止します。

```powershell
uv run python -m harness.main `
  --allowed-domain example.com `
  --session-id overnight-001 `
  --unattended `
  --max-runtime-hours 8 `
  --max-cycles 300 `
  --cycle-timeout-seconds 1200 `
  --retry-attempts 5 `
  --consecutive-error-limit 10 `
  --stall-limit 10 `
  "example.com だけを対象に、許可された範囲で調査を継続し、結果をまとめてください"
```

同じ `--session-id` と `--resume` を指定すると、同じ履歴ファイルとチェックポイントを使って再開します。

```powershell
uv run python -m harness.main `
  --allowed-domain example.com `
  --session-id example-run-001 `
  --resume `
  "example.com だけを対象に、検索結果を確認して要点をまとめてください"
```

各サイクルの最後に Agent は以下のステータスを返します。

- `CONTINUE`: 継続可能
- `COMPLETE`: 完了
- `NEEDS_USER`: ログイン、MFA、明示承認、追加入力が必要
- `BLOCKED`: ツール、環境、対象サイト都合で停止

`--enable-judge-loop` を付けると、各サイクル内で `AgentLoopMiddleware.with_judge()` による応答品質チェックを追加します。コストとレイテンシは増えます。

`--require-tool-approval` を付けると Harness の Tool Approval middleware を有効化します。ブラウザ操作のような高頻度ツールでは停止回数が増えるため、既定では無効です。送信、削除、購入、公開、権限変更などの意味的な承認境界はプロンプトで `NEEDS_USER` に停止させます。

実行成果物:

- `agent-framework/harness/runs/<session-id>.jsonl`: 監査ログ
- `agent-framework/harness/runs/<session-id>.checkpoint.json`: 最新チェックポイント
- `agent-framework/harness/runs/<session-id>.heartbeat.json`: 外部監視用ハートビート
- `agent-framework/harness/runs/history/<session-id>.jsonl`: Agent Framework の会話履歴

## 注意

この環境の Agent Framework 1.9.0 では、`HarnessAgent` という公開クラスではなく `create_harness_agent()` がハーネスエージェント作成APIです。

Toolbox MCP は Azure Playwright Workspace 上のブラウザセッション作成を担当します。返された CDP URL に Python Playwright で接続し、実際の URL 遷移、クリック、入力、スナップショット取得を `browser_tools.py` の function tool が担当します。Harness は Todo、履歴、コンテキスト圧縮、長時間タスク用の制御を担当します。

このサンプルの長時間稼働制御は、1回の `agent.run()` を無制限に伸ばす方式ではありません。外側の Python ループで短い作業サイクルに分割し、停止条件、タイムアウト、リトライ、進捗停滞検知、JSONLログを管理します。
