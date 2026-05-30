# Code Review Desk A2A Agent

OpenAI SDK で LLM を呼び出す「A2A で動くコードレビュー受付センター」です。

PR 概要、コード差分、設計メモ、障害報告を受け取り、リスク、レビュー観点、修正案、検証手順を返します。

同じ内容を HTML でも確認できます: [guide.html](./guide.html)

## 機能

- task 送信
- artifact_update によるストリーミング
- `context_id` によるレビュー会話の継続
- `get_task` による task 状態取得
- `cancel_task` による task キャンセル
- `reject` / `fail` によるエラー処理確認

## 起動

`.env` の `OPENAI_API_KEY` を `load_dotenv()` で読み込みます。未設定の場合は環境変数 `OPENAI_API_KEY` を設定してください。
設定例はルートの `.env.example` を参照してください。

```powershell
uv run python a2a/demo/server.py --model gpt-4.1-mini --port 41250
```

VS Code では `Demo A2A` launch configuration から起動できます。

## task 送信とストリーミング

```powershell
uv run python a2a/demo/client.py --text "このPRのリスクをレビューして: 認証ミドルウェアを追加し、全APIに適用した"
```

## context 継続

```powershell
uv run python a2a/demo/client.py --context-id review-session --text "このPRのリスクをレビューして: 認証ミドルウェアを追加し、全APIに適用した"
uv run python a2a/demo/client.py --context-id review-session --text "さっきのレビューに性能観点を追加してください。"
```

## task 状態取得

クライアントは送信直後に `get_task` を実行し、現在の状態を表示します。

```powershell
uv run python a2a/demo/client.py --text "この変更のテスト観点をレビューしてください: キャッシュ層を追加した"
```

## task キャンセル

```powershell
uv run python a2a/demo/client.py --text "このPRのリスクを詳しくレビューしてください: API ゲートウェイ配下の認証処理を全面的に見直した" --cancel-after 2
```

## 前提

- `a2a/demo` は `agent-framework/demo` から import しません。
- LLM 呼び出しには OpenAI SDK の `AsyncOpenAI` と Responses API を使います。
- Azure OpenAI を使うため、`AZURE_OPENAI_ENDPOINT` を設定し、事前に `az login` を実行してください。
- `OPENAI_MODEL` 環境変数を設定すると、`--model` 未指定時の既定モデルを変更できます。
