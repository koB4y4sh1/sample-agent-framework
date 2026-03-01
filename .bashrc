# Git bash での実行時に自動適用される　※ powershellでは反映されない
# MCPサーバー関連のエイリアス
alias mcp="python -m src.server"
alias mcptest="python -m pytest tests/ -v"
alias mcpformat="black src/ tests/ --line-length=100"
alias mcplint="ruff check src/ tests/"
alias mcpdev="MCP_LOG_LEVEL=DEBUG python -m src.server"


# プロジェクト環境の自動有効化
# cdコマンド実行時に自動反映される
cd() {
    builtin cd "$@"
    if [[ -f .venv/bin/activate ]]; then
        source .venv/bin/activate
    fi
}