"""
DocuMentor MCP Server - 社内コーディング規約サーバー

GitHubリポジトリから社内のコーディング規約を取得し、
Claude Codeに提供するMCPサーバー
"""

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("documenter")

# 設定
GITHUB_REPO_URL = "https://github.com/0xshooka/corporate-code-rules.git"
LOCAL_REPO_PATH = Path.home() / ".documenter" / "corporate-code-rules"
CACHE_FILE = Path.home() / ".documenter" / "index_cache.json"

@dataclass
class DocumentSection:
    """ドキュメントのセクション情報"""
    title: str
    content: str
    level: int
    file_path: str
    line_number: int

@dataclass
class CodeExample:
    """コード例の情報"""
    description: str
    code: str
    is_good_example: bool
    file_path: str

class DocumentIndexer:
    """ドキュメントのインデックス化を行うクラス"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.documents: Dict[str, List[DocumentSection]] = {}
        self.code_examples: List[CodeExample] = []
        
    def index_all_documents(self):
        """すべてのドキュメントをインデックス化"""
        print("ドキュメントのインデックス化を開始...")
        
        # Markdownファイルを探索
        md_files = list(self.repo_path.rglob("*.md"))
        for md_file in md_files:
            if ".git" not in str(md_file):
                self._index_markdown(md_file)
        
        # コード例を探索
        self._index_code_examples()
        
        print(f"インデックス完了: {len(self.documents)} ファイル, "
            f"{sum(len(sections) for sections in self.documents.values())} セクション")
    
    def _index_markdown(self, file_path: Path):
        """Markdownファイルをインデックス化"""
        relative_path = file_path.relative_to(self.repo_path)
        sections = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            # 見出しを検出
            if line.startswith('#'):
                # 前のセクションを保存
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)
                    current_content = []
                
                # 新しいセクションを開始
                level = len(line) - len(line.lstrip('#'))
                title = line.strip('#').strip()
                current_section = DocumentSection(
                    title=title,
                    content="",
                    level=level,
                    file_path=str(relative_path),
                    line_number=i + 1
                )
            else:
                current_content.append(line.rstrip())
        
        # 最後のセクションを保存
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        self.documents[str(relative_path)] = sections
    
    def _index_code_examples(self):
        """コード例をインデックス化"""
        # 良い例
        good_examples_dir = self.repo_path / "examples" / "good"
        if good_examples_dir.exists():
            for py_file in good_examples_dir.glob("*.py"):
                self._extract_code_example(py_file, is_good=True)
        
        # 悪い例
        bad_examples_dir = self.repo_path / "examples" / "bad"
        if bad_examples_dir.exists():
            for py_file in bad_examples_dir.glob("*.py"):
                self._extract_code_example(py_file, is_good=False)
    
    def _extract_code_example(self, file_path: Path, is_good: bool):
        """Pythonファイルからコード例を抽出"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # docstringから説明を抽出
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        description = docstring_match.group(1).strip() if docstring_match else ""
        
        relative_path = file_path.relative_to(self.repo_path)
        self.code_examples.append(CodeExample(
            description=description,
            code=content,
            is_good_example=is_good,
            file_path=str(relative_path)
        ))
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """ドキュメントを検索"""
        query_lower = query.lower()
        results = []
        
        for file_path, sections in self.documents.items():
            for section in sections:
                # タイトルまたは内容に検索語が含まれているか
                if (query_lower in section.title.lower() or 
                    query_lower in section.content.lower()):
                    results.append({
                        "file": file_path,
                        "title": section.title,
                        "content": section.content[:200] + "..." if len(section.content) > 200 else section.content,
                        "relevance": self._calculate_relevance(query_lower, section)
                    })
        
        # 関連度でソート
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:5]  # 上位5件を返す
    
    def _calculate_relevance(self, query: str, section: DocumentSection) -> float:
        """関連度を計算"""
        score = 0.0
        
        # タイトルに含まれる場合は高スコア
        if query in section.title.lower():
            score += 2.0
        
        # 内容に含まれる回数をカウント
        content_lower = section.content.lower()
        score += content_lower.count(query) * 0.1
        
        # セクションレベルが低い（重要度が高い）ほど高スコア
        score += (4 - section.level) * 0.5
        
        return score

class GitManager:
    """Gitリポジトリの管理を行うクラス"""
    
    @staticmethod
    def setup_repository() -> bool:
        """リポジトリをセットアップ"""
        if LOCAL_REPO_PATH.exists():
            # 既存のリポジトリを更新
            try:
                subprocess.run(
                    ["git", "pull"],
                    cwd=LOCAL_REPO_PATH,
                    check=True,
                    capture_output=True
                )
                print("リポジトリを更新しました")
                return True
            except subprocess.CalledProcessError as e:
                print(f"リポジトリの更新に失敗: {e}")
                return False
        else:
            # 新規クローン
            try:
                LOCAL_REPO_PATH.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "clone", GITHUB_REPO_URL, str(LOCAL_REPO_PATH)],
                    check=True,
                    capture_output=True
                )
                print("リポジトリをクローンしました")
                return True
            except subprocess.CalledProcessError as e:
                print(f"リポジトリのクローンに失敗: {e}")
                return False

# グローバルインデクサー
@lru_cache(maxsize=1)
def get_indexer() -> DocumentIndexer:
    """インデクサーを初期化"""
    global indexer
    
    # Gitリポジトリのセットアップ
    if not GitManager.setup_repository():
        raise Exception("リポジトリのセットアップに失敗しました")
    
    # インデックスの作成
    indexer = DocumentIndexer(LOCAL_REPO_PATH)
    indexer.index_all_documents()
    return indexer

# MCPツールの定義

@mcp.tool()
async def search_documentation(query: str, doc_type: Optional[str] = None) -> str:
    """
    社内ドキュメントを検索
    
    Args:
        query: 検索クエリ（例: "エラーハンドリング", "API設計"）
        doc_type: ドキュメントタイプ（例: "naming", "error", "api"）
    """
    indexer = get_indexer()
    
    results = indexer.search(query)
    
    if not results:
        return f"'{query}' に関する規約が見つかりませんでした。"
    
    # 結果を整形
    output = f"'{query}' に関する規約:\n\n"
    for i, result in enumerate(results, 1):
        output += f"{i}. **{result['title']}** ({result['file']})\n"
        output += f"   {result['content']}\n\n"
    
    return output

@mcp.tool()
async def get_coding_standard(topic: str) -> str:
    """
    特定トピックのコーディング規約を取得
    
    Args:
        topic: トピック（naming, error, testing, api, database）
    """
    # トピックとファイルのマッピング
    topic_file_map = {
        "naming": "docs/naming-conventions.md",
        "error": "docs/error-handling.md",
        "testing": "docs/testing-guidelines.md",
        "api": "docs/architecture/api-design.md",
        "database": "docs/architecture/database-patterns.md"
    }
    
    file_path = topic_file_map.get(topic.lower())
    if not file_path:
        return f"トピック '{topic}' の規約が見つかりません。利用可能: " + ", ".join(topic_file_map.keys())
    
    # 該当ファイルのセクションを取得
    if file_path not in indexer.documents:
        return f"ファイル {file_path} が見つかりません。"
    
    sections = indexer.documents[file_path]
    
    # 主要なセクションを返す
    output = f"# {topic.upper()} コーディング規約\n\n"
    for section in sections[:3]:  # 最初の3セクション
        output += f"## {section.title}\n"
        output += f"{section.content}\n\n"
    
    return output

@mcp.tool()
async def check_code_compliance(code: str, check_type: Optional[str] = None) -> str:
    """
    コードが規約に準拠しているかチェック
    
    Args:
        code: チェックするPythonコード
        check_type: チェックタイプ（naming, security, api）
    """
    violations = []
    warnings = []
    
    # 基本的なチェック（簡易版）
    
    # 1. ハードコードされた認証情報のチェック
    if re.search(r'(api_key|password|secret)\s*=\s*["\'][\w\-]+["\']', code, re.IGNORECASE):
        violations.append("🚨 セキュリティ違反: APIキーまたはパスワードがハードコードされています")
    
    # 2. 命名規則チェック
    # クラス名のチェック
    class_matches = re.findall(r'class\s+(\w+)', code)
    for class_name in class_matches:
        if not class_name[0].isupper():
            violations.append(f"❌ 命名規則違反: クラス名 '{class_name}' はPascalCaseにしてください")
    
    # 関数名のチェック
    func_matches = re.findall(r'def\s+(\w+)', code)
    for func_name in func_matches:
        if func_name != func_name.lower():
            violations.append(f"❌ 命名規則違反: 関数名 '{func_name}' はsnake_caseにしてください")
    
    # 3. エラーハンドリングチェック
    if "except Exception:" in code or "except:" in code:
        warnings.append("⚠️ エラーハンドリング: 具体的な例外を捕捉することを推奨します")
    
    # 4. ログ出力チェック
    if re.search(r'(print|logger\.\w+).*password', code, re.IGNORECASE):
        violations.append("🚨 セキュリティ違反: パスワードをログに出力している可能性があります")
    
    # 結果を整形
    if not violations and not warnings:
        return "✅ コードは規約に準拠しています！"
    
    output = "## コード規約チェック結果\n\n"
    
    if violations:
        output += "### 違反事項\n"
        for v in violations:
            output += f"- {v}\n"
        output += "\n"
    
    if warnings:
        output += "### 警告\n"
        for w in warnings:
            output += f"- {w}\n"
        output += "\n"
    
    # 良い例を提案
    if violations:
        output += "### 💡 ヒント\n"
        output += "良い実装例は `examples/good/` ディレクトリを参照してください。\n"
        output += "例: `get_coding_standard('error')` で詳細な規約を確認できます。"
    
    return output

@mcp.tool()
async def suggest_code_improvement(description: str) -> str:
    """
    規約に基づいたコード改善の提案
    
    Args:
        description: 実装したい機能の説明
    """
    indexer = get_indexer()
    
    # 関連する規約を検索
    relevant_docs = indexer.search(description)
    
    output = f"## '{description}' の実装ガイド\n\n"
    
    # 関連する規約をリスト
    if relevant_docs:
        output += "### 関連する規約\n"
        for doc in relevant_docs[:3]:
            output += f"- {doc['title']} ({doc['file']})\n"
        output += "\n"
    
    # 一般的な推奨事項
    output += "### 実装時の注意点\n"
    output += "1. **エラーハンドリング**: 適切な例外処理とログ出力\n"
    output += "2. **セキュリティ**: 認証情報のハードコーディング禁止\n"
    output += "3. **命名規則**: PEP 8に準拠したsnake_case/PascalCase\n"
    output += "4. **テスト**: 単体テストの作成（カバレッジ80%以上）\n"
    output += "5. **API設計**: 統一レスポンスフォーマットの使用\n\n"
    
    # 良い例の参照
    output += "### 参考実装\n"
    output += "- 認証機能の実装例: `examples/good/auth_example.py`\n"
    output += "- アンチパターン: `examples/bad/auth_antipattern.py`\n"
    
    return output

if __name__ == "__main__":
    # 初回実行時にインデックスを作成
    print("DocuMentor MCP Server を起動中...")
    try:
        get_indexer()
        print("準備が完了しました。Claude Codeからの接続を待機しています。")
    except Exception as e:
        print(f"初期化エラー: {e}")
    
    # MCPサーバーを起動
    mcp.run(transport="stdio")