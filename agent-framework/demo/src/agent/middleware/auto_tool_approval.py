from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agent_framework import Content

ApprovalScope = Literal["tool_with_arguments", "tool"]

ALLOW_TOOLS_FILE_NAME = "allow_tools.json"
DEFAULT_MEMORY_ROOT_DIR = Path(__file__).parents[3] / ".memory"


@dataclass(frozen=True, slots=True)
class AutoToolApprovalConfig:
    """ユーザーが明示許可したTool実行を永続的に自動承認する設定。

    想定ユースケース:
    - 承認画面で「同じ引数なら今後も許可」または「このTool自体を今後も許可」を選んだ内容を、
      セッションをまたいで `.memory/{user_name}/allow_tools.json` に保持したい。
    - 次回以降の `ToolApprovalMiddleware` の承認要求に対し、保存済みルールと一致するTool callだけを
      自動承認し、危険な未許可操作は引き続き人間確認に戻したい。

    user_name が未指定の場合は OS 環境変数 `USERNAME` / `USER` / `LOGNAME` を順に使い、
    取得できない場合は `demouser` を使う。
    """

    user_name: str | None = None
    memory_root_dir: Path | None = None


def build_auto_tool_approval_rule(
    config: AutoToolApprovalConfig | None = None,
) -> Callable[[Content], bool]:
    """保存済み allow_tools.json に基づく `ToolApprovalMiddleware` callback を作る。

    想定ユースケース:
    - 以前ユーザーが「引数の実行を許可する」を選んだTool callは、同一Tool名かつ同一引数の時だけ自動承認する。
    - 以前ユーザーが「このツール実行を自動許可する」を選んだTool callは、同一Tool名の時に自動承認する。
    - 保存済みルールに存在しないTool callは承認画面に戻す。
    """

    store = AllowToolsStore(config or AutoToolApprovalConfig())

    def approve_allowed_tool_call(function_call: Content) -> bool:
        return store.is_allowed(function_call)

    return approve_allowed_tool_call


class AllowToolsStore:
    """`.memory/{user_name}/allow_tools.json` の読書きを担う永続ストア。"""

    def __init__(self, config: AutoToolApprovalConfig | None = None) -> None:
        self.config = config or AutoToolApprovalConfig()

    @property
    def path(self) -> Path:
        user_name = _safe_user_name(self.config.user_name or _default_user_name())
        memory_root_dir = self.config.memory_root_dir or _default_memory_root_dir()
        return memory_root_dir / user_name / ALLOW_TOOLS_FILE_NAME

    def add_request(self, request: Content, scope: ApprovalScope) -> None:
        function_call = request.function_call
        if function_call is None:
            return
        self.add_function_call(function_call, scope)

    def add_function_call(self, function_call: Content, scope: ApprovalScope) -> None:
        rule = _rule_from_function_call(function_call, scope)
        if rule is None:
            return

        payload = self._load_payload()
        rules = payload.setdefault("rules", [])
        if not isinstance(rules, list):
            rules = []
            payload["rules"] = rules

        if not any(_same_rule(rule, existing) for existing in rules if isinstance(existing, Mapping)):
            rules.append(rule)
            self._save_payload(payload)

    def is_allowed(self, function_call: Content) -> bool:
        tool_name = _tool_call_name(function_call)
        if tool_name is None:
            return False

        payload = self._load_payload()
        rules = payload.get("rules", [])
        if not isinstance(rules, list):
            return False

        for raw_rule in rules:
            if not isinstance(raw_rule, Mapping):
                continue
            if _rule_matches(raw_rule, function_call):
                return True
        return False

    def _load_payload(self) -> dict[str, Any]:
        path = self.path
        if not path.exists():
            return {"version": 1, "rules": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "rules": []}
        return data if isinstance(data, dict) else {"version": 1, "rules": []}

    def _save_payload(self, payload: Mapping[str, Any]) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def allow_tool_request(request: Content, scope: ApprovalScope, config: AutoToolApprovalConfig | None = None) -> None:
    """承認画面で永続許可が選ばれたTool requestを allow_tools.json に保存する。"""

    AllowToolsStore(config).add_request(request, scope)


def _rule_from_function_call(function_call: Content, scope: ApprovalScope) -> dict[str, Any] | None:
    tool_name = _tool_call_name(function_call)
    if tool_name is None:
        return None
    rule: dict[str, Any] = {
        "scope": scope,
        "tool_name": tool_name,
        "server_label": _server_label(function_call),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if scope == "tool_with_arguments":
        rule["arguments"] = _serialize_arguments(function_call)
    return rule


def _rule_matches(rule: Mapping[str, Any], function_call: Content) -> bool:
    tool_name = _tool_call_name(function_call)
    if tool_name is None or rule.get("tool_name") != tool_name:
        return False
    if rule.get("server_label") != _server_label(function_call):
        return False

    scope = rule.get("scope")
    if scope == "tool":
        return True
    if scope == "tool_with_arguments":
        return rule.get("arguments") == _serialize_arguments(function_call)
    return False


def _same_rule(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("scope") == right.get("scope")
        and left.get("tool_name") == right.get("tool_name")
        and left.get("server_label") == right.get("server_label")
        and left.get("arguments") == right.get("arguments")
    )


def _serialize_arguments(function_call: Content) -> dict[str, str]:
    arguments = function_call.parse_arguments()
    return {
        str(key): json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        for key, value in dict(arguments or {}).items()
    }


def _server_label(function_call: Content) -> str | None:
    value = function_call.additional_properties.get("server_label")
    return value if isinstance(value, str) else None


def _tool_call_name(function_call: Content) -> str | None:
    name = getattr(function_call, "name", None)
    return str(name) if isinstance(name, str) and name.strip() else None


def _default_user_name() -> str:
    for env_name in ("USERNAME", "USER", "LOGNAME"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return "demouser"


def _default_memory_root_dir() -> Path:
    value = os.getenv("ALLOW_TOOLS_MEMORY_ROOT", "").strip()
    return Path(value) if value else DEFAULT_MEMORY_ROOT_DIR


def _safe_user_name(user_name: str) -> str:
    safe = user_name.strip().replace("/", "_").replace("\\", "_")
    return safe or "demouser"
