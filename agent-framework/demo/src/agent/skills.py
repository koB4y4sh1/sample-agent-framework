from __future__ import annotations

from pathlib import Path

from agent_framework import SkillsProvider


class DemoSkills:
    """デモ用 skills provider を構築するクラス。"""

    def __init__(self, skill_root: Path | None = None) -> None:
        self._skill_root = skill_root or (Path(__file__).parent  / "skills")
        self._skill_root.mkdir(parents=True, exist_ok=True)

    def build_provider(self) -> SkillsProvider:
        """demo 用の SkillsProvider を構築して返す。"""
        return SkillsProvider.from_paths(
            skill_paths=self._skill_root,
            source_id="demo_skills",
        )

    def describe(self) -> str:
        """ロード済み skill の概要を返す。"""
        skill_names = sorted(path.name for path in self._skill_root.iterdir() if path.is_dir())
        if not skill_names:
            return "No skills are loaded."
        return f"Loaded skills: {', '.join(skill_names)}"
