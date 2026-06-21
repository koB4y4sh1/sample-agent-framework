from __future__ import annotations

from pathlib import Path

from agent_framework import Content

from agent.contexts import create_cu_attachment_content


class AttachmentBuffer:
    """次のユーザーメッセージ送信まで添付ファイルを保持する。"""

    def __init__(self) -> None:
        self._contents: list[Content] = []

    def add_image(self, path: str) -> None:
        self._add_file(path)

    def add_file(self, path: str) -> None:
        self._add_file(path)

    def consume(self) -> list[Content]:
        contents = list(self._contents)
        self._contents.clear()
        return contents

    def clear(self) -> None:
        self._contents.clear()

    @property
    def size(self) -> int:
        return len(self._contents)

    def _add_file(self, path: str) -> None:
        file_path = Path(path)
        media_type = get_media_type(file_path)
        data = file_path.read_bytes()
        self._contents.append(
            create_cu_attachment_content(
                name=file_path.name,
                media_type=media_type,
                data=data,
            )
        )


def get_media_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if suffix == ".pptx":
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    raise ValueError(f"Unsupported attachment type: {file_path.suffix}")
