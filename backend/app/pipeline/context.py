"""Extraction context passed to every extractor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class ExtractContext:
    """Inputs and options for a single extraction attempt."""

    input_kind: Literal["file", "text", "url", "image", "html"]
    file_path: str | None = None
    text: str | None = None
    html: str | None = None
    url: str | None = None
    filename: str | None = None
    mime: str | None = None
    extension: str | None = None
    size_bytes: int | None = None
    use_ocr: bool = False
    describe_images: bool = False
    llm_client: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_extension(self) -> str:
        if self.extension:
            ext = self.extension.lower()
            return ext if ext.startswith(".") else f".{ext}"
        if self.filename:
            return Path(self.filename).suffix.lower()
        if self.file_path:
            return Path(self.file_path).suffix.lower()
        return ""
