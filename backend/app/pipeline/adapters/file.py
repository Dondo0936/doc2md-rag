"""File / text / image input adapters that build ExtractContext + SourceInfo."""

from __future__ import annotations

import mimetypes
import os
import tempfile
import uuid
from pathlib import Path

from backend.app.config import Settings, get_settings
from backend.app.models.blocks import SourceInfo
from backend.app.pipeline.adapters.url import FetchedResource, fetch_url
from backend.app.pipeline.context import ExtractContext

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_EXTS = {".txt", ".md", ".markdown"}
HTML_EXTS = {".html", ".htm"}


def _guess_mime(filename: str, fallback: str | None = None) -> str | None:
    mime, _ = mimetypes.guess_type(filename)
    return mime or fallback


def build_file_context(
    *,
    file_path: str,
    filename: str,
    size_bytes: int | None = None,
    mime: str | None = None,
    settings: Settings | None = None,
) -> tuple[ExtractContext, SourceInfo]:
    settings = settings or get_settings()
    size_bytes = size_bytes if size_bytes is not None else os.path.getsize(file_path)
    if size_bytes > settings.max_file_bytes:
        raise ValueError(
            f"File too large ({size_bytes / 1024 / 1024:.1f}MB). "
            f"Maximum is {settings.parser_max_file_mb}MB."
        )

    ext = Path(filename).suffix.lower()
    mime = mime or _guess_mime(filename)

    if ext in IMAGE_EXTS or (mime and mime.startswith("image/")):
        kind = "image"
        input_type = "image"
    elif ext in HTML_EXTS or mime == "text/html":
        kind = "html"
        input_type = "file"
    else:
        kind = "file"
        input_type = "file"

    ctx = ExtractContext(
        input_kind=kind,  # type: ignore[arg-type]
        file_path=file_path,
        filename=filename,
        mime=mime,
        extension=ext,
        size_bytes=size_bytes,
    )
    # For plain text / markdown files, also load text for extractors that prefer it
    if ext in TEXT_EXTS:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            ctx.text = f.read()
        ctx.input_kind = "file"
        if ext in {".md", ".markdown"}:
            ctx.extra["format"] = "markdown"
        else:
            ctx.extra["format"] = "plain"

    source = SourceInfo(
        input_type=input_type,  # type: ignore[arg-type]
        filename=filename,
        mime=mime,
        size_bytes=size_bytes,
    )
    return ctx, source


def build_text_context(
    text: str,
    *,
    fmt: str = "plain",
) -> tuple[ExtractContext, SourceInfo]:
    if text is None:
        raise ValueError("text is required")
    fmt = (fmt or "plain").lower()
    if fmt not in {"plain", "markdown", "html"}:
        raise ValueError("format must be plain, markdown, or html")

    ctx = ExtractContext(
        input_kind="text",
        text=text,
        html=text if fmt == "html" else None,
        mime={
            "plain": "text/plain",
            "markdown": "text/markdown",
            "html": "text/html",
        }[fmt],
        size_bytes=len(text.encode("utf-8")),
        extra={"format": fmt},
    )
    source = SourceInfo(
        input_type="text",
        mime=ctx.mime,
        size_bytes=ctx.size_bytes,
        format=fmt,  # type: ignore[arg-type]
    )
    return ctx, source


async def build_url_context(
    url: str,
    *,
    settings: Settings | None = None,
) -> tuple[ExtractContext, SourceInfo, str]:
    """Fetch URL and return (ctx, source, temp_path). Caller must clean temp_path if set."""
    settings = settings or get_settings()
    resource: FetchedResource = await fetch_url(url, settings=settings)

    ext = resource.extension or ""
    mime = resource.content_type
    tmp_path = ""

    # HTML pages → text/html chain without writing file (unless huge)
    if mime == "text/html" or ext in {".html", ".htm"}:
        try:
            html = resource.content.decode("utf-8")
        except UnicodeDecodeError:
            html = resource.content.decode("utf-8", errors="replace")
        ctx = ExtractContext(
            input_kind="html",
            html=html,
            text=html,
            url=resource.final_url,
            filename=resource.filename,
            mime="text/html",
            extension=".html",
            size_bytes=len(resource.content),
        )
        source = SourceInfo(
            input_type="url",
            url=resource.final_url,
            filename=resource.filename,
            mime="text/html",
            size_bytes=len(resource.content),
        )
        return ctx, source, tmp_path

    # Otherwise write to temp file and treat as file/image
    suffix = ext or ""
    fd, tmp_path = tempfile.mkstemp(prefix="parser_url_", suffix=suffix)
    os.close(fd)
    Path(tmp_path).write_bytes(resource.content)

    ctx, source = build_file_context(
        file_path=tmp_path,
        filename=resource.filename,
        size_bytes=len(resource.content),
        mime=mime,
        settings=settings,
    )
    ctx.url = resource.final_url
    source = SourceInfo(
        input_type="url",
        url=resource.final_url,
        filename=resource.filename,
        mime=mime,
        size_bytes=len(resource.content),
    )
    return ctx, source, tmp_path


def write_upload_temp(content: bytes, filename: str) -> str:
    """Persist an upload to a temp file; return path."""
    suffix = Path(filename).suffix or ""
    fd, path = tempfile.mkstemp(prefix=f"parser_{uuid.uuid4().hex[:8]}_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return path
