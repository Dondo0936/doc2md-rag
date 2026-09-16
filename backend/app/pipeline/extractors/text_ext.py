"""Plain text / markdown / HTML extractors."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from backend.app.models.blocks import Block
from backend.app.pipeline.context import ExtractContext
from backend.app.pipeline.extractors.base import markdown_result


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self.parts)).strip()


def _html_to_text(html: str) -> str:
    try:
        import trafilatura

        extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
        if extracted and extracted.strip():
            return extracted.strip()
    except Exception:
        pass

    try:
        from readability import Document

        doc = Document(html)
        title = doc.title() or ""
        summary_html = doc.summary()
        stripper = _HTMLStripper()
        stripper.feed(summary_html)
        body = stripper.get_text()
        if title and body:
            return f"# {title}\n\n{body}"
        return body or title
    except Exception:
        pass

    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


class PlainTextExtractor:
    name = "plain_text"
    supports = {"text", "plain", "txt", ".txt", "text/plain"}

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        text = ctx.text
        if text is None and ctx.file_path:
            with open(ctx.file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        if not text or not text.strip():
            raise RuntimeError("Empty text input")
        return markdown_result(text.strip(), self.name)


class MarkdownTextExtractor:
    name = "markdown_text"
    supports = {"markdown", "md", ".md", "text/markdown"}

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        text = ctx.text
        if text is None and ctx.file_path:
            with open(ctx.file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        if not text or not text.strip():
            raise RuntimeError("Empty markdown input")
        return markdown_result(text, self.name)


class HtmlExtractor:
    name = "html_trafilatura"
    supports = {"html", ".html", ".htm", "text/html", "application/xhtml+xml"}

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        html = ctx.html
        if html is None and ctx.text:
            html = ctx.text
        if html is None and ctx.file_path:
            with open(ctx.file_path, "r", encoding="utf-8", errors="replace") as f:
                html = f.read()
        if not html or not html.strip():
            raise RuntimeError("Empty HTML input")
        md = _html_to_text(html)
        if not md or len(md.strip()) < 10:
            raise RuntimeError("HTML extraction produced empty/minimal output")
        return markdown_result(md, self.name)


class HtmlRawFallbackExtractor:
    name = "html_raw"
    supports = {"html", ".html", ".htm", "text/html", "application/xhtml+xml"}

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        html = ctx.html or ctx.text
        if html is None and ctx.file_path:
            with open(ctx.file_path, "r", encoding="utf-8", errors="replace") as f:
                html = f.read()
        if not html:
            raise RuntimeError("Empty HTML input")
        stripper = _HTMLStripper()
        stripper.feed(html)
        text = stripper.get_text()
        if not text:
            raise RuntimeError("Raw HTML strip produced empty output")
        return markdown_result(text, self.name)
