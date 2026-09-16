"""Extractor registry — maps (input_kind, mime/ext) → ordered chain + retries."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.pipeline.context import ExtractContext
from backend.app.pipeline.extractors.base import Extractor
from backend.app.pipeline.extractors.csv_ext import PandasCsvExtractor, PandasExcelExtractor
from backend.app.pipeline.extractors.docling_ext import DoclingExtractor, DoclingOcrExtractor
from backend.app.pipeline.extractors.docx_ext import PandocDocxExtractor, PythonDocxExtractor
from backend.app.pipeline.extractors.image_ext import ClaudeVisionImageExtractor, RapidOcrImageExtractor
from backend.app.pipeline.extractors.pdf_legacy import PdfLegacyExtractor
from backend.app.pipeline.extractors.pptx_ext import MarkItDownPptxExtractor, PythonPptxExtractor
from backend.app.pipeline.extractors.text_ext import (
    HtmlExtractor,
    HtmlRawFallbackExtractor,
    MarkdownTextExtractor,
    PlainTextExtractor,
)


@dataclass(frozen=True)
class ChainStep:
    extractor: Extractor
    retries: int  # total attempts (1 = no retry)


# Shared singleton extractor instances
_DOCLING = DoclingExtractor()
_DOCLING_OCR = DoclingOcrExtractor()
_PDF_LEGACY = PdfLegacyExtractor()
_PANDOC = PandocDocxExtractor()
_PYTHON_DOCX = PythonDocxExtractor()
_MARKITDOWN = MarkItDownPptxExtractor()
_PYTHON_PPTX = PythonPptxExtractor()
_PANDAS_CSV = PandasCsvExtractor()
_PANDAS_XLSX = PandasExcelExtractor()
_PLAIN = PlainTextExtractor()
_MD = MarkdownTextExtractor()
_HTML = HtmlExtractor()
_HTML_RAW = HtmlRawFallbackExtractor()
_RAPIDOCR = RapidOcrImageExtractor()
_CLAUDE_VISION = ClaudeVisionImageExtractor()


def _ext(ctx: ExtractContext) -> str:
    return ctx.resolve_extension().lstrip(".").lower()


def resolve_chain(ctx: ExtractContext) -> list[ChainStep]:
    """Return ordered extractor chain for this context."""
    ext = _ext(ctx)
    mime = (ctx.mime or "").lower()
    kind = ctx.input_kind

    # Text paste
    if kind == "text":
        fmt = (ctx.extra.get("format") or "plain").lower()
        if fmt == "html":
            return [ChainStep(_HTML, 1), ChainStep(_HTML_RAW, 1)]
        if fmt == "markdown":
            return [ChainStep(_MD, 1)]
        return [ChainStep(_PLAIN, 1)]

    # HTML content (from URL page)
    if kind == "html" or ext in {"html", "htm"} or "text/html" in mime:
        return [ChainStep(_HTML, 1), ChainStep(_HTML_RAW, 1)]

    # Images
    if kind == "image" or ext in {"png", "jpg", "jpeg", "webp"} or mime.startswith("image/"):
        return [
            ChainStep(_DOCLING_OCR, 1),
            ChainStep(_RAPIDOCR, 1),
            ChainStep(_CLAUDE_VISION, 1),
        ]

    # CSV
    if ext == "csv" or "csv" in mime:
        return [ChainStep(_PANDAS_CSV, 1)]

    # Excel
    if ext in {"xlsx", "xls"} or "spreadsheet" in mime or mime.endswith("ms-excel"):
        return [ChainStep(_MARKITDOWN, 1), ChainStep(_PANDAS_XLSX, 1)]

    # Plain / markdown files
    if ext in {"txt"} or mime == "text/plain":
        return [ChainStep(_PLAIN, 1)]
    if ext in {"md", "markdown"} or mime == "text/markdown":
        return [ChainStep(_MD, 1)]

    # PDF
    if ext == "pdf" or mime == "application/pdf":
        if ctx.use_ocr:
            return [
                ChainStep(_DOCLING_OCR, 2),
                ChainStep(_PDF_LEGACY, 1),
            ]
        return [
            ChainStep(_DOCLING, 2),
            ChainStep(_PDF_LEGACY, 1),
        ]

    # DOCX
    if ext == "docx" or "wordprocessingml" in mime:
        return [
            ChainStep(_DOCLING, 2),
            ChainStep(_PANDOC, 1),
            ChainStep(_PYTHON_DOCX, 1),
        ]

    # PPTX
    if ext == "pptx" or "presentationml" in mime:
        return [
            ChainStep(_DOCLING, 2),
            ChainStep(_MARKITDOWN, 1),
            ChainStep(_PYTHON_PPTX, 1),
        ]

    raise ValueError(f"Unsupported input type: ext={ext!r} mime={mime!r} kind={kind!r}")


def list_extractors() -> list[str]:
    return [
        _DOCLING.name,
        _DOCLING_OCR.name,
        _PDF_LEGACY.name,
        _PANDOC.name,
        _PYTHON_DOCX.name,
        _MARKITDOWN.name,
        _PYTHON_PPTX.name,
        _PANDAS_CSV.name,
        _PANDAS_XLSX.name,
        _PLAIN.name,
        _MD.name,
        _HTML.name,
        _HTML_RAW.name,
        _RAPIDOCR.name,
        _CLAUDE_VISION.name,
    ]
