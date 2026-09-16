"""PPTX extractors — MarkItDown then python-pptx."""

from __future__ import annotations

from backend.app.models.blocks import Block
from backend.app.pipeline.context import ExtractContext
from backend.app.pipeline.extractors.base import markdown_result, run_sync


class MarkItDownPptxExtractor:
    name = "markitdown"
    supports = {
        "pptx",
        ".pptx",
        "xlsx",
        ".xlsx",
        "xls",
        ".xls",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    }

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        if not ctx.file_path:
            raise ValueError("MarkItDownPptxExtractor requires file_path")

        from markitdown import MarkItDown
        import converter

        def _run() -> str:
            mid = MarkItDown()
            result = mid.convert(ctx.file_path)
            md = result.text_content or ""
            md = converter._collapse_spaced_caps(md)
            md = converter._strip_page_numbers(md)
            return md

        md = await run_sync(_run)
        if not md or len(md.strip()) < 10:
            raise RuntimeError("MarkItDown produced empty/minimal output")
        return markdown_result(md, self.name)


class PythonPptxExtractor:
    name = "python_pptx"
    supports = {
        "pptx",
        ".pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        if not ctx.file_path:
            raise ValueError("PythonPptxExtractor requires file_path")

        import converter

        def _run() -> str:
            md = converter._extract_pptx_legacy(ctx.file_path)
            md = converter._collapse_spaced_caps(md)
            md = converter._strip_page_numbers(md)
            return md

        md = await run_sync(_run)
        return markdown_result(md, self.name)
