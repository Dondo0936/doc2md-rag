"""Legacy PDF extraction via pymupdf4llm + fitz tables."""

from __future__ import annotations

from backend.app.models.blocks import Block
from backend.app.pipeline.context import ExtractContext
from backend.app.pipeline.extractors.base import markdown_result, run_sync


class PdfLegacyExtractor:
    name = "pymupdf4llm"
    supports = {"pdf", ".pdf", "application/pdf"}

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        if not ctx.file_path:
            raise ValueError("PdfLegacyExtractor requires file_path")

        import converter

        def _run() -> str:
            md = converter._extract_pdf(ctx.file_path, embed_images=bool(ctx.llm_client))
            md = converter._collapse_spaced_caps(md)
            md = converter._strip_page_numbers(md)
            if ctx.use_ocr:
                # Scanned PDFs without OCR — surface a warning via metadata later
                pass
            return md

        md = await run_sync(_run)
        blocks = markdown_result(md, self.name)
        if ctx.use_ocr:
            for b in blocks:
                b.metadata = dict(b.metadata or {})
                b.metadata["warning"] = "scanned_pdf_without_ocr_fallback"
        return blocks
