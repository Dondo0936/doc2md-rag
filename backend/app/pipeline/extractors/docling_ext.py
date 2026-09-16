"""Docling-backed extractors (primary for PDF/DOCX/PPTX/images)."""

from __future__ import annotations

from backend.app.pipeline.context import ExtractContext
from backend.app.pipeline.extractors.base import markdown_result, run_sync
from backend.app.models.blocks import Block


class DoclingExtractor:
    """Primary Docling extraction — wraps converter._extract_with_docling."""

    name = "docling"
    supports = {
        "pdf",
        ".pdf",
        "docx",
        ".docx",
        "pptx",
        ".pptx",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "image",
        "png",
        "jpg",
        "jpeg",
        "webp",
        "image/png",
        "image/jpeg",
        "image/webp",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    def __init__(self, *, force_ocr: bool | None = None, name: str | None = None) -> None:
        self._force_ocr = force_ocr
        if name:
            self.name = name

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        if not ctx.file_path:
            raise ValueError("DoclingExtractor requires file_path")

        import converter

        use_ocr = self._force_ocr if self._force_ocr is not None else ctx.use_ocr
        describe = ctx.describe_images

        def _run() -> str:
            md = converter._extract_with_docling(
                ctx.file_path,
                ocr=use_ocr,
                describe_images=describe,
            )
            md = converter._collapse_spaced_caps(md)
            md = converter._strip_page_numbers(md)
            return md

        md = await run_sync(_run)
        return markdown_result(md, self.name)


class DoclingOcrExtractor(DoclingExtractor):
    """Docling with OCR forced on (scanned PDFs / images)."""

    name = "docling_ocr"

    def __init__(self) -> None:
        super().__init__(force_ocr=True, name="docling_ocr")
