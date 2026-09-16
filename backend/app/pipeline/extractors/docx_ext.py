"""DOCX extractors — pypandoc then python-docx."""

from __future__ import annotations

from backend.app.models.blocks import Block
from backend.app.pipeline.context import ExtractContext
from backend.app.pipeline.extractors.base import markdown_result, run_sync


class PandocDocxExtractor:
    name = "pypandoc"
    supports = {
        "docx",
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        if not ctx.file_path:
            raise ValueError("PandocDocxExtractor requires file_path")

        import pypandoc
        import converter

        def _run() -> str:
            md = pypandoc.convert_file(
                ctx.file_path,
                "gfm",
                extra_args=["--track-changes=accept", "--wrap=none", "--markdown-headings=atx"],
            )
            md = converter._html_tables_to_pipe(md)
            md = converter._collapse_spaced_caps(md)
            md = converter._strip_page_numbers(md)
            return md

        md = await run_sync(_run)
        return markdown_result(md, self.name)


class PythonDocxExtractor:
    name = "python_docx"
    supports = {
        "docx",
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        if not ctx.file_path:
            raise ValueError("PythonDocxExtractor requires file_path")

        import converter

        def _run() -> str:
            md = converter._extract_docx_legacy(ctx.file_path)
            md = converter._collapse_spaced_caps(md)
            md = converter._strip_page_numbers(md)
            return md

        md = await run_sync(_run)
        return markdown_result(md, self.name)
