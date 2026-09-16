"""CSV / spreadsheet extractors via pandas."""

from __future__ import annotations

from backend.app.models.blocks import Block
from backend.app.pipeline.context import ExtractContext
from backend.app.pipeline.extractors.base import markdown_result, run_sync


class PandasCsvExtractor:
    name = "pandas_csv"
    supports = {"csv", ".csv", "text/csv", "application/csv"}

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        if not ctx.file_path:
            raise ValueError("PandasCsvExtractor requires file_path")

        import converter

        md = await run_sync(converter._extract_csv, ctx.file_path)
        return markdown_result(md, self.name)


class PandasExcelExtractor:
    name = "pandas_excel"
    supports = {
        "xlsx",
        ".xlsx",
        "xls",
        ".xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    }

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        if not ctx.file_path:
            raise ValueError("PandasExcelExtractor requires file_path")

        import pandas as pd

        def _run() -> str:
            # Prefer openpyxl/xlrd via pandas; first sheet only for MVP
            df = pd.read_excel(ctx.file_path)
            return df.to_markdown(index=False)

        md = await run_sync(_run)
        return markdown_result(md, self.name)
