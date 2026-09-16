"""Pipeline integration — text/html extractors + orchestrator without Docling."""

from __future__ import annotations

import asyncio

from backend.app.pipeline.adapters.file import build_text_context
from backend.app.pipeline.orchestrator import run_pipeline


def test_orchestrator_plain_text():
    async def _run():
        ctx, source = build_text_context(
            "Enough plain text content for the quality gate to pass easily.",
            fmt="plain",
        )
        return await run_pipeline(ctx, source=source)

    result = asyncio.run(_run())
    assert result.status == "completed"
    assert result.blocks
    assert result.pipeline_trace
    assert result.pipeline_trace[0].status == "success"
    assert "quality gate" not in (result.error or "")


def test_orchestrator_markdown():
    async def _run():
        ctx, source = build_text_context(
            "# Doc\n\n## Section\n\nParagraph with substance.\n\n- item one\n- item two\n",
            fmt="markdown",
        )
        return await run_pipeline(ctx, source=source)

    result = asyncio.run(_run())
    assert result.status == "completed"
    types = {b.type.value for b in result.blocks}
    assert "heading" in types
    assert result.markdown.startswith("#")


def test_orchestrator_html():
    async def _run():
        html = """
        <html><head><title>Page</title></head>
        <body><h1>Welcome</h1><p>This is the main article content for testing HTML extraction.</p></body></html>
        """
        ctx, source = build_text_context(html, fmt="html")
        return await run_pipeline(ctx, source=source)

    result = asyncio.run(_run())
    assert result.status == "completed"
    assert len(result.markdown) >= 10
