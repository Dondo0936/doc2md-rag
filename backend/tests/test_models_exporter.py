"""Unit tests: models, markdown exporter, quality gate, retry."""

from __future__ import annotations

import asyncio

import pytest

from backend.app.exporters.markdown import (
    blocks_to_markdown,
    compute_stats,
    markdown_to_blocks,
)
from backend.app.models.blocks import Block, BlockType
from backend.app.pipeline.quality import passes_quality, total_text_chars
from backend.app.pipeline.retry import with_retries


class TestMarkdownRoundTrip:
    def test_heading_and_paragraph(self):
        md = "# Title\n\nHello world.\n"
        blocks = markdown_to_blocks(md, extractor="test")
        assert blocks[0].type == BlockType.HEADING
        assert blocks[0].level == 1
        assert blocks[0].text == "Title"
        assert blocks[1].type == BlockType.PARAGRAPH
        assert "Hello" in (blocks[1].text or "")
        assert blocks[0].metadata.get("extractor") == "test"

    def test_table_to_structured_rows(self):
        md = "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n"
        blocks = markdown_to_blocks(md)
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.TABLE
        assert blocks[0].rows is not None
        assert blocks[0].rows[0] == ["Name", "Age"]
        assert ["Alice", "30"] in blocks[0].rows

    def test_list_items(self):
        md = "- one\n- two\n"
        blocks = markdown_to_blocks(md)
        assert blocks[0].type == BlockType.LIST
        assert blocks[0].items == ["one", "two"]

    def test_export_heading(self):
        blocks = [
            Block(id="a", type=BlockType.HEADING, text="Hi", level=2),
            Block(id="b", type=BlockType.PARAGRAPH, text="Body"),
        ]
        md = blocks_to_markdown(blocks)
        assert md.startswith("## Hi")
        assert "Body" in md

    def test_export_table_as_lists(self):
        blocks = [
            Block(
                id="t",
                type=BlockType.TABLE,
                rows=[["Name", "Age"], ["Alice", "30"], ["Bob", "25"]],
            )
        ]
        md = blocks_to_markdown(blocks, tables_as_lists=True)
        assert "Alice" in md
        assert "| --- |" not in md

    def test_stats(self):
        blocks = markdown_to_blocks("# A\n\nPara\n\n- x\n")
        md = blocks_to_markdown(blocks)
        stats = compute_stats(blocks, md)
        assert stats.char_count == len(md)
        assert stats.block_counts.get("heading", 0) >= 1


class TestQualityGate:
    def test_rejects_empty(self):
        assert not passes_quality([])

    def test_rejects_trivial(self):
        blocks = [Block(id="1", type=BlockType.PARAGRAPH, text="hi")]
        assert not passes_quality(blocks, min_chars=10)

    def test_accepts_content(self):
        blocks = [Block(id="1", type=BlockType.PARAGRAPH, text="hello world this is enough")]
        assert passes_quality(blocks, min_chars=10)
        assert total_text_chars(blocks) >= 10

    def test_accepts_image_only(self):
        blocks = [Block(id="1", type=BlockType.IMAGE, text="chart", image_ref="data:x")]
        assert passes_quality(blocks, min_chars=10)


class TestRetry:
    def test_succeeds_first_try(self):
        async def ok():
            return 42

        assert asyncio.run(with_retries(ok, attempts=3)) == 42

    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("nope")
            return "ok"

        result = asyncio.run(with_retries(flaky, attempts=3, base_delay_s=0.01))
        assert result == "ok"
        assert calls["n"] == 3

    def test_exhausts_attempts(self):
        async def always_fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            asyncio.run(with_retries(always_fail, attempts=2, base_delay_s=0.01))
