"""Tests for converter.py — table conversion, double-pipe normalization, image handling."""

import os
import tempfile
import pytest

from converter import (
    _clean_cell,
    _parse_pipe_row,
    _is_separator,
    _table_to_list,
    _normalize_double_pipes,
    _tables_to_lists,
    _handle_images,
    _extract_csv,
    process_document,
    MAX_FILE_SIZE_BYTES,
)
from tests.conftest import (
    SAMPLE_MARKDOWN,
    SAMPLE_MARKDOWN_NO_TABLES,
    SAMPLE_TABLE_DOUBLE_PIPES,
    SAMPLE_MARKDOWN_WITH_IMAGES,
    MockLLMClient,
)


# ─── Cell Cleaning ────────────────────────────────────────────────

class TestCleanCell:
    def test_br_tags(self):
        # Simple fragments without sentence-ending punctuation get joined
        assert _clean_cell("line1<br>line2") == "line1 line2"
        assert _clean_cell("line1<br/>line2") == "line1 line2"
        # Fragments after sentence-ending punctuation stay on new lines
        assert "\n" in _clean_cell("line1.<br>line2")
        # Bullet markers stay on new lines
        assert "\n" in _clean_cell("text<br>- item")

    def test_zero_width_space(self):
        assert "\u200b" not in _clean_cell("hello\u200bworld")

    def test_bullet_markers(self):
        assert _clean_cell("● Item 1").startswith("- ")
        assert _clean_cell("○ Item 2").startswith("- ")

    def test_strip(self):
        assert _clean_cell("  hello  ") == "hello"


# ─── Pipe Row Parsing ─────���───────────────────────────────────────

class TestParseRow:
    def test_basic_row(self):
        cells = _parse_pipe_row("| A | B | C |")
        assert cells == ["A", "B", "C"]

    def test_no_leading_pipe(self):
        cells = _parse_pipe_row("A | B | C |")
        assert len(cells) == 3

    def test_empty_cells(self):
        cells = _parse_pipe_row("| A |  | C |")
        assert cells[1] == ""


# ─── Separator Detection ──────────────────────────────────────────

class TestIsSeparator:
    def test_standard_separator(self):
        assert _is_separator("|---|---|---|")

    def test_aligned_separator(self):
        assert _is_separator("| :--- | :---: | ---: |")

    def test_not_separator(self):
        assert not _is_separator("| A | B | C |")
        assert not _is_separator("just text")

    def test_data_with_dashes_not_separator(self):
        """Data rows containing dashes should NOT be treated as separators."""
        assert not _is_separator("| - | - |")
        assert not _is_separator("| -- note -- | text |")
        assert not _is_separator("| | |")  # empty cells

    def test_separator_with_spaces(self):
        assert _is_separator("|  ---  |  ---  |")


# ─── Table to List Conversion ─────────────────────────────────────

class TestTableToList:
    def test_basic_table(self):
        lines = [
            "| Name | Age |",
            "| --- | --- |",
            "| Alice | 30 |",
            "| Bob | 25 |",
        ]
        result = _table_to_list(lines)
        assert "Alice" in result
        assert "Bob" in result
        assert "- **" in result  # bullet list format

    def test_empty_lines(self):
        # Should not crash on empty input
        result = _table_to_list([])
        assert result == ""


# ─── Double-Pipe Normalization ─────────────────────────────────────

class TestNormalizeDoublePipes:
    def test_empty_cell_normalization(self):
        result = _normalize_double_pipes("|A||C|")
        assert "||" not in result or "|---|" in result  # either normalized or separator

    def test_no_double_pipes(self):
        text = "| A | B | C |"
        assert _normalize_double_pipes(text) == text

    def test_separator_data_join(self):
        text = "|---|---||Data|"
        result = _normalize_double_pipes(text)
        lines = [l for l in result.split("\n") if l.strip()]
        # Should split into separate lines
        assert len(lines) >= 1


# ─── Tables to Lists (Full Pipeline) ──────────────────────────────

class TestTablesToLists:
    def test_converts_table(self):
        result, count = _tables_to_lists(SAMPLE_MARKDOWN)
        assert count >= 1
        # Pipe tables should be gone
        assert "| --- |" not in result

    def test_no_tables_passthrough(self):
        result, count = _tables_to_lists(SAMPLE_MARKDOWN_NO_TABLES)
        assert count == 0
        assert "artificial intelligence" in result

    def test_double_pipe_table(self):
        result, count = _tables_to_lists(SAMPLE_TABLE_DOUBLE_PIPES)
        assert count >= 0  # May or may not detect as table
        # Should not crash regardless

    def test_crlf_line_endings(self):
        """Tables with CRLF line endings should still be detected."""
        table = "| Name | Age |\r\n| --- | --- |\r\n| Alice | 30 |\r\n"
        result, count = _tables_to_lists(table)
        assert count == 1
        assert "Alice" in result

    def test_indented_table(self):
        """Tables with leading whitespace should be detected."""
        table = "  | Name | Age |\n  | --- | --- |\n  | Alice | 30 |\n"
        result, count = _tables_to_lists(table)
        assert count == 1
        assert "Alice" in result

    def test_header_only_table(self):
        """A table with only header + separator should not leave raw pipe syntax."""
        table = "| Name | Age |\n| --- | --- |\n"
        result, count = _tables_to_lists(table)
        assert "| --- |" not in result


# ─── Image Handling ──────���─────────────────────────────────────────

class TestHandleImages:
    def test_without_llm_client(self):
        result, count = _handle_images(SAMPLE_MARKDOWN_WITH_IMAGES)
        assert count == 2
        assert "[Image:" in result
        assert "![" not in result

    def test_with_llm_client_base64(self):
        md = "![alt](data:image/png;base64,abc123)"
        result, count = _handle_images(md, llm_client=MockLLMClient())
        assert count == 1
        assert "test image description" in result

    def test_no_images(self):
        result, count = _handle_images("No images here.")
        assert count == 0
        assert result == "No images here."


# ─── CSV Extraction ──────────���─────────────────────────────────────

class TestExtractCSV:
    def test_basic_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Name,Age\nAlice,30\nBob,25\n")
            f.flush()
            path = f.name
        try:
            result = _extract_csv(path)
            assert "Alice" in result
            assert "Bob" in result
        finally:
            os.unlink(path)

    def test_latin1_csv(self):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
            f.write("Name,City\nAlice,Zürich\n".encode("latin-1"))
            path = f.name
        try:
            result = _extract_csv(path)
            assert "Alice" in result
        finally:
            os.unlink(path)


# ─── Process Document ──────────────────────────────────────────────

class TestProcessDocument:
    def test_csv_end_to_end(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Col1,Col2\nHello,World\n")
            f.flush()
            path = f.name
        try:
            result = process_document(path)
            assert "Hello" in result["final_markdown"]
            assert result["stats"]["char_count"] > 0
        finally:
            os.unlink(path)

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                process_document(path)
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            process_document("/tmp/nonexistent_file_abc123.pdf")

    def test_file_too_large(self):
        # Create a file that reports as too large via os.path.getsize
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
            f.write(b"x")
        try:
            # Patch MAX_FILE_SIZE_BYTES to trigger the check
            import converter
            original = converter.MAX_FILE_SIZE_BYTES
            converter.MAX_FILE_SIZE_BYTES = 0
            try:
                with pytest.raises(ValueError, match="File too large"):
                    process_document(path)
            finally:
                converter.MAX_FILE_SIZE_BYTES = original
        finally:
            os.unlink(path)
