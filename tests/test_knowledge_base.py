"""Tests for multi-format conversion and KnowledgeBase framework."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from converter import (
    _extract_csv,
    _extract_html_file,
    _extract_text_file,
    _extract_xlsx,
    process_document,
    process_source,
)
from knowledge_base import KnowledgeBase, KBConfig, create_knowledge_base
from tests.conftest import SAMPLE_MARKDOWN


# ─── New format extractors ─────────────────────────────────────

class TestXlsxExtract:
    def test_xlsx_to_markdown(self, tmp_path):
        pytest.importorskip("openpyxl")
        import pandas as pd

        path = tmp_path / "sales.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame({"Product": ["Widget", "Gadget"], "Revenue": [100, 250]}).to_excel(
                writer, sheet_name="Q1", index=False
            )
            pd.DataFrame({"Region": ["US", "EU"]}).to_excel(writer, sheet_name="Geo", index=False)

        md = _extract_xlsx(str(path))
        assert "Sheet: Q1" in md
        assert "Widget" in md
        assert "Sheet: Geo" in md
        assert "US" in md


class TestHtmlExtract:
    def test_html_file(self, tmp_path):
        path = tmp_path / "page.html"
        path.write_text(
            "<html><head><title>T</title><script>x=1</script></head>"
            "<body><h1>Hello</h1><p>World of refunds</p></body></html>",
            encoding="utf-8",
        )
        md = _extract_html_file(str(path))
        assert "Hello" in md
        assert "refunds" in md
        assert "x=1" not in md


class TestTextExtract:
    def test_md_file(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("# Title\n\nBody text about photosynthesis.", encoding="utf-8")
        assert "photosynthesis" in _extract_text_file(str(path))


class TestProcessSource:
    def test_process_txt(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("Alpha beta gamma unique-term-xyz", encoding="utf-8")
        result = process_document(str(path))
        assert "unique-term-xyz" in result["final_markdown"]
        assert result["source"].endswith("doc.txt")

    def test_process_source_dispatches_file(self, tmp_path):
        path = tmp_path / "a.md"
        path.write_text("# A\n\ncontent", encoding="utf-8")
        result = process_source(str(path))
        assert "content" in result["final_markdown"]

    def test_unsupported_url_scheme(self):
        with pytest.raises(ValueError, match="Unsupported URL"):
            process_source("ftp://example.com/file.pdf")


# ─── KnowledgeBase ─────────────────────────────────────────────

class TestKnowledgeBase:
    def test_ingest_markdown_and_search(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        kb.clear()
        kb.ingest_markdown(SAMPLE_MARKDOWN, source="sample")
        hits = kb.search("machine learning", top_k=3)
        assert len(hits) > 0
        assert hits[0]["source"] == "sample"
        assert "score" in hits[0]

    def test_multi_source(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        kb.clear()
        kb.ingest_markdown("Document about cats and feline behavior.", source="cats.md")
        kb.ingest_markdown("Document about quantum computing and qubits.", source="quantum.md")
        status = kb.status()
        assert status["source_count"] == 2
        assert status["chunk_count"] >= 2
        hits = kb.search("qubits", top_k=2)
        assert any(h.get("source") == "quantum.md" for h in hits)

    def test_build_prompt(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        kb.clear()
        kb.ingest_markdown(SAMPLE_MARKDOWN, source="sample")
        prompt = kb.build_prompt("neural networks")
        assert "Question: neural networks" in prompt
        assert "Context:" in prompt

    def test_save_and_load(self, tmp_path):
        persist = str(tmp_path / "kb")
        kb = KnowledgeBase(persist_dir=persist)
        kb.clear()
        kb.ingest_markdown(SAMPLE_MARKDOWN, source="sample")
        kb.save()

        kb2 = KnowledgeBase(persist_dir=persist)
        assert kb2.status()["chunk_count"] == kb.status()["chunk_count"]
        hits = kb2.search("photosynthesis", top_k=2)
        assert len(hits) > 0

    def test_ingest_file(self, tmp_path):
        path = tmp_path / "policy.txt"
        path.write_text(
            "Our refund policy allows returns within 30 days of purchase for unused items.",
            encoding="utf-8",
        )
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        kb.clear()
        result = kb.ingest(str(path))
        assert result.chunks_added >= 1
        hits = kb.search("refund policy", top_k=2)
        assert len(hits) > 0
        assert "refund" in hits[0]["text"].lower()

    def test_create_factory(self, tmp_path):
        kb = create_knowledge_base(persist_dir=str(tmp_path / "kb"), chunk_size=200)
        assert kb.config.chunk_size == 200

    def test_remove_source(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
        kb.clear()
        kb.ingest_markdown("Alpha content about zebras.", source="a")
        kb.ingest_markdown("Beta content about penguins.", source="b")
        assert kb.remove_source("a") is True
        sources = {s["source"] for s in kb.list_sources()}
        assert sources == {"b"}


class TestAddDocument:
    def test_add_document_appends(self):
        from rag_engine import RAGEngine

        engine = RAGEngine()
        engine.index_document("First document about apples.", source="a", chunk_size=100, overlap=10)
        n1 = len(engine.get_chunks())
        added = engine.add_document("Second document about oranges.", source="b", chunk_size=100, overlap=10)
        assert added >= 1
        assert len(engine.get_chunks()) == n1 + added
        results = engine.search("oranges", top_k=3)
        assert any(r.get("source") == "b" for r in results)
