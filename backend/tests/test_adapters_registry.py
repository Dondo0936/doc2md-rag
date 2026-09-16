"""Tests for URL SSRF guard and text adapter (no network for blocked hosts)."""

from __future__ import annotations

import pytest

from backend.app.pipeline.adapters.file import build_text_context
from backend.app.pipeline.adapters.url import UrlFetchError, validate_url
from backend.app.pipeline.registry import resolve_chain
from backend.app.pipeline.context import ExtractContext


class TestUrlSsrf:
    def test_rejects_localhost(self):
        with pytest.raises(UrlFetchError):
            validate_url("http://localhost/secret")

    def test_rejects_private_ip(self):
        with pytest.raises(UrlFetchError):
            validate_url("http://127.0.0.1/x")
        with pytest.raises(UrlFetchError):
            validate_url("http://10.0.0.1/x")
        with pytest.raises(UrlFetchError):
            validate_url("http://192.168.1.1/x")

    def test_rejects_non_http(self):
        with pytest.raises(UrlFetchError):
            validate_url("file:///etc/passwd")
        with pytest.raises(UrlFetchError):
            validate_url("ftp://example.com/a")

    def test_accepts_public_hostname_shape(self):
        # validate_url resolves DNS — example.com should be public
        url = validate_url("https://example.com/path")
        assert url.startswith("https://")


class TestTextAdapter:
    def test_plain(self):
        ctx, source = build_text_context("Hello parser platform", fmt="plain")
        assert ctx.input_kind == "text"
        assert source.format == "plain"
        chain = resolve_chain(ctx)
        assert chain[0].extractor.name == "plain_text"

    def test_markdown(self):
        ctx, _ = build_text_context("# Hi\n\nBody", fmt="markdown")
        chain = resolve_chain(ctx)
        assert chain[0].extractor.name == "markdown_text"

    def test_html(self):
        ctx, _ = build_text_context("<html><body><p>Hi</p></body></html>", fmt="html")
        chain = resolve_chain(ctx)
        assert chain[0].extractor.name == "html_trafilatura"


class TestRegistryChains:
    def test_pdf_digital(self):
        ctx = ExtractContext(input_kind="file", extension=".pdf", filename="a.pdf", use_ocr=False)
        names = [s.extractor.name for s in resolve_chain(ctx)]
        assert names[0] == "docling"
        assert "pymupdf4llm" in names

    def test_pdf_scanned(self):
        ctx = ExtractContext(input_kind="file", extension=".pdf", filename="a.pdf", use_ocr=True)
        names = [s.extractor.name for s in resolve_chain(ctx)]
        assert names[0] == "docling_ocr"

    def test_docx(self):
        ctx = ExtractContext(input_kind="file", extension=".docx", filename="a.docx")
        names = [s.extractor.name for s in resolve_chain(ctx)]
        assert names == ["docling", "pypandoc", "python_docx"]

    def test_csv(self):
        ctx = ExtractContext(input_kind="file", extension=".csv", filename="a.csv")
        names = [s.extractor.name for s in resolve_chain(ctx)]
        assert names == ["pandas_csv"]

    def test_image(self):
        ctx = ExtractContext(input_kind="image", extension=".png", filename="a.png")
        names = [s.extractor.name for s in resolve_chain(ctx)]
        assert names[0] == "docling_ocr"
        assert "rapidocr" in names
        assert "claude_vision" in names
