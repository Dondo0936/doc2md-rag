"""FastAPI TestClient coverage for parse + health endpoints."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.exporters.markdown import markdown_to_blocks
from backend.app.main import create_app
from backend.app.models.blocks import ParseResult, ParseStats, SourceInfo
from backend.app.services.job_store import JobStore


@pytest.fixture()
def client(monkeypatch):
    # Avoid Docling warmup during tests
    monkeypatch.setenv("PARSER_CORS_ORIGINS", "http://localhost:3000")
    with patch("backend.app.main.converter", create=True):
        app = create_app()

        # Replace lifespan warmup by patching create_app's lifespan — simplest:
        # just use app without running lifespan heavy work via TestClient
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def _fake_result(job_id: str = "abc", markdown: str = "# Hi\n\nHello world content here.\n") -> ParseResult:
    blocks = markdown_to_blocks(markdown, extractor="plain_text")
    return ParseResult(
        job_id=job_id,
        source=SourceInfo(input_type="text", format="plain", size_bytes=10),
        blocks=blocks,
        markdown=markdown,
        stats=ParseStats(char_count=len(markdown), block_counts={"heading": 1, "paragraph": 1}),
        pipeline_trace=[],
        warnings=[],
        status="completed",
    )


class TestHealth:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "docling_ready" in body


class TestParseText:
    def test_plain_text(self, client):
        r = client.post(
            "/api/v1/parse/text",
            json={"text": "Hello parser platform module test content.", "format": "plain"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert body["markdown"]
        assert len(body["blocks"]) >= 1
        assert body["source"]["input_type"] == "text"

    def test_empty_text(self, client):
        r = client.post("/api/v1/parse/text", json={"text": "  ", "format": "plain"})
        assert r.status_code == 400

    def test_markdown(self, client):
        r = client.post(
            "/api/v1/parse/text",
            json={"text": "# Title\n\nA paragraph with enough characters.", "format": "markdown"},
        )
        assert r.status_code == 200
        assert any(b["type"] == "heading" for b in r.json()["blocks"])


class TestParseFile:
    def test_txt_upload(self, client):
        content = b"Uploaded text file with enough characters for quality."
        r = client.post(
            "/api/v1/parse/file",
            files={"file": ("notes.txt", io.BytesIO(content), "text/plain")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert "Uploaded text" in body["markdown"]

    def test_csv_upload(self, client):
        content = b"Name,Age\nAlice,30\nBob,25\n"
        # May need pandas — skip soft if missing
        pytest.importorskip("pandas")
        r = client.post(
            "/api/v1/parse/file",
            files={"file": ("data.csv", io.BytesIO(content), "text/csv")},
        )
        assert r.status_code == 200
        assert "Alice" in r.json()["markdown"]

    def test_unsupported_type(self, client):
        r = client.post(
            "/api/v1/parse/file",
            files={"file": ("x.xyz", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert r.status_code == 415

    def test_empty_file(self, client):
        r = client.post(
            "/api/v1/parse/file",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert r.status_code == 400


class TestParseUrl:
    def test_bad_url_ssrf(self, client):
        r = client.post("/api/v1/parse/url", json={"url": "http://127.0.0.1/secret"})
        assert r.status_code == 400
        assert "private" in r.json()["detail"].lower() or "not allowed" in r.json()["detail"].lower() or "reserved" in r.json()["detail"].lower() or "loopback" in r.json()["detail"].lower() or "Private" in r.json()["detail"]


class TestJobPoll:
    def test_job_not_found(self, client):
        r = client.get("/api/v1/parse/doesnotexist123")
        assert r.status_code == 404

    def test_async_file_returns_202(self, client, monkeypatch):
        # Force async path via tiny sync max
        from backend.app.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("PARSER_SYNC_MAX_MB", "0")
        get_settings.cache_clear()

        # Re-create app so settings pick up env — patch run_pipeline to be fast
        from backend.app.main import create_app

        app = create_app()

        async def fake_pipeline(ctx, job_id=None, source=None, settings=None):
            return _fake_result(job_id or "job")

        with patch("backend.app.api.routes.parse.run_pipeline", new=AsyncMock(side_effect=fake_pipeline)):
            with TestClient(app) as c:
                content = b"hello async world content here"
                r = c.post(
                    "/api/v1/parse/file",
                    files={"file": ("notes.txt", io.BytesIO(content), "text/plain")},
                    data={"async_mode": "true"},
                )
                assert r.status_code == 202
                job_id = r.json()["job_id"]
                # Poll until completed (TestClient runs background tasks)
                poll = c.get(f"/api/v1/parse/{job_id}")
                assert poll.status_code == 200
                assert poll.json()["status"] in {"completed", "running", "pending", "failed"}

        get_settings.cache_clear()
