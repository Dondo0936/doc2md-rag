"""Smoke tests for MCP tool registration (no live stdio server)."""

import json

import pytest


def test_create_mcp_server_registers_tools():
    pytest.importorskip("mcp")
    from mcp_server import _tool_registry, create_mcp_server

    mcp = create_mcp_server()
    tool_names = set(_tool_registry(mcp).keys())
    if not tool_names:
        pytest.skip("Cannot introspect MCP tools in this mcp version")

    expected = {
        "kb_ingest",
        "kb_search",
        "kb_build_prompt",
        "kb_status",
        "kb_convert",
        "kb_list_formats",
        "kb_clear",
    }
    missing = expected - tool_names
    assert not missing, f"Missing MCP tools: {missing}"


def test_kb_list_formats_tool_runs(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("DOC2MD_KB_DIR", str(tmp_path / "kb"))
    from mcp_server import _get_kb, _tool_registry, create_mcp_server

    if hasattr(_get_kb, "_instance"):
        delattr(_get_kb, "_instance")

    mcp = create_mcp_server()
    tools = _tool_registry(mcp)
    assert "kb_list_formats" in tools
    tool = tools["kb_list_formats"]
    fn = getattr(tool, "fn", None) or getattr(tool, "handler", None) or tool
    result = fn() if callable(fn) else fn()
    if hasattr(result, "__await__"):
        pytest.skip("Async tool invocation needs event loop in this mcp version")
    data = json.loads(result)
    assert "pdf" in data["extensions"]
    assert "xlsx" in data["extensions"]
    assert "https://" in data["url_schemes"]
