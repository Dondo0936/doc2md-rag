"""MCP server — expose Doc2MD-RAG KnowledgeBase tools to agents.

Run:
    doc2md-rag mcp
    # or
    python -m mcp_server

Cursor / Claude Desktop example config:
{
  "mcpServers": {
    "doc2md-rag": {
      "command": "doc2md-rag",
      "args": ["mcp"],
      "env": { "DOC2MD_KB_DIR": "/path/to/.doc2md_kb" }
    }
  }
}
"""

from __future__ import annotations

import json
import logging
import os

from config import DEFAULT_KB_DIR, DEFAULT_TOP_K, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "Doc2MD-RAG knowledge base tools. "
    "Ingest documents (pdf/docx/pptx/csv/xlsx/html/txt/md/images) or http(s) URLs, "
    "then search with hybrid BM25+vector retrieval. "
    "Use kb_build_prompt to assemble grounded LLM context for the user's question."
)


def _get_kb():
    """Lazy singleton KnowledgeBase bound to DOC2MD_KB_DIR."""
    from knowledge_base import KBConfig, KnowledgeBase

    if not hasattr(_get_kb, "_instance"):
        cfg = KBConfig()
        from config import EMBEDDING_MODELS, EMBEDDING_API_KEY_ENV
        provider = EMBEDDING_MODELS.get(cfg.embedding_model, {}).get("provider")
        env_name = EMBEDDING_API_KEY_ENV.get(provider or "", "")
        if env_name:
            cfg.embedding_api_key = os.environ.get(env_name)

        llm_client = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from claude_wrapper import get_llm_client
                llm_client = get_llm_client(os.environ["ANTHROPIC_API_KEY"])
            except Exception:
                llm_client = None

        _get_kb._instance = KnowledgeBase(
            persist_dir=os.environ.get("DOC2MD_KB_DIR", DEFAULT_KB_DIR),
            config=cfg,
            llm_client=llm_client,
        )
    return _get_kb._instance


def _import_mcp_server_class():
    """Support mcp 2.x (MCPServer) and mcp 1.x (FastMCP)."""
    try:
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError as e:
        raise ImportError(
            "MCP support requires the `mcp` package. Install with: pip install 'mcp>=1.0'"
        ) from e


def create_mcp_server():
    """Build an MCP server with KB tools (mcp 1.x FastMCP or 2.x MCPServer)."""
    Server = _import_mcp_server_class()
    mcp = Server("doc2md-rag", instructions=INSTRUCTIONS)

    @mcp.tool()
    def kb_ingest(source: str, replace: bool = False) -> str:
        """Convert a local file path or http(s) URL and add it to the knowledge base.

        Supports: pdf, docx, pptx, csv, xlsx, html, txt, md, png/jpg/webp/gif, and URLs.
        """
        kb = _get_kb()
        result = kb.ingest(source, replace=replace)
        kb.save()
        return json.dumps(result.to_dict(), ensure_ascii=False)

    @mcp.tool()
    def kb_ingest_many(sources: list[str], replace: bool = False) -> str:
        """Ingest multiple file paths / URLs into the knowledge base."""
        kb = _get_kb()
        results = kb.ingest_many(sources, replace=replace)
        kb.save()
        return json.dumps([r.to_dict() for r in results], ensure_ascii=False)

    @mcp.tool()
    def kb_search(
        query: str,
        top_k: int = DEFAULT_TOP_K,
        mode: str = "Hybrid",
    ) -> str:
        """Search the knowledge base. mode: Hybrid | Vector (KNN) | Lexical (BM25)."""
        kb = _get_kb()
        hits = kb.search(query, top_k=top_k, mode=mode)
        slim = [
            {
                "text": h["text"],
                "score": h["score"],
                "source": h.get("source"),
                "chunk_id": h.get("chunk_id"),
            }
            for h in hits
        ]
        return json.dumps(slim, ensure_ascii=False)

    @mcp.tool()
    def kb_build_prompt(query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Retrieve relevant chunks and return a ready-to-use grounded LLM prompt."""
        kb = _get_kb()
        return kb.build_prompt(query, top_k=top_k)

    @mcp.tool()
    def kb_convert(source: str) -> str:
        """Convert a file/URL to markdown without indexing. Returns markdown + stats JSON."""
        from converter import process_source

        kb = _get_kb()
        result = process_source(source, llm_client=kb.llm_client)
        return json.dumps(
            {
                "source": result.get("source", source),
                "markdown": result["final_markdown"],
                "stats": result.get("stats", {}),
            },
            ensure_ascii=False,
        )

    @mcp.tool()
    def kb_status() -> str:
        """Return knowledge base chunk counts, sources, and config."""
        return json.dumps(_get_kb().status(), ensure_ascii=False, default=str)

    @mcp.tool()
    def kb_list_sources() -> str:
        """List indexed sources with chunk/char counts."""
        return json.dumps(_get_kb().list_sources(), ensure_ascii=False)

    @mcp.tool()
    def kb_list_formats() -> str:
        """List supported ingest file extensions and URL schemes."""
        return json.dumps(
            {
                "extensions": SUPPORTED_EXTENSIONS,
                "url_schemes": ["http://", "https://"],
            }
        )

    @mcp.tool()
    def kb_clear() -> str:
        """Clear all indexed sources from the knowledge base."""
        kb = _get_kb()
        kb.clear()
        kb.save()
        return json.dumps({"ok": True, "persist_dir": kb.persist_dir})

    @mcp.tool()
    def kb_ingest_markdown(markdown: str, source: str = "inline") -> str:
        """Index raw markdown text (for content your app already has)."""
        kb = _get_kb()
        result = kb.ingest_markdown(markdown, source=source)
        kb.save()
        return json.dumps(result.to_dict(), ensure_ascii=False)

    return mcp


def _tool_registry(mcp) -> dict:
    """Best-effort access to registered tools across mcp versions."""
    manager = getattr(mcp, "_tool_manager", None)
    if manager is not None:
        tools = getattr(manager, "_tools", None)
        if isinstance(tools, dict):
            return tools
        if hasattr(manager, "list_tools"):
            listed = manager.list_tools()
            if isinstance(listed, dict):
                return listed
    tools = getattr(mcp, "_tools", None)
    if isinstance(tools, dict):
        return tools
    return {}


def run_mcp(transport: str = "stdio") -> None:
    mcp = create_mcp_server()
    if transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


def main():
    logging.basicConfig(level=logging.INFO)
    run_mcp(transport=os.environ.get("DOC2MD_MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()
