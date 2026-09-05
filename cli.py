"""CLI for Doc2MD-RAG — agent-first ingest / search / MCP entry points.

Examples:
    doc2md-rag ingest ./docs/handbook.pdf https://example.com/pricing
    doc2md-rag search "refund policy" --top-k 5
    doc2md-rag convert ./data/sales.xlsx -o sales.md
    doc2md-rag status
    doc2md-rag mcp
    doc2md-rag ui
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from config import DEFAULT_KB_DIR, DEFAULT_TOP_K, SUPPORTED_EXTENSIONS

app = typer.Typer(
    name="doc2md-rag",
    help="CLI/MCP-first document → markdown → hybrid RAG knowledge base.",
    add_completion=False,
    no_args_is_help=True,
)


def _kb(persist_dir: Optional[str], embedding_model: Optional[str] = None):
    from knowledge_base import KBConfig, KnowledgeBase

    cfg = KBConfig()
    if embedding_model:
        cfg.embedding_model = embedding_model
        # Wire API keys from env when using remote providers
        from config import EMBEDDING_MODELS, EMBEDDING_API_KEY_ENV
        provider = EMBEDDING_MODELS.get(embedding_model, {}).get("provider")
        env_name = EMBEDDING_API_KEY_ENV.get(provider or "", "")
        if env_name:
            cfg.embedding_api_key = os.environ.get(env_name)

    # Optional vision client for image descriptions
    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from claude_wrapper import get_llm_client
            llm_client = get_llm_client(os.environ["ANTHROPIC_API_KEY"])
        except Exception:
            llm_client = None

    return KnowledgeBase(
        persist_dir=persist_dir or os.environ.get("DOC2MD_KB_DIR", DEFAULT_KB_DIR),
        config=cfg,
        llm_client=llm_client,
    )


def _print_json(data) -> None:
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))


@app.command()
def ingest(
    sources: list[str] = typer.Argument(..., help="File paths and/or http(s) URLs"),
    kb_dir: Optional[str] = typer.Option(None, "--kb", help="Knowledge base directory"),
    embedding_model: Optional[str] = typer.Option(None, "--model", help="Embedding model name"),
    replace: bool = typer.Option(False, "--replace", help="Replace existing source if present"),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist KB after ingest"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Convert and index one or more documents / URLs into the knowledge base."""
    kb = _kb(kb_dir, embedding_model)
    results = []
    for src in sources:
        try:
            result = kb.ingest(src, replace=replace)
            results.append(result.to_dict())
            if not json_out:
                typer.echo(f"✓ {result.source}  (+{result.chunks_added} chunks, {result.char_count} chars)")
        except Exception as e:
            results.append({"source": src, "error": str(e)})
            if not json_out:
                typer.secho(f"✗ {src}: {e}", fg=typer.colors.RED, err=True)
            else:
                pass

    if save:
        path = kb.save()
        if not json_out:
            typer.echo(f"Saved → {path}")

    if json_out:
        _print_json({"results": results, "status": kb.status()})


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    kb_dir: Optional[str] = typer.Option(None, "--kb", help="Knowledge base directory"),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", "-k", help="Number of results"),
    mode: Optional[str] = typer.Option(None, "--mode", help="Hybrid | Vector (KNN) | Lexical (BM25)"),
    prompt: bool = typer.Option(False, "--prompt", help="Emit a ready-to-use LLM prompt"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Search the knowledge base (hybrid BM25 + vector by default)."""
    kb = _kb(kb_dir)
    if kb.status()["chunk_count"] == 0:
        typer.secho("Knowledge base is empty. Run `doc2md-rag ingest ...` first.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1)

    if prompt:
        text = kb.build_prompt(query, top_k=top_k, mode=mode)
        typer.echo(text)
        return

    hits = kb.search(query, top_k=top_k, mode=mode)
    if json_out:
        _print_json(hits)
        return

    if not hits:
        typer.echo("No results.")
        return

    for i, hit in enumerate(hits, 1):
        typer.echo(f"\n[{i}] score={hit['score']:.3f}  source={hit.get('source', '?')}")
        snippet = hit["text"].replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        typer.echo(f"    {snippet}")


@app.command("convert")
def convert_cmd(
    source: str = typer.Argument(..., help="File path or URL"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write markdown to file"),
    json_out: bool = typer.Option(False, "--json", help="Include stats as JSON"),
):
    """Convert a document/URL to LLM-friendly markdown without indexing."""
    from converter import process_source

    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from claude_wrapper import get_llm_client
            llm_client = get_llm_client(os.environ["ANTHROPIC_API_KEY"])
        except Exception:
            pass

    result = process_source(source, llm_client=llm_client)
    md = result["final_markdown"]

    if output:
        output.write_text(md, encoding="utf-8")
        typer.echo(f"Wrote {output} ({len(md)} chars)")
    elif json_out:
        _print_json({
            "source": result.get("source", source),
            "markdown": md,
            "stats": result.get("stats", {}),
        })
    else:
        typer.echo(md)


@app.command()
def status(
    kb_dir: Optional[str] = typer.Option(None, "--kb", help="Knowledge base directory"),
    json_out: bool = typer.Option(True, "--json/--human", help="JSON or human output"),
):
    """Show knowledge base size, sources, and config."""
    kb = _kb(kb_dir)
    info = kb.status()
    if json_out:
        _print_json(info)
    else:
        typer.echo(f"KB: {info['persist_dir']}")
        typer.echo(f"Chunks: {info['chunk_count']}  Sources: {info['source_count']}")
        for src, meta in info["sources"].items():
            typer.echo(f"  - {src} ({meta.get('chunks', 0)} chunks)")


@app.command("list-formats")
def list_formats():
    """List supported ingest formats."""
    typer.echo("Supported extensions: " + ", ".join(SUPPORTED_EXTENSIONS))
    typer.echo("Also: http(s) URLs")


@app.command()
def clear(
    kb_dir: Optional[str] = typer.Option(None, "--kb", help="Knowledge base directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clear the knowledge base index."""
    kb = _kb(kb_dir)
    if not yes:
        confirm = typer.confirm(f"Clear KB at {kb.persist_dir}?")
        if not confirm:
            raise typer.Abort()
    kb.clear()
    kb.save()
    typer.echo("Cleared.")


@app.command()
def mcp(
    kb_dir: Optional[str] = typer.Option(None, "--kb", help="Knowledge base directory"),
    transport: str = typer.Option("stdio", "--transport", help="stdio or sse"),
):
    """Start the MCP server for agent tooling (default: stdio)."""
    os.environ["DOC2MD_KB_DIR"] = kb_dir or os.environ.get("DOC2MD_KB_DIR", DEFAULT_KB_DIR)
    from mcp_server import run_mcp

    run_mcp(transport=transport)


@app.command()
def ui():
    """Launch the optional Streamlit demo UI."""
    import subprocess

    app_path = Path(__file__).resolve().parent / "app.py"
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)]))


def main():
    app()


if __name__ == "__main__":
    main()
