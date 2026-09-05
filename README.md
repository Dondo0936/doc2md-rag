# Doc2MD-RAG

**CLI / MCP-first** document knowledge base + hybrid RAG — a framework you plug into existing apps and agents.

Convert many input types (PDF, DOCX, PPTX, CSV, **XLSX**, **HTML**, **images**, **URLs**, TXT/MD) to LLM-friendly markdown, then chunk, index, and retrieve with hybrid BM25 + vector search.

<p align="center">
  <img src="docs/demo.gif" alt="Doc2MD-RAG Demo" width="800" />
</p>

> [Watch full demo video (MP4)](docs/demo.mp4)

## Why this shape?

| Surface | Who it's for |
|---|---|
| **Python `KnowledgeBase`** | Drop into your existing backend / agent loop |
| **CLI (`doc2md-rag`)** | Scripts, CI, local indexing |
| **MCP server** | Cursor, Claude Desktop, any MCP agent |
| **Streamlit UI** (optional) | Interactive exploration / demos |

## Quick start

```bash
git clone https://github.com/Dondo0936/doc2md-rag.git
cd doc2md-rag
pip install -r requirements.txt
pip install -e .
```

### Library (connect to your app)

```python
from knowledge_base import KnowledgeBase

kb = KnowledgeBase(persist_dir="./.doc2md_kb")

# Any mix of local files + URLs
kb.ingest("./docs/handbook.pdf")
kb.ingest("./data/sales.xlsx")
kb.ingest("https://example.com/pricing")
kb.ingest("./screenshots/error.png")  # vision desc if ANTHROPIC_API_KEY set
kb.save()

hits = kb.search("refund policy", top_k=5)
prompt = kb.build_prompt("How do refunds work?")
```

### CLI

```bash
doc2md-rag ingest ./docs/handbook.pdf ./data/sales.xlsx https://example.com/pricing
doc2md-rag search "refund policy" --top-k 5
doc2md-rag search "refund policy" --prompt          # grounded LLM prompt
doc2md-rag convert ./report.pdf -o report.md
doc2md-rag status
doc2md-rag list-formats
```

### MCP (for agents)

```bash
doc2md-rag mcp
```

Cursor / Claude Desktop config:

```json
{
  "mcpServers": {
    "doc2md-rag": {
      "command": "doc2md-rag",
      "args": ["mcp"],
      "env": {
        "DOC2MD_KB_DIR": "/absolute/path/to/.doc2md_kb"
      }
    }
  }
}
```

MCP tools: `kb_ingest`, `kb_ingest_many`, `kb_search`, `kb_build_prompt`, `kb_convert`, `kb_status`, `kb_list_sources`, `kb_list_formats`, `kb_ingest_markdown`, `kb_clear`.

### Optional Streamlit UI

```bash
doc2md-rag ui
# or: streamlit run app.py
```

## Supported inputs

| Type | Extensions / schemes |
|---|---|
| Documents | `pdf`, `docx`, `pptx` |
| Tables | `csv`, `xlsx`, `xls` |
| Web | `html`, `htm`, `http://`, `https://` |
| Text | `txt`, `md`, `markdown` |
| Images | `png`, `jpg`, `jpeg`, `webp`, `gif`, `bmp`, `tiff` |

Images use Claude Vision when `ANTHROPIC_API_KEY` is set; otherwise MarkItDown / placeholders.

## Architecture

```
doc2md-rag/
├── knowledge_base.py   # Framework API (ingest / search / persist)
├── cli.py              # Typer CLI
├── mcp_server.py       # MCP tools for agents
├── converter.py        # Multi-format → markdown
├── rag_engine.py       # Chunking, FAISS+BM25, search
├── tracer.py           # Pipeline visualization (UI)
├── claude_wrapper.py   # Anthropic vision helper
├── config.py           # Defaults + supported formats
├── app.py              # Optional Streamlit UI
└── examples/           # Drop-in integration templates
```

Pipeline: **ingest → convert → chunk → embed → index (FAISS + BM25) → search / build_prompt**.

## Features

- **Multi-source knowledge base** — accumulate documents; persist/reload embeddings
- **4 chunking strategies** — Recursive, By Sentence, By Markdown Headers, Semantic
- **3 search modes** — Hybrid (BM25 + Vector), Vector-only (KNN), Lexical-only (BM25)
- **7 embedding models** — local SentenceTransformers or OpenAI / Voyage / Gemini APIs
- **Agent surfaces** — CLI + MCP first; Streamlit optional

## Configuration

| Parameter | Default | Description |
|---|---|---|
| Chunk size | 400 | Characters per chunk |
| Overlap | 50 | Characters shared between adjacent chunks |
| Search mode | Hybrid | BM25 + Vector, Vector-only, or BM25-only |
| BM25 weight | 0.4 | Keyword weight (Hybrid) |
| Vector weight | 0.6 | Semantic weight (Hybrid) |
| Top K | 5 | Results returned |
| Embedding model | all-MiniLM-L6-v2 | See list below |
| Persist dir | `.doc2md_kb` | Or `DOC2MD_KB_DIR` |

## Embedding Models

| Model | Provider | Dimensions | API Key |
|---|---|---|---|
| all-MiniLM-L6-v2 | Local | 384 | — |
| all-mpnet-base-v2 | Local | 768 | — |
| text-embedding-3-small | OpenAI | 1536 | `OPENAI_API_KEY` |
| text-embedding-3-large | OpenAI | 3072 | `OPENAI_API_KEY` |
| voyage-3.5-lite | Voyage AI | 1024 | `VOYAGE_API_KEY` |
| gemini-embedding-001 | Google | 768 | `GEMINI_API_KEY` |
| text-embedding-004 | Google | 768 | `GEMINI_API_KEY` |

## Environment

```bash
cp .env.example .env
# Optional:
# ANTHROPIC_API_KEY=...   # image descriptions
# OPENAI_API_KEY=...
# VOYAGE_API_KEY=...
# GEMINI_API_KEY=...
# DOC2MD_KB_DIR=./.doc2md_kb
# ENABLE_IMAGE_DESC=1     # local SmolVLM for PDF/DOCX/PPTX images
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
