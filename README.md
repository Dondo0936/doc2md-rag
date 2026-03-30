# Doc2MD-RAG

Convert documents (PDF, DOCX, PPTX, CSV) to LLM-friendly markdown, then chunk, index, and search with hybrid BM25 + vector retrieval — all in one interactive Streamlit app.

<p align="center">
  <img src="docs/demo.gif" alt="Doc2MD-RAG Demo" width="800" />
</p>

> [Watch full demo video (MP4)](docs/demo.mp4)

## Features

- **Document conversion** — PDF (pymupdf4llm), DOCX (python-docx), PPTX (python-pptx), CSV (pandas). Tables are converted to nested bullet lists for better LLM chunking. Images can be described by Claude Vision.
- **4 chunking strategies** — Recursive, By Sentence, By Markdown Headers, Semantic (embedding-based).
- **3 search modes** — Hybrid (BM25 + Vector), Vector-only (KNN), Lexical-only (BM25).
- **Tunable parameters** — Chunk size, overlap, search weights, score threshold, dedup threshold, embedding model — all with hover tooltips explaining their effect.
- **RAG pipeline tracer** — Visualize query tokenization, embedding space (PCA/t-SNE), BM25 token matches, and score fusion breakdown.
- **Prompt builder** — Auto-assembles retrieved chunks into a copy-paste-ready LLM prompt with token estimation.
- **Chunk comparison** — Side-by-side comparison of different chunking strategies on the same document.
- **Pipeline overview** — Educational tab explaining how each stage and parameter works.

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/doc2md-rag.git
cd doc2md-rag
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Optional: Claude Vision for image descriptions

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or enter your API key in the sidebar under "API Configuration". Without an API key, images will use placeholder descriptions instead.

## Architecture

```
doc2md-rag/
├── app.py              # Streamlit UI + orchestration
├── converter.py        # Doc → Markdown (tables→lists, images→descriptions)
├── rag_engine.py       # Chunking, FAISS+BM25 indexing, search, evaluation
├── tracer.py           # Pipeline visualization (Plotly charts)
├── claude_wrapper.py   # OpenAI-compatible wrapper for Anthropic API
├── config.py           # Constants + parameter defaults + tooltips
├── tests/              # Test suite
└── requirements.txt
```

## How It Works

1. **Upload** — Drop a PDF, DOCX, PPTX, or CSV (up to 100MB)
2. **Convert** — Document is parsed to markdown. Pipe-tables become nested bullet lists. Images get AI descriptions (with API key) or placeholders.
3. **Chunk** — Markdown is split into overlapping pieces using your chosen strategy.
4. **Embed** — Each chunk is encoded into a dense vector using SentenceTransformers.
5. **Index** — FAISS (vector) and BM25 (keyword) indexes are built in parallel.
6. **Search** — Your query is scored against both indexes. Scores are normalized, weighted, combined, filtered, and deduplicated.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| Chunk size | 400 | Characters per chunk |
| Overlap | 50 | Characters shared between adjacent chunks |
| Search mode | Hybrid | BM25 + Vector, Vector-only, or BM25-only |
| BM25 weight | 0.4 | Weight for keyword matching (Hybrid mode) |
| Vector weight | 0.6 | Weight for semantic similarity (Hybrid mode) |
| Top K | 5 | Number of results returned |
| Score threshold | 0.0 | Minimum score to include a result |
| Embedding model | all-MiniLM-L6-v2 | 384d, fast. Alternative: all-mpnet-base-v2 (768d) |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
