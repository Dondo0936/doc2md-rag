# Contributing to Doc2MD-RAG

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/Dondo0936/doc2md-rag.git
cd doc2md-rag

# Install dependencies
pip install -r requirements.txt
pip install pytest ruff  # dev tools

# Run the app
streamlit run app.py
```

## Running Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a PR.

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting: `ruff check .`
- Line length limit: 120 characters
- Use type hints where practical
- Add docstrings for public functions

## Submitting Changes

1. Fork the repo and create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Add tests for new functionality
4. Run `pytest tests/ -v` and `ruff check .` to verify
5. Open a PR with a clear description of what and why

## Reporting Issues

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your Python version and OS

## Architecture Overview

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI and orchestration |
| `converter.py` | Document to markdown conversion |
| `rag_engine.py` | Chunking, indexing, search, evaluation |
| `tracer.py` | Pipeline visualization (Plotly) |
| `claude_wrapper.py` | Anthropic API wrapper |
| `config.py` | Constants and defaults |
