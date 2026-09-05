"""Minimal template: wire Doc2MD-RAG into an existing Python app / agent loop.

Copy this pattern into your service — keep your own API/auth/UI, use KnowledgeBase
for convert + index + retrieve.
"""

from __future__ import annotations

import os
from pathlib import Path

from knowledge_base import KnowledgeBase, create_knowledge_base


def build_kb(persist_dir: str | None = None) -> KnowledgeBase:
    """Factory used by your app startup / DI container."""
    return create_knowledge_base(
        persist_dir=persist_dir or os.environ.get("DOC2MD_KB_DIR", "./.doc2md_kb"),
        embedding_model=os.environ.get("DOC2MD_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_api_key=os.environ.get("OPENAI_API_KEY")  # used only for OpenAI models
        or os.environ.get("VOYAGE_API_KEY")
        or os.environ.get("GEMINI_API_KEY"),
        chunk_size=400,
        chunk_overlap=50,
        search_mode="Hybrid",
        top_k=5,
    )


def sync_folder(kb: KnowledgeBase, folder: str) -> None:
    """Example: index every supported file under a directory."""
    from config import SUPPORTED_EXTENSIONS

    root = Path(folder)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS and path.is_file():
            try:
                kb.ingest(str(path), replace=True)
                print(f"indexed {path}")
            except Exception as e:
                print(f"skip {path}: {e}")
    kb.save()


def answer_with_rag(kb: KnowledgeBase, question: str) -> str:
    """Example: grounded prompt you pass to your existing LLM client."""
    return kb.build_prompt(
        question,
        system=(
            "You are a helpful assistant for our product. "
            "Answer using only the provided context. If missing, say you don't know."
        ),
    )


if __name__ == "__main__":
    kb = build_kb()
    demo = Path(__file__).resolve().parent / "sample_docs"
    if demo.is_dir():
        sync_folder(kb, str(demo))
    print(answer_with_rag(kb, "What is covered in the knowledge base?"))
