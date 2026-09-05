"""KnowledgeBase — drop-in RAG framework for existing apps, CLI, and MCP agents.

Connect your product to a multi-source hybrid BM25 + vector knowledge base:

    from knowledge_base import KnowledgeBase

    kb = KnowledgeBase(persist_dir="./.kb")
    kb.ingest("./docs/handbook.pdf")
    kb.ingest("https://example.com/pricing")
    kb.ingest("./data/sales.xlsx")
    hits = kb.search("refund policy")
    prompt = kb.build_prompt("How do refunds work?")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from config import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_KB_DIR,
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_WEIGHT,
)
from converter import process_source
from rag_engine import RAGEngine

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    source: str
    chunks_added: int
    char_count: int
    tables_found: int = 0
    images_found: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KBConfig:
    """Tunable retrieval / chunking settings for a KnowledgeBase instance."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_api_key: Optional[str] = None
    chunk_method: str = "Recursive"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    semantic_threshold: float = 0.5
    search_mode: str = "Hybrid"
    bm25_weight: float = DEFAULT_BM25_WEIGHT
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    top_k: int = DEFAULT_TOP_K
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    num_candidates: int = DEFAULT_NUM_CANDIDATES
    dedup_threshold: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "KBConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class KnowledgeBase:
    """Multi-source knowledge base with convert → chunk → index → search.

    Designed as a template you wire into an existing app:
    - Call ``ingest`` for any supported path/URL (pdf, docx, pptx, csv, xlsx,
      html, txt/md, images, http(s) URLs).
    - Call ``search`` / ``build_prompt`` from your agent loop or API.
    - Optionally ``save`` / ``load`` to a persist directory.
    """

    persist_dir: Optional[str] = None
    config: KBConfig = field(default_factory=KBConfig)
    llm_client: Any = None
    _engine: Optional[RAGEngine] = field(default=None, init=False, repr=False)
    _sources: dict = field(default_factory=dict, init=False, repr=False)
    _markdown_store: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.persist_dir is None:
            self.persist_dir = os.environ.get("DOC2MD_KB_DIR", DEFAULT_KB_DIR)
        self._engine = RAGEngine(
            embedding_model_name=self.config.embedding_model,
            embedding_api_key=self.config.embedding_api_key,
        )
        if os.path.isdir(self.persist_dir) and os.path.exists(self._meta_path()):
            try:
                self.load()
            except Exception as e:
                logger.warning("Failed to load existing KB at %s: %s", self.persist_dir, e)

    # ─── Public API ──────────────────────────────────────────────

    def ingest(self, source: str, *, replace: bool = False) -> IngestResult:
        """Convert a file path or URL and add it to the index.

        Args:
            source: Local path or http(s) URL.
            replace: If True and source already exists, drop its chunks first.
        """
        if replace and source in self._sources:
            self.remove_source(source)

        converted = process_source(source, llm_client=self.llm_client)
        markdown = converted["final_markdown"]
        label = converted.get("source") or source

        chunks_added = self._engine.add_document(
            markdown,
            source=label,
            method=self.config.chunk_method,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            semantic_threshold=self.config.semantic_threshold,
        )

        stats = converted.get("stats", {})
        self._sources[label] = {
            "chunks": chunks_added,
            "char_count": stats.get("char_count", len(markdown)),
            "tables_found": stats.get("tables_found", 0),
            "images_found": stats.get("images_found", 0),
        }
        self._markdown_store[label] = markdown

        result = IngestResult(
            source=label,
            chunks_added=chunks_added,
            char_count=stats.get("char_count", len(markdown)),
            tables_found=stats.get("tables_found", 0),
            images_found=stats.get("images_found", 0),
        )
        logger.info("Ingested %s → %d chunks", label, chunks_added)
        return result

    def ingest_many(self, sources: list[str], *, replace: bool = False) -> list[IngestResult]:
        """Ingest multiple sources; continues on individual failures."""
        results = []
        for src in sources:
            try:
                results.append(self.ingest(src, replace=replace))
            except Exception as e:
                logger.error("Failed to ingest %s: %s", src, e)
                results.append(IngestResult(source=src, chunks_added=0, char_count=0))
        return results

    def ingest_markdown(self, markdown: str, source: str = "inline") -> IngestResult:
        """Index raw markdown without conversion (for app-generated content)."""
        chunks_added = self._engine.add_document(
            markdown,
            source=source,
            method=self.config.chunk_method,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            semantic_threshold=self.config.semantic_threshold,
        )
        self._sources[source] = {
            "chunks": chunks_added,
            "char_count": len(markdown),
            "tables_found": 0,
            "images_found": 0,
        }
        self._markdown_store[source] = markdown
        return IngestResult(source=source, chunks_added=chunks_added, char_count=len(markdown))

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        **overrides,
    ) -> list[dict]:
        """Hybrid / vector / lexical search over the knowledge base."""
        return self._engine.search(
            query,
            mode=mode or self.config.search_mode,
            top_k=top_k if top_k is not None else self.config.top_k,
            bm25_weight=overrides.get("bm25_weight", self.config.bm25_weight),
            vector_weight=overrides.get("vector_weight", self.config.vector_weight),
            score_threshold=overrides.get("score_threshold", self.config.score_threshold),
            num_candidates=overrides.get("num_candidates", self.config.num_candidates),
            dedup_threshold=overrides.get("dedup_threshold", self.config.dedup_threshold),
        )

    def build_prompt(
        self,
        query: str,
        *,
        system: str = "Answer using only the provided context. Cite sources when possible.",
        top_k: Optional[int] = None,
        **search_kwargs,
    ) -> str:
        """Assemble a copy-paste / agent-ready prompt from retrieved chunks."""
        hits = self.search(query, top_k=top_k, **search_kwargs)
        if not hits:
            return f"{system}\n\nNo relevant context found.\n\nQuestion: {query}\n"

        context_blocks = []
        for i, hit in enumerate(hits, 1):
            src = hit.get("source", "unknown")
            context_blocks.append(
                f"[{i}] (source={src}, score={hit['score']:.3f})\n{hit['text']}"
            )
        context = "\n\n".join(context_blocks)
        return (
            f"{system}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
        )

    def status(self) -> dict:
        return {
            "persist_dir": self.persist_dir,
            "chunk_count": len(self._engine.get_chunks()),
            "source_count": len(self._sources),
            "sources": dict(self._sources),
            "config": self.config.to_dict(),
            "embedding_model": self.config.embedding_model,
        }

    def list_sources(self) -> list[dict]:
        return [{"source": k, **v} for k, v in self._sources.items()]

    def get_chunks(self, source: Optional[str] = None) -> list[dict]:
        chunks = self._engine.get_chunks()
        if source is None:
            return list(chunks)
        return [c for c in chunks if c.get("source") == source]

    def remove_source(self, source: str) -> bool:
        """Remove a source by rebuilding the index without its chunks."""
        if source not in self._sources:
            return False
        remaining = {
            src: md for src, md in self._markdown_store.items() if src != source
        }
        self.clear(keep_config=True)
        for src, md in remaining.items():
            self.ingest_markdown(md, source=src)
        return True

    def clear(self, keep_config: bool = True) -> None:
        self._engine.clear()
        self._sources.clear()
        self._markdown_store.clear()
        if not keep_config:
            self.config = KBConfig(
                embedding_model=self.config.embedding_model,
                embedding_api_key=self.config.embedding_api_key,
            )

    # ─── Persistence ─────────────────────────────────────────────

    def _meta_path(self) -> str:
        return os.path.join(self.persist_dir, "meta.json")

    def _embeddings_path(self) -> str:
        return os.path.join(self.persist_dir, "embeddings.npy")

    def _chunks_path(self) -> str:
        return os.path.join(self.persist_dir, "chunks.json")

    def save(self, persist_dir: Optional[str] = None) -> str:
        """Persist chunks, embeddings, sources, and config to disk."""
        path = persist_dir or self.persist_dir
        os.makedirs(path, exist_ok=True)
        self.persist_dir = path

        chunks = self._engine.get_chunks()
        with open(self._chunks_path(), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        if self._engine.chunk_embeddings is not None:
            np.save(self._embeddings_path(), self._engine.chunk_embeddings)

        meta = {
            "config": self.config.to_dict(),
            "sources": self._sources,
            "markdown_store": self._markdown_store,
            "total_doc_chars": self._engine._total_doc_chars,
        }
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("Saved KB (%d chunks, %d sources) → %s", len(chunks), len(self._sources), path)
        return path

    def load(self, persist_dir: Optional[str] = None) -> None:
        """Load a previously saved knowledge base."""
        import faiss
        from rank_bm25 import BM25Okapi

        path = persist_dir or self.persist_dir
        self.persist_dir = path

        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.config = KBConfig.from_dict(meta.get("config", {}))
        self._sources = meta.get("sources", {})
        self._markdown_store = meta.get("markdown_store", {})

        # Re-init engine with saved embedding model
        self._engine = RAGEngine(
            embedding_model_name=self.config.embedding_model,
            embedding_api_key=self.config.embedding_api_key,
        )

        with open(os.path.join(path, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self._engine.chunks = chunks
        self._engine._total_doc_chars = meta.get("total_doc_chars", 0)

        if chunks:
            tokenized = [self._engine._tokenize(c["text"]) for c in chunks]
            self._engine.bm25 = BM25Okapi(tokenized)
            self._engine._cache_idf(tokenized)

        emb_path = os.path.join(path, "embeddings.npy")
        if os.path.exists(emb_path) and chunks:
            embeddings = np.load(emb_path)
            self._engine.chunk_embeddings = embeddings
            dim = embeddings.shape[1]
            self._engine.faiss_index = faiss.IndexFlatIP(dim)
            self._engine.faiss_index.add(embeddings.astype(np.float32))

        logger.info("Loaded KB (%d chunks, %d sources) from %s", len(chunks), len(self._sources), path)


def create_knowledge_base(
    persist_dir: Optional[str] = None,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_api_key: Optional[str] = None,
    llm_client: Any = None,
    **config_overrides,
) -> KnowledgeBase:
    """Factory helper for apps that want a one-liner setup."""
    cfg = KBConfig(
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        **{k: v for k, v in config_overrides.items() if k in KBConfig.__dataclass_fields__},
    )
    return KnowledgeBase(persist_dir=persist_dir, config=cfg, llm_client=llm_client)
