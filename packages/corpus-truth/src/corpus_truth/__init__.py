"""corpus_truth -- versioned corpus truth for RAG indexes.

You parse, chunk, embed, retrieve and generate. corpus_truth sits between chunk
and index: it assigns revisions, tells you exactly what to add and remove, keeps
history, and hands generation a changelog an LLM can read.

    from corpus_truth import Ledger, TextChunk

    ledger = Ledger.open("./.corpus")
    plan = ledger.commit("policy", [TextChunk("Refunds take 14 days.", key="refund")])
    index.add(plan.add)        # embed only what changed
    index.remove(plan.remove)  # drop stale ids so they cannot retrieve
"""

from __future__ import annotations

from .adapters import IndexAdapter, InMemoryIndex, apply_plan
from .chunks import Chunk, ChunkKind, ImageChunk, OtherChunk, TableChunk, TextChunk
from .errors import (
    CorpusTruthError,
    DocumentNotFoundError,
    DuplicateChunkKeyError,
    DuplicateIdError,
    InvalidChunkError,
    RevisionNotFoundError,
)
from .ledger import Ledger
from .models import (
    ChunkChange,
    Diff,
    DocumentHead,
    IndexMutationPlan,
    Posting,
    ReconcileReport,
    Revision,
)

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "ChunkChange",
    "ChunkKind",
    "CorpusTruthError",
    "Diff",
    "DocumentHead",
    "DocumentNotFoundError",
    "DuplicateChunkKeyError",
    "DuplicateIdError",
    "ImageChunk",
    "InMemoryIndex",
    "IndexAdapter",
    "IndexMutationPlan",
    "InvalidChunkError",
    "Ledger",
    "OtherChunk",
    "Posting",
    "ReconcileReport",
    "Revision",
    "RevisionNotFoundError",
    "TableChunk",
    "TextChunk",
    "__version__",
    "apply_plan",
]
