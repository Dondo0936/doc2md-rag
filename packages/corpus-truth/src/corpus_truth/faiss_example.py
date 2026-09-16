"""Copy-paste FAISS adapter example.

``faiss`` and ``numpy`` are imported lazily so installing corpus_truth never
pulls a vector library in. Treat this as a 40-line template for whatever store
you actually run: the only contract is ``add`` / ``remove`` / ``list_ids``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .models import Posting


class FaissIndexAdapter:
    """Maps corpus_truth postings onto a ``faiss.IndexIDMap2``.

    FAISS addresses vectors by int64, so string ids get a stable integer alias.
    ``embed`` is yours -- corpus_truth never owns an embedding model.
    """

    def __init__(self, dim: int, embed: Callable[[Sequence[str]], Any]) -> None:
        import faiss

        self._faiss = faiss
        self.embed = embed
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._ids: dict[str, int] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._next_alias = 1

    def _alias(self, posting_id: str) -> int:
        if posting_id not in self._ids:
            self._ids[posting_id] = self._next_alias
            self._next_alias += 1
        return self._ids[posting_id]

    def add(self, postings: Sequence[Posting]) -> None:
        import numpy as np

        if not postings:
            return
        vectors = np.asarray(self.embed([posting.embed_text() for posting in postings]), dtype="float32")
        aliases = np.asarray([self._alias(posting.id) for posting in postings], dtype="int64")
        self.index.add_with_ids(vectors, aliases)
        for posting in postings:
            self._records[posting.id] = posting.to_index_record()

    def remove(self, posting_ids: Sequence[str]) -> None:
        import numpy as np

        aliases = [self._ids.pop(posting_id) for posting_id in posting_ids if posting_id in self._ids]
        for posting_id in posting_ids:
            self._records.pop(posting_id, None)
        if aliases:
            self.index.remove_ids(np.asarray(aliases, dtype="int64"))

    def list_ids(self) -> list[str]:
        return list(self._records)

    def get(self, posting_id: str) -> dict[str, Any] | None:
        return self._records.get(posting_id)


__all__ = ["FaissIndexAdapter"]
