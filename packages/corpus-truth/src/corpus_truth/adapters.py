"""Index adapters: plug your vector store in, don't fork it.

An adapter is three methods. corpus_truth never embeds, ranks, or searches --
it only tells you which postings to add and which ids to drop.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from .models import IndexMutationPlan, Posting


@runtime_checkable
class IndexAdapter(Protocol):
    """Minimum surface corpus_truth needs from a vector index."""

    def add(self, postings: Sequence[Posting]) -> None:
        """Upsert postings. Call ``posting.to_index_record()`` for a flat payload."""

    def remove(self, posting_ids: Sequence[str]) -> None:
        """Delete ids so they can no longer retrieve."""

    def list_ids(self) -> Iterable[str]:
        """Every id currently in the index. Used by :meth:`Ledger.reconcile`."""


def apply_plan(index: IndexAdapter, plan: IndexMutationPlan) -> IndexMutationPlan:
    """Apply a plan in the only safe order: add the new postings, then remove the old.

    Reversing this leaves a window where a query finds nothing for the slot.
    """
    if plan.add:
        index.add(list(plan.add))
    if plan.remove:
        index.remove(list(plan.remove))
    return plan


class InMemoryIndex:
    """Reference adapter. Stores index records in a dict -- no vectors, no ranking.

    Useful in tests and as the 15-line shape of a real adapter.
    """

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def add(self, postings: Sequence[Posting]) -> None:
        for posting in postings:
            self.records[posting.id] = posting.to_index_record()

    def remove(self, posting_ids: Sequence[str]) -> None:
        for posting_id in posting_ids:
            self.records.pop(posting_id, None)

    def list_ids(self) -> list[str]:
        return list(self.records)

    def get(self, posting_id: str) -> dict[str, Any] | None:
        """Optional hook: lets ``reconcile`` spot stale ``is_current`` / ``rev`` metadata."""
        return self.records.get(posting_id)

    def apply(self, plan: IndexMutationPlan) -> IndexMutationPlan:
        return apply_plan(self, plan)


__all__ = ["InMemoryIndex", "IndexAdapter", "apply_plan"]
