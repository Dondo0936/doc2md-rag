"""Value objects returned by the ledger."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Payload keys that describe rather than define a chunk. Used for readable diffs.
DESCRIBING_KEYS = ("uri", "caption", "description")


@dataclass(frozen=True)
class Posting:
    """One chunk as it exists in your index at one or more revisions.

    ``id`` is *your* id -- the UUID (or int, or string) you already use. We keep a
    sidecar record of it next to the payload hash and the id it replaced.
    """

    id: str
    doc_id: str
    chunk_key: str
    kind: str
    rev: int
    valid_from: int
    valid_to: int | None
    parent_rev: int | None
    replaces: str | None
    content_hash: str
    asset_hash: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_current(self) -> bool:
        return self.valid_to is None

    def to_index_record(self) -> dict[str, Any]:
        """Flat record to hand to your vector store alongside the vector.

        Payload fields are inlined (``text``, ``rows``, ``sha256``, ...) so a
        retriever hit carries everything it needs to filter and cite.
        """
        record: dict[str, Any] = {
            "id": self.id,
            "doc_id": self.doc_id,
            "key": self.chunk_key,
            "kind": self.kind,
            "rev": self.rev,
            "is_current": self.is_current,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "parent_rev": self.parent_rev,
            "replaces": self.replaces,
        }
        record.update(dict(self.payload))
        record.update(dict(self.metadata))
        return record

    def embed_text(self) -> str:
        """Best-effort text for your embedder. Empty when nothing readable was stored."""
        payload = dict(self.payload)
        if self.kind == "text":
            return str(payload.get("text", ""))
        if self.kind == "table":
            parts = []
            caption = payload.get("caption")
            if caption:
                parts.append(str(caption))
            rows = payload.get("rows")
            if rows:
                parts.extend(" | ".join(str(cell) for cell in row) for row in rows)
            return "\n".join(parts)
        if self.kind == "image":
            return str(payload.get("description") or payload.get("caption") or "")
        return ""


@dataclass(frozen=True)
class IndexMutationPlan:
    """What to do to your index, in order: add first, then remove.

    Adding the new postings before dropping the old ones means a query landing
    mid-apply may briefly see both revisions. A duplicate hit is better than a
    hole where the answer should be.
    """

    doc_id: str
    rev: int
    parent_rev: int | None
    add: tuple[Posting, ...] = ()
    remove: tuple[str, ...] = ()
    remap: Mapping[str, str] = field(default_factory=dict)
    unchanged: tuple[str, ...] = ()
    tombstone: bool = False

    @property
    def is_noop(self) -> bool:
        """True when hashes matched and your index needs no calls at all."""
        return not self.add and not self.remove

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "rev": self.rev,
            "parent_rev": self.parent_rev,
            "add": [posting.to_index_record() for posting in self.add],
            "remove": list(self.remove),
            "remap": dict(self.remap),
            "unchanged": list(self.unchanged),
            "tombstone": self.tombstone,
        }


@dataclass(frozen=True)
class Revision:
    doc_id: str
    rev: int
    parent_rev: int | None
    action: str
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentHead:
    doc_id: str
    rev: int
    action: str
    is_deleted: bool
    superseded_by: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ChunkChange:
    """One slot-level difference between two revisions.

    ``change`` is one of ``added``, ``removed``, ``changed``, ``replaced``,
    ``metadata_changed`` or ``kind_changed``.
    """

    key: str
    kind: str
    change: str
    from_id: str | None = None
    to_id: str | None = None
    from_kind: str | None = None
    to_kind: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"key": self.key, "kind": self.kind, "change": self.change}
        if self.from_id is not None:
            record["from_id"] = self.from_id
        if self.to_id is not None:
            record["to_id"] = self.to_id
        if self.change == "kind_changed":
            record["from_kind"] = self.from_kind
            record["to_kind"] = self.to_kind
        record.update(dict(self.detail))
        return record


@dataclass(frozen=True)
class Diff:
    doc_id: str
    from_rev: int | None
    to_rev: int
    added: tuple[ChunkChange, ...] = ()
    removed: tuple[ChunkChange, ...] = ()
    changed: tuple[ChunkChange, ...] = ()
    unchanged_count: int = 0
    deleted: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.added and not self.removed and not self.changed

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "from_rev": self.from_rev,
            "to_rev": self.to_rev,
            "added": [change.to_dict() for change in self.added],
            "removed": [change.to_dict() for change in self.removed],
            "changed": [change.to_dict() for change in self.changed],
            "unchanged_count": self.unchanged_count,
            "deleted": self.deleted,
        }


@dataclass(frozen=True)
class ReconcileReport:
    """Drift between the ledger's current view and what your index actually holds."""

    orphans: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    duplicate_hashes: tuple[Mapping[str, Any], ...] = ()
    stale: tuple[Mapping[str, Any], ...] = ()

    @property
    def is_clean(self) -> bool:
        return not (self.orphans or self.missing or self.unknown or self.duplicate_hashes or self.stale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "orphans": list(self.orphans),
            "missing": list(self.missing),
            "unknown": list(self.unknown),
            "duplicate_hashes": [dict(entry) for entry in self.duplicate_hashes],
            "stale": [dict(entry) for entry in self.stale],
            "is_clean": self.is_clean,
        }


def postings_by_key(postings: Sequence[Posting]) -> dict[str, Posting]:
    return {posting.chunk_key: posting for posting in postings}


__all__ = [
    "DESCRIBING_KEYS",
    "ChunkChange",
    "Diff",
    "DocumentHead",
    "IndexMutationPlan",
    "Posting",
    "ReconcileReport",
    "Revision",
    "postings_by_key",
]
