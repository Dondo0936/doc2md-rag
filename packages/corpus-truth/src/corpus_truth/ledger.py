"""The ledger: a sidecar that versions your chunks and plans your index writes.

Your vector store keeps ``id + vector + metadata`` and your own UUIDs. The
ledger records, next to it, which of your ids is current for each slot, which id
replaced which, what each payload hashed to, and both sides of every change.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .adapters import IndexAdapter
from .chunks import Chunk
from .diffing import compute_diff, render_prompt
from .errors import (
    DocumentNotFoundError,
    DuplicateChunkKeyError,
    DuplicateIdError,
    InvalidChunkError,
    RevisionNotFoundError,
)
from .models import Diff, DocumentHead, IndexMutationPlan, Posting, ReconcileReport, Revision
from .store import Store, resolve_db_path

TOMBSTONE_ACTIONS = frozenset({"delete", "supersede"})


class Ledger:
    """Versioned corpus truth for one SQLite file.

    Open it once per process::

        with Ledger.open("./.corpus") as ledger:
            plan = ledger.commit("policy", chunks)
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self._lock = threading.RLock()

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> Ledger:
        """Open (creating if needed) a ledger.

        ``path`` may be a directory -- ``./.corpus/ledger.db`` is used inside it --
        an explicit ``*.db`` / ``*.sqlite`` file, or ``":memory:"``.
        """
        return cls(Store(resolve_db_path(path)))

    @property
    def path(self) -> str:
        return self._store.db_path

    def close(self) -> None:
        self._store.close()

    def __enter__(self):  # typing.Self needs Python 3.11; inference is enough here
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- writes -----------------------------------------------------------

    def commit(
        self,
        doc_id: str,
        chunks: Iterable[Chunk],
        metadata: Mapping[str, Any] | None = None,
    ) -> IndexMutationPlan:
        """Record a revision of ``doc_id`` and return the index mutations it implies.

        Chunks whose payload hash matches the current revision are skipped: they
        keep their existing id and vector, and never appear in ``plan.add``. If
        nothing changed at all, no revision is created and the plan is a no-op.

        Apply the result in order: ``plan.add`` first, then ``plan.remove``.
        """
        doc_id = _require_doc_id(doc_id)
        prepared = _prepare_chunks(chunks)

        with self._lock:
            head = self._store.head(doc_id)
            existing = {posting.chunk_key: posting for posting in self._store.current_postings(doc_id)}

            replacements: list[tuple[Chunk, str, Posting | None]] = []
            retire: list[str] = []
            unchanged: list[str] = []
            seen: set[str] = set()

            for chunk, slot in prepared:
                seen.add(slot)
                previous = existing.get(slot)
                if previous is not None and previous.content_hash == chunk.content_hash:
                    unchanged.append(previous.id)
                    continue
                replacements.append((chunk, slot, previous))
                if previous is not None:
                    retire.append(previous.id)

            for slot, posting in existing.items():
                if slot not in seen:
                    retire.append(posting.id)

            if head is not None and not replacements and not retire:
                return IndexMutationPlan(
                    doc_id=doc_id,
                    rev=head.rev,
                    parent_rev=self._parent_rev(doc_id, head.rev),
                    unchanged=tuple(unchanged),
                )

            new_rev = 1 if head is None else head.rev + 1
            postings = self._build_postings(doc_id, replacements, new_rev)
            remap = {
                previous.id: posting.id
                for (_, _, previous), posting in zip(replacements, postings)
                if previous is not None
            }

            with self._store.transaction():
                self._store.add_revision(
                    doc_id,
                    rev=new_rev,
                    parent_rev=None if head is None else head.rev,
                    action="commit",
                    metadata=dict(metadata or {}),
                )
                if postings:
                    self._store.insert_postings(postings)
                if retire:
                    self._store.retire_postings(retire, rev=new_rev)
                self._store.write_head(
                    doc_id,
                    rev=new_rev,
                    is_deleted=False,
                    superseded_by=None,
                    created_at=None if head is None else head.created_at,
                )

            return IndexMutationPlan(
                doc_id=doc_id,
                rev=new_rev,
                parent_rev=None if head is None else head.rev,
                add=tuple(postings),
                remove=tuple(retire),
                remap=remap,
                unchanged=tuple(unchanged),
            )

    def delete(self, doc_id: str) -> IndexMutationPlan:
        """Tombstone a document. Its ids must be removed from the index so they cannot retrieve.

        History stays: ``as_of(doc_id, rev=<earlier>)`` still returns the postings.
        """
        doc_id = _require_doc_id(doc_id)
        with self._lock:
            head = self._require_head(doc_id)
            current = self._store.current_postings(doc_id)
            if head.is_deleted and not current:
                return IndexMutationPlan(
                    doc_id=doc_id,
                    rev=head.rev,
                    parent_rev=self._parent_rev(doc_id, head.rev),
                    tombstone=True,
                )

            new_rev = head.rev + 1
            remove = [posting.id for posting in current]
            with self._store.transaction():
                self._store.add_revision(
                    doc_id, rev=new_rev, parent_rev=head.rev, action="delete", metadata={}
                )
                if remove:
                    self._store.retire_postings(remove, rev=new_rev)
                self._store.write_head(
                    doc_id,
                    rev=new_rev,
                    is_deleted=True,
                    superseded_by=head.superseded_by,
                    created_at=head.created_at,
                )

            return IndexMutationPlan(
                doc_id=doc_id,
                rev=new_rev,
                parent_rev=head.rev,
                remove=tuple(remove),
                tombstone=True,
            )

    def supersede(self, old_doc_id: str, new_doc_id: str) -> IndexMutationPlan:
        """Retire ``old_doc_id`` in favour of an already-committed ``new_doc_id``.

        Commit the replacement first: the returned plan only removes the old
        document's ids, so the new revision is searchable before the old one goes.
        """
        old_doc_id = _require_doc_id(old_doc_id)
        new_doc_id = _require_doc_id(new_doc_id)
        if old_doc_id == new_doc_id:
            raise ValueError("a document cannot supersede itself")

        with self._lock:
            old_head = self._require_head(old_doc_id)
            self._require_head(new_doc_id)
            current = self._store.current_postings(old_doc_id)

            if old_head.is_deleted and old_head.superseded_by == new_doc_id and not current:
                return IndexMutationPlan(
                    doc_id=old_doc_id,
                    rev=old_head.rev,
                    parent_rev=self._parent_rev(old_doc_id, old_head.rev),
                    tombstone=True,
                )

            new_rev = old_head.rev + 1
            remove = [posting.id for posting in current]
            with self._store.transaction():
                self._store.add_revision(
                    old_doc_id,
                    rev=new_rev,
                    parent_rev=old_head.rev,
                    action="supersede",
                    metadata={"superseded_by": new_doc_id},
                )
                if remove:
                    self._store.retire_postings(remove, rev=new_rev)
                self._store.write_head(
                    old_doc_id,
                    rev=new_rev,
                    is_deleted=True,
                    superseded_by=new_doc_id,
                    created_at=old_head.created_at,
                )

            return IndexMutationPlan(
                doc_id=old_doc_id,
                rev=new_rev,
                parent_rev=old_head.rev,
                remove=tuple(remove),
                tombstone=True,
            )

    # -- views ------------------------------------------------------------

    def current(self, doc_id: str) -> list[Posting]:
        """Postings that should be searchable right now. Empty for a deleted document."""
        doc_id = _require_doc_id(doc_id)
        with self._lock:
            self._require_head(doc_id)
            return self._store.current_postings(doc_id)

    def as_of(self, doc_id: str, rev: int | None = None) -> list[Posting]:
        """Postings that were current at ``rev`` (default: the latest revision)."""
        doc_id = _require_doc_id(doc_id)
        with self._lock:
            head = self._require_head(doc_id)
            target = head.rev if rev is None else int(rev)
            self._require_revision(doc_id, target)
            return self._store.postings_as_of(doc_id, target)

    def lineage(self, doc_id: str, chunk_key: str) -> list[Posting]:
        """Every posting that has ever held one slot, oldest first -- the uuid-x -> uuid-y trail."""
        doc_id = _require_doc_id(doc_id)
        with self._lock:
            self._require_head(doc_id)
            return self._store.lineage(doc_id, chunk_key)

    def head(self, doc_id: str) -> DocumentHead:
        with self._lock:
            return self._require_head(_require_doc_id(doc_id))

    def revisions(self, doc_id: str) -> list[Revision]:
        doc_id = _require_doc_id(doc_id)
        with self._lock:
            self._require_head(doc_id)
            return self._store.revisions(doc_id)

    def documents(self) -> list[str]:
        with self._lock:
            return self._store.document_ids()

    # -- diffs ------------------------------------------------------------

    def diff(self, doc_id: str, from_rev: int | None = None, to_rev: int | None = None) -> Diff:
        """Chunk-level changelog between two revisions.

        Defaults compare the latest revision with its parent. Pass ``from_rev=0``
        to diff against an empty document.
        """
        doc_id = _require_doc_id(doc_id)
        with self._lock:
            head = self._require_head(doc_id)
            target = head.rev if to_rev is None else int(to_rev)
            to_revision = self._require_revision(doc_id, target)

            source = to_revision.parent_rev if from_rev is None else int(from_rev)
            if source is not None and source > target:
                raise ValueError(f"from_rev {source} is after to_rev {target}")

            if source is None or source == 0:
                from_postings: list[Posting] = []
            else:
                self._require_revision(doc_id, source)
                from_postings = self._store.postings_as_of(doc_id, source)

            return compute_diff(
                doc_id,
                from_rev=source,
                to_rev=target,
                from_postings=from_postings,
                to_postings=self._store.postings_as_of(doc_id, target),
                deleted=to_revision.action in TOMBSTONE_ACTIONS,
            )

    def diff_prompt(self, doc_id: str, from_rev: int | None = None, to_rev: int | None = None) -> str:
        """The same changelog as short markdown, ready to staple onto a prompt."""
        with self._lock:
            diff = self.diff(doc_id, from_rev=from_rev, to_rev=to_rev)
            return render_prompt(diff, current_rev=self._require_head(doc_id).rev)

    # -- drift ------------------------------------------------------------

    def reconcile(self, index: IndexAdapter, *, doc_id: str | None = None) -> ReconcileReport:
        """Compare the ledger's current view with what the index actually holds.

        * ``orphans`` -- ids in the index the ledger has retired (ghosts that can still retrieve)
        * ``missing`` -- current ids the index never received
        * ``unknown`` -- ids the ledger has never seen (only reported for a whole-ledger check)
        * ``duplicate_hashes`` -- current postings of one document sharing a payload hash
        * ``stale`` -- index metadata that disagrees with the ledger (needs an adapter ``get``)
        """
        with self._lock:
            index_ids = [str(posting_id) for posting_id in index.list_ids()]
            index_set = set(index_ids)

            if doc_id is None:
                current = self._store.all_current_postings()
            else:
                doc_id = _require_doc_id(doc_id)
                self._require_head(doc_id)
                current = self._store.current_postings(doc_id)
            current_by_id = {posting.id: posting for posting in current}

            known = self._store.postings_by_ids(index_ids)
            orphans: list[str] = []
            unknown: list[str] = []
            for posting_id in index_ids:
                if posting_id in current_by_id:
                    continue
                posting = known.get(posting_id)
                if posting is None:
                    if doc_id is None:
                        unknown.append(posting_id)
                elif doc_id is None or posting.doc_id == doc_id:
                    orphans.append(posting_id)

            missing = [posting_id for posting_id in current_by_id if posting_id not in index_set]

            hashes: dict[tuple[str, str], list[str]] = {}
            for posting in current:
                hashes.setdefault((posting.doc_id, posting.content_hash), []).append(posting.id)
            duplicate_hashes = [
                {"doc_id": key[0], "content_hash": key[1], "ids": sorted(ids)}
                for key, ids in sorted(hashes.items())
                if len(ids) > 1
            ]

            stale = _stale_records(index, current_by_id)

            return ReconcileReport(
                orphans=tuple(sorted(orphans)),
                missing=tuple(sorted(missing)),
                unknown=tuple(sorted(unknown)),
                duplicate_hashes=tuple(duplicate_hashes),
                stale=tuple(stale),
            )

    # -- internals --------------------------------------------------------

    def _require_head(self, doc_id: str) -> DocumentHead:
        head = self._store.head(doc_id)
        if head is None:
            raise DocumentNotFoundError(f"no document {doc_id!r} in the ledger")
        return head

    def _require_revision(self, doc_id: str, rev: int) -> Revision:
        revision = self._store.revision(doc_id, rev)
        if revision is None:
            raise RevisionNotFoundError(f"document {doc_id!r} has no revision {rev}")
        return revision

    def _parent_rev(self, doc_id: str, rev: int) -> int | None:
        revision = self._store.revision(doc_id, rev)
        return None if revision is None else revision.parent_rev

    def _build_postings(
        self, doc_id: str, replacements: Sequence[tuple[Chunk, str, Posting | None]], rev: int
    ) -> list[Posting]:
        requested = [chunk.id for chunk, _, _ in replacements if chunk.id is not None]
        duplicates = {posting_id for posting_id in requested if requested.count(posting_id) > 1}
        if duplicates:
            raise DuplicateIdError(f"ids repeated within one commit: {', '.join(sorted(duplicates))}")
        clashes = self._store.known_posting_ids(requested)
        if clashes:
            raise DuplicateIdError(
                "ids already recorded in the ledger (a changed chunk needs a new id): "
                + ", ".join(sorted(clashes))
            )

        postings: list[Posting] = []
        for chunk, slot, previous in replacements:
            postings.append(
                Posting(
                    id=chunk.id or uuid.uuid4().hex,
                    doc_id=doc_id,
                    chunk_key=slot,
                    kind=chunk.kind,
                    rev=rev,
                    valid_from=rev,
                    valid_to=None,
                    parent_rev=None if previous is None else previous.valid_from,
                    replaces=None if previous is None else previous.id,
                    content_hash=chunk.content_hash,
                    asset_hash=chunk.asset_hash(),
                    payload=chunk.payload(),
                    metadata=dict(chunk.metadata),
                )
            )
        return postings


def _stale_records(index: IndexAdapter, current_by_id: Mapping[str, Posting]) -> list[dict[str, Any]]:
    getter = getattr(index, "get", None)
    if not callable(getter):
        return []
    stale: list[dict[str, Any]] = []
    for posting_id, posting in current_by_id.items():
        record = getter(posting_id)
        if not record:
            continue
        if record.get("is_current") is False:
            stale.append({"id": posting_id, "field": "is_current", "index": False, "ledger": True})
        record_rev = record.get("rev")
        if record_rev is not None and int(record_rev) != posting.rev:
            stale.append({"id": posting_id, "field": "rev", "index": record_rev, "ledger": posting.rev})
    return sorted(stale, key=lambda entry: (entry["id"], entry["field"]))


def _require_doc_id(doc_id: str) -> str:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")
    return doc_id


def _prepare_chunks(chunks: Iterable[Chunk]) -> list[tuple[Chunk, str]]:
    prepared: list[tuple[Chunk, str]] = []
    seen: dict[str, int] = {}
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, Chunk):
            raise InvalidChunkError(
                f"chunk at position {position} is {type(chunk).__name__}, expected a TextChunk / "
                "TableChunk / ImageChunk / OtherChunk"
            )
        slot = chunk.slot_key
        if slot in seen:
            raise DuplicateChunkKeyError(
                f"chunk key {slot!r} appears twice in one commit (positions {seen[slot]} and {position})"
            )
        seen[slot] = position
        prepared.append((chunk, slot))
    return prepared


__all__ = ["Ledger"]
