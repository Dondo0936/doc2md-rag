"""SQLite storage for the ledger.

History has to survive a process restart, so revisions and the chunk manifest
live in SQLite. Your vectors stay in your vector store; nothing here holds an
embedding or an asset.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import DocumentHead, Posting, Revision

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    current_rev   INTEGER NOT NULL,
    is_deleted    INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT
);

CREATE TABLE IF NOT EXISTS revisions (
    doc_id     TEXT NOT NULL,
    rev        INTEGER NOT NULL,
    parent_rev INTEGER,
    action     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (doc_id, rev)
);

CREATE TABLE IF NOT EXISTS postings (
    posting_id   TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    chunk_key    TEXT NOT NULL,
    kind         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    asset_hash   TEXT NOT NULL,
    payload      TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    valid_from   INTEGER NOT NULL,
    valid_to     INTEGER,
    parent_rev   INTEGER,
    replaces     TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS postings_doc_valid ON postings (doc_id, valid_from, valid_to);
CREATE INDEX IF NOT EXISTS postings_doc_slot  ON postings (doc_id, chunk_key, valid_from);
CREATE INDEX IF NOT EXISTS postings_current   ON postings (valid_to);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def resolve_db_path(path: str | Path) -> str:
    """Accept a directory (the common case), an explicit db file, or ``:memory:``."""
    if str(path) == ":memory:":
        return ":memory:"
    resolved = Path(path).expanduser()
    if resolved.suffix in {".db", ".sqlite", ".sqlite3"}:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return str(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved / "ledger.db")


class Store:
    """Thin, typed wrapper over the SQLite tables. Not part of the public API."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
                )
            elif int(row["value"]) > SCHEMA_VERSION:
                raise RuntimeError(
                    f"ledger at {self.db_path} was written by a newer corpus_truth "
                    f"(schema {row['value']} > {SCHEMA_VERSION})"
                )

    def close(self) -> None:
        self._conn.close()

    # -- transactions -----------------------------------------------------

    def transaction(self) -> sqlite3.Connection:
        """``with store.transaction():`` commits on success, rolls back on error."""
        return self._conn

    # -- documents --------------------------------------------------------

    def head(self, doc_id: str) -> DocumentHead | None:
        row = self._conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        action_row = self._conn.execute(
            "SELECT action FROM revisions WHERE doc_id = ? AND rev = ?", (doc_id, row["current_rev"])
        ).fetchone()
        return DocumentHead(
            doc_id=row["doc_id"],
            rev=int(row["current_rev"]),
            action=action_row["action"] if action_row else "commit",
            is_deleted=bool(row["is_deleted"]),
            superseded_by=row["superseded_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def document_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()
        return [row["doc_id"] for row in rows]

    def write_head(
        self,
        doc_id: str,
        *,
        rev: int,
        is_deleted: bool,
        superseded_by: str | None,
        created_at: str | None = None,
    ) -> None:
        now = utc_now()
        self._conn.execute(
            """
            INSERT INTO documents (doc_id, created_at, updated_at, current_rev, is_deleted, superseded_by)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                current_rev = excluded.current_rev,
                is_deleted = excluded.is_deleted,
                superseded_by = excluded.superseded_by
            """,
            (doc_id, created_at or now, now, rev, int(is_deleted), superseded_by),
        )

    # -- revisions --------------------------------------------------------

    def add_revision(
        self, doc_id: str, *, rev: int, parent_rev: int | None, action: str, metadata: dict[str, Any] | None
    ) -> Revision:
        created_at = utc_now()
        self._conn.execute(
            "INSERT INTO revisions (doc_id, rev, parent_rev, action, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, rev, parent_rev, action, created_at, json.dumps(metadata or {}, sort_keys=True)),
        )
        return Revision(
            doc_id=doc_id,
            rev=rev,
            parent_rev=parent_rev,
            action=action,
            created_at=created_at,
            metadata=dict(metadata or {}),
        )

    def revisions(self, doc_id: str) -> list[Revision]:
        rows = self._conn.execute("SELECT * FROM revisions WHERE doc_id = ? ORDER BY rev", (doc_id,)).fetchall()
        return [
            Revision(
                doc_id=row["doc_id"],
                rev=int(row["rev"]),
                parent_rev=None if row["parent_rev"] is None else int(row["parent_rev"]),
                action=row["action"],
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    def revision(self, doc_id: str, rev: int) -> Revision | None:
        row = self._conn.execute("SELECT * FROM revisions WHERE doc_id = ? AND rev = ?", (doc_id, rev)).fetchone()
        if row is None:
            return None
        return Revision(
            doc_id=row["doc_id"],
            rev=int(row["rev"]),
            parent_rev=None if row["parent_rev"] is None else int(row["parent_rev"]),
            action=row["action"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
        )

    # -- postings ---------------------------------------------------------

    @staticmethod
    def _posting(row: sqlite3.Row) -> Posting:
        return Posting(
            id=row["posting_id"],
            doc_id=row["doc_id"],
            chunk_key=row["chunk_key"],
            kind=row["kind"],
            rev=int(row["valid_from"]),
            valid_from=int(row["valid_from"]),
            valid_to=None if row["valid_to"] is None else int(row["valid_to"]),
            parent_rev=None if row["parent_rev"] is None else int(row["parent_rev"]),
            replaces=row["replaces"],
            content_hash=row["content_hash"],
            asset_hash=row["asset_hash"],
            payload=json.loads(row["payload"]),
            metadata=json.loads(row["metadata"]),
        )

    def insert_postings(self, postings: Iterable[Posting]) -> None:
        created_at = utc_now()
        self._conn.executemany(
            """
            INSERT INTO postings (
                posting_id, doc_id, chunk_key, kind, content_hash, asset_hash,
                payload, metadata, valid_from, valid_to, parent_rev, replaces, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    posting.id,
                    posting.doc_id,
                    posting.chunk_key,
                    posting.kind,
                    posting.content_hash,
                    posting.asset_hash,
                    json.dumps(dict(posting.payload), sort_keys=True),
                    json.dumps(dict(posting.metadata), sort_keys=True),
                    posting.valid_from,
                    posting.valid_to,
                    posting.parent_rev,
                    posting.replaces,
                    created_at,
                )
                for posting in postings
            ],
        )

    def retire_postings(self, posting_ids: Sequence[str], *, rev: int) -> None:
        self._conn.executemany(
            "UPDATE postings SET valid_to = ? WHERE posting_id = ? AND valid_to IS NULL",
            [(rev, posting_id) for posting_id in posting_ids],
        )

    def current_postings(self, doc_id: str) -> list[Posting]:
        rows = self._conn.execute(
            "SELECT * FROM postings WHERE doc_id = ? AND valid_to IS NULL ORDER BY valid_from, rowid", (doc_id,)
        ).fetchall()
        return [self._posting(row) for row in rows]

    def all_current_postings(self) -> list[Posting]:
        rows = self._conn.execute(
            "SELECT * FROM postings WHERE valid_to IS NULL ORDER BY doc_id, valid_from, rowid"
        ).fetchall()
        return [self._posting(row) for row in rows]

    def postings_as_of(self, doc_id: str, rev: int) -> list[Posting]:
        rows = self._conn.execute(
            """
            SELECT * FROM postings
            WHERE doc_id = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
            ORDER BY valid_from, rowid
            """,
            (doc_id, rev, rev),
        ).fetchall()
        return [self._posting(row) for row in rows]

    def lineage(self, doc_id: str, chunk_key: str) -> list[Posting]:
        rows = self._conn.execute(
            "SELECT * FROM postings WHERE doc_id = ? AND chunk_key = ? ORDER BY valid_from, rowid",
            (doc_id, chunk_key),
        ).fetchall()
        return [self._posting(row) for row in rows]

    def known_posting_ids(self, posting_ids: Sequence[str]) -> set[str]:
        found: set[str] = set()
        for start in range(0, len(posting_ids), 500):
            batch = posting_ids[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT posting_id FROM postings WHERE posting_id IN ({placeholders})", tuple(batch)
            ).fetchall()
            found.update(row["posting_id"] for row in rows)
        return found

    def postings_by_ids(self, posting_ids: Sequence[str]) -> dict[str, Posting]:
        result: dict[str, Posting] = {}
        for start in range(0, len(posting_ids), 500):
            batch = posting_ids[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT * FROM postings WHERE posting_id IN ({placeholders})", tuple(batch)
            ).fetchall()
            for row in rows:
                posting = self._posting(row)
                result[posting.id] = posting
        return result


__all__ = ["SCHEMA_VERSION", "Store", "resolve_db_path", "utc_now"]
