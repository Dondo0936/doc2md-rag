import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

# Tests run straight from a checkout, installed or not.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from corpus_truth import ImageChunk, InMemoryIndex, Ledger, Posting, TableChunk, TextChunk


@pytest.fixture
def ledger(tmp_path):
    with Ledger.open(tmp_path / "corpus") as opened:
        yield opened


@pytest.fixture
def index():
    return InMemoryIndex()


class RecordingIndex(InMemoryIndex):
    """In-memory adapter that remembers the order of calls, for apply-order assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def add(self, postings: Sequence[Posting]) -> None:
        self.calls.append(("add", tuple(posting.id for posting in postings)))
        super().add(postings)

    def remove(self, posting_ids: Sequence[str]) -> None:
        self.calls.append(("remove", tuple(posting_ids)))
        super().remove(posting_ids)


@pytest.fixture
def recording_index():
    return RecordingIndex()


def policy_v1():
    """The three typed chunks from the spec walkthrough, revision 1."""
    return [
        TextChunk("Refunds take 30 days.", key="refund"),
        TableChunk([["item", "price"], ["A", "9"]], key="pricing"),
        ImageChunk(
            "aaa",
            uri="s3://old.png",
            caption="Fig 1",
            description="Wiring diagram with 3 connectors.",
            key="fig1",
        ),
    ]


def policy_v2():
    """Same slots, changed payloads."""
    return [
        TextChunk("Refunds take 14 days.", key="refund"),
        TableChunk([["item", "price"], ["A", "12"]], key="pricing"),
        ImageChunk(
            "bbb",
            uri="s3://new.png",
            caption="Fig 1",
            description="Wiring diagram with 5 connectors.",
            key="fig1",
        ),
    ]


def ids_by_key(postings) -> dict[str, str]:
    return {posting.chunk_key: posting.id for posting in postings}
