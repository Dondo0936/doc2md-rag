"""Atomicity, concurrent handles and threaded writers."""

import threading

import pytest
from conftest import policy_v1, policy_v2

from corpus_truth import Ledger, TextChunk


def test_a_failed_commit_leaves_no_partial_revision(ledger, monkeypatch):
    first = ledger.commit("policy", policy_v1())

    def explode(*args, **kwargs):
        raise RuntimeError("disk went away")

    monkeypatch.setattr(ledger._store, "retire_postings", explode)
    with pytest.raises(RuntimeError, match="disk went away"):
        ledger.commit("policy", policy_v2())
    monkeypatch.undo()

    assert ledger.head("policy").rev == 1
    assert [revision.rev for revision in ledger.revisions("policy")] == [1]
    assert {posting.id for posting in ledger.current("policy")} == {posting.id for posting in first.add}

    # And the ledger still accepts the retried commit.
    retried = ledger.commit("policy", policy_v2())
    assert retried.rev == 2
    assert set(retried.remove) == {posting.id for posting in first.add}


def test_two_handles_on_one_file_see_each_others_writes(tmp_path):
    directory = tmp_path / "corpus"
    with Ledger.open(directory) as writer, Ledger.open(directory) as reader:
        plan = writer.commit("policy", policy_v1())

        assert reader.documents() == ["policy"]
        assert {posting.id for posting in reader.current("policy")} == {posting.id for posting in plan.add}

        writer.delete("policy")
        assert reader.current("policy") == []
        assert reader.head("policy").is_deleted is True


def test_concurrent_commits_to_different_documents_all_land(tmp_path):
    with Ledger.open(tmp_path / "corpus") as ledger:
        errors: list[BaseException] = []
        revisions: dict[str, int] = {}
        barrier = threading.Barrier(8)

        def worker(number: int) -> None:
            try:
                barrier.wait(timeout=10)
                plan = ledger.commit(f"doc-{number}", [TextChunk(f"body {number}", key="body")])
                revisions[f"doc-{number}"] = plan.rev
            except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(number,)) for number in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert errors == []
        assert ledger.documents() == [f"doc-{number}" for number in range(8)]
        assert set(revisions.values()) == {1}
        assert all(len(ledger.current(f"doc-{number}")) == 1 for number in range(8))


def test_concurrent_commits_to_one_document_serialize(tmp_path):
    with Ledger.open(tmp_path / "corpus") as ledger:
        ledger.commit("policy", [TextChunk("rev 1", key="body")])
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def worker(number: int) -> None:
            try:
                barrier.wait(timeout=10)
                ledger.commit("policy", [TextChunk(f"body from writer {number}", key="body")])
            except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(number,)) for number in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert errors == []
        revisions = [revision.rev for revision in ledger.revisions("policy")]
        assert revisions == list(range(1, len(revisions) + 1))
        assert ledger.head("policy").rev == revisions[-1]
        assert len(ledger.current("policy")) == 1
