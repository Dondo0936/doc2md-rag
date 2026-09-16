"""current(), as_of(), revisions() and SQLite persistence."""

import pytest
from conftest import ids_by_key, policy_v1, policy_v2

from corpus_truth import (
    DocumentNotFoundError,
    Ledger,
    RevisionNotFoundError,
    TextChunk,
)


def test_as_of_returns_the_state_of_each_revision(ledger):
    first = ledger.commit("policy", policy_v1())
    second = ledger.commit(
        "policy",
        [TextChunk("Refunds take 14 days.", key="refund"), *policy_v1()[1:]],
    )

    at_one = ids_by_key(ledger.as_of("policy", rev=1))
    at_two = ids_by_key(ledger.as_of("policy", rev=2))

    assert at_one == ids_by_key(first.add)
    assert at_two["refund"] == ids_by_key(second.add)["refund"]
    # An unchanged slot appears at both revisions under the same id.
    assert at_one["pricing"] == at_two["pricing"]


def test_as_of_defaults_to_the_current_revision(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())

    assert ids_by_key(ledger.as_of("policy")) == ids_by_key(ledger.current("policy"))


def test_historical_postings_report_their_validity_window(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())

    retired = {posting.chunk_key: posting for posting in ledger.as_of("policy", rev=1)}
    assert all(posting.valid_from == 1 and posting.valid_to == 2 for posting in retired.values())
    assert all(posting.to_index_record()["is_current"] is False for posting in retired.values())

    current = {posting.chunk_key: posting for posting in ledger.current("policy")}
    assert all(posting.valid_from == 2 and posting.valid_to is None for posting in current.values())


def test_revisions_and_head_describe_the_document_timeline(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())
    ledger.delete("policy")

    revisions = ledger.revisions("policy")
    assert [(revision.rev, revision.parent_rev, revision.action) for revision in revisions] == [
        (1, None, "commit"),
        (2, 1, "commit"),
        (3, 2, "delete"),
    ]

    head = ledger.head("policy")
    assert (head.rev, head.action, head.is_deleted, head.superseded_by) == (3, "delete", True, None)
    assert head.created_at <= head.updated_at


def test_unknown_documents_and_revisions_raise(ledger):
    with pytest.raises(DocumentNotFoundError):
        ledger.current("nope")
    with pytest.raises(DocumentNotFoundError):
        ledger.as_of("nope", rev=1)
    with pytest.raises(DocumentNotFoundError):
        ledger.revisions("nope")
    with pytest.raises(DocumentNotFoundError):
        ledger.lineage("nope", "refund")

    ledger.commit("policy", policy_v1())
    with pytest.raises(RevisionNotFoundError, match="revision 7"):
        ledger.as_of("policy", rev=7)
    with pytest.raises(RevisionNotFoundError, match="revision 0"):
        ledger.as_of("policy", rev=0)


def test_ledger_survives_a_restart(tmp_path):
    directory = tmp_path / "corpus"
    with Ledger.open(directory) as ledger:
        first = ledger.commit("policy", policy_v1())
        second = ledger.commit("policy", policy_v2())
        expected_current = ids_by_key(second.add)
        expected_history = ids_by_key(first.add)
        prompt = ledger.diff_prompt("policy")

    assert (directory / "ledger.db").exists()

    with Ledger.open(directory) as reopened:
        assert reopened.documents() == ["policy"]
        assert ids_by_key(reopened.current("policy")) == expected_current
        assert ids_by_key(reopened.as_of("policy", rev=1)) == expected_history
        assert reopened.head("policy").rev == 2
        assert reopened.diff_prompt("policy") == prompt
        assert reopened.lineage("policy", "refund")[0].payload["text"] == "Refunds take 30 days."

        # And it keeps accepting writes on top of the restored state.
        third = reopened.commit("policy", [TextChunk("Refunds take 7 days.", key="refund"), *policy_v2()[1:]])
        assert third.rev == 3
        assert third.remove == (expected_current["refund"],)


def test_a_ledger_file_path_is_accepted_as_well_as_a_directory(tmp_path):
    with Ledger.open(tmp_path / "nested" / "corpus.sqlite") as ledger:
        ledger.commit("policy", policy_v1())
        assert ledger.path.endswith("corpus.sqlite")
    assert (tmp_path / "nested" / "corpus.sqlite").exists()


def test_an_in_memory_ledger_needs_no_files(tmp_path):
    with Ledger.open(":memory:") as ledger:
        ledger.commit("policy", policy_v1())
        assert len(ledger.current("policy")) == 3
    assert list(tmp_path.iterdir()) == []
