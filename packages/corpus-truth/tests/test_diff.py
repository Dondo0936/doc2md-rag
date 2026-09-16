"""Kind-aware diffs and the changelog handed to an LLM."""

import pytest
from conftest import policy_v1, policy_v2

from corpus_truth import ImageChunk, OtherChunk, TableChunk, TextChunk
from corpus_truth.diffing import a1_ref, summarize_text_change


def changes_by_key(diff):
    return {change.key: change for change in diff.changed}


def test_diff_defaults_to_the_latest_revision_against_its_parent(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())

    diff = ledger.diff("policy")

    assert (diff.doc_id, diff.from_rev, diff.to_rev) == ("policy", 1, 2)
    assert diff.added == () and diff.removed == ()
    assert set(changes_by_key(diff)) == {"refund", "pricing", "fig1"}


def test_text_change_carries_a_patch_and_both_sides(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())

    change = changes_by_key(ledger.diff("policy"))["refund"]

    assert change.kind == "text"
    assert change.change == "changed"
    assert change.detail["patch"] == "- Refunds take 30 days.\n+ Refunds take 14 days."
    assert change.detail["from_text"] == "Refunds take 30 days."
    assert change.detail["to_text"] == "Refunds take 14 days."
    assert change.to_dict()["key"] == "refund"


def test_table_change_reports_cells_not_a_text_dump(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())

    change = changes_by_key(ledger.diff("policy"))["pricing"]

    assert change.detail["cells"] == [{"row": 1, "col": 1, "ref": "B2", "from": "9", "to": "12"}]
    assert change.detail["rows_added"] == []
    assert change.detail["rows_removed"] == []
    assert change.detail["from_shape"] == [2, 2]


def test_table_row_insertion_is_not_a_cascade_of_cell_edits(ledger):
    ledger.commit("policy", [TableChunk([["item", "price"], ["A", "9"]], key="pricing")])
    ledger.commit("policy", [TableChunk([["item", "price"], ["B", "4"], ["A", "9"]], key="pricing")])

    detail = changes_by_key(ledger.diff("policy"))["pricing"].detail

    assert detail["cells"] == []
    assert detail["rows_added"] == [{"row": 1, "cells": ["B", "4"]}]
    assert detail["rows_removed"] == []


def test_table_without_rows_degrades_to_changed(ledger):
    ledger.commit("policy", [TableChunk(hash="csv-1", key="pricing")])
    ledger.commit("policy", [TableChunk(hash="csv-2", key="pricing")])

    change = changes_by_key(ledger.diff("policy"))["pricing"]

    assert change.change == "changed"
    assert "cells" not in change.detail
    assert change.detail["note"] == "table changed; rows not stored so no cell diff"
    assert (change.detail["from_hash"], change.detail["to_hash"]) == ("csv-1", "csv-2")
    assert "rows not stored" in ledger.diff_prompt("policy")


def test_image_change_is_a_replacement_with_both_uris_and_descriptions(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())

    change = changes_by_key(ledger.diff("policy"))["fig1"]

    assert change.change == "replaced"
    assert (change.detail["from_sha256"], change.detail["to_sha256"]) == ("aaa", "bbb")
    assert (change.detail["from_uri"], change.detail["to_uri"]) == ("s3://old.png", "s3://new.png")
    assert change.detail["description"] == {
        "from": "Wiring diagram with 3 connectors.",
        "to": "Wiring diagram with 5 connectors.",
    }


def test_image_without_a_description_degrades_to_replaced(ledger):
    ledger.commit("policy", [ImageChunk("aaa", key="fig1")])
    ledger.commit("policy", [ImageChunk("bbb", key="fig1")])

    change = changes_by_key(ledger.diff("policy"))["fig1"]

    assert change.change == "replaced"
    assert "description" not in change.detail
    assert change.detail["note"] == "image replaced; no description stored so no readable changelog"
    assert "no description stored" in ledger.diff_prompt("policy")


def test_a_caption_only_edit_is_a_metadata_change(ledger):
    ledger.commit("policy", [ImageChunk("aaa", caption="Fig 1", description="Three connectors.", key="fig1")])
    ledger.commit("policy", [ImageChunk("aaa", caption="Figure 1", description="Three connectors.", key="fig1")])

    change = changes_by_key(ledger.diff("policy"))["fig1"]

    assert change.change == "metadata_changed"
    assert change.detail["caption"] == {"from": "Fig 1", "to": "Figure 1"}
    assert "asset unchanged" in ledger.diff_prompt("policy")


def test_other_chunks_report_replacement_and_metadata(ledger):
    ledger.commit("policy", [OtherChunk("audio/wav", hash="h1", key="intro", metadata={"seconds": 30})])
    ledger.commit("policy", [OtherChunk("audio/wav", hash="h2", key="intro", metadata={"seconds": 45})])

    change = changes_by_key(ledger.diff("policy"))["intro"]

    assert (change.kind, change.change) == ("other", "replaced")
    assert change.detail["mime"] == "audio/wav"
    assert (change.detail["from_hash"], change.detail["to_hash"]) == ("h1", "h2")
    assert change.detail["metadata"] == {"from": {"seconds": 30}, "to": {"seconds": 45}}


def test_a_kind_change_in_one_slot_is_reported_as_such(ledger):
    ledger.commit("policy", [TableChunk([["item", "price"], ["A", "9"]], key="pricing")])
    ledger.commit("policy", [TextChunk("Price of A is 9.", key="pricing")])

    change = changes_by_key(ledger.diff("policy"))["pricing"]

    assert change.change == "kind_changed"
    assert (change.from_kind, change.to_kind) == ("table", "text")
    assert change.to_dict()["from_kind"] == "table"
    assert "kind changed table" in ledger.diff_prompt("policy")


def test_added_and_removed_slots_are_separated_from_changes(ledger):
    ledger.commit("policy", [TextChunk("Intro.", key="intro"), TextChunk("Old clause.", key="clause")])
    ledger.commit("policy", [TextChunk("Intro.", key="intro"), TextChunk("Appendix.", key="appendix")])

    diff = ledger.diff("policy")

    assert [change.key for change in diff.added] == ["appendix"]
    assert [change.key for change in diff.removed] == ["clause"]
    assert diff.changed == ()
    assert diff.unchanged_count == 1
    assert diff.added[0].detail["payload"]["text"] == "Appendix."
    assert diff.removed[0].detail["payload"]["text"] == "Old clause."


def test_unkeyed_chunks_read_as_a_removal_plus_an_addition(ledger):
    ledger.commit("policy", [TextChunk("Refunds take 30 days.")])
    ledger.commit("policy", [TextChunk("Refunds take 14 days.")])

    diff = ledger.diff("policy")

    assert diff.changed == ()
    assert len(diff.added) == 1 and len(diff.removed) == 1


def test_diff_from_rev_zero_shows_a_document_built_from_nothing(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())

    diff = ledger.diff("policy", from_rev=0)

    assert diff.from_rev == 0
    assert len(diff.added) == 3
    assert diff.changed == () and diff.removed == ()


def test_diff_spans_more_than_one_revision(ledger):
    ledger.commit("policy", [TextChunk("30 days", key="refund")])
    ledger.commit("policy", [TextChunk("21 days", key="refund")])
    ledger.commit("policy", [TextChunk("14 days", key="refund")])

    diff = ledger.diff("policy", from_rev=1, to_rev=3)

    assert (diff.from_rev, diff.to_rev) == (1, 3)
    assert changes_by_key(diff)["refund"].detail["from_text"] == "30 days"
    assert changes_by_key(diff)["refund"].detail["to_text"] == "14 days"


def test_first_revision_diffs_against_an_empty_document(ledger):
    ledger.commit("policy", policy_v1())

    diff = ledger.diff("policy")

    assert diff.from_rev is None
    assert len(diff.added) == 3
    assert "first revision" in ledger.diff_prompt("policy")


def test_diff_rejects_an_inverted_range(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())

    with pytest.raises(ValueError, match="after to_rev"):
        ledger.diff("policy", from_rev=2, to_rev=1)


def test_diff_prompt_names_every_change_in_llm_readable_terms(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v2())

    prompt = ledger.diff_prompt("policy")

    assert prompt.splitlines() == [
        "Document policy is now rev 2 (was 1).",
        '- text/refund: "30 days." \u2192 "14 days."',
        "- table/pricing cell B2: 9 \u2192 12",
        '- image/fig1: "Wiring diagram with 3 connectors." \u2192 "Wiring diagram with 5 connectors."',
        "  old: s3://old.png",
        "  new: s3://new.png",
        "Current revision is 2.",
    ]


def test_diff_prompt_counts_what_stayed_and_states_the_current_rev(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", [TextChunk("Refunds take 14 days.", key="refund"), *policy_v1()[1:]])
    ledger.commit("policy", policy_v1())

    prompt = ledger.diff_prompt("policy", from_rev=1, to_rev=2)

    assert prompt.endswith("2 chunks unchanged. Current revision is 3.")


def test_diff_prompt_reports_a_deletion(ledger):
    ledger.commit("policy", policy_v1())
    ledger.delete("policy")

    prompt = ledger.diff_prompt("policy")

    assert prompt.startswith("Document policy was deleted at rev 2 (was rev 1).")
    assert "- text/refund: removed" in prompt
    assert ledger.diff("policy").deleted is True


def test_diff_prompt_for_an_unchanged_document_says_so(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("policy", policy_v1())

    assert ledger.diff_prompt("policy", from_rev=1, to_rev=1) == (
        "Document policy has no recorded changes at rev 1 (current)."
    )


def test_a_long_text_edit_falls_back_to_the_patch_block(ledger):
    old = "\n".join(f"clause {number} stays the same" for number in range(6))
    new = "\n".join(f"clause {number} has been rewritten entirely for emphasis" for number in range(6))
    ledger.commit("policy", [TextChunk(old, key="body")])
    ledger.commit("policy", [TextChunk(new, key="body")])

    prompt = ledger.diff_prompt("policy")

    assert "- text/body: changed" in prompt
    assert "    - clause 0 stays the same" in prompt
    assert "    + clause 0 has been rewritten entirely for emphasis" in prompt


def test_many_changed_cells_are_summarized(ledger):
    wide = [[f"c{column}" for column in range(12)]]
    ledger.commit("policy", [TableChunk(wide, key="grid")])
    ledger.commit("policy", [TableChunk([[f"d{column}" for column in range(12)]], key="grid")])

    prompt = ledger.diff_prompt("policy")

    assert "- table/grid cell A1: c0 \u2192 d0" in prompt
    assert "and 4 more cells changed" in prompt


@pytest.mark.parametrize(
    "row, col, expected", [(0, 0, "A1"), (1, 1, "B2"), (9, 25, "Z10"), (0, 26, "AA1"), (0, 27, "AB1")]
)
def test_a1_references(row, col, expected):
    assert a1_ref(row, col) == expected


def test_word_level_summary_narrows_or_gives_up():
    assert summarize_text_change("Refunds take 30 days.", "Refunds take 14 days.") == (
        '"30 days." \u2192 "14 days."'
    )
    assert summarize_text_change("a b c", "a b c") is None
    assert summarize_text_change("one two three", "one two three four") == 'added "four"'
    assert summarize_text_change("one two three four", "one two") == 'removed "three four"'
    # A span too wide to inline gives up, so the caller falls back to the patch block.
    assert summarize_text_change(" ".join("abcdefghijklm"), " ".join("nopqrstuvwxyz")) is None
