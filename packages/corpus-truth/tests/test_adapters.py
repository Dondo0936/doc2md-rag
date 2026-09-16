"""The adapter contract and the two reference adapters."""

import pytest
from conftest import policy_v1, policy_v2

from corpus_truth import IndexAdapter, InMemoryIndex, apply_plan


def test_in_memory_index_satisfies_the_protocol():
    assert isinstance(InMemoryIndex(), IndexAdapter)


def test_in_memory_index_stores_flat_records(ledger, index):
    plan = ledger.commit("policy", policy_v1())
    index.apply(plan)

    record = index.get(plan.add[0].id)
    assert record["key"] == "refund"
    assert record["kind"] == "text"
    assert record["is_current"] is True
    assert index.get("never-added") is None
    assert sorted(index.list_ids()) == sorted(posting.id for posting in plan.add)


def test_removing_an_absent_id_is_not_an_error(index):
    index.remove(["missing-1", "missing-2"])
    assert index.list_ids() == []


def test_apply_plan_returns_the_plan_for_chaining(ledger, index):
    plan = ledger.commit("policy", policy_v1())
    assert apply_plan(index, plan) is plan


def test_an_index_only_ever_holds_the_current_revision(ledger, index):
    index.apply(ledger.commit("policy", policy_v1()))
    index.apply(ledger.commit("policy", policy_v2()))

    assert sorted(index.list_ids()) == sorted(posting.id for posting in ledger.current("policy"))
    assert all(record["rev"] == 2 for record in index.records.values())


def test_faiss_example_adapter_round_trips_a_plan(ledger):
    pytest.importorskip("faiss")
    numpy = pytest.importorskip("numpy")

    from corpus_truth.faiss_example import FaissIndexAdapter

    def embed(texts):
        # Deterministic stand-in for a real embedder.
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    adapter = FaissIndexAdapter(dim=4, embed=embed)
    first = ledger.commit("policy", policy_v1())
    apply_plan(adapter, first)
    assert adapter.index.ntotal == 3

    second = ledger.commit("policy", policy_v2())
    apply_plan(adapter, second)

    assert adapter.index.ntotal == 3
    assert sorted(adapter.list_ids()) == sorted(posting.id for posting in second.add)
    assert ledger.reconcile(adapter).is_clean

    query = numpy.asarray(embed(["Refunds take 14 days."]), dtype="float32")
    _, aliases = adapter.index.search(query, 1)
    assert aliases[0][0] != -1
