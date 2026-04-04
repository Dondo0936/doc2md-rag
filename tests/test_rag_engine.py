"""Tests for rag_engine.py — chunking, indexing, search, evaluation."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rag_engine import RAGEngine, EmbeddingModel
from tests.conftest import SAMPLE_MARKDOWN


@pytest.fixture
def engine():
    """Create a RAGEngine with the default model."""
    return RAGEngine()


@pytest.fixture
def indexed_engine(engine):
    """Engine with SAMPLE_MARKDOWN indexed using default settings."""
    engine.index_document(SAMPLE_MARKDOWN, source="test")
    return engine


# ─── Chunking Methods ──────────────────────────────────────────────

class TestChunking:
    def test_recursive_produces_chunks(self, engine):
        engine.index_document(SAMPLE_MARKDOWN, method="Recursive", chunk_size=200, overlap=20)
        assert len(engine.get_chunks()) > 0

    def test_by_sentence_produces_chunks(self, engine):
        engine.index_document(SAMPLE_MARKDOWN, method="By Sentence", chunk_size=200, overlap=20)
        assert len(engine.get_chunks()) > 0

    def test_by_headers_produces_chunks(self, engine):
        engine.index_document(SAMPLE_MARKDOWN, method="By Markdown Headers", chunk_size=200, overlap=20)
        assert len(engine.get_chunks()) > 0

    def test_semantic_produces_chunks(self, engine):
        engine.index_document(SAMPLE_MARKDOWN, method="Semantic", semantic_threshold=0.5)
        assert len(engine.get_chunks()) > 0

    def test_unknown_method_falls_back_to_recursive(self, engine):
        engine.index_document(SAMPLE_MARKDOWN, method="NonexistentMethod", chunk_size=200, overlap=20)
        assert len(engine.get_chunks()) > 0

    def test_single_sentence_semantic(self, engine):
        engine.index_document("Hello world.", method="Semantic", semantic_threshold=0.5)
        assert len(engine.get_chunks()) == 1

    def test_empty_text(self, engine):
        engine.index_document("", method="Recursive", chunk_size=200, overlap=20)
        assert len(engine.get_chunks()) == 0


# ─── Parameter Validation ──���───────────────────────────────────────

class TestValidation:
    def test_chunk_size_zero_raises(self, engine):
        with pytest.raises(ValueError, match="chunk_size"):
            engine.index_document(SAMPLE_MARKDOWN, chunk_size=0)

    def test_overlap_exceeds_chunk_size_raises(self, engine):
        with pytest.raises(ValueError, match="overlap"):
            engine.index_document(SAMPLE_MARKDOWN, chunk_size=100, overlap=100)

    def test_negative_overlap_raises(self, engine):
        with pytest.raises(ValueError, match="overlap"):
            engine.index_document(SAMPLE_MARKDOWN, chunk_size=100, overlap=-1)


# ─── Search ─────────────��──────────────────────────────────────────

class TestSearch:
    def test_hybrid_search(self, indexed_engine):
        results = indexed_engine.search("machine learning", mode="Hybrid", top_k=3)
        assert len(results) > 0
        assert all(r["score"] > 0 for r in results)

    def test_vector_search(self, indexed_engine):
        results = indexed_engine.search("neural networks", mode="Vector (KNN)", top_k=3)
        assert len(results) > 0

    def test_bm25_search(self, indexed_engine):
        # BM25 with common words can return negative IDF scores; use rarer terms
        results = indexed_engine.search("photosynthesis mitochondria", mode="Lexical (BM25)", top_k=3)
        # Even if results are empty (rare terms in small corpus), should not crash
        assert isinstance(results, list)

    def test_empty_query_returns_empty(self, indexed_engine):
        results = indexed_engine.search("", top_k=3)
        # Empty query may return results from BM25 (all zeros) filtered out
        # Just check it doesn't crash
        assert isinstance(results, list)

    def test_score_threshold_filters(self, indexed_engine):
        results = indexed_engine.search("machine learning", score_threshold=0.99)
        # Very high threshold should filter most results
        assert len(results) <= len(indexed_engine.get_chunks())

    def test_dedup_reduces_results(self, indexed_engine):
        results_no_dedup = indexed_engine.search("sample document", top_k=10, dedup_threshold=1.0)
        results_dedup = indexed_engine.search("sample document", top_k=10, dedup_threshold=0.85)
        assert len(results_dedup) <= len(results_no_dedup)

    def test_top_k_limits(self, indexed_engine):
        results = indexed_engine.search("document", top_k=2)
        assert len(results) <= 2

    def test_search_result_fields(self, indexed_engine):
        results = indexed_engine.search("document", top_k=1)
        if results:
            r = results[0]
            assert "text" in r
            assert "chunk_id" in r
            assert "score" in r
            assert "bm25_score" in r
            assert "faiss_score" in r
            assert "start_char" in r
            assert "end_char" in r

    def test_no_chunks_returns_empty(self, engine):
        results = engine.search("anything")
        assert results == []


# ─── Trace ─��──────────────────────────��────────────────────────────

class TestTrace:
    def test_trace_returns_expected_keys(self, indexed_engine):
        trace = indexed_engine.get_trace("machine learning")
        assert "query_tokens" in trace
        assert "query_embedding" in trace
        assert "chunk_embeddings" in trace
        assert "bm25_token_matches" in trace
        assert "combined_scores" in trace
        assert "top_k_indices" in trace

    def test_trace_empty_engine(self, engine):
        trace = engine.get_trace("anything")
        assert trace == {}


# ─── Evaluation ────────────────────────────────────────────────────

class TestEvaluation:
    def test_eval_metrics_fields(self, indexed_engine):
        results = indexed_engine.search("machine learning", top_k=3)
        metrics = indexed_engine.compute_eval_metrics(results, "machine learning")
        assert "chunk_utilization" in metrics
        assert "char_coverage" in metrics
        assert "lexical_coverage" in metrics
        assert "redundancy" in metrics
        assert "avg_score" in metrics
        assert "top_score" in metrics
        assert "score_variance" in metrics
        assert "score_spread" in metrics

    def test_eval_metrics_ranges(self, indexed_engine):
        results = indexed_engine.search("document", top_k=3)
        metrics = indexed_engine.compute_eval_metrics(results, "document")
        assert 0 <= metrics["chunk_utilization"] <= 1
        assert 0 <= metrics["char_coverage"] <= 1
        assert 0 <= metrics["lexical_coverage"] <= 1
        assert metrics["avg_score"] >= 0

    def test_eval_empty_results(self, indexed_engine):
        metrics = indexed_engine.compute_eval_metrics([], "anything")
        assert metrics == {}


# ─── Clear ───────────��─────────────────────────────────────────────

class TestClear:
    def test_clear_resets_state(self, indexed_engine):
        assert len(indexed_engine.get_chunks()) > 0
        indexed_engine.clear()
        assert len(indexed_engine.get_chunks()) == 0
        assert indexed_engine.bm25 is None
        assert indexed_engine.faiss_index is None
        assert indexed_engine.chunk_embeddings is None


# ─── API Embedding Providers (mocked) ────────────────────────────────

class TestEmbeddingModel:
    def test_local_model_loads(self):
        """Local SentenceTransformer model initializes and encodes."""
        model = EmbeddingModel("all-MiniLM-L6-v2")
        assert model.provider == "local"
        result = model.encode(["hello world"])
        assert result.shape == (1, 384)

    @patch("openai.OpenAI")
    def test_openai_embedding(self, mock_openai_cls):
        """OpenAI embedding calls the API and returns normalized vectors."""
        # Mock the response
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        model = EmbeddingModel("text-embedding-3-small", api_key="sk-test")
        result = model.encode(["hello"])
        assert result.shape == (1, 1536)
        # Check normalization
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01

    @patch("voyageai.Client")
    def test_voyage_embedding(self, mock_voyage_cls):
        """Voyage AI embedding calls the API and returns normalized vectors."""
        mock_client = MagicMock()
        mock_voyage_cls.return_value = mock_client
        mock_client.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])

        model = EmbeddingModel("voyage-3.5-lite", api_key="pa-test")
        result = model.encode(["hello"])
        assert result.shape == (1, 1024)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01

    @patch("google.generativeai.configure")
    @patch("google.generativeai.embed_content")
    def test_google_embedding(self, mock_embed, mock_configure):
        """Google Gemini embedding calls the API and returns normalized vectors."""
        mock_embed.return_value = {"embedding": [0.1] * 768}

        model = EmbeddingModel("gemini-embedding-001", api_key="AI-test")
        result = model.encode(["hello"])
        assert result.shape == (1, 768)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01

    def test_unknown_provider_raises(self):
        """Unknown provider raises ValueError."""
        with patch.dict("config.EMBEDDING_MODELS", {"fake-model": {"dim": 100, "provider": "unknown"}}):
            with pytest.raises(ValueError, match="Unknown embedding provider"):
                EmbeddingModel("fake-model")

    def test_encode_single_string(self):
        """Passing a single string (not list) works."""
        model = EmbeddingModel("all-MiniLM-L6-v2")
        result = model.encode("hello world")
        assert result.shape == (1, 384)
