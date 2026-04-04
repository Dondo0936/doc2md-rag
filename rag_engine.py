# rag_engine.py — Chunking (4 methods), indexing (FAISS + BM25), search (3 modes), trace + eval

import re
import math
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import faiss

from config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS

logger = logging.getLogger(__name__)


# ─── Embedding Wrapper ─────────────────────────────────────────

class EmbeddingModel:
    """Unified interface for local and API-based embedding models."""

    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self._api_key = api_key
        model_info = EMBEDDING_MODELS.get(model_name, {})
        self.provider = model_info.get("provider", "local")
        self.dim = model_info.get("dim", 384)
        self._client = None

        if self.provider == "local":
            from sentence_transformers import SentenceTransformer
            self._client = SentenceTransformer(model_name)
        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        elif self.provider == "voyage":
            import voyageai
            self._client = voyageai.Client(api_key=api_key)
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def encode(self, texts, normalize_embeddings=True):
        """Encode texts to numpy array of embeddings. Returns shape (n, dim)."""
        if isinstance(texts, str):
            texts = [texts]

        if self.provider == "local":
            return self._client.encode(texts, normalize_embeddings=normalize_embeddings)

        elif self.provider == "openai":
            response = self._client.embeddings.create(
                model=self.model_name, input=texts
            )
            embeddings = np.array([d.embedding for d in response.data], dtype=np.float32)

        elif self.provider == "voyage":
            response = self._client.embed(texts, model=self.model_name)
            embeddings = np.array(response.embeddings, dtype=np.float32)

        elif self.provider == "google":
            result = self._client.embed_content(
                model=f"models/{self.model_name}",
                content=texts if len(texts) > 1 else texts[0],
            )
            if isinstance(result["embedding"][0], list):
                embeddings = np.array(result["embedding"], dtype=np.float32)
            else:
                embeddings = np.array([result["embedding"]], dtype=np.float32)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Normalize if requested (API providers don't always return normalized vectors)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return embeddings


class RAGEngine:
    def __init__(self, embedding_model_name=DEFAULT_EMBEDDING_MODEL, embedding_api_key=None):
        self.model = EmbeddingModel(embedding_model_name, api_key=embedding_api_key)
        self.model_name = embedding_model_name
        self.chunks = []
        self.bm25 = None
        self.faiss_index = None
        self.chunk_embeddings = None
        self._idf_cache = {}
        self._total_doc_chars = 0

    # ─── Chunking ───────────────────────────────────────────────

    def index_document(
        self,
        markdown: str,
        source: str = "document",
        method: str = "Recursive",
        chunk_size: int = 400,
        overlap: int = 50,
        semantic_threshold: float = 0.5,
    ) -> None:
        """Chunk the markdown and build both BM25 and FAISS indexes."""
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")
        self.clear()
        self._total_doc_chars = len(markdown)

        texts = self._chunk(markdown, method, chunk_size, overlap, semantic_threshold)

        # Compute character offsets by finding each chunk in the original text
        offset = 0
        for i, text in enumerate(texts):
            start = markdown.find(text[:80], offset)
            if start == -1:
                start = offset
            end = start + len(text)
            self.chunks.append({
                "text": text,
                "chunk_id": i,
                "source": source,
                "start_char": start,
                "end_char": end,
            })
            offset = start + 1

        if not self.chunks:
            return

        # Build BM25 index
        tokenized = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        self._cache_idf(tokenized)

        # Build FAISS index
        texts_list = [c["text"] for c in self.chunks]
        self.chunk_embeddings = self.model.encode(texts_list, normalize_embeddings=True)
        dim = self.chunk_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.chunk_embeddings.astype(np.float32))

    def _chunk(self, text, method, chunk_size, overlap, semantic_threshold):
        if method == "Recursive":
            return self._chunk_recursive(text, chunk_size, overlap)
        elif method == "By Sentence":
            return self._chunk_by_sentence(text, chunk_size, overlap)
        elif method == "By Markdown Headers":
            return self._chunk_by_headers(text, chunk_size, overlap)
        elif method == "Semantic":
            return self._chunk_semantic(text, semantic_threshold)
        return self._chunk_recursive(text, chunk_size, overlap)

    def _chunk_recursive(self, text, chunk_size, overlap):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, length_function=len,
        )
        return [doc.page_content for doc in splitter.create_documents([text])]

    def _chunk_by_sentence(self, text, chunk_size, overlap):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks, current, current_len = [], [], 0

        for sent in sentences:
            if current_len + len(sent) > chunk_size and current:
                chunks.append(" ".join(current))
                overlap_text, overlap_sents = "", []
                for s in reversed(current):
                    if len(overlap_text) + len(s) > overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_text = " ".join(overlap_sents)
                current = overlap_sents
                current_len = len(overlap_text)
            current.append(sent)
            current_len += len(sent) + 1

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _chunk_by_headers(self, text, chunk_size, overlap):
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        docs = splitter.split_text(text)
        sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap,
        )

        chunks = []
        for doc in docs:
            content = doc.page_content
            header_parts = []
            for key in ["h1", "h2", "h3"]:
                if key in doc.metadata:
                    header_parts.append("#" * int(key[1]) + " " + doc.metadata[key])
            if header_parts:
                content = "\n".join(header_parts) + "\n\n" + content

            if len(content) > chunk_size:
                chunks.extend(d.page_content for d in sub_splitter.create_documents([content]))
            else:
                chunks.append(content)
        return chunks if chunks else [text]

    def _chunk_semantic(self, text, threshold):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        chunks, group = [], [sentences[0]]

        for i in range(1, len(sentences)):
            if float(np.dot(embeddings[i - 1], embeddings[i])) < threshold:
                chunks.append(" ".join(group))
                group = [sentences[i]]
            else:
                group.append(sentences[i])
        if group:
            chunks.append(" ".join(group))
        return chunks

    # ─── Search ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        mode: str = "Hybrid",
        top_k: int = 5,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        score_threshold: float = 0.0,
        num_candidates: int = 50,
        dedup_threshold: float = 1.0,
    ) -> list[dict]:
        """Search indexed chunks with score filtering, candidate oversampling, and dedup."""
        if not self.chunks or top_k < 1:
            return []

        n = len(self.chunks)
        num_candidates = max(num_candidates, top_k)
        num_candidates = min(num_candidates, n)

        bm25_scores = np.zeros(n)
        vector_scores = np.zeros(n)

        # BM25 scores
        if mode in ("Hybrid", "Lexical (BM25)") and self.bm25 is not None:
            raw = self.bm25.get_scores(self._tokenize(query))
            max_bm25 = raw.max() if raw.max() > 0 else 1.0
            bm25_scores = raw / max_bm25

        # Vector scores — retrieve num_candidates, not all
        if mode in ("Hybrid", "Vector (KNN)") and self.faiss_index is not None:
            q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
            k_search = min(num_candidates, n)
            scores, indices = self.faiss_index.search(q_emb, k_search)
            for rank, idx in enumerate(indices[0]):
                if idx >= 0:
                    vector_scores[idx] = scores[0][rank]

        # Combine
        if mode == "Hybrid":
            combined = bm25_weight * bm25_scores + vector_weight * vector_scores
        elif mode == "Vector (KNN)":
            combined = vector_scores
        else:
            combined = bm25_scores

        # Sort and filter
        sorted_indices = np.argsort(combined)[::-1]

        results = []
        seen_embeddings = []

        for idx in sorted_indices:
            if combined[idx] <= 0 or combined[idx] < score_threshold:
                continue

            # Dedup: skip if too similar to already-selected chunk
            if dedup_threshold < 1.0 and self.chunk_embeddings is not None:
                emb = self.chunk_embeddings[idx]
                is_dup = False
                for seen_emb in seen_embeddings:
                    sim = float(np.dot(emb, seen_emb))
                    if sim >= dedup_threshold:
                        is_dup = True
                        break
                if is_dup:
                    continue
                seen_embeddings.append(emb)

            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "chunk_id": chunk["chunk_id"],
                "score": float(combined[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "faiss_score": float(vector_scores[idx]),
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
            })

            if len(results) >= top_k:
                break

        return results

    # ─── Trace Data ─────────────────────────────────────────────

    def get_trace(
        self,
        query: str,
        mode: str = "Hybrid",
        top_k: int = 5,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> dict:
        if not self.chunks:
            return {}

        n = len(self.chunks)
        query_tokens = self._tokenize(query)
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

        bm25_scores_raw = np.zeros(n)
        bm25_token_matches = {}

        if self.bm25 is not None:
            bm25_scores_raw = self.bm25.get_scores(query_tokens)
            for i, chunk in enumerate(self.chunks):
                chunk_set = set(self._tokenize(chunk["text"]))
                matched = [t for t in query_tokens if t in chunk_set]
                if matched:
                    bm25_token_matches[i] = matched

        vector_scores_raw = np.zeros(n)
        if self.faiss_index is not None:
            q_emb = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self.faiss_index.search(q_emb, n)
            for rank, idx in enumerate(indices[0]):
                if idx >= 0:
                    vector_scores_raw[idx] = scores[0][rank]

        max_bm25 = bm25_scores_raw.max() if bm25_scores_raw.max() > 0 else 1.0
        bm25_norm = bm25_scores_raw / max_bm25

        if mode == "Hybrid":
            combined = bm25_weight * bm25_norm + vector_weight * vector_scores_raw
        elif mode == "Vector (KNN)":
            combined = vector_scores_raw
        else:
            combined = bm25_norm

        top_indices = np.argsort(combined)[::-1][:top_k].tolist()
        idf_values = {t: self._idf_cache.get(t, 0.0) for t in query_tokens}

        return {
            "query_tokens": query_tokens,
            "query_embedding": query_embedding,
            "chunk_embeddings": self.chunk_embeddings,
            "bm25_token_matches": bm25_token_matches,
            "bm25_scores_raw": bm25_scores_raw.tolist(),
            "vector_scores_raw": vector_scores_raw.tolist(),
            "combined_scores": combined.tolist(),
            "top_k_indices": top_indices,
            "idf_values": idf_values,
        }

    # ─── Evaluation Metrics ────────────────────────────────────

    def compute_eval_metrics(self, search_results: list[dict], query: str) -> dict:
        """Compute retrieval quality metrics for the current search results."""
        if not search_results or not self.chunks:
            return {}

        n_total = len(self.chunks)
        n_retrieved = len(search_results)
        retrieved_ids = [r["chunk_id"] for r in search_results]

        # 1. Chunk utilization: what fraction of chunks were retrieved
        chunk_utilization = n_retrieved / n_total if n_total > 0 else 0

        # 2. Character coverage: how much of the original doc is covered
        retrieved_chars = sum(len(r["text"]) for r in search_results)
        char_coverage = retrieved_chars / self._total_doc_chars if self._total_doc_chars > 0 else 0

        # 3. Score variance: higher = clearer signal
        scores = [r["score"] for r in search_results]
        score_variance = float(np.var(scores)) if len(scores) > 1 else 0.0

        # 4. Score spread: gap between best and worst result
        score_spread = max(scores) - min(scores) if scores else 0.0

        # 5. Lexical coverage: what % of query tokens appear in retrieved chunks
        query_tokens = set(self._tokenize(query))
        found_tokens = set()
        for r in search_results:
            chunk_tokens = set(self._tokenize(r["text"]))
            found_tokens.update(query_tokens & chunk_tokens)
        lexical_coverage = len(found_tokens) / len(query_tokens) if query_tokens else 0

        # 6. Redundancy: avg pairwise cosine similarity between retrieved chunks
        redundancy = 0.0
        if self.chunk_embeddings is not None and len(retrieved_ids) > 1:
            embs = self.chunk_embeddings[retrieved_ids]
            sims = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sims.append(float(np.dot(embs[i], embs[j])))
            redundancy = float(np.mean(sims)) if sims else 0.0

        # 7. Score drop-off: ratio of top to 2nd score (higher = clearer winner)
        score_dropoff = (scores[0] / scores[1]) if len(scores) > 1 and scores[1] > 0 else 0.0

        return {
            "chunk_utilization": chunk_utilization,
            "char_coverage": char_coverage,
            "score_variance": score_variance,
            "score_spread": score_spread,
            "lexical_coverage": lexical_coverage,
            "redundancy": redundancy,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "top_score": max(scores) if scores else 0.0,
            "score_dropoff": score_dropoff,
        }

    # ─── Helpers ────────────────────────────────────────────────

    def get_chunks(self):
        return self.chunks

    def clear(self):
        self.chunks = []
        self.bm25 = None
        self.faiss_index = None
        self.chunk_embeddings = None
        self._idf_cache = {}
        self._total_doc_chars = 0

    @staticmethod
    def _tokenize(text):
        return text.lower().split()

    def _cache_idf(self, tokenized_corpus):
        n_docs = len(tokenized_corpus)
        df = {}
        for tokens in tokenized_corpus:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self._idf_cache = {
            t: math.log((n_docs - count + 0.5) / (count + 0.5) + 1)
            for t, count in df.items()
        }
