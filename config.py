# config.py — All constants and parameter defaults for Doc2MD-RAG

# ─── Chunking ───────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50
CHUNKING_METHODS = ["Recursive", "By Sentence", "By Markdown Headers", "Semantic"]

CHUNKING_TOOLTIPS = {
    "Recursive": "Splits text hierarchically — tries paragraphs first, then sentences, then words. Best general-purpose option.",
    "By Sentence": "Groups complete sentences together. Preserves grammatical boundaries but may create uneven chunk sizes.",
    "By Markdown Headers": "Uses document headings as natural split points. Best for well-structured documents with clear sections.",
    "Semantic": "Groups semantically similar sentences together using embeddings. Slower but creates the most coherent chunks.",
}

# ─── Search / Retrieval ─────────────────────────────────────────
SEARCH_MODES = ["Hybrid", "Vector (KNN)", "Lexical (BM25)"]
DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.0
DEFAULT_NUM_CANDIDATES = 50

SEARCH_TOOLTIPS = {
    "Hybrid": "Combines keyword matching and semantic similarity. Usually the best choice — catches both exact terms and related concepts.",
    "Vector (KNN)": "Finds semantically similar content even with different wording. Best for conceptual queries.",
    "Lexical (BM25)": "Matches exact keywords weighted by rarity. Best for specific terms, names, or technical jargon.",
}

# ─── Embedding ───────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {"dim": 384, "desc": "Fast, 80MB. Good balance of speed and quality."},
    "all-mpnet-base-v2": {"dim": 768, "desc": "Higher quality, 420MB. Better semantic understanding."},
}
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Parameter Tooltips ─────────────────────────────────────────
PARAM_TOOLTIPS = {
    # Chunking
    "chunk_method": "How text gets split into pieces for search. Different methods preserve different types of context.",
    "chunk_size": "Number of characters per chunk. Smaller = more precise matches, larger = more context per result.",
    "chunk_overlap": "Characters shared between adjacent chunks. Prevents context loss at chunk boundaries.",
    "semantic_threshold": "(Semantic chunking only) Similarity cutoff for splitting. Lower = larger chunks, higher = tighter grouping.",
    # Retrieval
    "search_mode": "How the system finds relevant chunks. Hybrid combines keyword + semantic search for best results.",
    "bm25_weight": "How much keyword matching influences results. Higher = more weight on exact word matches.",
    "vector_weight": "How much semantic similarity influences results. Higher = more weight on meaning, less on exact words.",
    "top_k": "Number of final results returned. More results = broader context but more noise in the prompt.",
    "score_threshold": "Minimum score to include a chunk. Filters out low-relevance noise. 0 = no filter.",
    "num_candidates": "Number of candidates FAISS retrieves before final ranking. Higher = better recall but slower. Must be >= Top K.",
    # Embedding
    "embedding_model": "The model that converts text to vectors. Larger models understand meaning better but run slower.",
    # Advanced
    "similarity_metric": "How vector distances are computed. Cosine is standard. Dot product favors longer texts.",
    "dim_reduction": "Dimensionality reduction for the tracer scatter plot. PCA is fast. t-SNE preserves local clusters better.",
    "dedup_threshold": "Cosine similarity above which two chunks are considered duplicates. 1.0 = off, 0.95 = aggressive dedup.",
}

# ─── Evaluation Metrics ─────────────────────────────────────────
EVAL_METRICS = {
    "chunk_utilization": "What percentage of the document is covered by retrieved chunks vs total chunks.",
    "score_variance": "How spread out the scores are. Low variance = all chunks equally relevant (bad). High = clear signal.",
    "lexical_coverage": "What percentage of query tokens appear in at least one retrieved chunk.",
    "redundancy_score": "Average pairwise similarity between retrieved chunks. High = redundant results.",
}

# ─── Indexing Methods ───────────────────────────────────────────
INDEXING_EXPLAINERS = {
    "FAISS (Flat Inner Product)": {
        "how": "Stores all chunk vectors in a flat array. Computes inner product (cosine similarity for normalized vectors) between the query vector and every chunk vector.",
        "pros": "Exact search — guaranteed to find the true nearest neighbors. No approximation error.",
        "cons": "Scales linearly with corpus size. For >1M chunks, consider IVF or HNSW indexes.",
        "complexity": "O(n \u00d7 d) per query, where n = chunks, d = embedding dimensions.",
    },
    "BM25 (Okapi)": {
        "how": "Tokenizes text into words, computes term frequency (TF) and inverse document frequency (IDF) for each token. Ranks by TF-IDF with length normalization.",
        "pros": "Fast, no GPU needed, excellent for exact keyword matching and rare terms.",
        "cons": "No semantic understanding — 'car' won't match 'automobile'. Sensitive to tokenization.",
        "complexity": "O(n \u00d7 q) per query, where n = chunks, q = query tokens.",
    },
}

# ─── UI Colors ───────────────────────────────────────────────────
CHUNK_COLORS = ["#EBF5FB", "#FDFEFE"]
MATCH_COLOR = "#FEF9E7"
MATCH_BORDER = "#F39C12"
CHUNK_BORDER = "#AED6F1"

# ─── File Config ─────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = ["pdf", "pptx", "docx", "csv"]
CLAUDE_VISION_MODEL = "claude-haiku-4-5-20251001"
