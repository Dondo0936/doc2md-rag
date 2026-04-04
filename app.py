# app.py — Doc2MD-RAG: Editorial Data Pipeline Design
# Run with: streamlit run app.py

import html
import logging
import os
import sys
import hashlib
import tempfile
import traceback
import streamlit as st
import pandas as pd
import markdown as md_lib

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 500
MAX_VISIBLE_CHUNKS = 200

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CHUNKING_METHODS, CHUNKING_TOOLTIPS, SEARCH_MODES, SEARCH_TOOLTIPS,
    EMBEDDING_MODELS, EMBEDDING_API_KEY_ENV, PARAM_TOOLTIPS, CHUNK_COLORS,
    MATCH_COLOR, MATCH_BORDER, CHUNK_BORDER, SUPPORTED_EXTENSIONS,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_BM25_WEIGHT,
    DEFAULT_VECTOR_WEIGHT, DEFAULT_TOP_K, DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SCORE_THRESHOLD, DEFAULT_NUM_CANDIDATES, INDEXING_EXPLAINERS, EVAL_METRICS,
)
from converter import process_document
from claude_wrapper import get_llm_client
from rag_engine import RAGEngine
from tracer import (
    render_query_tokens, render_embedding_space,
    render_bm25_highlights, render_score_fusion,
)

st.set_page_config(page_title="Doc2MD-RAG", layout="wide", page_icon="📄")

MAX_FILE_SIZE_MB = 100

# ─── Editorial CSS ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.ed-title { font-size:1.75rem; font-weight:700; color:#0F1419; letter-spacing:-0.02em; margin:0 0 0.25rem 0; }
.ed-sub { font-size:0.95rem; color:#6B7280; font-weight:400; margin:0 0 1.5rem 0; }
.section-label { font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#9CA3AF; margin:0.75rem 0 0.5rem 0; }

.pipeline-box { background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:1.25rem 2rem; margin:1rem 0 1.5rem 0; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.pipeline-track { display:flex; align-items:center; justify-content:space-between; position:relative; }
.pipeline-track::before { content:''; position:absolute; top:20px; left:40px; right:40px; height:2px; background:#E5E7EB; z-index:0; }
.pipeline-stage { display:flex; flex-direction:column; align-items:center; z-index:1; flex:1; }
.pipeline-dot { width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:600; margin-bottom:0.5rem; }
.dot-pending { background:#F3F4F6; border:2px solid #D1D5DB; color:#9CA3AF; }
.dot-done { background:#10B981; border:2px solid #10B981; color:white; }
.dot-active { background:#3B82F6; border:2px solid #3B82F6; color:white; animation:pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%,100%{transform:scale(1);} 50%{transform:scale(1.05);box-shadow:0 0 0 8px rgba(59,130,246,0);} }
.stage-label { font-size:0.75rem; font-weight:500; color:#6B7280; text-transform:uppercase; letter-spacing:0.05em; }
.stage-label-done { color:#10B981; } .stage-label-active { color:#3B82F6; }

.card { background:#fff; border:1px solid #E5E7EB; border-radius:12px; padding:1.5rem; margin:0.75rem 0; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.metric-row { display:flex; gap:1.5rem; margin:0.5rem 0; flex-wrap:wrap; }
.metric { display:flex; flex-direction:column; }
.metric-value { font-size:1.5rem; font-weight:700; color:#0F1419; }
.metric-label { font-size:0.75rem; color:#9CA3AF; font-weight:500; }

.chunk-block { padding:10px 14px; margin:6px 0; border-radius:6px; position:relative; font-size:13px; line-height:1.6; border-left:3px solid; }
.chunk-id { position:absolute; top:6px; right:10px; font-size:10px; color:#9CA3AF; font-family:monospace; font-weight:500; }

.score-badge { display:inline-flex; align-items:center; gap:4px; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:500; font-family:monospace; }
.badge-bm25 { background:#DBEAFE; color:#1D4ED8; }
.badge-vector { background:#D1FAE5; color:#065F46; }
.badge-combined { background:#FEE2E2; color:#991B1B; }

.empty-state { text-align:center; padding:80px 20px; color:#9CA3AF; }
.empty-state h2 { font-size:1.5rem; font-weight:600; color:#374151; margin-bottom:0.5rem; }
.empty-state p { font-size:0.9rem; line-height:1.6; max-width:500px; margin:0 auto; }

.api-hint { background:#FFF7ED; border:1px solid #FDBA74; border-radius:8px; padding:0.75rem 1rem; font-size:0.8rem; color:#9A3412; margin:0.5rem 0; }

.search-container { position:relative; }
.search-container input { padding-right:70px !important; }
div[data-testid="stTextInput"] input { border:2px solid #3B82F6 !important; border-radius:10px !important; padding:0.75rem 1rem 0.75rem 2.5rem !important; font-size:1rem !important; background:#F0F7FF url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%233B82F6' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") no-repeat 0.75rem center !important; }
div[data-testid="stTextInput"] input:focus { box-shadow:0 0 0 4px rgba(59,130,246,0.15) !important; border-color:#2563EB !important; background-color:#fff !important; }
div[data-testid="stTextInput"] input::placeholder { color:#6B7280 !important; font-weight:500 !important; }

.param-impact { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1rem 0; }
.param-card { background:#fff; border:1px solid #E5E7EB; border-radius:10px; padding:1.25rem; }
.param-card h4 { margin:0 0 0.5rem 0; font-size:0.95rem; color:#1F2937; }
.param-card .param-icon { font-size:1.5rem; margin-bottom:0.5rem; display:block; }
.param-card p { font-size:0.8rem; color:#6B7280; line-height:1.5; margin:0; }
.param-card .param-effect { background:#F0F9FF; border:1px solid #BAE6FD; border-radius:6px; padding:0.5rem 0.75rem; margin-top:0.75rem; font-size:0.78rem; color:#0369A1; }

.flow-diagram { display:flex; align-items:center; gap:0; margin:1.5rem 0; overflow-x:auto; padding:0.5rem 0; }
.flow-step { background:#fff; border:1px solid #E5E7EB; border-radius:10px; padding:1rem 0.75rem; min-width:140px; text-align:center; flex-shrink:0; }
.flow-step .step-icon { font-size:1.5rem; display:block; margin-bottom:0.25rem; }
.flow-step .step-name { font-size:0.8rem; font-weight:600; color:#1F2937; }
.flow-step .step-detail { font-size:0.7rem; color:#9CA3AF; margin-top:0.25rem; }
.flow-arrow { font-size:1.25rem; color:#D1D5DB; flex-shrink:0; padding:0 0.15rem; }

.embed-demo { background:#F9FAFB; border:1px solid #E5E7EB; border-radius:10px; padding:1.25rem; margin:1rem 0; }
.embed-demo h4 { margin:0 0 0.75rem 0; font-size:0.95rem; }
.vector-bar { display:flex; align-items:center; gap:0.5rem; margin:0.25rem 0; font-size:0.78rem; }
.vector-bar .vb-label { width:100px; color:#6B7280; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.vector-bar .vb-bar { flex:1; height:18px; border-radius:4px; position:relative; }
.vector-bar .vb-val { font-family:monospace; font-size:0.7rem; color:#6B7280; width:40px; text-align:right; }

.explainer-card { background:#F9FAFB; border:1px solid #E5E7EB; border-radius:8px; padding:1rem; margin:0.5rem 0; font-size:0.85rem; line-height:1.5; }
.explainer-card h4 { margin:0 0 0.5rem 0; font-size:0.95rem; color:#1F2937; }
.explainer-card .label { font-weight:600; color:#6B7280; }

.eval-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:0.75rem; margin:0.75rem 0; }
.eval-card { background:#F9FAFB; border:1px solid #E5E7EB; border-radius:8px; padding:0.75rem; }
.eval-value { font-size:1.25rem; font-weight:700; color:#0F1419; }
.eval-label { font-size:0.7rem; color:#9CA3AF; font-weight:500; text-transform:uppercase; }

.sidebar-section { margin:0.5rem 0; padding:0.25rem 0; }
.sidebar-divider { border:none; border-top:1px solid #E5E7EB; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Pipeline Progress ──────────────────────────────────

def render_pipeline(stages):
    dots = ""
    icons = {"pending": "○", "done": "✓", "active": "◉"}
    for s in stages:
        st_ = s["status"]
        lbl_cls = f"stage-label-{st_}" if st_ != "pending" else "stage-label"
        dots += f'<div class="pipeline-stage"><div class="pipeline-dot dot-{st_}">{icons[st_]}</div><div class="stage-label {lbl_cls}">{s["name"]}</div></div>'
    st.markdown(f'<div class="pipeline-box"><div class="pipeline-track">{dots}</div></div>', unsafe_allow_html=True)


# ─── Session State ──────────────────────────────────────────────

defaults = {
    "rag_engine": None, "conversion_result": None, "indexed_params": None,
    "pipeline_log": [], "file_hash": None, "pipeline_stage": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="ed-title">Parameters</p>', unsafe_allow_html=True)

    # ── API Keys (stored in session state, never in os.environ)
    with st.expander("API Configuration", expanded=False):
        st.markdown('<div class="section-label">CLAUDE VISION</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "Anthropic API Key", type="password",
            value=st.session_state.get("_api_key", os.environ.get("ANTHROPIC_API_KEY", "")),
            help="Required for AI-powered image descriptions. Get yours at console.anthropic.com.",
        )
        if api_key:
            st.session_state["_api_key"] = api_key
            st.markdown('<span style="color:#10B981;font-size:0.8rem;">✓ Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-hint">Set via sidebar or terminal:<br><code>export ANTHROPIC_API_KEY=sk-ant-...</code></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">EMBEDDING PROVIDERS</div>', unsafe_allow_html=True)
        st.caption("Only needed if you select an API-based embedding model below.")

        openai_key = st.text_input(
            "OpenAI API Key", type="password",
            value=st.session_state.get("_openai_key", os.environ.get("OPENAI_API_KEY", "")),
            help="Required for text-embedding-3-small/large. Get yours at platform.openai.com.",
        )
        if openai_key:
            st.session_state["_openai_key"] = openai_key
            st.markdown('<span style="color:#10B981;font-size:0.8rem;">✓ OpenAI connected</span>', unsafe_allow_html=True)

        voyage_key = st.text_input(
            "Voyage AI API Key", type="password",
            value=st.session_state.get("_voyage_key", os.environ.get("VOYAGE_API_KEY", "")),
            help="Required for voyage-3.5-lite. Anthropic-recommended embeddings. Get yours at dash.voyageai.com.",
        )
        if voyage_key:
            st.session_state["_voyage_key"] = voyage_key
            st.markdown('<span style="color:#10B981;font-size:0.8rem;">✓ Voyage AI connected</span>', unsafe_allow_html=True)

        gemini_key = st.text_input(
            "Google Gemini API Key", type="password",
            value=st.session_state.get("_gemini_key", os.environ.get("GEMINI_API_KEY", "")),
            help="Required for gemini-embedding-001 / text-embedding-004. Get yours at aistudio.google.com.",
        )
        if gemini_key:
            st.session_state["_gemini_key"] = gemini_key
            st.markdown('<span style="color:#10B981;font-size:0.8rem;">✓ Gemini connected</span>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── CHUNKING SECTION
    st.markdown('<div class="section-label">CHUNKING</div>', unsafe_allow_html=True)
    chunk_method = st.selectbox("Method", CHUNKING_METHODS, help=PARAM_TOOLTIPS["chunk_method"])
    if chunk_method in CHUNKING_TOOLTIPS:
        st.caption(CHUNKING_TOOLTIPS[chunk_method])
    chunk_size = st.slider("Chunk size", 100, 1000, DEFAULT_CHUNK_SIZE, step=50, help=PARAM_TOOLTIPS["chunk_size"])
    chunk_overlap = st.slider("Overlap", 0, 200, DEFAULT_CHUNK_OVERLAP, step=10, help=PARAM_TOOLTIPS["chunk_overlap"])
    semantic_threshold = 0.5
    if chunk_method == "Semantic":
        semantic_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, step=0.05, help=PARAM_TOOLTIPS["semantic_threshold"])

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── RETRIEVAL SECTION
    st.markdown('<div class="section-label">RETRIEVAL</div>', unsafe_allow_html=True)
    search_mode = st.selectbox("Mode", SEARCH_MODES, help=PARAM_TOOLTIPS["search_mode"])
    if search_mode in SEARCH_TOOLTIPS:
        st.caption(SEARCH_TOOLTIPS[search_mode])

    bm25_weight, vector_weight = DEFAULT_BM25_WEIGHT, DEFAULT_VECTOR_WEIGHT
    if search_mode == "Hybrid":
        bm25_weight = st.slider("BM25 weight", 0.0, 1.0, DEFAULT_BM25_WEIGHT, step=0.05, help=PARAM_TOOLTIPS["bm25_weight"])
        vector_weight = st.slider("Vector weight", 0.0, 1.0, DEFAULT_VECTOR_WEIGHT, step=0.05, help=PARAM_TOOLTIPS["vector_weight"])

    top_k = st.slider("Top K", 1, 20, DEFAULT_TOP_K, help=PARAM_TOOLTIPS["top_k"])
    score_threshold = st.slider("Score threshold", 0.0, 1.0, DEFAULT_SCORE_THRESHOLD, step=0.05, help=PARAM_TOOLTIPS["score_threshold"])
    num_candidates = st.slider("Num candidates", 10, 200, DEFAULT_NUM_CANDIDATES, step=10, help=PARAM_TOOLTIPS["num_candidates"])

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── EMBEDDING SECTION
    st.markdown('<div class="section-label">EMBEDDING</div>', unsafe_allow_html=True)
    emb_model = st.selectbox("Model", list(EMBEDDING_MODELS.keys()), help=PARAM_TOOLTIPS["embedding_model"])
    emb_info = EMBEDDING_MODELS[emb_model]
    st.caption(emb_info["desc"])

    # Warn if API-based model selected without the required key
    _emb_provider = emb_info.get("provider", "local")
    _emb_api_key = None
    if _emb_provider == "openai":
        _emb_api_key = st.session_state.get("_openai_key", os.environ.get("OPENAI_API_KEY"))
        if not _emb_api_key:
            st.warning("OpenAI API key required. Enter it under API Configuration above.")
    elif _emb_provider == "voyage":
        _emb_api_key = st.session_state.get("_voyage_key", os.environ.get("VOYAGE_API_KEY"))
        if not _emb_api_key:
            st.warning("Voyage AI API key required. Enter it under API Configuration above.")
    elif _emb_provider == "google":
        _emb_api_key = st.session_state.get("_gemini_key", os.environ.get("GEMINI_API_KEY"))
        if not _emb_api_key:
            st.warning("Google Gemini API key required. Enter it under API Configuration above.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── ADVANCED / RESEARCHER SECTION
    with st.expander("Advanced (Researcher)", expanded=False):
        st.markdown('<div class="section-label">QUALITY CONTROL</div>', unsafe_allow_html=True)
        dedup_threshold = st.slider(
            "Dedup threshold", 0.80, 1.0, 1.0, step=0.01,
            help=PARAM_TOOLTIPS["dedup_threshold"],
        )

        st.markdown('<div class="section-label">VISUALIZATION</div>', unsafe_allow_html=True)
        show_overlap_zones = st.toggle("Show chunk overlap zones", value=False, help="Highlights overlapping text between adjacent chunks.")
        dim_reduction = st.selectbox("Tracer projection", ["PCA", "t-SNE"], help=PARAM_TOOLTIPS["dim_reduction"])
        similarity_metric = st.selectbox("Similarity metric", ["Cosine", "Dot Product", "Euclidean"], help=PARAM_TOOLTIPS["similarity_metric"])

    # ── Pipeline Log
    with st.expander("Pipeline log"):
        if st.session_state.pipeline_log:
            for entry in st.session_state.pipeline_log:
                st.markdown(f"- {entry}")
        else:
            st.caption("Upload a file to see the pipeline log.")


# ═══════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════

st.markdown('<p class="ed-title">Doc2MD-RAG</p><p class="ed-sub">Convert documents to markdown, chunk, index, and search with hybrid retrieval</p>', unsafe_allow_html=True)

# Pipeline Progress Bar
stage = st.session_state.pipeline_stage
pipeline_stages = [
    {"name": "Upload",  "status": "done" if stage >= 1 else "pending"},
    {"name": "Convert", "status": "done" if stage >= 2 else ("active" if stage == 1 else "pending")},
    {"name": "Chunk",   "status": "done" if stage >= 3 else ("active" if stage == 2 else "pending")},
    {"name": "Index",   "status": "done" if stage >= 4 else ("active" if stage == 3 else "pending")},
    {"name": "Search",  "status": "done" if stage >= 5 else ("active" if stage == 4 else "pending")},
]
render_pipeline(pipeline_stages)

# Top Controls
col_upload, col_search = st.columns([1, 1.2], gap="medium")
with col_upload:
    uploaded = st.file_uploader("Upload document", type=SUPPORTED_EXTENSIONS, label_visibility="collapsed")
with col_search:
    query_raw = st.text_input("Search", placeholder="Search chunks...", label_visibility="collapsed", key="search_input", max_chars=MAX_QUERY_LENGTH)
    query = query_raw.strip()[:MAX_QUERY_LENGTH] if query_raw else ""


# ─── Process Document ───────────────────────────────────────────

current_params = (chunk_method, chunk_size, chunk_overlap, semantic_threshold, emb_model, _emb_provider, bool(_emb_api_key), bool(st.session_state.get("_api_key")))

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({file_size_mb:.1f}MB). Maximum is {MAX_FILE_SIZE_MB}MB.")
        st.stop()

    file_hash = hashlib.md5(file_bytes).hexdigest()
    file_changed = file_hash != st.session_state.file_hash
    params_changed = st.session_state.indexed_params != current_params

    if file_changed or params_changed or st.session_state.rag_engine is None:
        st.session_state.pipeline_stage = 1
        with st.spinner("Converting and indexing..."):
            log = []
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                st.session_state.pipeline_stage = 2
                llm_client = get_llm_client(api_key=st.session_state.get("_api_key"))
                result = process_document(tmp_path, llm_client=llm_client)
                st.session_state.conversion_result = result
                stats = result["stats"]
                log.append(f"**Extracted** markdown ({stats['char_count']:,} chars)")
                log.append(f"**Converted** {stats['tables_found']} table(s) to list format")
                log.append(f"**Processed** {stats['images_found']} image reference(s)")

                st.session_state.pipeline_stage = 3
                engine = RAGEngine(embedding_model_name=emb_model, embedding_api_key=_emb_api_key)
                engine.index_document(
                    result["final_markdown"], source=uploaded.name,
                    method=chunk_method, chunk_size=chunk_size,
                    overlap=chunk_overlap, semantic_threshold=semantic_threshold,
                )
                st.session_state.rag_engine = engine
                st.session_state.indexed_params = current_params
                st.session_state.file_hash = file_hash
                st.session_state.pipeline_stage = 4

                n_chunks = len(engine.get_chunks())
                log.append(f"**Chunked** into {n_chunks} pieces ({chunk_method}, size={chunk_size})")
                log.append(f"**Indexed** with FAISS + BM25 ({emb_model})")
                st.session_state.pipeline_log = log
            except Exception as e:
                logger.error("Pipeline failed: %s", traceback.format_exc())
                st.error(f"Processing failed: {e}")
                st.session_state.pipeline_stage = 0
                st.stop()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# CONTENT
# ═══════════════════════════════════════════════════════════════

if st.session_state.conversion_result is not None:
    engine = st.session_state.rag_engine
    chunks = engine.get_chunks() if engine else []
    result = st.session_state.conversion_result
    stats = result["stats"]

    # ── Stats + Export Row
    stat_col, export_col = st.columns([4, 1])
    with stat_col:
        st.markdown(f"""<div class="card"><div class="metric-row">
            <div class="metric"><span class="metric-value">{stats['char_count']:,}</span><span class="metric-label">Characters</span></div>
            <div class="metric"><span class="metric-value">{len(chunks)}</span><span class="metric-label">Chunks</span></div>
            <div class="metric"><span class="metric-value">{stats['tables_found']}</span><span class="metric-label">Tables converted</span></div>
            <div class="metric"><span class="metric-value">{stats['images_found']}</span><span class="metric-label">Images processed</span></div>
        </div></div>""", unsafe_allow_html=True)
    with export_col:
        st.download_button(
            "Export Markdown",
            result["final_markdown"],
            file_name=f"{os.path.splitext(uploaded.name)[0]}_converted.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # ── Search
    search_results = []
    trace_data = {}
    eval_metrics = {}
    matched_ids = set()

    if query and engine and chunks:
        st.session_state.pipeline_stage = 5
        search_results = engine.search(
            query, mode=search_mode, top_k=top_k,
            bm25_weight=bm25_weight, vector_weight=vector_weight,
            score_threshold=score_threshold, num_candidates=num_candidates,
            dedup_threshold=dedup_threshold,
        )
        trace_data = engine.get_trace(
            query, mode=search_mode, top_k=top_k,
            bm25_weight=bm25_weight, vector_weight=vector_weight,
        )
        eval_metrics = engine.compute_eval_metrics(search_results, query)
        matched_ids = {r["chunk_id"] for r in search_results}

    # ── Two Column Layout
    doc_col, prompt_col = st.columns([3, 2], gap="medium")

    with doc_col:
        st.markdown('<div class="section-label">DOCUMENT VIEW</div>', unsafe_allow_html=True)

        # Limit visible chunks to prevent browser freeze on large documents
        visible_chunks = chunks[:MAX_VISIBLE_CHUNKS]
        truncated = len(chunks) > MAX_VISIBLE_CHUNKS

        html_parts = []
        for i, chunk in enumerate(visible_chunks):
            cid = chunk["chunk_id"]
            is_match = cid in matched_ids
            bg = MATCH_COLOR if is_match else CHUNK_COLORS[cid % 2]
            border_color = MATCH_BORDER if is_match else CHUNK_BORDER
            # Sanitize: escape HTML in chunk text, then render markdown
            safe_text = html.escape(chunk["text"])
            chunk_html = md_lib.markdown(safe_text)
            overlap_style = "border-top:2px dashed #D1D5DB;" if show_overlap_zones and i > 0 else ""

            html_parts.append(f'<div class="chunk-block" id="chunk-{cid}" style="background:{bg};border-left-color:{border_color};{overlap_style}"><span class="chunk-id">#{cid}</span>{chunk_html}</div>')

        if truncated:
            html_parts.append(f'<div style="text-align:center;padding:12px;color:#9CA3AF;font-size:0.85rem;">Showing first {MAX_VISIBLE_CHUNKS} of {len(chunks)} chunks</div>')

        scroll_js = ""
        if matched_ids:
            first = min(matched_ids)
            scroll_js = f"<script>setTimeout(function(){{var el=document.getElementById('chunk-{first}');if(el)el.scrollIntoView({{behavior:'smooth',block:'center'}})}},100);</script>"

        st.components.v1.html(f'<div style="max-height:550px;overflow-y:auto;padding:4px;font-family:Inter,sans-serif;">{"".join(html_parts)}</div>{scroll_js}', height=570, scrolling=True)

    with prompt_col:
        st.markdown('<div class="section-label">PROMPT BUILDER</div>', unsafe_allow_html=True)

        if search_results:
            context_parts = [f"--- Chunk #{r['chunk_id']} (score: {r['score']:.2f}) ---\n{r['text']}" for r in search_results]
            prompt = f"Use the following context to answer the question.\n\nContext:\n{chr(10).join(context_parts)}\n\nQuestion: {query}\nAnswer:"

            est_tokens = int(len(prompt) / 3.5)  # approximate; use tiktoken for exact counts
            st.caption(f"~{est_tokens:,} tokens | {len(search_results)} chunks | threshold: {score_threshold}")
            st.code(prompt, language="text")

            dl_col, _ = st.columns([1, 1])
            with dl_col:
                st.download_button("Download prompt", prompt, file_name="rag_prompt.txt", mime="text/plain", use_container_width=True)

            st.markdown('<div class="section-label">RESULTS</div>', unsafe_allow_html=True)
            for r in search_results:
                st.markdown(f'**Chunk #{r["chunk_id"]}** &nbsp;<span class="score-badge badge-combined">{r["score"]:.2f}</span> <span class="score-badge badge-bm25">BM25 {r["bm25_score"]:.2f}</span> <span class="score-badge badge-vector">Vec {r["faiss_score"]:.2f}</span>', unsafe_allow_html=True)

            # Eval Metrics
            if eval_metrics:
                st.markdown('<div class="section-label" style="margin-top:1rem;">EVAL METRICS</div>', unsafe_allow_html=True)
                st.markdown(f"""<div class="eval-grid">
                    <div class="eval-card"><div class="eval-value">{eval_metrics['lexical_coverage']:.0%}</div><div class="eval-label">Lexical Coverage</div></div>
                    <div class="eval-card"><div class="eval-value">{eval_metrics['redundancy']:.2f}</div><div class="eval-label">Redundancy</div></div>
                    <div class="eval-card"><div class="eval-value">{eval_metrics['score_spread']:.2f}</div><div class="eval-label">Score Spread</div></div>
                    <div class="eval-card"><div class="eval-value">{eval_metrics['char_coverage']:.1%}</div><div class="eval-label">Doc Coverage</div></div>
                    <div class="eval-card"><div class="eval-value">{eval_metrics['avg_score']:.2f}</div><div class="eval-label">Avg Score</div></div>
                    <div class="eval-card"><div class="eval-value">{eval_metrics['score_variance']:.4f}</div><div class="eval-label">Score Variance</div></div>
                </div>""", unsafe_allow_html=True)
        else:
            if query:
                st.info("No chunks above the score threshold. Try lowering it in the sidebar.")
            else:
                st.markdown('<div style="text-align:center;padding:40px 20px;color:#9CA3AF;"><p style="font-size:1.5rem;margin-bottom:0.5rem;">Search to build a prompt</p><p style="font-size:0.85rem;">Matching chunks will be assembled into a copy-paste-ready LLM prompt.</p></div>', unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════
    # BOTTOM TABS
    # ═══════════════════════════════════════════════════════════

    tab_chunks, tab_tracer, tab_pipeline, tab_compare = st.tabs([
        "Chunks", "Tracer", "Pipeline Overview", "Chunk Comparison",
    ])

    # ── TAB: Chunks
    with tab_chunks:
        if chunks:
            chunk_data = [{
                "ID": c["chunk_id"],
                "Preview": c["text"][:140].replace("\n", " "),
                "Chars": len(c["text"]),
                "Start": c["start_char"],
                "End": c["end_char"],
            } for c in chunks]
            st.dataframe(pd.DataFrame(chunk_data), use_container_width=True, hide_index=True)

    # ── TAB: Tracer
    with tab_tracer:
        if trace_data and query:
            st.markdown('<div class="section-label">RAG PIPELINE TRACE</div>', unsafe_allow_html=True)
            st.plotly_chart(render_query_tokens(trace_data), use_container_width=True)
            st.plotly_chart(render_embedding_space(trace_data, chunks, method=dim_reduction), use_container_width=True)

            st.markdown('<div class="section-label">BM25 TOKEN MATCHES</div>', unsafe_allow_html=True)
            bm25_highlights = render_bm25_highlights(trace_data, chunks)
            for item in bm25_highlights:
                st.markdown(f'**Chunk #{item["chunk_id"]}** <span class="score-badge badge-bm25">BM25 {item["score"]:.3f}</span> &nbsp;matched: {", ".join(item["matched_tokens"]) if item["matched_tokens"] else "none"}', unsafe_allow_html=True)
                st.markdown(item["html"], unsafe_allow_html=True)
                st.divider()

            st.plotly_chart(render_score_fusion(trace_data, chunks), use_container_width=True)

            # Score distribution
            if trace_data.get("combined_scores"):
                import plotly.graph_objects as go
                fig = go.Figure(go.Histogram(x=trace_data["combined_scores"], nbinsx=30, marker_color="#3B82F6", opacity=0.8))
                fig.update_layout(title="Score Distribution (all chunks)", xaxis_title="Combined Score", yaxis_title="Count", height=250, margin=dict(t=40,b=40,l=40,r=20), plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Search for something to see the RAG pipeline tracer.")

    # ── TAB: Pipeline Overview
    with tab_pipeline:
        # ── Visual Flow Diagram
        st.markdown('<div class="section-label">PIPELINE FLOW</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="flow-diagram">
            <div class="flow-step"><span class="step-icon">📎</span><span class="step-name">Upload</span><span class="step-detail">PDF, DOCX, PPTX, CSV</span></div>
            <span class="flow-arrow">→</span>
            <div class="flow-step"><span class="step-icon">📝</span><span class="step-name">Convert</span><span class="step-detail">Tables → Lists<br>Images → Descriptions</span></div>
            <span class="flow-arrow">→</span>
            <div class="flow-step"><span class="step-icon">✂️</span><span class="step-name">Chunk</span><span class="step-detail">Split by strategy<br>+ overlap</span></div>
            <span class="flow-arrow">→</span>
            <div class="flow-step" style="border-color:#3B82F6;"><span class="step-icon">🧠</span><span class="step-name">Embed</span><span class="step-detail">Text → Vectors<br>(384d or 768d)</span></div>
            <span class="flow-arrow">→</span>
            <div class="flow-step"><span class="step-icon">📊</span><span class="step-name">Index</span><span class="step-detail">FAISS + BM25<br>dual index</span></div>
            <span class="flow-arrow">→</span>
            <div class="flow-step"><span class="step-icon">🔍</span><span class="step-name">Search</span><span class="step-detail">Hybrid scoring<br>+ dedup + filter</span></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Detailed Pipeline Steps
        st.markdown('<div class="section-label">STEP-BY-STEP BREAKDOWN</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="font-size:0.88rem;line-height:1.8;">
        <strong>1. Upload</strong> — Document loaded into memory (max 100MB). Format detected from extension.<br><br>
        <strong>2. Convert</strong> — Each format has a dedicated parser:<br>
        &nbsp;&nbsp;• <strong>PDF</strong>: pymupdf4llm extracts text layout, tables, and images. Tables with <code>&lt;br&gt;</code> tags and cross-page splits are normalized.<br>
        &nbsp;&nbsp;• <strong>DOCX</strong>: python-docx maps Word heading styles to markdown headers, extracts tables.<br>
        &nbsp;&nbsp;• <strong>PPTX</strong>: python-pptx processes slide-by-slide, extracting text frames and tables.<br>
        &nbsp;&nbsp;• <strong>CSV</strong>: pandas reads the file and converts to markdown table.<br>
        All pipe-tables are then converted to nested bullet lists (better for LLM chunking). If an Anthropic API key is configured, images are described using Claude Vision.<br><br>
        <strong>3. Chunk</strong> — The clean markdown is split into overlapping pieces. Each chunk records its character offset (start/end) in the original text so we can highlight matches in the document view.<br><br>
        <strong>4. Embed</strong> — The <strong>embedding model</strong> converts each chunk's text into a dense vector. This is the key step that enables <em>semantic</em> search — similar meaning maps to nearby points in vector space, regardless of exact wording.<br><br>
        <strong>5. Index</strong> — Two parallel indexes are built simultaneously:<br>
        &nbsp;&nbsp;• <strong>FAISS</strong> (vector) — stores all chunk embeddings for cosine similarity search.<br>
        &nbsp;&nbsp;• <strong>BM25</strong> (keyword) — tokenizes chunks and builds term frequency / inverse document frequency index.<br><br>
        <strong>6. Search</strong> — Your query goes through both indexes. Scores are normalized, weighted, combined, filtered by threshold, and deduplicated before returning the top-K results.
        </div>
        """, unsafe_allow_html=True)

        # ── How the Embedding Model Works
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">HOW EMBEDDING MODELS WORK</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="embed-demo">
            <h4>Text → Vector: What the model actually does</h4>
            <p style="font-size:0.85rem;color:#6B7280;margin-bottom:1rem;">
                The embedding model reads a chunk of text and compresses its <em>meaning</em> into a fixed-size vector
                (a list of numbers). Texts with similar meaning end up with similar vectors, even if they use different words.
            </p>
            <p style="font-size:0.8rem;color:#374151;font-weight:600;margin-bottom:0.5rem;">Example: 3 sentences → 384-dimensional vectors (showing first 8 dims)</p>
            <div class="vector-bar">
                <span class="vb-label">"create voucher"</span>
                <div class="vb-bar" style="background:linear-gradient(90deg,#3B82F6 72%, #93C5FD 72%);opacity:0.9;"></div>
                <span class="vb-val">[0.23, -0.11, 0.45, 0.08, ...]</span>
            </div>
            <div class="vector-bar">
                <span class="vb-label">"make a new gift"</span>
                <div class="vb-bar" style="background:linear-gradient(90deg,#3B82F6 68%, #93C5FD 68%);opacity:0.9;"></div>
                <span class="vb-val">[0.21, -0.09, 0.42, 0.10, ...]</span>
            </div>
            <div class="vector-bar">
                <span class="vb-label">"server config"</span>
                <div class="vb-bar" style="background:linear-gradient(90deg,#EF4444 30%, #FCA5A5 30%);opacity:0.9;"></div>
                <span class="vb-val">[-0.15, 0.33, -0.07, 0.28, ...]</span>
            </div>
            <p style="font-size:0.78rem;color:#6B7280;margin-top:0.75rem;">
                ↑ "create voucher" and "make a new gift" have similar vectors (high cosine similarity ~0.85)
                because they share semantic meaning. "server config" is far away (~0.12) — different topic entirely.
                This is how <strong>Vector (KNN)</strong> search finds relevant chunks without exact keyword matches.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Embedding model comparison
        st.markdown("""
        <div class="card" style="font-size:0.85rem;">
            <h4 style="margin-top:0;">Embedding Model Comparison</h4>
            <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
                <tr style="border-bottom:2px solid #E5E7EB;">
                    <th style="text-align:left;padding:6px;">Model</th>
                    <th style="text-align:center;padding:6px;">Dimensions</th>
                    <th style="text-align:center;padding:6px;">Size</th>
                    <th style="text-align:center;padding:6px;">Speed</th>
                    <th style="text-align:left;padding:6px;">Best for</th>
                </tr>
                <tr style="border-bottom:1px solid #F3F4F6;">
                    <td style="padding:6px;font-weight:600;">all-MiniLM-L6-v2</td>
                    <td style="text-align:center;padding:6px;">384</td>
                    <td style="text-align:center;padding:6px;">~80MB</td>
                    <td style="text-align:center;padding:6px;color:#10B981;">Fast</td>
                    <td style="padding:6px;">General use, quick iteration</td>
                </tr>
                <tr>
                    <td style="padding:6px;font-weight:600;">all-mpnet-base-v2</td>
                    <td style="text-align:center;padding:6px;">768</td>
                    <td style="text-align:center;padding:6px;">~420MB</td>
                    <td style="text-align:center;padding:6px;color:#F59E0B;">Slower</td>
                    <td style="padding:6px;">Higher accuracy, nuanced queries</td>
                </tr>
            </table>
            <p style="font-size:0.78rem;color:#6B7280;margin-top:0.5rem;">Higher dimensions capture more nuance but require more memory and computation.
            For most documents, MiniLM-L6 is sufficient. Switch to mpnet for longer, more complex texts.</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Parameter Impact Guide
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">HOW EACH PARAMETER AFFECTS RESULTS</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="param-impact">
            <div class="param-card">
                <span class="param-icon">✂️</span>
                <h4>Chunk Size</h4>
                <p><strong>Small (100-200)</strong>: Very precise matches, but each chunk may lack context. Good for FAQ-style retrieval.<br>
                <strong>Large (600-1000)</strong>: More context per chunk, but may include irrelevant text. Good for narrative documents.</p>
                <div class="param-effect">Impact: Changes how much text the LLM sees per retrieved piece. Directly affects prompt token count.</div>
            </div>
            <div class="param-card">
                <span class="param-icon">🔗</span>
                <h4>Chunk Overlap</h4>
                <p>Characters shared between adjacent chunks. Prevents information from being split across a boundary and lost.<br>
                <strong>0</strong>: No overlap — fastest, but risks cutting sentences in half.<br>
                <strong>50-100</strong>: Safe overlap — duplicates some text but preserves context.</p>
                <div class="param-effect">Impact: Higher overlap = more chunks (redundancy), but fewer missed boundary cases.</div>
            </div>
            <div class="param-card">
                <span class="param-icon">⚖️</span>
                <h4>BM25 / Vector Weights</h4>
                <p><strong>High BM25</strong> (0.7/0.3): Prioritizes exact keyword matches. Best when users search for specific terms, names, or codes.<br>
                <strong>High Vector</strong> (0.3/0.7): Prioritizes semantic meaning. Best when users ask natural-language questions.</p>
                <div class="param-effect">Impact: Controls whether "voucher giảm giá" matches only those exact words (BM25) or also "discount coupon" (Vector).</div>
            </div>
            <div class="param-card">
                <span class="param-icon">🎯</span>
                <h4>Top K</h4>
                <p>Number of chunks returned. More chunks = broader context for the LLM, but also more noise and higher token cost.<br>
                <strong>3-5</strong>: Focused, low-noise results.<br>
                <strong>10-20</strong>: Broad recall, good for complex questions spanning multiple sections.</p>
                <div class="param-effect">Impact: Directly controls prompt size. 5 chunks × 400 chars ≈ 500 tokens of context.</div>
            </div>
            <div class="param-card">
                <span class="param-icon">📏</span>
                <h4>Score Threshold</h4>
                <p>Minimum score a chunk needs to be included. Filters out low-relevance noise.<br>
                <strong>0.0</strong>: No filter — return Top K regardless of quality.<br>
                <strong>0.3-0.5</strong>: Only return chunks with meaningful relevance.</p>
                <div class="param-effect">Impact: Prevents the LLM from seeing irrelevant context that could confuse its answer.</div>
            </div>
            <div class="param-card">
                <span class="param-icon">🔢</span>
                <h4>Num Candidates</h4>
                <p>How many chunks FAISS retrieves before final ranking. Like casting a wider net before picking the best fish.<br>
                <strong>Low (10-20)</strong>: Faster but may miss good chunks ranked just outside Top K.<br>
                <strong>High (100-200)</strong>: Better recall, especially for Hybrid mode.</p>
                <div class="param-effect">Impact: Higher candidates improve Hybrid search quality at the cost of slightly more computation.</div>
            </div>
            <div class="param-card">
                <span class="param-icon">🧬</span>
                <h4>Embedding Model</h4>
                <p>Converts text to vectors. The model's "understanding" of language determines how well semantic search works.<br>
                A better model can distinguish "create a gift" from "gift wrapping" — smaller models may conflate them.</p>
                <div class="param-effect">Impact: Affects ALL vector operations — indexing, search, dedup, and tracer visualization quality.</div>
            </div>
            <div class="param-card">
                <span class="param-icon">🔄</span>
                <h4>Dedup Threshold</h4>
                <p>Cosine similarity above which two chunks are considered duplicates. Prevents returning near-identical text.<br>
                <strong>1.0</strong>: Off — no dedup.<br>
                <strong>0.90-0.95</strong>: Removes very similar chunks, improving result diversity.</p>
                <div class="param-effect">Impact: Reduces redundancy in the prompt, giving the LLM more diverse context to work with.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Search Mode Deep Dive
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">SEARCH MODES EXPLAINED</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="font-size:0.85rem;line-height:1.7;">
            <h4 style="margin-top:0;">How Hybrid Search Combines Two Signals</h4>
            <p>When you search in Hybrid mode, your query goes through two completely different systems in parallel:</p>
            <div style="display:flex;gap:1rem;margin:1rem 0;">
                <div style="flex:1;background:#DBEAFE;border-radius:8px;padding:1rem;">
                    <strong style="color:#1D4ED8;">BM25 (Keyword)</strong><br>
                    <span style="font-size:0.8rem;">Tokenizes query → counts term frequency in each chunk → weighs by rarity (IDF) → ranks by TF-IDF score.</span><br>
                    <span style="font-size:0.78rem;color:#1E40AF;">Catches: exact terms, names, IDs, technical jargon</span>
                </div>
                <div style="flex:1;background:#D1FAE5;border-radius:8px;padding:1rem;">
                    <strong style="color:#065F46;">Vector (KNN)</strong><br>
                    <span style="font-size:0.8rem;">Embeds query → computes cosine similarity against all chunk vectors → ranks by semantic closeness.</span><br>
                    <span style="font-size:0.78rem;color:#065F46;">Catches: paraphrases, related concepts, synonyms</span>
                </div>
            </div>
            <p><strong>Fusion formula:</strong> <code>final_score = bm25_weight × norm(bm25) + vector_weight × cosine_sim</code></p>
            <p style="font-size:0.82rem;color:#6B7280;">BM25 scores are normalized to [0,1] by dividing by the max score. Vector scores (cosine similarity of normalized vectors) are already in [0,1]. The weighted sum produces the final ranking.</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Chunking Strategy Visual
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">CHUNKING STRATEGIES VISUALIZED</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="font-size:0.85rem;line-height:1.7;">
            <div style="margin-bottom:1rem;">
                <strong>Recursive</strong> (default) — Tries to split at natural boundaries in order: <code>\\n\\n</code> → <code>\\n</code> → <code>. </code> → <code> </code> → character. Falls through each level until chunks fit within the size limit.
                <div style="background:#F3F4F6;border-radius:6px;padding:0.5rem;margin-top:0.5rem;font-family:monospace;font-size:0.75rem;">
                    <span style="background:#DBEAFE;padding:2px 4px;border-radius:3px;">Paragraph 1 text here...</span>
                    <span style="color:#D1D5DB;"> | </span>
                    <span style="background:#D1FAE5;padding:2px 4px;border-radius:3px;">Paragraph 2 text here...</span>
                    <span style="color:#D1D5DB;"> | </span>
                    <span style="background:#FEF3C7;padding:2px 4px;border-radius:3px;">Paragraph 3...</span>
                </div>
            </div>
            <div style="margin-bottom:1rem;">
                <strong>By Sentence</strong> — Groups complete sentences until the chunk size limit. Never breaks mid-sentence.
                <div style="background:#F3F4F6;border-radius:6px;padding:0.5rem;margin-top:0.5rem;font-family:monospace;font-size:0.75rem;">
                    <span style="background:#DBEAFE;padding:2px 4px;border-radius:3px;">Sent 1. Sent 2. Sent 3.</span>
                    <span style="color:#D1D5DB;"> | </span>
                    <span style="background:#D1FAE5;padding:2px 4px;border-radius:3px;">Sent 4. Sent 5.</span>
                    <span style="color:#D1D5DB;"> | </span>
                    <span style="background:#FEF3C7;padding:2px 4px;border-radius:3px;">Sent 6. Sent 7. Sent 8.</span>
                </div>
            </div>
            <div style="margin-bottom:1rem;">
                <strong>By Markdown Headers</strong> — Splits at <code>#</code>, <code>##</code>, <code>###</code> headings. Each section becomes a chunk (sub-split if too large). Preserves document structure.
                <div style="background:#F3F4F6;border-radius:6px;padding:0.5rem;margin-top:0.5rem;font-family:monospace;font-size:0.75rem;">
                    <span style="background:#DBEAFE;padding:2px 4px;border-radius:3px;"># Section 1 content...</span>
                    <span style="color:#D1D5DB;"> | </span>
                    <span style="background:#D1FAE5;padding:2px 4px;border-radius:3px;">## Section 1.1 content...</span>
                    <span style="color:#D1D5DB;"> | </span>
                    <span style="background:#FEF3C7;padding:2px 4px;border-radius:3px;">## Section 1.2 content...</span>
                </div>
            </div>
            <div>
                <strong>Semantic</strong> — Embeds each sentence, compares consecutive similarities. Splits where meaning shifts (cosine sim drops below threshold). Creates the most coherent chunks but is slower.
                <div style="background:#F3F4F6;border-radius:6px;padding:0.5rem;margin-top:0.5rem;font-family:monospace;font-size:0.75rem;">
                    <span style="background:#DBEAFE;padding:2px 4px;border-radius:3px;">Topic A sentences...</span>
                    <span style="color:#EF4444;font-weight:bold;"> ✂ sim=0.3 </span>
                    <span style="background:#D1FAE5;padding:2px 4px;border-radius:3px;">Topic B sentences...</span>
                    <span style="color:#EF4444;font-weight:bold;"> ✂ sim=0.2 </span>
                    <span style="background:#FEF3C7;padding:2px 4px;border-radius:3px;">Topic C sentences...</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Indexing method explainers
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">INDEXING METHODS</div>', unsafe_allow_html=True)

        for name, info in INDEXING_EXPLAINERS.items():
            st.markdown(f"""<div class="explainer-card">
                <h4>{name}</h4>
                <p><span class="label">How it works:</span> {info['how']}</p>
                <p><span class="label">Strengths:</span> {info['pros']}</p>
                <p><span class="label">Weaknesses:</span> {info['cons']}</p>
                <p><span class="label">Complexity:</span> <code>{info['complexity']}</code></p>
            </div>""", unsafe_allow_html=True)

        # ── Current config summary
        if st.session_state.rag_engine:
            eng = st.session_state.rag_engine
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">CURRENT CONFIGURATION</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="card" style="font-size:0.85rem;">
                <div class="metric-row">
                    <div class="metric"><span class="metric-value">{len(eng.get_chunks())}</span><span class="metric-label">Total chunks</span></div>
                    <div class="metric"><span class="metric-value">{EMBEDDING_MODELS[emb_model]['dim']}d</span><span class="metric-label">Vector dims</span></div>
                    <div class="metric"><span class="metric-value">{chunk_size}</span><span class="metric-label">Chunk size</span></div>
                    <div class="metric"><span class="metric-value">{chunk_overlap}</span><span class="metric-label">Overlap</span></div>
                    <div class="metric"><span class="metric-value">{num_candidates}</span><span class="metric-label">Candidates</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── TAB: Chunk Comparison
    with tab_compare:
        st.markdown('<div class="section-label">CHUNKING STRATEGY COMPARISON</div>', unsafe_allow_html=True)
        st.caption("Compare how different chunking methods split your document.")

        if st.session_state.conversion_result:
            md_text = st.session_state.conversion_result["final_markdown"]
            compare_methods = st.multiselect("Compare methods", CHUNKING_METHODS, default=["Recursive", "By Sentence"])

            if compare_methods and len(compare_methods) >= 2:
                compare_cols = st.columns(len(compare_methods))
                temp_engine = RAGEngine(embedding_model_name=emb_model)

                for i, method in enumerate(compare_methods):
                    with compare_cols[i]:
                        st.markdown(f"**{method}**")
                        temp_engine.clear()
                        temp_engine.index_document(
                            md_text, source="compare", method=method,
                            chunk_size=chunk_size, overlap=chunk_overlap,
                            semantic_threshold=semantic_threshold,
                        )
                        tc = temp_engine.get_chunks()
                        sizes = [len(c["text"]) for c in tc]
                        avg = sum(sizes) / len(sizes) if sizes else 0
                        st.metric("Chunks", len(tc))
                        st.metric("Avg size", f"{avg:.0f} chars")
                        for c in tc[:5]:
                            st.markdown(f'<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:6px;padding:8px;margin:4px 0;font-size:12px;line-height:1.4;">{c["text"][:200]}...</div>', unsafe_allow_html=True)
                        if len(tc) > 5:
                            st.caption(f"+ {len(tc) - 5} more chunks")
            else:
                st.caption("Select at least 2 methods to compare.")

else:
    st.markdown('<div class="empty-state"><h2>Drop a document to begin</h2><p>Upload a PDF, PPTX, DOCX, or CSV. It will be converted to clean markdown (tables become lists, images become descriptions), then chunked and indexed for hybrid BM25 + vector search.</p><p style="margin-top:1rem;font-size:0.8rem;color:#D1D5DB;">Supports files up to 100MB</p></div>', unsafe_allow_html=True)
