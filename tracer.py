# tracer.py — RAG pipeline visualization (4 panels)
#
# Panel 1: Query tokenization + IDF weights
# Panel 2: Embedding space (PCA 2D scatter)
# Panel 3: BM25 token matches (highlighted text)
# Panel 4: Score fusion (grouped bar chart)

import numpy as np
import plotly.graph_objects as go


def render_query_tokens(trace: dict) -> go.Figure:
    """Panel 1: Bar chart of query token IDF weights."""
    tokens = trace["query_tokens"]
    idf = trace["idf_values"]

    values = [idf.get(t, 0.0) for t in tokens]

    fig = go.Figure(
        go.Bar(
            x=tokens,
            y=values,
            marker_color=["#3498DB" if v > 0 else "#BDC3C7" for v in values],
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Query Token IDF Weights",
        xaxis_title="Token",
        yaxis_title="IDF (inverse document frequency)",
        height=300,
        margin=dict(t=40, b=40, l=40, r=20),
        plot_bgcolor="white",
    )
    return fig


def render_embedding_space(trace: dict, chunks: list[dict], method: str = "PCA") -> go.Figure:
    """Panel 2: 2D scatter of query + chunk embeddings (PCA or t-SNE)."""
    query_emb = trace["query_embedding"].reshape(1, -1)
    chunk_embs = trace["chunk_embeddings"]
    top_k = set(trace["top_k_indices"])

    # Combine and reduce to 2D
    all_embs = np.vstack([query_emb, chunk_embs])
    n_samples = all_embs.shape[0]
    n_features = all_embs.shape[1]
    if min(n_samples, n_features) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for visualization", showarrow=False)
        return fig

    if method == "t-SNE" and n_samples > 3:
        from sklearn.manifold import TSNE
        perplexity = min(30, n_samples - 1)
        reduced = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(all_embs)
    else:
        from sklearn.decomposition import PCA as _PCA
        reduced = _PCA(n_components=2).fit_transform(all_embs)

    query_pt = reduced[0]
    chunk_pts = reduced[1:]

    # Chunk colors: matched = amber, unmatched = light blue
    colors = []
    hover_texts = []
    for i, chunk in enumerate(chunks):
        if i in top_k:
            colors.append("#F39C12")
        else:
            colors.append("#AED6F1")
        preview = chunk["text"][:100].replace("\n", " ") + "..."
        hover_texts.append(f"Chunk #{chunk['chunk_id']}<br>{preview}")

    fig = go.Figure()

    # Chunk points
    fig.add_trace(go.Scatter(
        x=chunk_pts[:, 0],
        y=chunk_pts[:, 1],
        mode="markers+text",
        marker=dict(size=10, color=colors, line=dict(width=1, color="#2C3E50")),
        text=[f"#{c['chunk_id']}" for c in chunks],
        textposition="top center",
        textfont=dict(size=9),
        hovertext=hover_texts,
        hoverinfo="text",
        name="Chunks",
    ))

    # Query point (star)
    fig.add_trace(go.Scatter(
        x=[query_pt[0]],
        y=[query_pt[1]],
        mode="markers+text",
        marker=dict(size=16, color="#E74C3C", symbol="star"),
        text=["Query"],
        textposition="top center",
        name="Query",
    ))

    fig.update_layout(
        title=f"Embedding Space ({method} 2D)",
        height=400,
        margin=dict(t=40, b=40, l=40, r=20),
        plot_bgcolor="white",
        showlegend=True,
        xaxis=dict(title="Dimension 1"),
        yaxis=dict(title="Dimension 2"),
    )
    return fig


def render_bm25_highlights(trace: dict, chunks: list[dict]) -> list[dict]:
    """Panel 3: Return data for BM25 token match highlighting.

    Returns list of {chunk_id, html, matched_tokens, score} for top-K chunks.
    """
    top_k = trace["top_k_indices"]
    bm25_matches = trace["bm25_token_matches"]
    bm25_scores = trace["bm25_scores_raw"]

    results = []
    for idx in top_k:
        chunk = chunks[idx]
        matched_tokens = set(bm25_matches.get(idx, []))
        text = chunk["text"]

        # Highlight matched tokens in the text
        if matched_tokens:
            words = text.split()
            highlighted_words = []
            for word in words:
                clean = word.lower().strip(".,!?;:\"'()[]{}—-")
                if clean in matched_tokens:
                    highlighted_words.append(
                        f'<span style="background-color:#F9E79F;padding:1px 3px;border-radius:3px;font-weight:600">{word}</span>'
                    )
                else:
                    highlighted_words.append(word)
            html = " ".join(highlighted_words)
        else:
            html = text

        results.append({
            "chunk_id": chunk["chunk_id"],
            "html": html,
            "matched_tokens": list(matched_tokens),
            "score": bm25_scores[idx],
        })

    return results


def render_score_fusion(trace: dict, chunks: list[dict]) -> go.Figure:
    """Panel 4: Grouped bar chart showing BM25/Vector/Combined scores for top-K."""
    top_k = trace["top_k_indices"]
    bm25_raw = trace["bm25_scores_raw"]
    vector_raw = trace["vector_scores_raw"]
    combined = trace["combined_scores"]

    labels = [f"#{chunks[i]['chunk_id']}" for i in top_k]
    bm25_vals = [bm25_raw[i] for i in top_k]
    vector_vals = [vector_raw[i] for i in top_k]
    combined_vals = [combined[i] for i in top_k]

    # Normalize BM25 for display
    max_bm25 = max(bm25_vals) if bm25_vals and max(bm25_vals) > 0 else 1.0
    bm25_norm = [v / max_bm25 for v in bm25_vals]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="BM25 (normalized)",
        x=labels,
        y=bm25_norm,
        marker_color="#3498DB",
        text=[f"{v:.2f}" for v in bm25_norm],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        name="Vector (cosine sim)",
        x=labels,
        y=vector_vals,
        marker_color="#2ECC71",
        text=[f"{v:.2f}" for v in vector_vals],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        name="Combined",
        x=labels,
        y=combined_vals,
        marker_color="#E74C3C",
        text=[f"{v:.2f}" for v in combined_vals],
        textposition="outside",
    ))

    fig.update_layout(
        title="Score Fusion Breakdown",
        barmode="group",
        height=350,
        margin=dict(t=40, b=40, l=40, r=20),
        plot_bgcolor="white",
        xaxis_title="Chunk",
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
