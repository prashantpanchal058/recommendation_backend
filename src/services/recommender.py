import numpy as np
import faiss
from scipy.sparse import issparse


# ── Built once in loader.py, passed via artifacts ─────────────────────
# artifacts["faiss_index"]   → faiss.IndexFlatIP
# artifacts["matrix_dense"]  → np.ndarray float32 (normalized)


def recommend_by_asin(asin: str, top_n: int = 10, artifacts: dict = None) -> list[dict]:
    """
    Same-category recommendations for a given ASIN.
    Stage 1 → FAISS retrieves 500 candidates fast
    Stage 2 → relevance score re-ranks them
    Ranking: 0.6 × similarity + 0.4 × popularity_score
    """
    faiss_index  = artifacts["faiss_index"]
    matrix_dense = artifacts["matrix_dense"]
    asin_to_idx  = artifacts["asin_to_idx"]
    df           = artifacts["df"]

    if asin not in asin_to_idx:
        return []

    idx      = asin_to_idx[asin]
    category = df.iloc[idx]["category_name"]

    # ── Stage 1: FAISS — retrieve top 500 candidates ──────────────────
    query               = matrix_dense[idx].reshape(1, -1)          # (1, n_features)
    similarities, idxs  = faiss_index.search(query, 501)            # +1 to exclude self
    similarities        = similarities[0]                            # flatten
    idxs                = idxs[0]

    # Exclude self
    mask         = idxs != idx
    idxs         = idxs[mask][:500]
    similarities = similarities[mask][:500]

    # ── Stage 2: Re-rank candidates ───────────────────────────────────
    candidates                     = df.iloc[idxs].copy()
    candidates["similarity_score"] = similarities

    # Hard filter: same category only
    candidates = candidates[candidates["category_name"] == category].copy()

    candidates["relevance_score"] = (
        0.6 * candidates["similarity_score"] +
        0.4 * candidates["popularity_score"]
    )

    top = candidates.sort_values("relevance_score", ascending=False).head(top_n)

    return _format_products(top)


def recommend_for_user(viewed_asins: list[str], top_n: int = 10, artifacts: dict = None) -> list[dict]:
    """
    Average the vectors of last-viewed ASINs → FAISS finds closest products.
    Used for homepage 'Based on your history' section.
    """
    faiss_index  = artifacts["faiss_index"]
    matrix_dense = artifacts["matrix_dense"]
    asin_to_idx  = artifacts["asin_to_idx"]
    df           = artifacts["df"]

    valid_idxs = [asin_to_idx[a] for a in viewed_asins if a in asin_to_idx]

    if not valid_idxs:
        # Cold start: return top products by popularity
        return _format_products(
            df.sort_values("popularity_score", ascending=False).head(top_n)
        )

    # Average vector of all viewed products → already normalized rows, re-normalize after mean
    user_vector = matrix_dense[valid_idxs].mean(axis=0, keepdims=True).astype("float32")  # (1, n_features)
    faiss.normalize_L2(user_vector)                                                         # keep on unit sphere

    # ── FAISS search ──────────────────────────────────────────────────
    similarities, idxs = faiss_index.search(user_vector, top_n + len(valid_idxs))
    similarities       = similarities[0]
    idxs               = idxs[0]

    # Exclude already-viewed products
    mask         = ~np.isin(idxs, valid_idxs)
    idxs         = idxs[mask][:top_n]
    similarities = similarities[mask][:top_n]

    result                     = df.iloc[idxs].copy()
    result["relevance_score"]  = similarities

    return _format_products(result)


def _format_products(df_slice) -> list[dict]:
    """Return clean list of dicts for API response."""
    cols = [
        "asin", "title", "category_name", "stars", "price",
        "boughtInLastMonth", "isBestSeller",
        "popularity_score", "purchase_likelihood", "relevance_score","imgUrl", "productURL","reviews", 
        "listPrice"
    ]
    cols = [c for c in cols if c in df_slice.columns]
    return df_slice[cols].fillna(0).to_dict(orient="records")

# import numpy as np
# from scipy.sparse import issparse
# from sklearn.metrics.pairwise import cosine_similarity


# def recommend_by_asin(asin: str, top_n: int = 10, artifacts: dict = None) -> list[dict]:
#     combined_matrix = artifacts["combined_matrix"]
#     asin_to_idx     = artifacts["asin_to_idx"]
#     df              = artifacts["df"]

#     if asin not in asin_to_idx:
#         return []

#     idx      = asin_to_idx[asin]
#     category = df.iloc[idx]["category_name"]

#     # ── Fix: ensure row is 2D for cosine_similarity ──────────────────
#     row = combined_matrix[idx]
#     if issparse(row):
#         row = row  # sparse row is already 2D (1, n_features)
#     else:
#         row = row.reshape(1, -1)

#     scores      = cosine_similarity(row, combined_matrix).flatten()
#     scores[idx] = 0  # exclude self

#     result                     = df.copy()
#     result["similarity_score"] = scores

#     result = result[result["category_name"] == category].copy()

#     result["relevance_score"] = (
#         0.6 * result["similarity_score"] +
#         0.4 * result["popularity_score"]
#     )

#     top = result.sort_values("relevance_score", ascending=False).head(top_n)
#     return _format_products(top)


# def recommend_for_user(viewed_asins: list[str], top_n: int = 10, artifacts: dict = None) -> list[dict]:
#     combined_matrix = artifacts["combined_matrix"]
#     asin_to_idx     = artifacts["asin_to_idx"]
#     df              = artifacts["df"]

#     valid_idxs = [asin_to_idx[a] for a in viewed_asins if a in asin_to_idx]

#     if not valid_idxs:
#         return _format_products(
#             df.sort_values("popularity_score", ascending=False).head(top_n)
#         )

#     # Average viewed vectors
#     user_vector = combined_matrix[valid_idxs].mean(axis=0)

#     # ── Fix: ensure user_vector is 2D dense for cosine_similarity ────
#     user_vector = np.asarray(user_vector).reshape(1, -1)

#     scores = cosine_similarity(user_vector, combined_matrix).flatten()

#     for idx in valid_idxs:
#         scores[idx] = 0
#     print("hello")
#     result                    = df.copy()
#     print("hello1.5")
#     print(len(scores))
#     print(len(df))
#     result["relevance_score"] = scores
#     print("hello2")

#     top = result.sort_values("relevance_score", ascending=False).head(top_n)
#     return _format_products(top)


# def _format_products(df_slice) -> list[dict]:
    # cols = [
    #     "asin", "title", "category_name", "stars", "price",
    #     "boughtInLastMonth", "isBestSeller",
    #     "popularity_score", "purchase_likelihood", "relevance_score","imgUrl", "productURL","reviews", 
    #     "listPrice"
    # ]
#     cols = [c for c in cols if c in df_slice.columns]
#     return df_slice[cols].fillna(0).to_dict(orient="records")