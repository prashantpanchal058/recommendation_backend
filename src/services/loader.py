import os
import pickle
import joblib
import numpy as np
import faiss
import pandas as pd
from sklearn.decomposition import TruncatedSVD

_artifacts = {}

HF_REPO_ID = os.getenv("HF_REPO_ID", "prashantpanchal058/recommendation-models")


def download_artifacts(save_dir: str):
    """Download model files from HuggingFace Hub if not present locally."""
    from huggingface_hub import snapshot_download

    required_files = [
        "combined_matrix.joblib",
        "lgbm_model.joblib",
        "tfidf_recommender.joblib",
        "num_scaler.joblib",
        "products_df.parquet",
        "asin_to_idx.pkl",
    ]

    all_exist = all(os.path.exists(f"{save_dir}/{f}") for f in required_files)
    if all_exist:
        print("✅ Model files already present, skipping download")
        return

    hf_token = os.getenv("HF_TOKEN")
    print(f"⬇️  Downloading model artifacts from HuggingFace: {HF_REPO_ID} ...")
    os.makedirs(save_dir, exist_ok=True)

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=save_dir,
        token=hf_token,
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print("✅ Download complete")

    # Verify all files exist after download
    for f in required_files:
        path = f"{save_dir}/{f}"
        if not os.path.exists(path):
            raise RuntimeError(f"❌ File missing after download: {path}")
    print("✅ All required files verified")


def load_artifacts(save_dir: str = "src/model_artifacts"):
    global _artifacts

    if _artifacts:
        return _artifacts

    # ── Step 1: Download from HuggingFace if needed ─────────────────
    download_artifacts(save_dir)

    # ── Step 2: Load base artifacts ──────────────────────────────────
    print("⏳ Loading ML artifacts...")

    combined_matrix               = joblib.load(f"{save_dir}/combined_matrix.joblib")
    _artifacts["lgbm_model"]      = joblib.load(f"{save_dir}/lgbm_model.joblib")
    _artifacts["tfidf"]           = joblib.load(f"{save_dir}/tfidf_recommender.joblib")
    _artifacts["num_scaler"]      = joblib.load(f"{save_dir}/num_scaler.joblib")
    _artifacts["df"]              = pd.read_parquet(f"{save_dir}/products_df.parquet")
    _artifacts["combined_matrix"] = combined_matrix

    with open(f"{save_dir}/asin_to_idx.pkl", "rb") as f:
        _artifacts["asin_to_idx"] = pickle.load(f)

    print("✅ Base artifacts loaded")

    # ── Step 3: SVD — load cache or compute & save ───────────────────
    svd_cache_path    = f"{save_dir}/svd_model.joblib"
    matrix_cache_path = f"{save_dir}/matrix_dense.npy"

    if os.path.exists(svd_cache_path) and os.path.exists(matrix_cache_path):
        print("⚡ Loading cached SVD + dense matrix...")
        svd          = joblib.load(svd_cache_path)
        matrix_dense = np.load(matrix_cache_path)
        print(f"✅ Cache loaded: {matrix_dense.shape}")
    else:
        print("⏳ Compressing matrix with SVD (first run, may take a few minutes)...")
        svd          = TruncatedSVD(n_components=256, random_state=42)
        matrix_dense = svd.fit_transform(combined_matrix).astype("float32")
        print(f"✅ Compressed: {combined_matrix.shape} → {matrix_dense.shape}")

        print("💾 Saving SVD cache for next startup...")
        joblib.dump(svd, svd_cache_path)
        np.save(matrix_cache_path, matrix_dense)
        print("✅ SVD cache saved")

    # ── Step 4: Build FAISS index ────────────────────────────────────
    print("⏳ Building FAISS index...")
    faiss.normalize_L2(matrix_dense)

    n_features = matrix_dense.shape[1]
    index      = faiss.IndexFlatIP(n_features)
    index.add(matrix_dense)

    _artifacts["faiss_index"]  = index
    _artifacts["matrix_dense"] = matrix_dense
    _artifacts["svd"]          = svd

    print(f"✅ FAISS ready — {index.ntotal:,} vectors @ {n_features} dims")
    print(f"✅ {len(_artifacts['df']):,} products loaded")

    return _artifacts


def get_artifacts():
    if not _artifacts:
        load_artifacts()
    return _artifacts
