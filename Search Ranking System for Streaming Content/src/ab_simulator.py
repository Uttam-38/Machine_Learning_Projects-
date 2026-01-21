import numpy as np
import pandas as pd
import joblib

from src.config import ensure_dirs, TrainConfig
from src.utils import set_seed

def ndcg_at_k(y_true, y_score, k):
    order = np.argsort(-y_score)
    yt = np.asarray(y_true)[order][:k]
    if yt.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, yt.size + 2))
    dcg = float(np.sum((2 ** yt - 1) * discounts))
    ideal = np.sort(y_true)[::-1][:k]
    ideal_dcg = float(np.sum((2 ** ideal - 1) * discounts))
    return 0.0 if ideal_dcg == 0 else dcg / ideal_dcg

def main():
    paths = ensure_dirs()
    cfg = TrainConfig()
    set_seed(cfg.seed)

    df = pd.read_parquet(paths.processed / "features.parquet")

    bundle = joblib.load(paths.models / "ranker.joblib")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # scores
    df["score_ltr"] = model.predict(df[feature_cols].to_numpy())

    # baseline: use popularity proxy (exists in features but scaled); approximate using one column
    # If popularity is scaled, it still works for relative ranking within a session.
    if "popularity" in df.columns:
        df["score_base"] = df["popularity"]
    else:
        df["score_base"] = 0.0

    ks = [5, 10]
    for k in ks:
        ndcg_ltr = []
        ndcg_base = []

        for sid, g in df.groupby("session_id"):
            y = g["relevance"].to_numpy()
            ndcg_ltr.append(ndcg_at_k(y, g["score_ltr"].to_numpy(), k))
            ndcg_base.append(ndcg_at_k(y, g["score_base"].to_numpy(), k))

        print(f"\nNDCG@{k}")
        print(f"  Baseline (popularity): {np.mean(ndcg_base):.4f}")
        print(f"  LTR (XGBoost):         {np.mean(ndcg_ltr):.4f}")
        print(f"  Delta:                {(np.mean(ndcg_ltr) - np.mean(ndcg_base)):.4f}")

if __name__ == "__main__":
    main()
