import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

from src.config import ensure_dirs, TrainConfig
from src.utils import set_seed, save_json

def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2 ** rels - 1) * discounts))

def ndcg_at_k(y_true, y_score, k):
    order = np.argsort(-y_score)
    dcg = dcg_at_k(y_true[order], k)
    ideal = dcg_at_k(np.sort(y_true)[::-1], k)
    return 0.0 if ideal == 0 else dcg / ideal

def mrr_at_k(y_true, y_score, k, positive_threshold=2):
    order = np.argsort(-y_score)[:k]
    rels = y_true[order]
    hits = np.where(rels >= positive_threshold)[0]
    if hits.size == 0:
        return 0.0
    return float(1.0 / (hits[0] + 1))

def precision_recall_at_k(y_true, y_score, k, positive_threshold=2):
    order = np.argsort(-y_score)[:k]
    rels = y_true[order]
    relevant = (y_true >= positive_threshold).sum()
    retrieved_relevant = (rels >= positive_threshold).sum()
    precision = retrieved_relevant / k
    recall = 0.0 if relevant == 0 else retrieved_relevant / relevant
    return float(precision), float(recall)

def main():
    paths = ensure_dirs()
    cfg = TrainConfig()
    set_seed(cfg.seed)

    bundle = joblib.load(paths.models / "ranker.joblib")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_parquet(paths.processed / "features.parquet")
    X = df[feature_cols].to_numpy()
    y = df[cfg.label_col].to_numpy()
    groups = df[cfg.group_col].to_numpy()

    # reproduce a test split (group-aware)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    _, test_idx = next(gss.split(X, y, groups=groups))
    df_test = df.iloc[test_idx].copy()

    # predict scores
    df_test["score"] = model.predict(df_test[feature_cols].to_numpy())

    ks = [3, 5, 10, 20]
    ndcgs, mrrs, precs, recs = {}, {}, {}, {}

    for k in ks:
        per_sess = []
        per_sess_mrr = []
        per_sess_p = []
        per_sess_r = []

        for sid, g in df_test.groupby("session_id"):
            yt = g["relevance"].to_numpy()
            ys = g["score"].to_numpy()
            per_sess.append(ndcg_at_k(yt, ys, k))
            per_sess_mrr.append(mrr_at_k(yt, ys, k))
            p, r = precision_recall_at_k(yt, ys, k)
            per_sess_p.append(p)
            per_sess_r.append(r)

        ndcgs[f"ndcg@{k}"] = float(np.mean(per_sess))
        mrrs[f"mrr@{k}"] = float(np.mean(per_sess_mrr))
        precs[f"precision@{k}"] = float(np.mean(per_sess_p))
        recs[f"recall@{k}"] = float(np.mean(per_sess_r))

    metrics = {}
    metrics.update(ndcgs)
    metrics.update(mrrs)
    metrics.update(precs)
    metrics.update(recs)

    save_json(paths.reports / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))

    # Plot NDCG by k
    plt.figure()
    plt.plot(ks, [ndcgs[f"ndcg@{k}"] for k in ks], marker="o")
    plt.xlabel("K")
    plt.ylabel("NDCG@K")
    plt.title("NDCG by cutoff K")
    out = paths.reports / "ndcg_by_k.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"[OK] Saved plot: {out}")

if __name__ == "__main__":
    main()
