import pandas as pd
from typing import Dict, List, Set
from tqdm import tqdm

from .metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, average_precision_at_k, mean_of
)

def build_user_history(train_df: pd.DataFrame) -> Dict[int, List[int]]:
    hist = train_df.groupby("userId")["movieId"].apply(list).to_dict()
    return {int(k): [int(x) for x in v] for k, v in hist.items()}

def build_user_seen(train_df: pd.DataFrame) -> Dict[int, Set[int]]:
    seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    return {int(k): set(int(x) for x in v) for k, v in seen.items()}

def build_user_ground_truth(test_df: pd.DataFrame) -> Dict[int, Set[int]]:
    gt = test_df.groupby("userId")["movieId"].apply(set).to_dict()
    return {int(k): set(int(x) for x in v) for k, v in gt.items()}

def evaluate_model(recommender, train_df: pd.DataFrame, test_df: pd.DataFrame, k: int) -> Dict[str, float]:
    user_history = build_user_history(train_df)
    user_seen = build_user_seen(train_df)
    user_gt = build_user_ground_truth(test_df)

    precs, recs, ndcgs, maps = [], [], [], []

    users = sorted(user_gt.keys())
    for uid in tqdm(users, desc="Evaluating users"):
        gt = user_gt.get(uid, set())
        seen = user_seen.get(uid, set())
        hist = user_history.get(uid, [])

        # recommender interface: recommend(user_id, user_history, seen, k) for hybrid
        # baseline models may ignore user_history
        try:
            rec_list = recommender.recommend(user_id=uid, user_history=hist, seen=seen, k=k)
        except TypeError:
            # fallback for simpler recommenders: recommend(user_id, seen, k)
            rec_list = recommender.recommend(user_id=uid, seen=seen, k=k)

        precs.append(precision_at_k(rec_list, gt, k))
        recs.append(recall_at_k(rec_list, gt, k))
        ndcgs.append(ndcg_at_k(rec_list, gt, k))
        maps.append(average_precision_at_k(rec_list, gt, k))

    return {
        f"Precision@{k}": mean_of(precs),
        f"Recall@{k}": mean_of(recs),
        f"NDCG@{k}": mean_of(ndcgs),
        f"MAP@{k}": mean_of(maps),
        "UsersEvaluated": float(len(users))
    }
