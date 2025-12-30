import numpy as np
from typing import List, Set

def precision_at_k(recs: List[int], ground_truth: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    recs_k = recs[:k]
    if not recs_k:
        return 0.0
    hits = sum(1 for r in recs_k if r in ground_truth)
    return hits / k

def recall_at_k(recs: List[int], ground_truth: Set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    recs_k = recs[:k]
    hits = sum(1 for r in recs_k if r in ground_truth)
    return hits / len(ground_truth)

def dcg_at_k(recs: List[int], ground_truth: Set[int], k: int) -> float:
    recs_k = recs[:k]
    dcg = 0.0
    for i, item in enumerate(recs_k, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 1)
    return dcg

def ndcg_at_k(recs: List[int], ground_truth: Set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    dcg = dcg_at_k(recs, ground_truth, k)
    ideal_hits = min(k, len(ground_truth))
    ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def average_precision_at_k(recs: List[int], ground_truth: Set[int], k: int) -> float:
    recs_k = recs[:k]
    if not ground_truth or not recs_k:
        return 0.0
    hit_count = 0
    ap_sum = 0.0
    for i, item in enumerate(recs_k, start=1):
        if item in ground_truth:
            hit_count += 1
            ap_sum += hit_count / i
    return ap_sum / min(len(ground_truth), k)

def mean_of(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0
