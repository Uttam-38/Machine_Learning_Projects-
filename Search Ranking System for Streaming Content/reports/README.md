# Evaluation Reports

This directory contains **offline evaluation outputs** used for analysis and model comparison.

## Typical Artifacts
- `metrics.json` – NDCG, MRR, Precision@K, Recall@K
- `ndcg_by_k.png` – ranking performance visualization

## How Reports Are Generated
```bash
python -m src.evaluate
python -m src.ab_simulator
