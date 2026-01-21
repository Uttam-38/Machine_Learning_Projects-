# Machine Learning Projects

A curated portfolio of **end-to-end machine learning systems** spanning **ranking, recommendation, robustness under distribution shift, and quality-of-experience prediction**.  
Each project is structured to demonstrate strong ML fundamentals, evaluation rigor, and production-shaped engineering practices (reproducible pipelines, modular code, artifact outputs).

---

## What’s in this repository

This repository currently contains the following projects:

| Project | Domain | Core ML Task | Key Metrics |
|--------|--------|--------------|-------------|
| **Search Ranking System for Streaming Content** | Search & Discovery | Learning-to-Rank | NDCG@K, MRR@K, Precision/Recall@K |
| **personalized-recsys** | Personalization | Recommendation & Ranking | MAP@K, NDCG@K, Recall@K (project-specific) |
| **robustness-under-shift-qoe** | Reliability | Robustness under Distribution Shift | Calibration, performance under shift, error analysis |
| **streaming-quality-prediction** | Media / QoE | Prediction & Monitoring | RMSE/MAE (or classification metrics), trend/anomaly analysis |

> Repo folders: `Search Ranking System for Streaming Content/`, `personalized-recsys/`, `robustness-under-shift-qoe/`, `streaming-quality-prediction/`.  

---

## Technical Focus Areas

- **Learning-to-Rank (LTR)**: query/session grouping, pairwise objectives, offline ranking metrics
- **Recommender Systems**: personalization signals, candidate ranking, top-K evaluation
- **Robust ML**: distribution shift evaluation, stability analysis, diagnostic reporting
- **Quality-of-Experience Modeling**: feature engineering, prediction, monitoring-style evaluation
- **Engineering Practices**: reproducible runs, deterministic seeds, modular pipeline stages, artifact outputs

---

## Tech Stack (common)

- **Language**: Python
- **Core Libraries**: NumPy, pandas, scikit-learn
- **Modeling**: XGBoost / tree models / classical ML (project-dependent)
- **Evaluation**: ranking metrics (NDCG/MRR/MAP), regression/classification metrics
- **Artifacts**: parquet/CSV outputs, serialized models, metric JSON reports

---

## Projects

### 1) Search Ranking System for Streaming Content
**Goal**: Build a learning-to-rank pipeline that ranks content results for a given query/session using a multi-stage workflow:
**data → sessions → features → ranker training → offline evaluation → inference.**

**What it demonstrates**
- Query/session-aware ranking (grouped training)
- Feature engineering spanning relevance + personalization + popularity signals
- Pairwise ranking model training and evaluation with ranking metrics

**Typical outputs**
- Sessionized training data (e.g., `sessions.parquet`)
- Feature table (e.g., `features.parquet`)
- Model artifact (e.g., `ranker.joblib`)
- Offline metrics report (e.g., `metrics.json`)

**Metrics**
- NDCG@K, MRR@K, Precision@K, Recall@K

---

### 2) personalized-recsys
**Goal**: Build a personalized recommendation workflow that ranks items per user using interaction signals and content metadata.

**What it demonstrates**
- Personalization signal design (user affinity / item popularity / content similarity)
- Top-K evaluation that mirrors recommendation workloads
- Modular separation of retrieval vs ranking (if implemented in the project)

**Metrics (examples)**
- NDCG@K, MAP@K, Recall@K, HitRate@K (based on the project implementation)

---

### 3) robustness-under-shift-qoe
**Goal**: Study model performance under **distribution shift** and build diagnostics that explain failure modes.

**What it demonstrates**
- Shift-aware evaluation (train vs test mismatch)
- Robustness analysis (performance stratified by segments)
- Error analysis and metric reporting for reliability-focused ML

**Metrics (examples)**
- Performance deltas under shift (e.g., ΔAUC/ΔRMSE), calibration error, slice-based analysis

---

### 4) streaming-quality-prediction
**Goal**: Predict streaming quality/QoE signals using engineered features and evaluate model quality for monitoring use-cases.

**What it demonstrates**
- Feature engineering for operational prediction tasks
- Regression/classification modeling (depending on labels)
- Reporting that aligns with monitoring/decision-support workflows

**Metrics (examples)**
- RMSE/MAE for regression; Precision/Recall/F1/AUC for classification (based on the project implementation)

---

## Getting Started

### 1) Clone the repository

git clone https://github.com/Uttam-38/Machine_Learning_Projects-.git
cd Machine_Learning_Projects-

### 2) Create an environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

### 3) Install dependencies
pip install -U pip
pip install numpy pandas scikit-learn matplotlib xgboost
