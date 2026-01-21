# Search Ranking System for Streaming Content (Learning-to-Rank)

An end-to-end **search ranking (Learning-to-Rank)** pipeline for a streaming content catalog.  
This project simulates realistic search sessions, engineers relevance and personalization features, trains a **pairwise ranking model**, and evaluates ranking quality using industry-standard metrics such as **NDCG@K** and **MRR@K**.

The repository is designed to reflect **production-shaped machine learning workflows**, including reproducible data generation, feature pipelines, offline evaluation, and inference.

---

## Project Motivation

Search ranking is a critical surface in large-scale content platforms:
- High-quality ranking reduces **time-to-content**
- Better relevance improves **user satisfaction and engagement**
- Ranking systems must combine **text relevance**, **personalization**, and **content popularity**

This project demonstrates how such systems are built and evaluated in practice using a Learning-to-Rank approach.

---

## Key Highlights

### End-to-End Ranking Pipeline
- Data ingestion and preprocessing
- Search session simulation with graded relevance labels
- Feature engineering for relevance, personalization, and popularity
- Pairwise Learning-to-Rank model training
- Offline ranking evaluation
- Inference and baseline comparison

### Learning-to-Rank Model
- Pairwise ranking objective
- Query/session-aware grouping
- Optimized for ranking quality rather than pointwise prediction

### Evaluation Metrics
- **NDCG@K** – graded relevance ranking quality
- **MRR@K** – speed of retrieving the first relevant item
- **Precision@K / Recall@K** – top-K usefulness and coverage

---

## Tech Stack

- **Programming Language**: Python
- **Data Processing**: pandas, NumPy
- **Feature Engineering**: scikit-learn (TF-IDF, scaling)
- **Modeling**: XGBoost (pairwise Learning-to-Rank)
- **Evaluation & Visualization**: matplotlib

---

## Repository Structure

search-ranking-system/
├── README.md
├── requirements.txt
├── .gitignore
├── data/ # created locally, not committed
│ ├── raw/
│ └── processed/
├── models/ # trained model artifacts
├── reports/ # metrics and plots
└── src/
├── config.py
├── utils.py
├── download_data.py
├── make_dataset.py
├── features.py
├── train.py
├── evaluate.py
├── inference.py
└── ab_simulator.py

## Dataset

The project uses the **MovieLens 1M** dataset to simulate search behavior:
- User interactions are converted into search sessions
- Queries are generated from genres or title tokens
- Candidate items are retrieved and ranked
- Ordinal relevance labels (0–3) are assigned using ratings and popularity proxies

> Note: Behavioral signals such as clicks and dwell time are simulated for offline experimentation.

---

## How It Works (High-Level)

### 1. Data Ingestion
- Download and parse MovieLens user, item, and rating data

### 2. Search Session Simulation
- Generate search queries
- Retrieve candidate items
- Assign graded relevance labels
- Simulate user interaction signals with position bias

### 3. Feature Engineering
- Text relevance via **TF-IDF cosine similarity**
- Genre overlap and query intent detection
- User–genre affinity from implicit feedback
- Popularity, item age, and hybrid signals

### 4. Model Training
- Train a **pairwise Learning-to-Rank model**
- Group samples by search session
- Optimize ranking quality metrics

### 5. Evaluation
- Compute NDCG@K, MRR@K, Precision@K, Recall@K
- Generate metric plots for analysis

### 6. Inference
- Rank candidates for a new query using the trained model
- Return top-K ranked results

### 7. Baseline Comparison
- Compare Learning-to-Rank model against a popularity-based baseline
- Offline A/B-style evaluation

---

## Quickstart

### 1. Environment Setup

python -m venv .venv
source .venv/bin/activate     # Mac/Linux
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 2. Download Dataset
python -m src.download_data

# 3. Generate Sessions and Features
python -m src.make_dataset
python -m src.features

# 4. Train Ranking Model
python -m src.train

# 5. Evaluate Ranking Quality
python -m src.evaluate

# 6. Run Inference
python -m src.inference --user_id 42 --query "action"

# 7. Run Baseline Comparison
python -m src.ab_simulator
