# Personalized Content Recommendation System (Hybrid)

This project implements a **Netflix-style personalized recommendation system** using a **hybrid approach** that combines **collaborative filtering** and **content-based filtering**, built as a **clean, reproducible, end-to-end machine learning pipeline**.

The system is designed with **software engineering best practices**, modular code structure, and offline evaluation using ranking metrics.

---

## Problem Statement

Personalized recommendations are a core component of modern streaming and content platforms.  
The goal of this project is to:

- Recommend relevant content to users based on historical interactions
- Handle cold-start users and items gracefully
- Evaluate recommendation quality using ranking-based metrics
- Build a production-style ML pipeline rather than a single notebook

---

## Approach Overview

### Collaborative Filtering
- Matrix Factorization using **Truncated SVD**
- Learns latent user and item representations from interaction data
- Captures implicit user preferences

### Content-Based Filtering
- TF-IDF vectorization on **movie titles + genres**
- Computes similarity between items
- Effective for cold-start scenarios

### Hybrid Recommendation
- Weighted combination of collaborative and content-based scores
- Popularity-based fallback for extreme cold-start users
- Produces robust and interpretable recommendations

---

## Evaluation Strategy

The system is evaluated using **offline ranking metrics** on the MovieLens 1M dataset:

- **Precision@K**
- **Recall@K**
- **NDCG@K**
- **MAP@K**

Train/test split is performed **per user**, ensuring each user retains interaction history in training data.

### Results (MovieLens 1M)

| Metric        | Value   |
|--------------|---------|
| Precision@10 | 0.1596  |
| Recall@10    | 0.0674  |
| NDCG@10      | 0.1936  |
| MAP@10       | 0.1072  |

---

## Project Structure

personalized-recsys/
├── requirements.txt
├── README.md
├── scripts/
│ ├── download_movielens.py
│ ├── run_pipeline.py
│ └── demo_recommendations.py
└── src/
├── config.py
├── data_load.py
├── preprocess.py
├── recommenders.py
├── metrics.py
├── evaluate.py
└── utils.py

**Design Principles**
- Modular, testable components
- Configuration-driven parameters
- Clear separation between data, modeling, and evaluation

---

## Dataset

This project uses the **MovieLens 1M dataset**.

- Source: https://grouplens.org/datasets/movielens/
- Contains ~1M user-movie ratings
- Dataset is **downloaded automatically** using a script

 Dataset files are **not committed** to the repository.

---

## Setup & Installation

### Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate
