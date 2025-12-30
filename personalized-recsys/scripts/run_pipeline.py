import os
from joblib import dump

from src.config import Config
from src.data_load import load_movielens
from src.preprocess import filter_sparse_users, per_user_train_test_split
from src.recommenders import (
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeSVDRecommender,
    HybridRecommender
)
from src.evaluate import evaluate_model
from src.utils import ensure_dir, save_json, pretty_print_metrics

def main():
    cfg = Config()

    ensure_dir(cfg.model_dir)
    ensure_dir(cfg.reports_dir)

    print("1) Loading data...")
    ratings, movies = load_movielens(cfg.data_dir)

    print("2) Filtering sparse users...")
    ratings = filter_sparse_users(ratings, cfg.min_user_interactions)

    print("3) Train/test split (per user)...")
    train_df, test_df = per_user_train_test_split(ratings, cfg.test_ratio, cfg.random_state)

    print("4) Training recommenders...")
    pop = PopularityRecommender()
    cb = ContentBasedRecommender(max_features=cfg.tfidf_max_features)
    cf = CollaborativeSVDRecommender(n_components=cfg.svd_components)

    hybrid = HybridRecommender(
        cf=cf, cb=cb, pop=pop,
        cf_weight=cfg.cf_weight,
        content_weight=cfg.content_weight
    )
    hybrid.fit(train_df, movies)

    print("5) Evaluating hybrid recommender...")
    metrics = evaluate_model(hybrid, train_df, test_df, k=cfg.k)
    print("\n" + pretty_print_metrics(metrics))

    # Save artifacts
    print("6) Saving models + metrics...")
    dump(hybrid, os.path.join(cfg.model_dir, "hybrid_recommender.joblib"))
    save_json(os.path.join(cfg.reports_dir, "metrics.json"), metrics)

    print("\nDone ✅")
    print(f"Metrics saved to: {cfg.reports_dir}/metrics.json")
    print(f"Model saved to: {cfg.model_dir}/hybrid_recommender.joblib")

if __name__ == "__main__":
    main()
