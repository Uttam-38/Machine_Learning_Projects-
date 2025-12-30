import pandas as pd
from joblib import load

from src.data_load import load_movielens
from src.preprocess import filter_sparse_users, per_user_train_test_split
from src.config import Config

def main():
    cfg = Config()
    ratings, movies = load_movielens(cfg.data_dir)
    ratings = filter_sparse_users(ratings, cfg.min_user_interactions)
    train_df, _ = per_user_train_test_split(
        ratings, cfg.test_ratio, cfg.random_state
    )

    model = load("models/hybrid_recommender.joblib")

    # Pick a sample user
    user_id = int(train_df["userId"].iloc[0])
    user_history = train_df[train_df["userId"] == user_id]["movieId"].tolist()
    seen = set(user_history)

    recs = model.recommend(
        user_id=user_id,
        user_history=user_history,
        seen=seen,
        k=10
    )

    print("\nUser ID:", user_id)

    print("\nRecently Watched:")
    print(
        movies[movies["movieId"].isin(user_history[-5:])][["title"]]
        .to_string(index=False)
    )

    print("\nTop-10 Recommendations:")
    print(
        movies[movies["movieId"].isin(recs)][["title"]]
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()

