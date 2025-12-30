import numpy as np
import pandas as pd
from typing import Tuple

def filter_sparse_users(ratings: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    counts = ratings.groupby("userId")["movieId"].count()
    keep_users = counts[counts >= min_interactions].index
    return ratings[ratings["userId"].isin(keep_users)].copy()

def per_user_train_test_split(
    ratings: pd.DataFrame, test_ratio: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits per user so every user keeps some interactions in train.
    """
    rng = np.random.default_rng(random_state)

    train_parts = []
    test_parts = []

    for uid, grp in ratings.groupby("userId"):
        grp = grp.sample(frac=1.0, random_state=random_state)  # shuffle
        n = len(grp)
        n_test = max(1, int(n * test_ratio))
        test = grp.iloc[:n_test]
        train = grp.iloc[n_test:]
        # ensure train non-empty
        if len(train) == 0:
            train = grp.iloc[: n - 1]
            test = grp.iloc[n - 1 :]
        train_parts.append(train)
        test_parts.append(test)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df
