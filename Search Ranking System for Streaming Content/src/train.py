import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit

from xgboost import XGBRanker

from src.config import ensure_dirs, TrainConfig
from src.utils import set_seed, save_json

def get_group_sizes(df: pd.DataFrame, group_col: str):
    return df.groupby(group_col).size().to_numpy()

def main():
    paths = ensure_dirs()
    cfg = TrainConfig()
    set_seed(cfg.seed)

    df = pd.read_parquet(paths.processed / "features.parquet")

    feature_cols = [c for c in df.columns if c not in ["session_id", "user_id", "query", "movie_id", cfg.label_col]]
    X = df[feature_cols].to_numpy()
    y = df[cfg.label_col].to_numpy()
    groups = df[cfg.group_col].to_numpy()

    # split by session_id (group-aware)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # validation split from train
    gss2 = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.seed)
    tr_idx, val_idx = next(gss2.split(
        df_train[feature_cols].to_numpy(),
        df_train[cfg.label_col].to_numpy(),
        groups=df_train[cfg.group_col].to_numpy()
    ))
    df_tr = df_train.iloc[tr_idx].copy()
    df_val = df_train.iloc[val_idx].copy()

    X_tr, y_tr = df_tr[feature_cols].to_numpy(), df_tr[cfg.label_col].to_numpy()
    X_val, y_val = df_val[feature_cols].to_numpy(), df_val[cfg.label_col].to_numpy()

    group_tr = get_group_sizes(df_tr, cfg.group_col)
    group_val = get_group_sizes(df_val, cfg.group_col)

    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        random_state=cfg.seed,
        tree_method="hist",
    )

    model.fit(
        X_tr,
        y_tr,
        group=group_tr,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        verbose=50
    )

    model_path = paths.models / "ranker.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    split_info = {
        "rows_train": int(len(df_tr)),
        "rows_val": int(len(df_val)),
        "rows_test": int(len(df_test)),
        "sessions_train": int(df_tr["session_id"].nunique()),
        "sessions_val": int(df_val["session_id"].nunique()),
        "sessions_test": int(df_test["session_id"].nunique()),
        "feature_cols": feature_cols,
    }
    save_json(paths.reports / "train_split.json", split_info)

    print(f"[OK] Saved model: {model_path}")
    print(f"[OK] Saved split info: {paths.reports / 'train_split.json'}")

if __name__ == "__main__":
    main()
