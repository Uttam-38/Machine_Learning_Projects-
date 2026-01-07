from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
from xgboost import XGBRegressor, XGBClassifier

from streaming_quality.constants import (
    TARGETS, CATEGORICAL_COLS, NUMERIC_COLS, ID_COLS
)
from streaming_quality.models.registry import ModelBundle, save_bundle, new_version
from streaming_quality.utils import load_config, Paths, ensure_dirs, write_json

def split_groups(df: pd.DataFrame, group_col: str = "user_id", test_size: float = 0.2, seed: int = 42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train, idx_test = next(gss.split(df, groups=df[group_col]))
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = Paths()
    ensure_dirs(paths)

    df = pd.read_parquet(args.data)

    # Drop ID columns from X, but keep for grouping
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS + ["hour", "dow", "is_weekend", "log_throughput", "log_rtt", "loss_x_rtt",
                                                     "demand_mbps", "demand_over_throughput", "is_4k", "is_tv", "is_mobile", "codec_efficiency"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    required = set(CATEGORICAL_COLS + NUMERIC_COLS + TARGETS + ID_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    train_df, test_df = split_groups(df, "user_id", cfg["split"]["test_size"], cfg["split"]["random_state"])

    # Create a small val split from train groups
    train_df2, val_df = split_groups(train_df, "user_id", cfg["split"]["val_size"], cfg["split"]["random_state"])

    X_train = train_df2[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train_reb = train_df2["rebuffer_ratio"].astype(float)
    y_val_reb = val_df["rebuffer_ratio"].astype(float)
    y_test_reb = test_df["rebuffer_ratio"].astype(float)

    y_train_start = train_df2["startup_time_ms"].astype(float)
    y_val_start = val_df["startup_time_ms"].astype(float)
    y_test_start = test_df["startup_time_ms"].astype(float)

    y_train_q = train_df2["quality_label"].astype(str)
    y_val_q = val_df["quality_label"].astype(str)
    y_test_q = test_df["quality_label"].astype(str)

    pre = build_preprocessor()

    reg_params = cfg["model"]["reg_params"]
    clf_params = cfg["model"]["clf_params"]

    reg_rebuffer = XGBRegressor(**reg_params, objective="reg:squarederror", n_jobs=-1)
    reg_startup = XGBRegressor(**reg_params, objective="reg:squarederror", n_jobs=-1)
    clf_quality = XGBClassifier(**clf_params, objective="multi:softprob", num_class=3, n_jobs=-1)

    pipe_reb = Pipeline([("pre", pre), ("m", reg_rebuffer)])
    pipe_start = Pipeline([("pre", pre), ("m", reg_startup)])
    pipe_q = Pipeline([("pre", pre), ("m", clf_quality)])

    pipe_reb.fit(X_train, y_train_reb, m__eval_set=[(X_val, y_val_reb)], m__verbose=False)
    pipe_start.fit(X_train, y_train_start, m__eval_set=[(X_val, y_val_start)], m__verbose=False)
    pipe_q.fit(X_train, y_train_q, m__eval_set=[(X_val, y_val_q)], m__verbose=False)

    # Evaluate quick
    pred_reb = pipe_reb.predict(X_test)
    pred_start = pipe_start.predict(X_test)
    pred_q = pipe_q.predict(X_test)

    metrics = {
        "rebuffer_mae": float(mean_absolute_error(y_test_reb, pred_reb)),
        "rebuffer_rmse": float(mean_squared_error(y_test_reb, pred_reb, squared=False)),
        "startup_mae": float(mean_absolute_error(y_test_start, pred_start)),
        "startup_rmse": float(mean_squared_error(y_test_start, pred_start, squared=False)),
        "quality_f1_macro": float(f1_score(y_test_q, pred_q, average="macro")),
        "n_train": int(len(train_df2)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }

    version = new_version()
    bundle = ModelBundle(
        version=version,
        preprocessor=pre,               # stored separately inside pipelines too, but kept for clarity
        reg_rebuffer=pipe_reb,
        reg_startup=pipe_start,
        clf_quality=pipe_q,
        feature_columns=feature_cols,
        categorical_columns=[c for c in CATEGORICAL_COLS if c in feature_cols],
        numeric_columns=[c for c in NUMERIC_COLS if c in feature_cols],
    )

    model_path = f"{paths.models}/model_{version}.joblib"
    save_bundle(bundle, model_path)

    write_json(f"{paths.reports}/train_metrics_{version}.json", metrics)

    print("Saved:", model_path)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
