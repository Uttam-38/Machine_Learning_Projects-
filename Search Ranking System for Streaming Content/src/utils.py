import json
from pathlib import Path
import numpy as np
import pandas as pd

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def standardize_columns(df: pd.DataFrame):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def safe_div(a, b, eps=1e-9):
    return a / (b + eps)
