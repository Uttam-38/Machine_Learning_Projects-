from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import time

@dataclass
class ModelBundle:
    version: str
    preprocessor: object
    reg_rebuffer: object
    reg_startup: object
    clf_quality: object
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]

def save_bundle(bundle: ModelBundle, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

def load_bundle(path: str) -> ModelBundle:
    return joblib.load(path)

def new_version() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
