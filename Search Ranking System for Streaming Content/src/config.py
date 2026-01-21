from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data: Path = root / "data"
    raw: Path = data / "raw"
    processed: Path = data / "processed"
    models: Path = root / "models"
    reports: Path = root / "reports"

@dataclass(frozen=True)
class DataConfig:
    movielens_url: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_name: str = "ml-1m.zip"
    folder_name: str = "ml-1m"

@dataclass(frozen=True)
class SessionConfig:
    seed: int = 42
    sessions_per_user: int = 8
    candidates_per_session: int = 30
    min_candidates: int = 10
    max_query_terms_per_item: int = 4
    click_position_bias: float = 1.25  # higher => stronger position bias
    noise: float = 0.15  # randomness for behavior simulation

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    group_col: str = "session_id"
    label_col: str = "relevance"
    # XGBRanker params (safe defaults)
    n_estimators: int = 600
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0

def ensure_dirs():
    p = Paths()
    p.raw.mkdir(parents=True, exist_ok=True)
    p.processed.mkdir(parents=True, exist_ok=True)
    p.models.mkdir(parents=True, exist_ok=True)
    p.reports.mkdir(parents=True, exist_ok=True)
    return p
