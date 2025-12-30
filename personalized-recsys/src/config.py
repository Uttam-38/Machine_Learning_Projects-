from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Data
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    # Split
    test_ratio: float = 0.2
    min_user_interactions: int = 10  # filter very sparse users
    random_state: int = 42

    # Collaborative Filtering (SVD)
    svd_components: int = 50
    svd_reg: float = 0.0  # keep 0; regularization done implicitly via truncation

    # Content-based
    tfidf_max_features: int = 5000
    content_weight: float = 0.35
    cf_weight: float = 0.65

    # Evaluation
    k: int = 10

    # Output
    model_dir: str = "models"
    reports_dir: str = "reports"
