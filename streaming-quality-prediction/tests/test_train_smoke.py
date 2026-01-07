import pandas as pd
from streaming_quality.data.generate_synthetic import make
from streaming_quality.features.build_features import add_session_features, add_time_features

def test_smoke_pipeline():
    df = make(5000, seed=1)
    df = add_session_features(df)
    df = add_time_features(df)
    assert "rebuffer_ratio" in df.columns
    assert df["startup_time_ms"].notna().all()
