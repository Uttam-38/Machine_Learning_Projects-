from __future__ import annotations
import numpy as np
import pandas as pd
from streaming_quality.models.registry import ModelBundle

def predict_one(bundle: ModelBundle, row: dict) -> dict:
    df = pd.DataFrame([row])
    X = df[bundle.feature_columns].copy()

    rebuffer = float(bundle.reg_rebuffer.predict(X)[0])
    startup = float(bundle.reg_startup.predict(X)[0])
    quality = str(bundle.clf_quality.predict(X)[0])

    # Optional: return probabilities
    proba = bundle.clf_quality.predict_proba(X)[0]
    classes = list(bundle.clf_quality.classes_)
    prob_map = {c: float(p) for c, p in zip(classes, proba)}

    return {
        "rebuffer_ratio_pred": max(0.0, rebuffer),
        "startup_time_ms_pred": max(0.0, startup),
        "quality_label_pred": quality,
        "quality_proba": prob_map,
    }
