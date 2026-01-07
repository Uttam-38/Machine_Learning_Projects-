from __future__ import annotations
import argparse
import glob
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from streaming_quality.models.registry import load_bundle
from streaming_quality.utils import Paths, write_json

def latest_model(models_dir: str) -> str:
    paths = sorted(glob.glob(f"{models_dir}/model_*.joblib"))
    if not paths:
        raise FileNotFoundError("No trained model found in artifacts/models")
    return paths[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models_dir", default="artifacts/models")
    args = ap.parse_args()

    mp = latest_model(args.models_dir)
    bundle = load_bundle(mp)

    df = pd.read_parquet(args.data)
    X = df[bundle.feature_columns]
    yq = df["quality_label"].astype(str)

    pred = bundle.clf_quality.predict(X)
    rep = classification_report(yq, pred, output_dict=True)
    cm = confusion_matrix(yq, pred).tolist()

    out = {
        "model_path": mp,
        "version": bundle.version,
        "classification_report": rep,
        "confusion_matrix": cm
    }
    paths = Paths()
    write_json(f"{paths.reports}/eval_{bundle.version}.json", out)
    print(f"Wrote artifacts/reports/eval_{bundle.version}.json")

if __name__ == "__main__":
    main()
