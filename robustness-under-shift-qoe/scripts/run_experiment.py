import argparse
import json
import os
import time
from pathlib import Path

import yaml
import pandas as pd

from shiftguard.seed import set_seed
from shiftguard.data.qoe_synthetic import make_qoe_splits
from shiftguard.train import train_and_predict
from shiftguard.eval import evaluate_all, save_plots


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--method", type=str, required=True,
                    choices=["erm", "groupdro", "coral", "iw"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(cfg["outputs"]["out_dir"]) / f"{ts}_{args.method}"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) data
    splits = make_qoe_splits(
        n_train=cfg["data"]["n_train"],
        n_val=cfg["data"]["n_val"],
        n_test=cfg["data"]["n_test"],
        n_groups=cfg["data"]["n_groups"],
        shift_type=cfg["data"]["shift"]["type"],
        severity=float(cfg["data"]["shift"]["severity"]),
    )

    # 2) train + predict
    preds = train_and_predict(cfg, splits, method=args.method)

    # 3) evaluate
    metrics = evaluate_all(splits, preds)
    (out_root / "plots").mkdir(exist_ok=True)

    with open(out_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # save predictions
    df = pd.DataFrame({
        "y_true": splits["test"]["y"],
        "y_prob": preds["test_prob"],
        "y_pred": (preds["test_prob"] >= 0.5).astype(int),
        "group": splits["test"]["group"],
        "env": splits["test"]["env"],
    })
    df.to_csv(out_root / "preds.csv", index=False)

    save_plots(splits, preds, out_dir=str(out_root / "plots"))

    print("\n=== DONE ===")
    print("Output:", out_root)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
