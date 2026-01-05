import argparse
import json
import time
from pathlib import Path

import yaml
import pandas as pd

from shiftguard.seed import set_seed
from shiftguard.data.qoe_synthetic import make_qoe_splits
from shiftguard.train import train_and_predict
from shiftguard.eval import evaluate_all, save_plots


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--method", type=str, required=True, choices=["erm", "groupdro", "coral", "iw"])
    ap.add_argument("--shift_type", type=str, default=None)
    ap.add_argument("--severity", type=float, default=None)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_yaml(cfg_path)

    set_seed(int(cfg["seed"]))

    # optional CLI overrides
    if args.shift_type is not None:
        cfg["data"]["shift"]["type"] = args.shift_type
    if args.severity is not None:
        cfg["data"]["shift"]["severity"] = float(args.severity)

    ts = time.strftime("%Y%m%d-%H%M%S")
    shift_type = cfg["data"]["shift"]["type"]
    severity = float(cfg["data"]["shift"]["severity"])

    out_root = project_root / cfg["outputs"]["out_dir"] / f"{ts}_{args.method}_{shift_type}_s{severity}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(exist_ok=True)

    # data
    splits = make_qoe_splits(
        n_train=cfg["data"]["n_train"],
        n_val=cfg["data"]["n_val"],
        n_test=cfg["data"]["n_test"],
        n_groups=cfg["data"]["n_groups"],
        shift_type=shift_type,
        severity=severity,
    )

    # train + predict
    preds = train_and_predict(cfg, splits, method=args.method)

    # evaluate
    metrics = evaluate_all(splits, preds)

    # save metrics + summary
    with open(out_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "method": args.method,
        "shift_type": shift_type,
        "severity": severity,
        "test_accuracy": metrics["test"]["accuracy"],
        "test_worst_group_accuracy": metrics["test"]["worst_group_accuracy"],
        "test_auroc": metrics["test"]["auroc"],
        "test_ece": metrics["test"]["ece"],
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # save predictions (test)
    df = pd.DataFrame({
        "idx": splits["test"]["idx"],
        "y_true": splits["test"]["y"],
        "y_prob": preds["test_prob"],
        "y_pred": (preds["test_prob"] >= 0.5).astype(int),
        "group": splits["test"]["group"],
        "env": splits["test"]["env"],
    })
    df.to_csv(out_root / "preds_test.csv", index=False)

    # plots
    save_plots(splits, preds, out_dir=str(out_root / "plots"))

    print("\n=== DONE ===")
    print("Output:", out_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
