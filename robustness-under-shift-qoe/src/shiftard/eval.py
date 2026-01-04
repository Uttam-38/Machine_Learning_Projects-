import os
import json
import numpy as np
import matplotlib.pyplot as plt

from shiftguard.metrics import accuracy, auroc, ece, worst_group_accuracy


def evaluate_all(splits, preds):
    out = {}

    for split_name in ["train", "val", "test"]:
        y = splits[split_name]["y"]
        p = preds[f"{split_name}_prob"]
        yp = (p >= 0.5).astype(int)
        g = splits[split_name]["group"]

        out[split_name] = {
            "accuracy": accuracy(y, yp),
            "auroc": auroc(y, p),
            "ece": ece(y, p),
            "worst_group_accuracy": worst_group_accuracy(y, yp, g),
            "positive_rate": float(np.mean(y)),
        }

    return out


def save_plots(splits, preds, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Reliability diagram (test)
    y = splits["test"]["y"]
    p = preds["test_prob"]

    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(p, bins) - 1
    bin_acc = []
    bin_conf = []
    for b in range(10):
        m = bin_ids == b
        if m.sum() == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
        else:
            bin_acc.append(y[m].mean())
            bin_conf.append(p[m].mean())

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.scatter(bin_conf, bin_acc)
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.title("Reliability (Test)")
    plt.savefig(os.path.join(out_dir, "reliability_test.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Group accuracy bars (test)
    g = splits["test"]["group"]
    yp = (p >= 0.5).astype(int)
    group_ids = np.unique(g)
    group_acc = [(y[g == gg] == yp[g == gg]).mean() for gg in group_ids]

    plt.figure()
    plt.bar(group_ids.astype(str), group_acc)
    plt.xlabel("group")
    plt.ylabel("accuracy")
    plt.title("Per-Group Accuracy (Test)")
    plt.savefig(os.path.join(out_dir, "group_acc_test.png"), dpi=160, bbox_inches="tight")
    plt.close()
