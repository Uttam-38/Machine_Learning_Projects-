from shiftguard.data.qoe_synthetic import make_qoe_splits
from shiftguard.train import train_and_predict
from shiftguard.eval import evaluate_all

def test_smoke_run():
    cfg = {
        "seed": 1,
        "data": {"shift": {"type": "region_device", "severity": 0.7}},
        "model": {"hidden_sizes": [16, 16], "dropout": 0.0},
        "train": {"epochs": 2, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0, "early_stop_patience": 1},
        "robust": {"groupdro": {"eta": 0.05}, "coral": {"lambda": 0.5}, "iw": {"clip": 10.0}},
        "outputs": {"out_dir": "runs"},
    }

    splits = make_qoe_splits(
        n_train=2000, n_val=500, n_test=1000, n_groups=4,
        shift_type=cfg["data"]["shift"]["type"], severity=cfg["data"]["shift"]["severity"]
    )
    preds = train_and_predict(cfg, splits, method="erm")
    metrics = evaluate_all(splits, preds)
    assert "test" in metrics
    assert 0.0 <= metrics["test"]["accuracy"] <= 1.0
