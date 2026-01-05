from shiftguard.data.qoe_synthetic import make_qoe_splits
from shiftguard.train import train_and_predict
from shiftguard.eval import evaluate_all


def _cfg():
    return {
        "seed": 1,
        "model": {"hidden_sizes": [16, 16], "dropout": 0.0},
        "train": {"epochs": 2, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0, "early_stop_patience": 1},
        "robust": {"groupdro": {"eta": 0.05}, "coral": {"lambda": 0.5}, "iw": {"clip": 10.0}},
        "outputs": {"out_dir": "runs"},
        "data": {"shift": {"type": "region_device", "severity": 0.7}},
    }


def test_smoke_erm():
    cfg = _cfg()
    splits = make_qoe_splits(2000, 500, 1000, 4, cfg["data"]["shift"]["type"], cfg["data"]["shift"]["severity"])
    preds = train_and_predict(cfg, splits, method="erm")
    m = evaluate_all(splits, preds)
    assert "test" in m and 0.0 <= m["test"]["accuracy"] <= 1.0


def test_smoke_iw():
    cfg = _cfg()
    splits = make_qoe_splits(2000, 500, 1000, 4, cfg["data"]["shift"]["type"], cfg["data"]["shift"]["severity"])
    preds = train_and_predict(cfg, splits, method="iw")
    m = evaluate_all(splits, preds)
    assert "test" in m


def test_smoke_groupdro():
    cfg = _cfg()
    splits = make_qoe_splits(2000, 500, 1000, 4, cfg["data"]["shift"]["type"], cfg["data"]["shift"]["severity"])
    preds = train_and_predict(cfg, splits, method="groupdro")
    m = evaluate_all(splits, preds)
    assert "test" in m
