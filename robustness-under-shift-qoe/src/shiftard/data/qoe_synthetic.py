import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_qoe_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    n_groups: int,
    shift_type: str,
    severity: float,
):
    """
    Synthetic streaming QoE-like binary task:
      y=1 => "bad QoE event" (e.g., rebuffer/low MOS bucket)

    Key detail:
      - Adds stable global indices (idx) so sample-wise IW weights are aligned correctly.
    """
    from shiftguard.shifts import apply_shift

    rng = np.random.default_rng(123)

    def sample_group(n):
        return rng.integers(0, n_groups, size=n)

    def gen_base(n, group):
        g = group.astype(float)

        throughput = rng.lognormal(mean=1.5 - 0.10 * (g % 3), sigma=0.35, size=n)
        rtt = rng.lognormal(mean=4.0 + 0.06 * (g // 3), sigma=0.18, size=n)
        rtt = np.clip(rtt, 10, 400)

        loss = rng.beta(a=1.5 + 0.1 * (g % 2), b=30.0 - 0.2 * (g % 3), size=n)
        buffer = rng.gamma(shape=2.0 + 0.2 * (g % 3), scale=2.0, size=n)
        bitrate = np.clip(0.6 * throughput + rng.normal(0, 0.8, size=n), 0.2, 25)
        dropped = rng.poisson(lam=0.5 + 0.15 * (g // 2), size=n)
        device_score = (g % 3) / 2.0
        tod = rng.uniform(0, 24, size=n)

        return np.column_stack([throughput, rtt, loss, buffer, bitrate, dropped, device_score, tod])

    def gen_y(X):
        throughput, rtt, loss, buffer, bitrate, dropped, device_score, tod = X.T
        peak = np.cos((tod - 20) / 24 * 2 * np.pi)

        logit = (
            -1.2
            - 0.35 * np.log1p(throughput)
            + 0.010 * rtt
            + 6.0 * loss
            - 0.22 * buffer
            + 0.15 * np.maximum(0, bitrate - throughput)
            + 0.25 * dropped
            - 0.8 * device_score
            + 0.25 * peak
        )
        p = _sigmoid(logit)
        return (rng.uniform(0, 1, size=len(p)) < p).astype(int)

    def make_split(n, env, start_idx):
        group = sample_group(n)
        X = gen_base(n, group)
        y = gen_y(X)

        Xs, ys = apply_shift(
            X=X, y=y, group=group, env=env,
            shift_type=shift_type, severity=severity, rng=rng
        )

        idx = np.arange(start_idx, start_idx + n, dtype=np.int64)

        return {
            "X": Xs.astype(np.float32),
            "y": ys.astype(int),
            "group": group.astype(int),
            "env": np.array([env] * n),
            "idx": idx,
        }

    train = make_split(n_train, "train", 0)
    val = make_split(n_val, "val", n_train)
    test = make_split(n_test, "test", n_train + n_val)

    # standardize using train stats only
    mu = train["X"].mean(axis=0, keepdims=True)
    sd = train["X"].std(axis=0, keepdims=True) + 1e-6
    for split in (train, val, test):
        split["X"] = (split["X"] - mu) / sd

    return {
        "train": train,
        "val": val,
        "test": test,
        "feature_names": [
            "throughput_mbps", "rtt_ms", "loss_rate", "buffer_s",
            "bitrate_mbps", "dropped_frames", "device_score", "time_of_day"
        ],
    }
