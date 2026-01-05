import os
import numpy as np

from shiftguard.shifts import apply_shift


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
    Synthetic QoE-like binary task:
      y=1 means "bad QoE event" (e.g., rebuffering / low MOS bucket)

    Features (examples):
      - throughput_mbps
      - rtt_ms
      - loss_rate
      - buffer_s
      - bitrate_mbps
      - dropped_frames
      - device_score (TV=stable, mobile=noisy)
      - time_of_day (proxy)
    Groups: represent {region, device_class, isp bucket...}
    Envs: "train"/"val"/"test" with controlled shifts
    """

    # Allow sweep overrides without rewriting config
    shift_type = os.getenv("SHIFT_TYPE", shift_type)
    severity = float(os.getenv("SHIFT_SEVERITY", severity))

    rng = np.random.default_rng(123)

    # group assignment
    def sample_group(n):
        return rng.integers(0, n_groups, size=n)

    # base features generator (train distribution)
    def gen_base(n, group):
        # group-specific offsets (like region/device)
        g = group.astype(float)

        throughput = rng.lognormal(mean=1.5 - 0.10 * (g % 3), sigma=0.35, size=n)  # Mbps
        rtt = rng.lognormal(mean=4.0 + 0.06 * (g // 3), sigma=0.18, size=n)        # ms-ish (lognormal)
        rtt = np.clip(rtt, 10, 400)

        loss = rng.beta(a=1.5 + 0.1 * (g % 2), b=30.0 - 0.2 * (g % 3), size=n)     # small %
        buffer = rng.gamma(shape=2.0 + 0.2 * (g % 3), scale=2.0, size=n)           # seconds
        bitrate = np.clip(0.6 * throughput + rng.normal(0, 0.8, size=n), 0.2, 25)  # Mbps
        dropped = rng.poisson(lam=0.5 + 0.15 * (g // 2), size=n)                   # frames
        device_score = (g % 3) / 2.0                                               # 0..1
        tod = rng.uniform(0, 24, size=n)                                           # hour

        X = np.column_stack([throughput, rtt, loss, buffer, bitrate, dropped, device_score, tod])
        return X

    # label generator (train concept)
    def gen_y(X):
        throughput, rtt, loss, buffer, bitrate, dropped, device_score, tod = X.T

        # "bad QoE" rises with high rtt/loss, low buffer, bitrate near throughput cap, dropped frames
        # device_score: higher => more stable (reduces bad events)
        # time of day: mild congestion peak
        peak = np.cos((tod - 20) / 24 * 2 * np.pi)  # evening-ish
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
        y = (rng.uniform(0, 1, size=len(p)) < p).astype(int)
        return y

    def make_split(n, env):
        group = sample_group(n)
        X = gen_base(n, group)
        y = gen_y(X)

        Xs, ys = apply_shift(
            X=X, y=y, group=group, env=env,
            shift_type=shift_type, severity=severity, rng=rng
        )
        return {"X": Xs.astype(np.float32), "y": ys.astype(int), "group": group.astype(int), "env": np.array([env]*n)}

    train = make_split(n_train, "train")
    val = make_split(n_val, "val")
    test = make_split(n_test, "test")

    # standardize using train stats only
    mu = train["X"].mean(axis=0, keepdims=True)
    sd = train["X"].std(axis=0, keepdims=True) + 1e-6
    for split in [train, val, test]:
        split["X"] = (split["X"] - mu) / sd

    return {"train": train, "val": val, "test": test, "feature_names": [
        "throughput_mbps", "rtt_ms", "loss_rate", "buffer_s",
        "bitrate_mbps", "dropped_frames", "device_score", "time_of_day"
    ]}
