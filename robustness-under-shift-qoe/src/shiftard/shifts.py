import numpy as np

def apply_shift(X, y, group, env, shift_type: str, severity: float, rng):
    """
    Implements common shifts:
      - covariate shift: X distribution changes
      - label shift: P(y) changes via thresholding (simulated)
      - concept shift: P(y|X) changes (drift in relationship)
    """
    if shift_type == "none" or env in ("train", "val"):
        return X, y

    Xs = X.copy()
    ys = y.copy()

    # Helpers
    def add_noise(cols, scale):
        Xs[:, cols] += rng.normal(0, scale, size=(len(Xs), len(cols)))

    if shift_type in ("region", "region_device"):
        # Region shift: throughput drops + RTT rises for some groups
        # severity controls magnitude
        g3 = (group % 3)
        thr_drop = (0.20 + 0.50 * severity) * (g3 == 0) + (0.10 + 0.25 * severity) * (g3 == 1)
        rtt_up = (0.10 + 0.60 * severity) * (g3 == 0) + (0.05 + 0.30 * severity) * (g3 == 2)

        # throughput col 0, rtt col 1
        Xs[:, 0] *= (1.0 - thr_drop)
        Xs[:, 1] *= (1.0 + rtt_up)
        add_noise([2], 0.01 * severity)  # loss noise

    if shift_type in ("device", "region_device"):
        # Device shift: buffer and dropped frames behave differently
        device_bucket = (group // 3)  # 0/1
        # buffer col 3, dropped col 5
        Xs[:, 3] *= (1.0 - (0.05 + 0.35 * severity) * (device_bucket == 1))
        Xs[:, 5] += rng.poisson(lam=(0.2 + 1.5 * severity) * (device_bucket == 1), size=len(Xs))
        add_noise([6], 0.10 * severity)  # device_score noise

    if shift_type == "concept":
        # Concept shift: relationship changes (e.g., ABR logic update).
        # Simulate by flipping labels with probability depending on rtt/loss.
        rtt = Xs[:, 1]
        loss = Xs[:, 2]
        # higher rtt/loss -> higher flip probability
        flip_p = np.clip(0.03 + severity * (0.05 * (rtt > np.median(rtt)) + 0.25 * (loss > np.median(loss))), 0, 0.35)
        flip = rng.uniform(0, 1, size=len(ys)) < flip_p
        ys = ys.copy()
        ys[flip] = 1 - ys[flip]

    return Xs, ys
