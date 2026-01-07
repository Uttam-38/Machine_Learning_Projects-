from __future__ import annotations
import numpy as np

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    # Population Stability Index
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf

    e_hist, _ = np.histogram(expected, bins=qs)
    a_hist, _ = np.histogram(actual, bins=qs)

    e = e_hist / (e_hist.sum() + eps)
    a = a_hist / (a_hist.sum() + eps)

    val = np.sum((a - e) * np.log((a + eps) / (e + eps)))
    return float(val)

def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    # Simple W1 distance for 1D
    x = np.sort(x)
    y = np.sort(y)
    n = min(len(x), len(y))
    return float(np.mean(np.abs(x[:n] - y[:n])))
