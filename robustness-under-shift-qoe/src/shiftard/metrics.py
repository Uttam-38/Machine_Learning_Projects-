import numpy as np
from sklearn.metrics import roc_auc_score

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def auroc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    # handle degenerate case
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))

def ece(y_true, y_prob, n_bins=15):
    """
    Expected Calibration Error for binary classifier.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece_val += (mask.mean()) * abs(acc - conf)
    return float(ece_val)

def worst_group_accuracy(y_true, y_pred, group):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    accs = []
    for g in np.unique(group):
        m = group == g
        accs.append((y_true[m] == y_pred[m]).mean())
    return float(np.min(accs)) if accs else float("nan")
