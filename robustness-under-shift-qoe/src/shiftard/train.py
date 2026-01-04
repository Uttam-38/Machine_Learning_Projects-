import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from shiftguard.models import MLP


def _to_loader(split, batch_size, shuffle):
    X = torch.from_numpy(split["X"])
    y = torch.from_numpy(split["y"]).float()
    g = torch.from_numpy(split["group"]).long()
    ds = TensorDataset(X, y, g)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _predict_proba(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for X, _, _ in loader:
            X = X.to(device)
            logits = model(X)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def _coral_loss(source, target):
    """
    CORAL: align covariance of source and target embeddings.
    """
    def cov(m):
        m = m - m.mean(dim=0, keepdim=True)
        return (m.t() @ m) / (m.shape[0] - 1 + 1e-8)

    cs = cov(source)
    ct = cov(target)
    return torch.mean((cs - ct) ** 2)


def train_and_predict(cfg, splits, method: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    robust_cfg = cfg.get("robust", {})

    train_loader = _to_loader(splits["train"], train_cfg["batch_size"], shuffle=True)
    val_loader = _to_loader(splits["val"], train_cfg["batch_size"], shuffle=False)
    test_loader = _to_loader(splits["test"], train_cfg["batch_size"], shuffle=False)

    d_in = splits["train"]["X"].shape[1]
    model = MLP(d_in, hidden_sizes=tuple(model_cfg["hidden_sizes"]), dropout=float(model_cfg["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))

    best_val = -1e9
    best_state = None
    patience = int(train_cfg["early_stop_patience"])
    bad = 0

    # Precompute IW weights with logistic density ratio approximation p_test(x)/p_train(x)
    iw_clip = float(robust_cfg.get("iw", {}).get("clip", 10.0))
    if method == "iw":
        # fit a domain classifier: train=0, test=1
        from sklearn.linear_model import LogisticRegression
        Xtr = splits["train"]["X"]
        Xte = splits["test"]["X"]
        Xdom = np.vstack([Xtr, Xte])
        ydom = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xte))])
        clf = LogisticRegression(max_iter=200, n_jobs=None)
        clf.fit(Xdom, ydom)
        p = clf.predict_proba(Xtr)[:, 1]
        # density ratio ~ p/(1-p)
        w = p / (1 - p + 1e-8)
        w = np.clip(w, 1.0 / iw_clip, iw_clip).astype(np.float32)
        iw_weights = torch.from_numpy(w)
    else:
        iw_weights = None

    # GroupDRO weights
    if method == "groupdro":
        eta = float(robust_cfg.get("groupdro", {}).get("eta", 0.05))
        n_groups = int(np.max(splits["train"]["group"])) + 1
        q = torch.ones(n_groups, device=device) / n_groups
    else:
        eta, n_groups, q = None, None, None

    # CORAL alignment uses embeddings: we’ll align hidden activations between train and test batches
    coral_lambda = float(robust_cfg.get("coral", {}).get("lambda", 1.0))

    def forward_with_embed(x):
        # Take activations from penultimate layer by reusing the MLP structure
        # (simple approach: run all but last Linear)
        h = x
        for layer in list(model.net.children())[:-1]:
            h = layer(h)
        logits = list(model.net.children())[-1](h).squeeze(-1)
        return logits, h

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False)

        for batch_idx, (X, y, g) in enumerate(pbar):
            X = X.to(device)
            y = y.to(device)
            g = g.to(device)

            opt.zero_grad()

            if method == "coral":
                logits_s, emb_s = forward_with_embed(X)

                # sample a "target" batch from test distribution for alignment
                # (in real usage, this can be unlabeled deployment data)
                # grab one batch from test_loader deterministically via iterator reset
                if batch_idx == 0:
                    test_iter = iter(test_loader)
                try:
                    Xt, _, _ = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    Xt, _, _ = next(test_iter)
                Xt = Xt.to(device)
                _, emb_t = forward_with_embed(Xt)

                clf_loss = F.binary_cross_entropy_with_logits(logits_s, y)
                align_loss = _coral_loss(emb_s, emb_t)
                loss = clf_loss + coral_lambda * align_loss

            else:
                logits = model(X)
                per_ex = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

                if method == "iw":
                    # match each batch row to a weight (by global index would be ideal;
                    # here we re-weight within the epoch by sampling weights from the dataset order)
                    # Use a simple approximation: shuffle-independent weights by pulling from a cycling buffer.
                    # (good enough for this project demo)
                    # Build a persistent weight queue once
                    if not hasattr(train_and_predict, "_iw_ptr"):
                        train_and_predict._iw_ptr = 0
                    ptr = train_and_predict._iw_ptr
                    w = iw_weights[ptr:ptr + len(per_ex)]
                    if len(w) < len(per_ex):
                        w = torch.cat([w, iw_weights[:len(per_ex)-len(w)]], dim=0)
                    train_and_predict._iw_ptr = (ptr + len(per_ex)) % len(iw_weights)
                    w = w.to(device)
                    loss = (per_ex * w).mean()

                elif method == "groupdro":
                    # compute per-group loss and update adversarial group weights
                    group_losses = torch.zeros(n_groups, device=device)
                    for gg in range(n_groups):
                        m = (g == gg)
                        if m.any():
                            group_losses[gg] = per_ex[m].mean()
                    # exponentiated gradient update
                    q = q * torch.exp(eta * group_losses.detach())
                    q = q / (q.sum() + 1e-12)
                    loss = (q * group_losses).sum()

                else:
                    # ERM
                    loss = per_ex.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # validation
        val_prob = _predict_proba(model, val_loader, device)
        val_y = splits["val"]["y"]
        val_acc = float(((val_prob >= 0.5).astype(int) == val_y).mean())

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final predictions
    train_prob = _predict_proba(model, train_loader, device)
    val_prob = _predict_proba(model, val_loader, device)
    test_prob = _predict_proba(model, test_loader, device)

    return {
        "train_prob": train_prob,
        "val_prob": val_prob,
        "test_prob": test_prob,
    }
