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
    idx = torch.from_numpy(split["idx"]).long()
    ds = TensorDataset(X, y, g, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def _predict_proba(model, loader, device):
    model.eval()
    probs = []
    for X, _, _, _ in loader:
        X = X.to(device)
        logits = model(X)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def _coral_loss(source, target):
    """CORAL: ||Cov(S) - Cov(T)||_F^2"""
    def cov(m):
        m = m - m.mean(dim=0, keepdim=True)
        return (m.t() @ m) / (m.shape[0] - 1 + 1e-8)
    return torch.mean((cov(source) - cov(target)) ** 2)


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
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    # ---- Importance weighting (aligned by idx) ----
    iw_clip = float(robust_cfg.get("iw", {}).get("clip", 10.0))
    iw_w_by_idx = None
    if method == "iw":
        from sklearn.linear_model import LogisticRegression

        Xtr = splits["train"]["X"]
        Xte = splits["test"]["X"]

        Xdom = np.vstack([Xtr, Xte])
        ydom = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xte))])

        clf = LogisticRegression(max_iter=300)
        clf.fit(Xdom, ydom)

        p = clf.predict_proba(Xtr)[:, 1]
        w = p / (1 - p + 1e-8)  # density ratio approx
        w = np.clip(w, 1.0 / iw_clip, iw_clip).astype(np.float32)

        max_idx = int(max(splits["train"]["idx"].max(), splits["test"]["idx"].max(), splits["val"]["idx"].max()))
        iw_w_by_idx = np.ones(max_idx + 1, dtype=np.float32)
        iw_w_by_idx[splits["train"]["idx"]] = w
        iw_w_by_idx = torch.from_numpy(iw_w_by_idx)

    # ---- GroupDRO ----
    if method == "groupdro":
        eta = float(robust_cfg.get("groupdro", {}).get("eta", 0.05))
        n_groups = int(np.max(splits["train"]["group"])) + 1
        q = torch.ones(n_groups, device=device) / n_groups

    # ---- CORAL ----
    coral_lambda = float(robust_cfg.get("coral", {}).get("lambda", 1.0))

    def forward_with_embed(x):
        h = x
        for layer in list(model.net.children())[:-1]:
            h = layer(h)
        logits = list(model.net.children())[-1](h).squeeze(-1)
        return logits, h

    best_val = -1.0
    best_state = None
    patience = int(train_cfg["early_stop_patience"])
    bad = 0

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False)

        test_iter = iter(test_loader)

        for X, y, g, idx in pbar:
            X = X.to(device)
            y = y.to(device)
            g = g.to(device)
            idx = idx.to(device)

            opt.zero_grad()

            if method == "coral":
                logits_s, emb_s = forward_with_embed(X)
                clf_loss = F.binary_cross_entropy_with_logits(logits_s, y)

                try:
                    Xt, _, _, _ = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    Xt, _, _, _ = next(test_iter)

                Xt = Xt.to(device)
                _, emb_t = forward_with_embed(Xt)

                loss = clf_loss + coral_lambda * _coral_loss(emb_s, emb_t)

            else:
                logits = model(X)
                per_ex = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

                if method == "iw":
                    w = iw_w_by_idx[idx].to(device)
                    loss = (per_ex * w).mean()

                elif method == "groupdro":
                    group_losses = torch.zeros_like(q)
                    for gg in range(len(q)):
                        m = (g == gg)
                        if m.any():
                            group_losses[gg] = per_ex[m].mean()
                    q = q * torch.exp(eta * group_losses.detach())
                    q = q / (q.sum() + 1e-12)
                    loss = (q * group_losses).sum()

                else:
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

    return {
        "train_prob": _predict_proba(model, train_loader, device),
        "val_prob": _predict_proba(model, val_loader, device),
        "test_prob": _predict_proba(model, test_loader, device),
    }
