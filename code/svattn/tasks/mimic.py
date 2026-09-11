"""MIMIC-IV ICU next-horizon deterioration prediction (causal).

Loads the locally-built tokenized cohort
(clinical_seq/cache/icu_vitals_n1500.npz) -- a PhysioNet-DUA-covered derivative,
used locally and never redistributed. Task: at each hour t, predict whether any
vital crosses a deterioration threshold within the next `horizon` hours, from the
stream so far. This is a FUTURE-outcome prediction (the gate and the label do not
read the same hour's vitals), which sidesteps the retrospective-selection critique
of the original paper's clinical result.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CACHE = "clinical_seq/cache/icu_vitals_n1500.npz"


def load_cohort(path=CACHE, t_fixed=56, n_max=400, seed=0):
    d = np.load(path, allow_pickle=True)
    seqs, labs = d["sequences"], d["det_labels"]
    idx = [i for i in range(len(seqs)) if len(seqs[i]) >= t_fixed]
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    idx = idx[:n_max]
    X = np.stack([np.asarray(seqs[i][:t_fixed], dtype=np.float32) for i in idx])
    det = np.stack([np.asarray(labs[i][:t_fixed], dtype=np.float32) for i in idx])
    return X, det                                         # (N, T, 6), (N, T)


def next_h_labels(det, horizon=6):
    N, T = det.shape
    y = np.zeros((N, T), np.float32)
    valid = np.zeros((N, T), np.float32)
    for t in range(T - 1):
        end = min(t + 1 + horizon, T)
        y[:, t] = (det[:, t + 1:end].sum(1) > 0).astype(np.float32)
        valid[:, t] = 1.0
    return y, valid


class ClinicalModel(nn.Module):
    """embed(6 vitals) -> causal sequence layer -> per-hour logit."""

    def __init__(self, layer: nn.Module, n_feat=6, d_model=32):
        super().__init__()
        self.embed = nn.Linear(n_feat, d_model)
        self.layer = layer
        self.head = nn.Linear(d_model, 1)

    def forward(self, X):                                  # (B, T, 6) -> (B, T)
        return self.head(self.layer(self.embed(X))).squeeze(-1)


class LSTMLayer(nn.Module):
    """Causal LSTM with the (B, T, d_model) -> (B, T, d_model) layer interface."""

    def __init__(self, d_model, n_heads=None, head_dim=None):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

    def forward(self, X):
        return self.lstm(X)[0]


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auprc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y.sum(), 1)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def train_clinical(model, X, y, valid, steps=120, batch=8, lr=2e-3, seed=0):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt, vt = torch.tensor(X), torch.tensor(y), torch.tensor(valid)
    npos = max((y * valid).sum(), 1.0)
    pos_w = torch.tensor([(valid.sum() - npos) / npos], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")
    N = len(X)
    model.train()
    last = float("nan")
    for _ in range(steps):
        bi = rng.choice(N, min(batch, N), replace=False)
        opt.zero_grad()
        logit = model(Xt[bi])
        l = (lossf(logit, yt[bi]) * vt[bi]).sum() / vt[bi].sum().clamp_min(1)
        l.backward()
        opt.step()
        last = float(l.item())
    return last


@torch.no_grad()
def eval_clinical(model, X, y, valid):
    model.eval()
    logit = model(torch.tensor(X)).numpy()
    m = valid.astype(bool)
    return auroc(y[m], logit[m]), auprc(y[m], logit[m])
