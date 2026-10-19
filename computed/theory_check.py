"""Controlled numerical check of the trends predicted by Theorem 1 (J5).

Synthetic federated softmax regression (not the detection task): 30 clients with
Dirichlet(0.5) class mixtures, local SGD with S steps per edge round, clients regrouped at
random into 6 RSU groups every edge round (mobility), group averaging after every edge round
and global averaging every kappa edge rounds. Pseudo-label noise flips a fraction eps of the
training labels (biased gradients); DP-SGD clips per-example gradients to C and adds
N(0, z^2 C^2 I) to the batch sum. The total number of local steps is fixed for every run.

Reported per setting (mean and std over seeds): excess clean objective F(w) - F* and clean
squared gradient norm of the final global model, where F* is the clean centralised optimum.

Usage: python theory_check.py > theory_check.json
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize

K, GROUPS, N_PER, CLASSES = 30, 6, 400, 5
S, BATCH, LR, LAM = 10, 16, 0.05, 1e-3
TOTAL_STEPS = 3000
SEEDS = 5


def make_data(rng, d_in):
    mu = rng.normal(0, 1.0, size=(CLASSES, d_in)) * (2.0 / np.sqrt(d_in / 20))
    X, Y = [], []
    for _ in range(K):
        p = rng.dirichlet(0.5 * np.ones(CLASSES))
        y = rng.choice(CLASSES, size=N_PER, p=p)
        x = mu[y] + rng.normal(0, 1.5, size=(N_PER, d_in))
        X.append(np.hstack([x, np.ones((N_PER, 1))]))
        Y.append(y)
    return X, Y


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss_grad(W, X, y):
    P = softmax(X @ W)
    n = len(y)
    loss = -np.log(P[np.arange(n), y] + 1e-12).mean() + 0.5 * LAM * np.sum(W * W)
    P[np.arange(n), y] -= 1
    return loss, X.T @ P / n + LAM * W


def global_obj(W, X, Y):
    ls, gs = zip(*(loss_grad(W, x, y) for x, y in zip(X, Y)))
    return float(np.mean(ls)), np.mean(gs, axis=0)


def run(rng, X, Y, eps, kappa, z, C=1.0, clip=None):
    clip = (z > 0) if clip is None else clip
    d = X[0].shape[1]
    Yn = []
    for y in Y:  # pseudo-label noise: symmetric flips
        y2 = y.copy()
        flip = rng.random(len(y)) < eps
        y2[flip] = (y[flip] + rng.integers(1, CLASSES, size=flip.sum())) % CLASSES
        Yn.append(y2)
    W = np.zeros((d, CLASSES))
    local = [W.copy() for _ in range(K)]
    steps = 0
    edge_round = 0
    while steps < TOTAL_STEPS:
        for k in range(K):
            Wk = local[k]
            for _ in range(S):
                idx = rng.integers(0, N_PER, size=BATCH)
                x, y = X[k][idx], Yn[k][idx]
                P = softmax(x @ Wk)
                P[np.arange(BATCH), y] -= 1
                per = x[:, :, None] * P[:, None, :]          # per-example gradients (B, d, C)
                if clip:
                    norms = np.sqrt((per ** 2).sum(axis=(1, 2)))
                    per = per / np.maximum(1.0, norms / C)[:, None, None]
                    g = (per.sum(axis=0) + (rng.normal(0, z * C, size=Wk.shape) if z > 0 else 0.0)) / BATCH
                else:
                    g = per.mean(axis=0)
                Wk = Wk - LR * (g + LAM * Wk)
            local[k] = Wk
        steps += S
        edge_round += 1
        perm = rng.permutation(K)
        for gidx in np.array_split(perm, GROUPS):          # mobility-driven regrouping
            m = np.mean([local[k] for k in gidx], axis=0)
            for k in gidx:
                local[k] = m.copy()
        if edge_round % kappa == 0:
            m = np.mean(local, axis=0)
            local = [m.copy() for _ in range(K)]
    Wg = np.mean(local, axis=0)
    f, g = global_obj(Wg, X, Y)
    return f, float(np.sum(g * g))


def optimum(X, Y):
    d = X[0].shape[1]
    fun = lambda w: global_obj(w.reshape(d, CLASSES), X, Y)
    res = minimize(lambda w: fun(w)[0], np.zeros(d * CLASSES), jac=lambda w: fun(w)[1].ravel(), method="L-BFGS-B", options=dict(maxiter=2000, gtol=1e-10))
    return float(res.fun)


def main():
    out = dict(K=K, groups=GROUPS, S=S, batch=BATCH, lr=LR, total_steps=TOTAL_STEPS, seeds=SEEDS, sweeps={})
    sweeps = dict(eps=[dict(eps=e, kappa=2, z=0.0, d_in=20) for e in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)],
                  kappa=[dict(eps=0.1, kappa=k, z=0.0, d_in=20) for k in (1, 2, 4, 8, 16)],
                  dp=[dict(eps=0.1, kappa=2, z=zz, d_in=d, clip=True) for d in (20, 200) for zz in (0.0, 0.5, 1.0, 2.0)])
    only = sys.argv[1:]  # optional: names of sweeps to run
    data = {}
    for name, cfgs in sweeps.items():
        if only and name not in only:
            continue
        rows = []
        for cfg in cfgs:
            ex, gn = [], []
            for s in range(SEEDS):
                key = (cfg["d_in"], s)
                if key not in data:
                    X, Y = make_data(np.random.default_rng(100 + s), cfg["d_in"])
                    data[key] = (X, Y, optimum(X, Y))
                X, Y, fstar = data[key]
                f, g2 = run(np.random.default_rng(500 + s), X, Y, cfg["eps"], cfg["kappa"], cfg["z"], clip=cfg.get("clip"))
                ex.append(f - fstar)
                gn.append(g2)
            row = dict(cfg, excess=[float(np.mean(ex)), float(np.std(ex, ddof=1))], grad2=[float(np.mean(gn)), float(np.std(gn, ddof=1))])
            rows.append(row)
            print(name, row, file=sys.stderr, flush=True)
        out["sweeps"][name] = rows
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
