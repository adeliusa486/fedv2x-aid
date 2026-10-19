"""Statistics of FedV2X-Aid's learning experiments: bootstrap clustered by aid, paired Wilcoxon with Holm.

Frames of one aid are correlated, so every interval resamples aids (clusters), never frames.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def cluster_bootstrap(per_aid_values: dict, statistic=np.mean, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile interval of ``statistic`` over aids, resampling aids with replacement.

    ``per_aid_values`` maps aid -> value (e.g. that aid's mean recall). Returns (estimate, low, high).
    """
    keys = sorted(per_aid_values, key=str)
    x = np.array([per_aid_values[k] for k in keys], dtype=np.float64)
    est = float(statistic(x))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    if statistic is np.mean:
        boot = x[idx].mean(axis=1)
    else:
        boot = np.array([statistic(x[i]) for i in idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def paired_cluster_bootstrap(a: dict, b: dict, n_boot=10_000, alpha=0.05, seed=0):
    """Interval of mean(a - b) over aids present in both, resampling aids. Returns (est, low, high)."""
    keys = sorted(set(a) & set(b), key=str)
    d = {k: a[k] - b[k] for k in keys}
    return cluster_bootstrap(d, np.mean, n_boot, alpha, seed)


def cluster_bootstrap_pooled(per_aid_counts: dict, n_boot=10_000, alpha=0.05, seed=0):
    """Interval of a pooled ratio sum(num)/sum(den) (e.g. pooled recall), resampling aids.

    ``per_aid_counts`` maps aid -> (numerator, denominator).
    """
    keys = sorted(per_aid_counts, key=str)
    num = np.array([per_aid_counts[k][0] for k in keys], dtype=np.float64)
    den = np.array([per_aid_counts[k][1] for k in keys], dtype=np.float64)
    est = float(num.sum() / den.sum()) if den.sum() else float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    nb, db = num[idx].sum(1), den[idx].sum(1)
    boot = np.divide(nb, db, out=np.full_like(nb, np.nan), where=db > 0)
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def wilcoxon_paired(a: dict, b: dict):
    """Two-sided Wilcoxon signed-rank test of a vs. b over aids present in both.

    Returns (statistic, p_value, median_difference, n). Zero differences are dropped (``zero_method="wilcox"``),
    as in Wilcoxon (1945); if all differences are zero, p = 1.
    """
    keys = sorted(set(a) & set(b), key=str)
    d = np.array([a[k] - b[k] for k in keys], dtype=np.float64)
    med = float(np.median(d)) if len(d) else float("nan")
    if len(d) == 0 or np.all(d == 0):
        return 0.0, 1.0, med, len(d)
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue), med, len(d)


def holm(pvalues: dict, alpha=0.05):
    """Holm (1979) step-down adjustment. Returns {name: (adjusted_p, reject)}."""
    names = sorted(pvalues, key=lambda k: (pvalues[k], str(k)))
    m = len(names)
    out, running = {}, 0.0
    for i, k in enumerate(names):
        adj = min(1.0, (m - i) * pvalues[k])
        running = max(running, adj)          # adjusted p-values are monotone
        out[k] = running
    return {k: (out[k], out[k] <= alpha) for k in pvalues}
