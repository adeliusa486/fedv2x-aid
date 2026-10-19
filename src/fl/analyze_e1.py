#!/usr/bin/env python3
"""Statistics of experiment E1 as pre-registered (docs/preregistration_E1.md, Sections 4-5), from the run logs only.

* Per method: mean and standard deviation over seeds of the primary metric (mean pooled R@P80 over the six
  hazard groups), per-group R@P80, the five-group mean without stairs, mAP50 over hazard classes, mean per-aid
  recall and worst-aid recall.
* Per aid: mean R@P80 over the groups evaluated on that aid, averaged over seeds.
* H1 FedV2X-Aid > Local-only; H2 FedV2X-Aid > each self-training baseline on the drop-off groups (curb, stairs,
  pothole; per-aid mean over the drop-off groups evaluated on the aid); H3 FedV2X-Aid not worse than FedAvg by more
  than 1 point (lower 95 % bootstrap bound of the pooled primary-metric difference above -0.01; its Wilcoxon p is
  reported and corrected with the others); H4 FedV2X-Aid > Pretrained. Paired two-sided Wilcoxon tests over aids,
  Holm correction over the seven tests, median per-aid difference as effect size, 10,000 bootstrap resamples of
  aids (seed 0) for intervals of pooled metrics (thresholds fixed from validation data in every run).

    python src/fl/analyze_e1.py --dir results/phase6/E1
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.metrics.stats import holm, wilcoxon_paired  # noqa: E402

GROUPS = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]
DROP = ["curb", "stairs", "pothole"]
SELF = ["hier_st", "hier_mt", "fedsto", "semifl"]


def load(d):
    runs = defaultdict(dict)
    for f in sorted(Path(d).glob("*_s[0-9].json")):
        r = json.loads(f.read_text())
        runs[f.stem.rsplit("_s", 1)[0]][r["seed"]] = r          # result tag (method, or method-ablation)
    return runs


def per_aid_counts(r):
    """(aid, group) -> (true positives, instances) at the run's validation thresholds."""
    out = {}
    for key, v in r["per_aid_group"].items():
        aid, g = key.split("|")
        out[(aid, g)] = (v["recall"] * v["n"], v["n"])
    return out


def aid_means(runs, groups):
    """aid -> mean over seeds of the per-aid mean recall over ``groups`` evaluated on the aid (>= 20 instances)."""
    acc = defaultdict(list)
    for r in runs.values():
        per = defaultdict(list)
        for key, v in r["per_aid_group"].items():
            aid, g = key.split("|")
            if g in groups and v["n"] >= 20:
                per[aid].append(v["recall"])
        for aid, vals in per.items():
            acc[aid].append(float(np.mean(vals)))
    return {a: float(np.mean(v)) for a, v in acc.items()}


def pooled_boot(runs_a, runs_b, n_boot=10_000, seed=0, groups=GROUPS):
    """Bootstrap over aids of the seed-averaged pooled primary metric of a and of a - b."""
    def counts(runs):
        c = defaultdict(lambda: np.zeros(2))
        for r in runs.values():
            for k, v in per_aid_counts(r).items():
                c[k] += np.array(v) / len(runs)
        return c
    ca = counts(runs_a)
    cb = counts(runs_b) if runs_b is not None else None
    aids = sorted({k[0] for k in ca})
    idx = {a: i for i, a in enumerate(aids)}

    def mat(c):
        tp = np.zeros((len(aids), len(groups)))
        n = np.zeros((len(aids), len(groups)))
        for (a, g), (t, m) in c.items():
            if g in groups:
                tp[idx[a], groups.index(g)] = t
                n[idx[a], groups.index(g)] = m
        return tp, n
    ta, na = mat(ca)
    rng = np.random.default_rng(seed)
    samp = rng.integers(0, len(aids), size=(n_boot, len(aids)))

    def metric(tp, n, s):
        T, N = tp[s].sum(1), n[s].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nanmean(np.where(N > 0, T / N, np.nan), axis=-1)
    est_a = float(np.nanmean(np.where(na.sum(0) > 0, ta.sum(0) / np.maximum(na.sum(0), 1), np.nan)))
    boot_a = metric(ta, na, samp)
    res = {"estimate": est_a, "ci95": [float(np.nanquantile(boot_a, 0.025)), float(np.nanquantile(boot_a, 0.975))]}
    if cb is not None:
        tb, nb = mat(cb)
        est_b = float(np.nanmean(np.where(nb.sum(0) > 0, tb.sum(0) / np.maximum(nb.sum(0), 1), np.nan)))
        d = boot_a - metric(tb, nb, samp)
        res["difference"] = est_a - est_b
        res["difference_ci95"] = [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results/phase6/E1"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = load(a.dir)
    summary = {"methods": {}, "tests": {}}
    for m, rs in sorted(runs.items()):
        seeds = sorted(rs)

        def ms(f):
            v = np.array([f(rs[s]) for s in seeds], dtype=float)
            return [float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0]
        summary["methods"][m] = {
            "seeds": seeds,
            "primary": ms(lambda r: r["primary_mean_r_at_p80"]),
            "five_groups": ms(lambda r: r["mean_r_at_p80_without_stairs"]),
            "per_group": {g: ms(lambda r, g=g: r["per_group"][g]["test_recall"]) for g in GROUPS},
            "map50": ms(lambda r: r["map50_hazard_classes"]),
            "mean_aid": ms(lambda r: r["mean_aid_recall"]),
            "worst_aid": ms(lambda r: r["worst_aid_recall"]),
            "seconds": ms(lambda r: r["seconds"]),
            "comm": {k: float(np.mean([rs[s]["comm"].get(k, 0) for s in seeds])) for k in rs[seeds[0]].get("comm", {})},
            "pooled_bootstrap": pooled_boot(rs, None),
        }
    if "fedv2x" in runs:
        F = runs["fedv2x"]
        pv = {}
        tests = {}

        def add(name, other, groups):
            if other not in runs:
                return
            st, p, med, n = wilcoxon_paired(aid_means(F, groups), aid_means(runs[other], groups))
            tests[name] = {"vs": other, "groups": groups, "wilcoxon_stat": st, "p": p, "median_diff": med, "aids": n}
            pv[name] = p
        add("H1_vs_local", "local", GROUPS)
        for b in SELF:
            add(f"H2_vs_{b}", b, DROP)
        add("H3_vs_fedavg", "fedavg", GROUPS)
        add("H4_vs_pretrained", "pretrained", GROUPS)
        for name, (padj, rej) in holm(pv).items():
            tests[name]["p_holm"] = padj
            tests[name]["reject_holm_0.05"] = bool(rej)
        if "fedavg" in runs:
            tests["H3_vs_fedavg"]["pooled"] = pooled_boot(F, runs["fedavg"])
            tests["H3_vs_fedavg"]["non_inferior"] = tests["H3_vs_fedavg"]["pooled"]["difference_ci95"][0] > -0.01
        for other in ("local", "pretrained", "central", *SELF):
            if other in runs:
                tests.setdefault(f"pooled_vs_{other}", {})["pooled"] = pooled_boot(F, runs[other])
        summary["tests"] = tests
    out = Path(a.out) if a.out else Path(a.dir).with_name(Path(a.dir).name + "_summary.json")
    out.write_text(json.dumps(summary, indent=1))
    for m, v in summary["methods"].items():
        print(f"{m:12s} primary {v['primary'][0]:.3f}+-{v['primary'][1]:.3f}  mAP50 {v['map50'][0]:.3f}  worst {v['worst_aid'][0]:.3f}  "
              + " ".join(f"{g[:4]} {v['per_group'][g][0]:.2f}" for g in GROUPS) + f"  ({len(v['seeds'])} seeds)")
    for k, t in summary["tests"].items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in t.items() if kk != "pooled"})


if __name__ == "__main__":
    main()
