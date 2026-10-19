#!/usr/bin/env python3
"""Statistics of experiment E7 and of the Phase 7 communication and participation studies, from the run logs only
(docs/preregistration_E7.md).

* E7: each ablation (A1-A8) and interaction run (I1, I2) against FedV2X-Aid on seeds 0-2: primary metric, pooled
  difference with a 95 % bootstrap interval over aids, paired Wilcoxon test over aids (seed-averaged per-aid mean
  R@P80) with Holm correction over the 10 tests; interactions (F - A) - (B - AB) with a bootstrap interval. A9 (no RSU
  tier) is the FedAvg row of E1 on the same seeds, reported without a test of its own (H3 of E1 covers it).
* E3: per method of E1, wide-area bytes, uplink bytes and uploads per run, and the simulated time (edge rounds of
  15 minutes) and bytes to the target, 90 % of FedAvg's final validation mean R@P80 (mean over E1 seeds), linearly
  interpolated between validation-curve points (every 5 cloud rounds); a run that never reaches the target is
  reported as such.
* Participation 0.5 and 0.25 against full participation (E1 seeds 0-2): primary metric, deferral share, time and
  bytes to the target.

    python src/fl/analyze_e7.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.fl.analyze_e1 import GROUPS, aid_means, load, pooled_boot  # noqa: E402
from src.metrics.stats import holm, wilcoxon_paired  # noqa: E402

EDGE_ROUND_MIN = 15.0
ABL = {"A1": "fedv2x-no_v2x", "A2": "fedv2x-pm", "A3": "fedv2x-c", "A4": "fedv2x-no_range", "A5": "fedv2x-no_scope",
       "A6": "fedv2x-no_quality", "A7": "fedv2x-fixed_rsu", "A8": "fedv2x-no_normbound",
       "I1": "fedv2x-pm-no_scope", "I2": "fedv2x-no_range-no_scope"}
INTER = {"I1": ("A2", "A5"), "I2": ("A4", "A5")}
PART = {"P50": "fedv2x-part0.50", "P25": "fedv2x-part0.25"}


def counts_matrix(runs, aids):
    """Seed-averaged true positives and instances per (aid, group) at each run's validation thresholds."""
    idx = {a: i for i, a in enumerate(aids)}
    tp = np.zeros((len(aids), len(GROUPS)))
    n = np.zeros((len(aids), len(GROUPS)))
    for r in runs.values():
        for key, v in r["per_aid_group"].items():
            a, g = key.split("|")
            if g in GROUPS and a in idx:
                tp[idx[a], GROUPS.index(g)] += v["recall"] * v["n"] / len(runs)
                n[idx[a], GROUPS.index(g)] += v["n"] / len(runs)
    return tp, n


def pooled_metric(tp, n, samp=None):
    if samp is None:
        T, N = tp.sum(0), n.sum(0)
        return float(np.nanmean(np.where(N > 0, T / np.maximum(N, 1e-12), np.nan)))
    T, N = tp[samp].sum(1), n[samp].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(np.where(N > 0, T / N, np.nan), axis=-1)


def interaction(F, A, B, AB, n_boot=10_000, seed=0):
    aids = sorted({k.split("|")[0] for r in F.values() for k in r["per_aid_group"]})
    mats = [counts_matrix(x, aids) for x in (F, A, B, AB)]
    est = [pooled_metric(*m) for m in mats]
    rng = np.random.default_rng(seed)
    samp = rng.integers(0, len(aids), size=(n_boot, len(aids)))
    bs = [pooled_metric(*m, samp) for m in mats]
    d = (bs[0] - bs[1]) - (bs[2] - bs[3])
    return {"estimate": (est[0] - est[1]) - (est[2] - est[3]),
            "ci95": [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]}


def crossing(curve, target):
    """Edge round, simulated hours, wide-area and uplink bytes at the first crossing of ``target``, linearly
    interpolated between curve points; None when the curve never reaches it."""
    prev = None
    for c in curve:
        if c["val_mean_r_at_p80"] >= target:
            if prev is None:
                return {"edge_round": c["round"], "hours": c["round"] * EDGE_ROUND_MIN / 60,
                        "wan_bytes": c["comm"].get("wan_bytes", 0), "uplink_bytes": c["comm"].get("uplink_bytes", 0),
                        "at_first_point": True}
            f = (target - prev["val_mean_r_at_p80"]) / max(c["val_mean_r_at_p80"] - prev["val_mean_r_at_p80"], 1e-12)

            def lerp(x0, x1):
                return x0 + f * (x1 - x0)
            r = lerp(prev["round"], c["round"])
            return {"edge_round": r, "hours": r * EDGE_ROUND_MIN / 60,
                    "wan_bytes": lerp(prev["comm"].get("wan_bytes", 0), c["comm"].get("wan_bytes", 0)),
                    "uplink_bytes": lerp(prev["comm"].get("uplink_bytes", 0), c["comm"].get("uplink_bytes", 0)),
                    "at_first_point": False}
        prev = c
    return None


def to_target(runs, target):
    per = {s: crossing(r["log"]["curve"], target) for s, r in runs.items()}
    hit = [v for v in per.values() if v is not None]
    out = {"reached": f"{len(hit)}/{len(per)}", "per_seed": per}
    if hit:
        for k in ("edge_round", "hours", "wan_bytes", "uplink_bytes"):
            out[k] = float(np.mean([v[k] for v in hit]))
    return out


def comm_summary(runs):
    keys = sorted({k for r in runs.values() for k in r.get("comm", {})})
    out = {k: float(np.mean([r["comm"].get(k, 0) for r in runs.values()])) for k in keys}
    if out.get("rsu_rounds"):
        out["deferred_share"] = out["deferred_rsu_rounds"] / out["rsu_rounds"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default=str(ROOT / "results/phase6/E1"))
    ap.add_argument("--e7", default=str(ROOT / "results/phase7/E7"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "results/phase7/E7_summary.json"))
    a = ap.parse_args()
    e1, e7 = load(a.e1), load(a.e7)
    sub = {k: {s: r for s, r in v.items() if s in a.seeds} for k, v in {**e1, **e7}.items()}
    F = sub["fedv2x"]
    res = {"seeds": a.seeds, "fedv2x_primary": [float(np.mean([r["primary_mean_r_at_p80"] for r in F.values()])),
                                               len(F)], "ablations": {}, "interactions": {}, "e3": {}, "participation": {}}
    pv = {}
    for k, tag in ABL.items():
        if tag not in sub or not sub[tag]:
            continue
        R = sub[tag]
        prim = [r["primary_mean_r_at_p80"] for r in R.values()]
        st, p, med, n = wilcoxon_paired(aid_means(F, GROUPS), aid_means(R, GROUPS))
        res["ablations"][k] = {"tag": tag, "seeds": sorted(R), "primary": [float(np.mean(prim)), float(np.std(prim, ddof=1)) if len(prim) > 1 else 0.0],
                               "pooled_fedv2x_minus_ablation": pooled_boot(F, R), "wilcoxon_stat": st, "p": p,
                               "median_diff": med, "aids": n}
        pv[k] = p
    for k, (padj, rej) in holm(pv).items():
        res["ablations"][k]["p_holm"] = padj
        res["ablations"][k]["reject_holm_0.05"] = bool(rej)
    if "fedavg" in sub:
        res["ablations"]["A9"] = {"tag": "fedavg (E1)", "seeds": sorted(sub["fedavg"]),
                                  "primary": [float(np.mean([r["primary_mean_r_at_p80"] for r in sub["fedavg"].values()])), 0.0],
                                  "pooled_fedv2x_minus_ablation": pooled_boot(F, sub["fedavg"])}
    for k, (ka, kb) in INTER.items():
        tags = [ABL[ka], ABL[kb], ABL[k]]
        if all(t in sub and sub[t] for t in tags):
            res["interactions"][k] = interaction(F, *[sub[t] for t in tags])
    # E3: communication and time to target, from all E1 seeds
    if "fedavg" in e1:
        finals = [r["log"]["curve"][-1]["val_mean_r_at_p80"] for r in e1["fedavg"].values() if r["log"]["curve"]]
        target = 0.9 * float(np.mean(finals))
        res["e3"]["target"] = target
        res["e3"]["fedavg_final_val"] = [float(np.mean(finals)), len(finals)]
        for m, R in sorted(e1.items()):
            res["e3"][m] = {"comm": comm_summary(R), "to_target": to_target(R, target) if any(r["log"]["curve"] for r in R.values()) else None}
        for k, tag in PART.items():
            if tag in sub and sub[tag]:
                R = sub[tag]
                res["participation"][k] = {"tag": tag, "seeds": sorted(R),
                                           "primary": float(np.mean([r["primary_mean_r_at_p80"] for r in R.values()])),
                                           "pooled_fedv2x_minus_part": pooled_boot(F, R),
                                           "comm": comm_summary(R), "to_target": to_target(R, target)}
        res["participation"]["full"] = {"seeds": sorted(F), "primary": res["fedv2x_primary"][0],
                                        "comm": comm_summary(F), "to_target": to_target(F, target)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["ablations"].items():
        d = v["pooled_fedv2x_minus_ablation"]
        print(f"{k} {v['tag']:28s} primary {v['primary'][0]:.4f}  F-A {d['difference']:+.4f} [{d['difference_ci95'][0]:+.4f}, {d['difference_ci95'][1]:+.4f}]"
              + (f"  p_holm {v['p_holm']:.4f}" if "p_holm" in v else ""))
    for k, v in res["interactions"].items():
        print(k, f"{v['estimate']:+.4f}", [round(x, 4) for x in v["ci95"]])
    if res["e3"]:
        print("target", round(res["e3"]["target"], 4))
        for m, v in res["e3"].items():
            if isinstance(v, dict) and v.get("to_target"):
                print(f"  {m:12s} reached {v['to_target']['reached']}  hours {v['to_target'].get('hours', float('nan')):.1f}  WAN MB {v['comm'].get('wan_bytes', 0) / 1e6:.1f}")
    for k, v in res["participation"].items():
        print(k, round(v["primary"], 4), v["comm"].get("deferred_share"), v["to_target"].get("hours") if v["to_target"] else None)


if __name__ == "__main__":
    main()
