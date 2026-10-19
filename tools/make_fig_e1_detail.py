"""Portrait figure with the E1 detail (per hazard group, per aid, per seed), from results/phase6/E1/*.json.

    .venv-gpu/Scripts/python tools/make_fig_e1_detail.py --out ../figures/fig_e1_detail.pdf

Row 1: per hazard group, test recall at 80% precision for every method (bar = min..max over seeds, tick = mean).
Row 2: distribution of the 30 per-aid mean recalls (averaged over seeds) per method.
Row 3: validation recall over edge rounds, one panel per seed.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_fig_results import STYLE

E1 = Path(__file__).resolve().parents[1] / "results" / "phase6" / "E1"
METHODS = [  # key, label, color (by family: RSU tier blue, flat orange, self-training pink, no federation gray)
    ("fedv2x", "FedV2X-Aid", "#184f95"),
    ("hier_mt", "HierFAVG + MT", "#5598e7"),
    ("hier_st", "HierFAVG + ST", "#9ec5f4"),
    ("fedavg", "FedAvg", "#eb6834"),
    ("fedprox", "FedProx", "#f29a6b"),
    ("scaffold", "SCAFFOLD", "#c24d1c"),
    ("fedrep", "FedRep-style", "#c98500"),
    ("semifl", "SemiFL-style", "#1baf7a"),
    ("fedsto", "FedSTO-style", "#e87ba4"),
    ("central", "Pooled", "#8a8a8a"),
    ("local", "Local only", "#008300"),
    ("pretrained", "Pretrained", "#555555"),
]
GROUPS = ["vehicle", "person", "stairs", "pothole"]


def runs(m):
    return [json.loads(p.read_text()) for p in sorted(E1.glob(f"{m}_s[0-9].json"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(13.1 / 2.54, 19.0 / 2.54))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1.25, 0.85], hspace=0.75, wspace=0.12)
    data = {m: runs(m) for m, _, _ in METHODS}
    ys = np.arange(len(METHODS))[::-1]

    for j, g in enumerate(GROUPS):
        ax = fig.add_subplot(gs[0, j])
        for (m, label, c), y in zip(METHODS, ys):
            v = np.array([r["per_group"][g]["test_recall"] for r in data[m]])
            ax.add_patch(plt.Rectangle((v.min(), y - 0.32), max(v.max() - v.min(), 0.004), 0.64, color=c, alpha=0.55, lw=0))
            ax.plot([v.mean()] * 2, [y - 0.36, y + 0.36], color=c, lw=1.4)
        ax.set_ylim(-0.7, len(METHODS) - 0.3)
        ax.set_yticks(ys, [l for _, l, _ in METHODS] if j == 0 else [], fontsize=5.8)
        ax.set_title(f"({'abcd'[j]}) {g}", fontsize=7.5)
        hi = 0.62 if g in ("vehicle", "person") else 0.085
        ax.set_xlim(-0.03 * hi, hi)
        ax.set_xticks([0, 0.25, 0.5] if hi > 0.1 else [0, 0.04, 0.08])
        ax.tick_params(axis="x", labelsize=5.8)

    ax = fig.add_subplot(gs[1, :])
    vals = []
    for m, label, c in METHODS:
        a = np.array([[r["aid_means"][k] for k in sorted(r["aid_means"])] for r in data[m]]).mean(0)
        vals.append(a)
    bp = ax.boxplot(vals, positions=range(len(METHODS)), widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "lw": 1.0}, whiskerprops={"lw": 0.8}, capprops={"lw": 0.8})
    rng = np.random.default_rng(0)
    for i, ((m, label, c), a) in enumerate(zip(METHODS, vals)):
        bp["boxes"][i].set(facecolor=c, alpha=0.45, lw=0.6)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(a)), a, s=4, color=c, edgecolor="none", zorder=3)
    ax.set_xticks(range(len(METHODS)), [l for _, l, _ in METHODS], rotation=35, ha="right", fontsize=6.0)
    ax.set_ylabel("Per-aid mean recall")
    ax.set_title("(e) Per-aid mean recall over the 30 aids (mean of three seeds)")

    curves = {}
    for m, label, c in METHODS:
        if m == "central":  # pooled training logs optimizer steps, not edge rounds
            continue
        for r in data[m]:
            if r["log"].get("curve"):
                curves.setdefault(m, []).append(r["log"]["curve"])
    axs = [fig.add_subplot(gs[2, s]) for s in range(3)]
    for s, ax in enumerate(axs):
        for m, label, c in METHODS:
            if m not in curves or s >= len(curves[m]):
                continue
            cv = curves[m][s]
            ax.plot([p["round"] for p in cv], [p["val_mean_r_at_p80"] for p in cv], color=c, marker="o", ms=2.2, lw=1.0,
                    label=label)
        ax.set_title(f"({'fgh'[s]}) validation, seed {s}", fontsize=7.2)
        ax.set_xlabel("Edge round", fontsize=6.6)
        ax.set_ylim(-0.01, 0.24)
        ax.tick_params(labelsize=5.8)
        if s:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Val. mean R@P80", fontsize=6.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.add_subplot(gs[2, 3]).axis("off")
    fig.legend(h, l, loc="lower right", bbox_to_anchor=(0.995, 0.035), fontsize=5.6, ncol=1, handlelength=1.4)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
