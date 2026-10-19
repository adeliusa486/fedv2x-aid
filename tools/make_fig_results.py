"""Generate figures/fig_results.pdf from the E1, E7 and E6 result files (matplotlib, seaborn-like grid style).

    .venv/Scripts/python tools/make_fig_results.py --out ../figures/fig_results.pdf

Every plotted value is read from results/; nothing is typed by hand.
(a) E1 validation recall over edge rounds (mean over three seeds, band = min..max)
(b) E1 validation recall against cumulative wide-area traffic (log x), the communication/accuracy trade-off
(c) E7 ablations: FedV2X-Aid minus each variant with 95% bootstrap interval
(d) E6 client-level DP: mean R@P80 against epsilon
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
E1 = RES / "phase6" / "E1"

# Validated categorical order (dataviz reference palette; yellow uses its darker step for print contrast).
SERIES = [  # key, label, color, marker
    ("fedv2x", "FedV2X-Aid (RSU tier)", "#2a78d6", "o"),
    ("fedavg", "FedAvg", "#eb6834", "s"),
    ("fedprox", "FedProx", "#1baf7a", "^"),
    ("semifl", "SemiFL-style", "#c98500", "D"),
    ("hier_mt", "HierFAVG + mean teacher", "#e87ba4", "v"),
    ("local", "Local only", "#008300", "P"),
    ("fedsto", "FedSTO-style", "#6b6b6b", "X"),
]
ABL = {"A1": "No V2X labels", "A2": "Matched projection", "A3": "Per-frame consensus", "A4": "No range labels",
       "A5": "No source-scoped loss", "A6": "No quality weights", "A7": "Fixed RSU association",
       "A8": "No norm bound", "I1": "PM, no source scoping", "I2": "No range labels, no source scoping"}

STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.3, "axes.facecolor": "#eaeaf2", "axes.edgecolor": "white", "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": "#333333",
    "ytick.color": "#333333", "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": True,
    "legend.facecolor": "#eaeaf2", "legend.edgecolor": "#cccccc", "lines.linewidth": 1.3, "lines.markersize": 3.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def curves(method):
    runs = [json.loads(p.read_text())["log"] for p in sorted(E1.glob(f"{method}_s[0-9].json"))]
    rounds = [c["round"] for c in runs[0]["curve"]]
    val = np.array([[c["val_mean_r_at_p80"] for c in r["curve"]] for r in runs])
    wan = np.array([[c["comm"]["wan_bytes"] for c in r["curve"]] for r in runs]) / 1e6
    return np.array(rounds), val, wan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    fig, axs = plt.subplots(2, 2, figsize=(13.1 / 2.54, 10.6 / 2.54), gridspec_kw={"hspace": 0.62, "wspace": 0.30})
    (ax_a, ax_b), (ax_c, ax_d) = axs

    pre = json.loads((RES / "phase6" / "E1_summary.json").read_text())["methods"]
    handles = []
    for key, label, color, marker in SERIES:
        r, val, wan = curves(key)
        m = val.mean(0)
        h, = ax_a.plot(r, m, color=color, marker=marker, label=label)
        ax_a.fill_between(r, val.min(0), val.max(0), color=color, alpha=0.18, lw=0)
        handles.append(h)
        if wan.max() > 0:
            ax_b.plot(wan.mean(0), m, color=color, marker=marker)
    ax_a.set(title="(a) Validation recall over training", xlabel="Edge round", ylabel="Mean R@P80 (validation)",
             xticks=[10, 20, 30, 40, 50], ylim=(-0.01, 0.235))
    ax_b.set_xscale("log")
    ax_b.set(title="(b) Recall against wide-area traffic", xlabel="Cumulative wide-area traffic (MB, log)",
             ylabel="Mean R@P80 (validation)", ylim=(-0.01, 0.235))
    ax_b.grid(True, which="minor", color="white", linewidth=0.4)

    a = json.loads((RES / "phase7" / "E7_summary.json").read_text())["ablations"]
    keys = sorted((k for k in ABL if k in a), key=lambda k: a[k]["pooled_fedv2x_minus_ablation"]["difference"])
    for i, k in enumerate(keys):
        p = a[k]["pooled_fedv2x_minus_ablation"]
        d, (lo, hi) = p["difference"], p["difference_ci95"]
        ax_c.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color="#2a78d6", ms=3.6, capsize=1.8, lw=1.0)
    ax_c.axvline(0, color="#555555", lw=0.8)
    ax_c.set_yticks(range(len(keys)), [f"{k}: {ABL[k]}" for k in keys], fontsize=6.3)
    ax_c.set(title="(c) E7: effect of each component", xlabel="FedV2X-Aid minus variant (95% CI)")

    for sub, color, marker, label, filled in (("", "#2a78d6", "o", "hazard classifier (7,740 param.)", True),
                                              ("full_head", "#eb6834", "s", "whole head (878,216 param.)", False)):
        acc = json.loads((RES / "phase10" / sub / "dp_accounting.json").read_text())
        pts = sorted((row["eps_rdp"], json.loads((RES / "phase10" / sub / row["file"]).read_text())["primary_mean_r_at_p80"])
                     for row in acc)
        e, r = zip(*pts)
        ax_d.plot(e, r, color=color, marker=marker, label=label, mfc=color if filled else "none",
                  ls="-" if filled else "none", mew=1.1)
    ax_d.axhline(pre["fedv2x"]["primary"][0], color="#555555", ls="--", lw=0.9, label="no DP (secure aggregation only)")
    ax_d.set_xscale("log")
    ax_d.set_xticks([10, 23, 59, 168], ["10", "23", "59", "168"])
    ax_d.minorticks_off()
    ax_d.set(title="(d) E6: client-level DP at the RSU", xlabel=r"$\varepsilon$ at $\delta=10^{-5}$ (log)",
             ylabel="Mean R@P80 (test)", ylim=(-0.01, 0.21))
    ax_d.legend(loc="lower right", bbox_to_anchor=(0.995, 0.10), fontsize=5.9)

    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.015), columnspacing=1.0,
               handlelength=1.8)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(Path(args.out).with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
