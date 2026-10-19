"""Generate two further result figures from the archived computed studies (results/published/*.json).

    .venv-gpu/Scripts/python tools/make_fig_extra.py --outdir ../figures

fig_v2x_labels.pdf (portrait, 6 panels): V2X-as-teacher in the static and sequential Monte-Carlo studies.
fig_system.pdf (4 panels): RSU deferral and wide-area share under SUMO mobility; safety-gate operating characteristic.
Every plotted value is read from the result files; style matches make_fig_results.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_fig_results import STYLE

PUB = Path(__file__).resolve().parents[1] / "results" / "published"
COL = {"P": "#6b6b6b", "PM": "#e87ba4", "C": "#2a78d6", "SM": "#1baf7a", "CT": "#eb6834"}
MRK = {"P": "^", "PM": "s", "C": "o", "SM": "D", "CT": "P"}
NAME = {"P": "P: projection", "PM": "PM: matching, no correction", "C": "C: consensus",
        "SM": "SM: smoothed pose", "CT": "CT: temporal consensus"}


def load(name):
    return json.loads((PUB / name).read_text())


def mv(metric):
    return (metric[0], metric[1]) if isinstance(metric, list) else (metric, 0.0)


def fig_labels(out):
    st = load("v2x_pseudolabel_mc.json")
    tm = load("mc_temporal.json")["results"]
    fig, axs = plt.subplots(3, 2, figsize=(13.1 / 2.54, 17.0 / 2.54), gridspec_kw={"hspace": 0.62, "wspace": 0.32})
    g = [0, 1, 3, 5, 10]
    for key in ("P", "PM", "C"):
        for ax, met in ((axs[0, 0], "precision"), (axs[0, 1], "recall_connected")):
            m = np.array([mv(st[f"gnss_{x}"]["metrics"][key][met]) for x in g])
            ax.errorbar(g, m[:, 0], yerr=m[:, 1], color=COL[key], marker=MRK[key], capsize=1.5, label=NAME[key])
    axs[0, 0].set(title="(a) Label precision", xlabel="GNSS error, 2-D RMS (m)", ylabel="Precision", ylim=(0, 1.02))
    axs[0, 1].set(title="(b) Recall of connected vehicles", xlabel="GNSS error, 2-D RMS (m)", ylabel="Recall", ylim=(0, 1.02))
    axs[0, 0].legend(loc="lower left", fontsize=5.8)

    rho = [0.1, 0.25, 0.5, 0.75, 1.0]
    ax = axs[1, 0]
    ax.plot([0.1, 1], [0.1, 1], ls=":", color="#555555", lw=0.9, label="connected share")
    for key in ("PM", "C"):
        m = np.array([mv(st[f"rho_{r}"]["metrics"][key]["recall_all"]) for r in rho])
        ax.errorbar(rho, m[:, 0], yerr=m[:, 1], color=COL[key], marker=MRK[key], capsize=1.5, label=key)
    ax.set(title="(c) Recall of all vehicles, 5 m", xlabel="V2X penetration", ylabel="Recall", ylim=(0, 1.02))
    ax.legend(loc="upper left", fontsize=5.8)

    ax = axs[1, 1]
    sets = [("pose_iid_g5", "i.i.d."), ("pose_gm_g5", "Gauss-Markov"), ("pose_gm_bias_g5", "GM + bias")]
    keys = ("PM", "C", "SM", "CT")
    w = 0.19
    for j, key in enumerate(keys):
        m = np.array([mv(tm[s]["metrics"][key]["recall_connected"]) for s, _ in sets])
        ax.bar(np.arange(3) + (j - 1.5) * w, m[:, 0], w * 0.92, yerr=m[:, 1], color=COL[key], label=key,
               error_kw={"lw": 0.7, "capsize": 1.2})
    ax.set_xticks(range(3), [n for _, n in sets])
    ax.set(title="(d) Correlated pose error, 5 m", ylabel="Recall (connected)", ylim=(0, 1.0))
    ax.legend(ncol=4, loc="upper center", fontsize=5.6, columnspacing=0.6, handlelength=1.0)

    ax = axs[2, 0]
    rr = [0.1, 0.25, 0.5, 1.0]
    for key in ("C", "CT"):
        m = np.array([mv(tm[f"rho_{r}"]["metrics"][key]["recall_connected"]) for r in rr])
        ax.errorbar(rr, m[:, 0], yerr=m[:, 1], color=COL[key], marker=MRK[key], capsize=1.5, label=NAME[key])
    ax.set(title="(e) Temporal gain vs. penetration (GM, 5 m)", xlabel="V2X penetration", ylabel="Recall (connected)",
           ylim=(0.3, 0.95))
    ax.legend(loc="lower right", fontsize=5.8)

    ax = axs[2, 1]
    att = [("pose_gm_g5", "none"), ("ghost_0.1", "ghosts\n10%"), ("ghost_0.3", "ghosts\n30%"),
           ("shift_0.1", "shifted\n10%"), ("shift_0.3", "shifted\n30%")]
    for j, key in enumerate(("PM", "C", "CT")):
        m = np.array([mv(tm[s]["metrics"][key]["precision"]) for s, _ in att])
        ax.bar(np.arange(len(att)) + (j - 1) * 0.26, m[:, 0], 0.24, yerr=m[:, 1], color=COL[key], label=key,
               error_kw={"lw": 0.7, "capsize": 1.2})
    ax.set_xticks(range(len(att)), [n for _, n in att], fontsize=5.8)
    ax.set(title="(f) Spoofed V2X messages", ylabel="Precision", ylim=(0.5, 1.0))
    ax.legend(ncol=3, loc="upper right", fontsize=5.8, columnspacing=0.6, handlelength=1.0)
    fig.savefig(out / "fig_v2x_labels.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig_v2x_labels.png", dpi=200, bbox_inches="tight", pad_inches=0.02)


def fig_system(out):
    runs = [r for r in load("mobility_sumo.json")["runs"] if r["loss"] == 0.0 and r.get("range_m", 150) == 150]
    oc = load("gate_oc.json")["uniform"]
    fig, axs = plt.subplots(2, 2, figsize=(13.1 / 2.54, 10.4 / 2.54), gridspec_kw={"hspace": 0.62, "wspace": 0.32})
    for fixed, color, marker, label in ((False, "#2a78d6", "o", "mobility-driven association"),
                                        (True, "#eb6834", "s", "fixed home RSU")):
        rs = sorted((r for r in runs if r["fixed_association"] == fixed), key=lambda r: r["n_aids"] / r["rsus"])
        x = [r["n_aids"] / r["rsus"] for r in rs]
        axs[0, 0].plot(x, [r["deferred_share"] for r in rs], marker=marker, ls="none", color=color, label=label, ms=4.5)
        axs[0, 1].plot(x, [r["wan_ratio"] for r in rs], marker=marker, ls="none", color=color, label=label, ms=4.5)
        for r, xi in zip(rs, x):
            if not fixed:
                axs[0, 0].annotate(f"{int(r['n_aids'])}/{int(r['rsus'])}", (xi, r["deferred_share"]), fontsize=5.4,
                                   xytext=(3, 3), textcoords="offset points")
    axs[0, 0].set(title="(a) Deferred RSU rounds", xlabel="Aids per RSU", ylabel="Share of RSU rounds", ylim=(-0.03, 0.75))
    axs[0, 1].set(title="(b) Wide-area traffic vs. flat FL", xlabel="Aids per RSU", ylabel="WAN ratio", ylim=(0, 0.14))
    axs[0, 0].legend(fontsize=5.8, loc="upper right")

    deltas = sorted({u["delta"] for u in oc})
    cmap = plt.get_cmap("Blues")
    for i, n in enumerate((500, 1000, 2000, 4000)):
        y = [next(u["p_release_shared"] for u in oc if u["n"] == n and u["churn"] == 0.05 and u["delta"] == d) for d in deltas]
        axs[1, 0].plot(np.array(deltas) * 100, y, marker="o", color=cmap(0.35 + 0.17 * i), label=f"n = {n}")
    axs[1, 0].set(title="(c) Gate: safety-set size (5% churn)", xlabel="True gain per group (points)",
                  ylabel="P(release)", ylim=(-0.03, 1.03))
    axs[1, 0].legend(fontsize=5.8, loc="lower right")
    cmap = plt.get_cmap("Oranges")
    for i, c in enumerate((0.02, 0.05, 0.1)):
        y = [next(u["p_release_shared"] for u in oc if u["n"] == 2000 and u["churn"] == c and u["delta"] == d) for d in deltas]
        axs[1, 1].plot(np.array(deltas) * 100, y, marker="s", color=cmap(0.45 + 0.2 * i), label=f"churn {int(c * 100)}%")
    axs[1, 1].set(title="(d) Gate: churn (n = 2,000)", xlabel="True gain per group (points)", ylabel="P(release)",
                  ylim=(-0.03, 1.03))
    axs[1, 1].legend(fontsize=5.8, loc="lower right")
    fig.savefig(out / "fig_system.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig_system.png", dpi=200, bbox_inches="tight", pad_inches=0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    out = Path(ap.parse_args().outdir)
    plt.rcParams.update(STYLE)
    fig_labels(out)
    fig_system(out)
    print("wrote", out / "fig_v2x_labels.pdf", out / "fig_system.pdf")


if __name__ == "__main__":
    main()
