"""Check the computed numbers printed in the FedV2X-Aid manuscript against the archived result files.

Every table cell, plotted coordinate and listed sentence that reports a computed result is recomputed
from results/published/*.json and compared with the printed value at the printed precision. Values
inside \\est{...} (old orange targets) or \\pend{...} (pending experiments) are skipped: they are not
results.

Usage:
    python tools/verify_manuscript_numbers.py --tex path/to/FedV2X-Aid_journal.tex [--report docs/number_check_report.md]

Result per item: PASS (printed = recomputed at printed precision), FAIL (they differ), MISSING (the
row, series or sentence the check looks for is not in the manuscript; the check needs updating).
Exit status is 1 if anything FAILs.
"""
import argparse
import os
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "results" / "published"
NUM = r"-?\d[\d,]*(?:\.\d+)?"


def load(name):
    return json.loads((PUB / name).read_text())


# ----------------------------------------------------------------------------------------------
# LaTeX helpers
# ----------------------------------------------------------------------------------------------
def _braced(s, cmd, keep):
    out, i, key = [], 0, "\\" + cmd + "{"
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k, depth = j + len(key), 1
        while k < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        out.append(s[j + len(key):k - 1] if keep else " TARGET ")
        i = k


def prep(s):
    """Normalise number formatting so that 1{,}285 reads as 1,285."""
    return s.replace("{,}", ",")


def env_around(tex, label, kind):
    pos = tex.find("\\label{" + label + "}")
    if pos < 0:
        return None
    start = max(tex.rfind("\\begin{" + kind + "}", 0, pos), tex.rfind("\\begin{" + kind + "*}", 0, pos))
    end = tex.find("\\end{" + kind, pos)
    return tex[start:end] if start >= 0 and end > 0 else None


def normalise(cell):
    c = cell
    m = re.match(r"\s*\\multicolumn\{\d+\}\{(?:[^{}]|\{[^{}]*\})*\}\{(.*)\}\s*$", c, re.S)
    if m:
        c = m.group(1)
    for cmd in ("textbf", "emph", "mathrm", "texttt"):
        c = _braced(c, cmd, keep=True)
    for cmd in ("est", "pend"):
        c = _braced(c, cmd, keep=False)
    for a, b in {r"$\times$": "x", r"\times": "x", r"\rho": "rho", r"\omega^{rel}": "omega_rel", r"\,": " ", r"\%": "%",
                 r"\quad": " ", "$": "", r"\ ": " ", "~": " ", r"\pm": "+-", r"\mu": "mu"}.items():
        c = c.replace(a, b)
    c = c.replace("{", "").replace("}", "")
    c = re.sub(r"\\[a-zA-Z]+", "", c)
    return re.sub(r"\s+", " ", c).strip()


def table_rows(tex, label):
    env = env_around(tex, label, "table")
    if env is None:
        return None
    m = re.search(r"\\begin\{tabular\}\{.*?\}\n(.*?)\\end\{tabular\}", env, re.S)
    body = re.sub(r"\\(toprule|midrule|bottomrule)", "", m.group(1))
    body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
    return [[normalise(c) for c in re.split(r"(?<!\\)&", raw)] for raw in body.split("\\\\") if raw.strip()]


def figure_series(tex, label):
    env = env_around(tex, label, "figure")
    if env is None:
        return None
    series = []
    for m in re.finditer(r"\\addplot(?:\[[^\]]*\])?\s*coordinates\s*\{(.*?)\};", env, re.S):
        series.append(re.findall(r"\(\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*\)", m.group(1)))
    return series


# ----------------------------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------------------------
def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def agrees(printed, expected):
    """expected: float (compare at printed precision), (float, decimals), ('le', x): printed >= x,
    or ('lt', x): printed > x (for 'at most' / 'less than' statements)."""
    p = float(printed.replace(",", ""))
    if isinstance(expected, tuple) and expected[0] in ("le", "lt"):
        return p >= expected[1] - 1e-12 if expected[0] == "le" else p > expected[1]
    v, d = (expected if isinstance(expected, tuple) else (expected, None))
    d = decimals(printed) if d is None else d
    return abs(p - v) <= 0.5 * 10 ** (-d) + 1e-9


def shown(e):
    if isinstance(e, tuple):
        return e if isinstance(e[0], str) else round(e[0], 6)
    return round(e, 6)


class Report:
    def __init__(self):
        self.items = []

    def add(self, group, item, status, detail=""):
        self.items.append((group, item, status, detail))

    def counts(self):
        c = {}
        for item in self.items:
            c[item[2]] = c.get(item[2], 0) + 1
        return c


def check_table(rep, tex, label, specs, key_cells=1):
    rows = table_rows(tex, label)
    if rows is None:
        rep.add(label, "(table)", "MISSING", "label not found")
        return
    keyed = [(" ".join(r[:key_cells]).strip(), r) for r in rows]
    for key, expected in specs:
        row = next((r for k, r in keyed if k == key), None) or next((r for k, r in keyed if k.startswith(key)), None)
        if row is None:
            rep.add(label, key, "MISSING", "row not found")
            continue
        got = [n for c in row for n in re.findall(NUM, c)]
        if len(got) != len(expected):
            rep.add(label, key, "FAIL", f"{len(got)} numbers printed {got}, {len(expected)} expected")
            continue
        bad = [(g, shown(e)) for g, e in zip(got, expected) if not agrees(g, e)]
        rep.add(label, key, "FAIL" if bad else "PASS", f"(printed, computed): {bad}" if bad else f"{len(got)} values")


def check_figure(rep, tex, label, specs, dec=3):
    series = figure_series(tex, label)
    if series is None:
        rep.add(label, "(figure)", "MISSING", "label not found")
        return
    for i, (name, expected) in enumerate(specs):
        if expected is None:
            continue
        if i >= len(series):
            rep.add(label, name, "MISSING", "series not found")
            continue
        got = series[i]
        if len(got) != len(expected):
            rep.add(label, name, "FAIL", f"{len(got)} points printed, {len(expected)} computed")
            continue
        bad = [(x, y, round(v, 6)) for (x, y), v in zip(got, expected) if not agrees(y, (v, max(dec, decimals(y))))]
        rep.add(label, name, "FAIL" if bad else "PASS", f"(x, printed, computed): {bad}" if bad else f"{len(got)} points")


# Claims whose text was removed when the paper was shortened (18 October 2026); the numbers are no longer printed.
RETIRED = {"related work: trained share", "conclusion: update share", "conclusion: gate per round",
           "conclusion: gate over 100 rounds", "device: conclusion rate"}


def check_text(rep, tex, claims):
    for cid, template, value in claims:
        if cid in RETIRED:
            continue
        parts = prep(template).split("{v}")
        regex = "".join(re.escape(p) + ("(" + NUM + ")" if k < len(parts) - 1 else "") for k, p in enumerate(parts))
        m = re.search(regex, tex)
        if not m:
            rep.add("text", cid, "MISSING", template.replace("{v}", "#"))
            continue
        vals = list(value) if isinstance(value, list) else [value]
        bad = [(g, shown(v)) for g, v in zip(m.groups(), vals) if not agrees(g, v)]
        rep.add("text", cid, "FAIL" if bad else "PASS", f"(printed, computed): {bad}" if bad else "")


# ----------------------------------------------------------------------------------------------
# Specifications
# ----------------------------------------------------------------------------------------------
def specs():
    det = load("mc_det.json")["results"]
    hgt = load("mc_height.json")["results"]
    tmp = load("mc_temporal.json")["results"]
    mis = load("mc_temporal_misspec.json")["results"]
    mob = load("mobility_sumo.json")["runs"]
    mod = load("mobility_modes.json")["runs"]
    gdr = load("gate_drift.json")
    goc = load("gate_oc.json")
    st = load("v2x_pseudolabel_mc.json")
    grd = load("mc_grid.json")["results"]
    thc = load("theory_check.json")["sweeps"]
    thd = load("theory_check_dp.json")["sweeps"]["dp"]
    dpb = load("dp_budget.json")
    dpu = load("dp_userlevel.json")
    bud = load("yolo11_budget.json")["nc152"]

    def pr(m, meth):
        return [(m[meth]["precision"][0], 3), (m[meth]["recall_connected"][0], 3)]

    def run(n, r, rng, fixed=False, loss=0.0):
        return next(x for x in mob if x["n_aids"] == n and x["rsus"] == r and x["range_m"] == rng
                    and x["fixed_association"] == fixed and abs(x["loss"] - loss) < 1e-9)

    def mrun(n, mode, duty, fixed):
        return next(x for x in mod if x["n_aids"] == n and x["mode"] == mode and x["duty"] == duty and x["fixed_association"] == fixed)

    def dr(s, v):
        return next(d for d in gdr["drift"] if d["scenario"] == s and d["variant"] == v)

    def gk(g, r, h):
        return grd[f"g{g}_r{r}_h{h}"]["metrics"]

    tables = {}
    rows = [("Default", "default", []), ("Recall x0.9", "rec_scale_0.9", [0.9]), ("Recall x0.8", "rec_scale_0.8", [0.8]),
            ("0.5 false", "fp_rate_0.5", [0.5]), ("2.5 false", "fp_rate_2.5", [2.5]), ("4.0 false", "fp_rate_4.0", [4.0]),
            ("Box noise 2%", "box_sig_0.02", [2]), ("Box noise 8%", "box_sig_0.08", [8]),
            ("Class confusion 5%", "confusion_0.05", [5]), ("Class confusion 25%", "confusion_0.25", [25])]
    t = [(key, lead + pr(det[n]["metrics"], "PM") + pr(det[n]["metrics"], "C") + [(det[n]["metrics"]["C"]["hard_share"][0], 3)]) for key, n, lead in rows]
    t.append(("Camera at 1.3", [1.3] + pr(hgt["hc1.3_g5"]["metrics"], "PM") + pr(hgt["hc1.3_g5"]["metrics"], "C")))
    tables["tab:det"] = (t, 1)

    t = []
    for err, prefix in (("i.i.d.", "pose_iid_g"), ("GM", "pose_gm_g"), ("GM + bias", "pose_gm_bias_g")):
        for g in (3, 5, 10):
            m = tmp[f"{prefix}{g}"]["metrics"]
            t.append((f"{err} {g} m", [g] + sum((pr(m, k) for k in ("PM", "C", "SM", "CT")), [])))
    for r in ("0.1", "0.25", "1.0"):
        m = tmp[f"rho_{r}"]["metrics"]
        t.append((f"GM 5 m, rho={r}", [5, float(r)] + sum((pr(m, k) for k in ("PM", "C", "SM", "CT")), [])))
    p = tmp["pose_gm_g5"]["metrics"]["pose"]
    q = mis["ms_base"]["metrics"]["pose"]
    t.append(("Position RMSE, GM 5 m", [5, (p["raw"]["rms"], 2), (p["C"]["rms"], 2), (p["CT"]["rms"], 2)]))
    t.append(("raw GNSS, same frames", [(q["raw_on_C"]["rms"], 2), (q["raw_on_CT"]["rms"], 2)]))
    tables["tab:temporal"] = (t, 2)

    t = []
    for key, name, lead in (("As assumed", "ms_base", [5, 30, 60]), ("3x faster", "ms_tau_short", [3, 10, 20]), ("4x slower", "ms_tau_long", [4, 120, 240]),
                            ("Under-reported, 8", "ms_sig_under", [8]), ("Over-reported, 3", "ms_sig_over", [3]), ("8 m and 3x faster", "ms_under_short", [8, 3])):
        m = mis[name]["metrics"]
        t.append((key, lead + pr(m, "C") + pr(m, "CT") + [(m["pose"]["CT"]["rms"], 2), (m["pose"]["raw_on_CT"]["rms"], 2)]))
    tables["tab:misspec"] = (t, 1)

    t = []
    for key, name, lead in (("No latency, no loss", "lat_none", []), ("Latency, dead reckoning", "lat_dr", []), ("Latency, raw reports", "lat_raw", []),
                            ("30% loss, dead reckoning", "lat_dr_loss30", [30]), ("30% loss, raw reports", "lat_raw_loss30", [30])):
        m = tmp[name]["metrics"]
        t.append((key, lead + pr(m, "C") + pr(m, "CT")))
    rho = 0.5
    for key, name in (("Ghosts, 17%", "ghost_0.1"), ("Ghosts, 38%", "ghost_0.3"), ("Shifted, 10%", "shift_0.1"), ("Shifted, 30%", "shift_0.3")):
        m, c = tmp[name]["metrics"], tmp[name]["config"]
        lead = [(100 * c["ghost_rate"] / (rho + c["ghost_rate"]), 0)] if "ghost_rate" in c else [(100 * c["shift_frac"], 0)]
        t.append((key, lead + pr(m, "C") + pr(m, "CT") + [(m["C"]["wrong_from_spoof_per_1k"][0], 1), (m["CT"]["wrong_from_spoof_per_1k"][0], 1)]))
    tables["tab:robust"] = (t, 1)

    t = []
    for key, args, lead in (("30 / 6, 150 m", (30, 6, 150, False), [30, 6, 150]), ("30 / 6, fixed", (30, 6, 150, True), [30, 6]),
                            ("30 / 6, 100 m", (30, 6, 100, False), [30, 6, 100]), ("30 / 6, 250 m", (30, 6, 250, False), [30, 6, 250]),
                            ("30 / 12, 150 m", (30, 12, 150, False), [30, 12, 150]), ("120 / 12, 150 m", (120, 12, 150, False), [120, 12, 150]),
                            ("300 / 24, 150 m", (300, 24, 150, False), [300, 24, 150])):
        x = run(*args)
        t.append((key, lead + [(x["p_upload_per_aid_round"], 3), (x["cross_rsu_share"], 2), (x["deferred_share"], 3),
                               (x["wan_mb_per_cloud_round_hier"], 1), (x["wan_ratio"], 3)]))
    tables["tab:mobility"] = (t, 1)

    t = []
    for n, r, label, mode, duty in ((30, 6, "random", "iid", 1.0), (30, 6, "continuous", "continuous", 1.0), (30, 6, "habitual", "habitual", 1.0),
                                    (30, 6, "habitual", "habitual", 0.25), (120, 12, "habitual", "habitual", 1.0), (120, 12, "habitual", "habitual", 0.25),
                                    (300, 24, "habitual", "habitual", 1.0), (300, 24, "habitual", "habitual", 0.25)):
        x, f = mrun(n, mode, duty, False), mrun(n, mode, duty, True)
        dkey = "1.0" if duty == 1.0 else "0.25"
        t.append((f"{n} / {r} {label} {dkey}", [n, r, duty, (x["p_upload_per_active_round"], 3), (x["reporters_mean"], 2), (x["deferred_share"], 3),
                                                (x["wan_ratio"], 3), (f["p_upload_per_active_round"], 3), (f["deferred_share"], 3)]))
    tables["tab:modes"] = (t, 3)

    t = []
    for key, v in (("vs. omega_rel", "prev"), ("+ fallback", "prev+fallback"), ("+ anchor", "prev+anchor"), ("+ fallback + anchor", "prev+anchor+fallback")):
        vals = []
        for s in ("plateau", "one_group_decay", "learning"):
            vals += [(dr(s, v)["p_some_group_below_eps"], 3), (dr(s, v)["releases_mean"], 1)]
        t.append((key, vals))
    tables["tab:gatedrift"] = (t, 1)

    figures = {}
    gn = [f"gnss_{e}" for e in (0, 1, 3, 5, 10)]
    rh = [f"rho_{r}" for r in (0.1, 0.25, 0.5, 0.75, 1.0)]
    # fig:mc is no longer typed coordinates: tools/make_fig_extra.py plots it straight from
    # v2x_pseudolabel_mc.json and mc_temporal.json (fig_v2x_labels.pdf), so there is nothing to compare here.
    del gn, rh
    G = (0, 1, 3, 5, 10)
    fig = []
    for pn, key in (("a", "recall_connected"), ("b", "precision")):
        for h in (2, 5, 10):
            fig += [(f"({pn}) C {h} deg", [gk(g, 0.5, h)["C"][key][0] for g in G]), (f"({pn}) PM {h} deg", [gk(g, 0.5, h)["PM"][key][0] for g in G])]
    fig += [(f"(c) C rho {r}", [gk(g, r, 5)["C"]["recall_connected"][0] for g in G]) for r in (0.1, 0.25, 0.5, 0.75, 1.0)]
    figures["fig:grid"] = fig

    def oc(churn, delta, key="p_release_shared"):
        return [next(u for u in goc["uniform"] if u["eps"] == 0.01 and u["churn"] == churn and u["n"] == n and u["delta"] == delta)[key]
                for n in (500, 1000, 2000, 4000)]

    def ocreg(delta_reg):
        return [next(u for u in goc["regression"] if u["eps"] == 0.01 and u["churn"] == 0.05 and u["n"] == n and abs(u["delta_reg"] - delta_reg) < 1e-12)["p_release_shared"]
                for n in (500, 1000, 2000, 4000)]

    figures["fig:gate"] = [("+1 pt churn 2%", oc(0.02, 0.01)), ("+1 pt churn 5%", oc(0.05, 0.01)), ("+1 pt churn 5% indep", oc(0.05, 0.01, "p_release_indep")),
                           ("+1 pt churn 10%", oc(0.10, 0.01)), ("no change churn 5%", oc(0.05, 0.0)), ("one group -2 pt churn 5%", ocreg(-0.02))]
    eps_rows = {r["eps"]: r for r in thc["eps"]}

    def dp_excess(d_in, z):
        base = next(r for r in thd if r["d_in"] == d_in and r["z"] == 0.0)["excess"][0]
        return next(r for r in thd if r["d_in"] == d_in and r["z"] == z)["excess"][0] - base

    figures["fig:theory"] = [("(a) grad norm vs eps", [eps_rows[e]["grad2"][0] for e in (0.05, 0.1, 0.2, 0.3, 0.4)]),
                             ("(b) d=1005", [dp_excess(200, z) for z in (0.5, 1.0, 2.0)]),
                             ("(b) d=105", [dp_excess(20, z) for z in (0.5, 1.0, 2.0)])]

    # Sentences ---------------------------------------------------------------------------------
    g0, g5, g10 = (st[f"gnss_{e}"]["metrics"] for e in (0, 5, 10))
    tg5 = tmp["pose_gm_g5"]["metrics"]
    msb = mis["ms_base"]["metrics"]
    ratio = g5["C"]["recall_connected"][0] / g5["PM"]["recall_connected"][0]
    allg = list(grd.values())
    n_more = sum(v["metrics"]["C"]["recall_connected"][0] > v["metrics"]["PM"]["recall_connected"][0] for v in allg)
    n_twice = sum(v["metrics"]["C"]["recall_connected"][0] >= 2 * v["metrics"]["PM"]["recall_connected"][0] for v in allg)
    n_prec = sum(v["metrics"]["C"]["precision"][0] >= v["metrics"]["PM"]["precision"][0] for v in allg)
    worst_prec_gap = max(100 * (v["metrics"]["PM"]["precision"][0] - v["metrics"]["C"]["precision"][0]) for v in allg)
    worst_rec_gap = max(100 * (v["metrics"]["PM"]["recall_connected"][0] - v["metrics"]["C"]["recall_connected"][0]) for v in allg)
    det_ratio = min(d["metrics"]["C"]["recall_connected"][0] / d["metrics"]["PM"]["recall_connected"][0] for d in det.values())
    base, fixed = run(30, 6, 150), run(30, 6, 150, True)
    dp = {r["noise_multiplier"]: r for r in dpb["results"]}
    dpu30 = next(r for r in dpu["rows"] if r["n_aids"] == 30 and r["eps"] == 8.0 and r["q"] == 1.0)
    dpu300 = next(r for r in dpu["rows"] if r["n_aids"] == 300 and r["eps"] == 8.0 and r["q"] == 1.0)
    fbk = {(f["regression"], f["variant"]): f for f in gdr["fallback"]}
    eps_g = [eps_rows[e]["grad2"][0] for e in (0.05, 0.1, 0.2, 0.4)]
    kap = [r["grad2"][0] for r in thc["kappa"]]
    kap_var = 100 * (max(kap) - min(kap)) / min(kap)
    habit25 = [mrun(n, "habitual", 0.25, False)["deferred_share"] for n in (30, 120, 300)]
    ocfb = {f["alpha"]: f["p_release_shared"] for f in goc["fallback"] if f["n"] == 2000}  # churn and gain scale with alpha
    optimism = max(100 * d["safety_set_optimism_mean"] for d in gdr["drift"])
    C = lambda name, key="recall_connected", meth="C": tmp[name]["metrics"][meth][key][0]  # noqa: E731

    claims = [
        ("C4: update share", r"whose FP16 updates are {v}\% of the FP32 model's size", (100 * bud["head_update_bytes_fp16"] / bud["full_model_bytes_fp32"], 1)),
        ("related work: trained share", r"it trains under a tenth of the parameters", []),
        ("conclusion: update share", r"keeps the trained dimension and every update small ({v}\% of the model)", (100 * bud["head_update_bytes_fp16"] / bud["full_model_bytes_fp32"], 1)),
        ("conclusion: gate per round", r"with probability at most {v}\% per round", ("le", 100 * max(ocreg(-0.02)))),
        ("conclusion: gate over 100 rounds", r"it keeps at least {v}\% of runs within the tolerance of the first release",
         (min(100 * (1 - dr(s, "prev+anchor+fallback")["p_some_group_below_eps"]) for s in ("plateau", "one_group_decay", "learning")), 0)),
        ("discussion: participation", r"raises participation from between {v} and {v} to between {v} and {v} under all three mobility models",
         [(min(mrun(30, m, 1.0, True)["p_upload_per_active_round"] for m in ("iid", "continuous", "habitual")), 2),
          (max(mrun(30, m, 1.0, True)["p_upload_per_active_round"] for m in ("iid", "continuous", "habitual")), 2),
          (min(mrun(30, m, 1.0, False)["p_upload_per_active_round"] for m in ("iid", "continuous", "habitual")), 2),
          (max(mrun(30, m, 1.0, False)["p_upload_per_active_round"] for m in ("iid", "continuous", "habitual")), 2)]),
        ("RQ3: modes participation and deferral", r"reproduce the participation ({v} to {v}) and deferral ({v} to {v})",
         [(min(mrun(30, m, 1.0, False)["p_upload_per_active_round"] for m in ("iid", "continuous", "habitual")), 2),
          (max(mrun(30, m, 1.0, False)["p_upload_per_active_round"] for m in ("iid", "continuous", "habitual")), 2),
          (min(mrun(30, m, 1.0, False)["deferred_share"] for m in ("iid", "continuous", "habitual")), 2),
          (max(mrun(30, m, 1.0, False)["deferred_share"] for m in ("iid", "continuous", "habitual")), 2)]),
        # main configuration (aids may leave their home town, Table tab:mobility) vs the home-town models of tab:modes
        ("RQ3: modes cross-RSU", r"falls from {v} to between {v} and {v}",
         [(base["cross_rsu_share"], 2), (min(mrun(30, m, 1.0, False)["cross_rsu_share"] for m in ("iid", "continuous", "habitual")), 2),
          (max(mrun(30, m, 1.0, False)["cross_rsu_share"] for m in ("iid", "continuous", "habitual")), 2)]),
        ("budget: 80-class parameters", r"reproduces Ultralytics' published {v} parameters of YOLO11-s for 80 classes", load("yolo11_budget.json")["nc80"]["total_params"]),
        ("model budget", r"backbone and neck, {v}\,M parameters for YOLO11-s) and $\omega$ the detection head ({v}\,M parameters, {v}\% of the model",
         [(bud["trunk_params"] / 1e6, 2), (bud["head_params"] / 1e6, 2), (100 * bud["head_fraction"], 1)]),
        ("C2: precision and ratio", r"keeps pseudo-label precision at {v} under 5\,m GNSS error while labeling {v}$\times$", [(g5["C"]["precision"][0], 2), (ratio, 1)]),
        ("corollary: floor ratio", r"$(0.813/0.123)^2\approx{v}$", ((1 - g5["P"]["precision"][0]) / (1 - g5["C"]["precision"][0])) ** 2),
        ("corollary: precisions", r"at the {v} precision of consensus labels than at the {v} precision of direct projection", [g5["C"]["precision"][0], g5["P"]["precision"][0]]),
        ("corollary: PM", r"Against matching without pose correction (precision {v}) the ratio is only $(0.158/0.123)^2\approx{v}$",
         [g5["PM"]["precision"][0], ((1 - g5["PM"]["precision"][0]) / (1 - g5["C"]["precision"][0])) ** 2]),
        ("corollary: head params", r"Training the {v} head parameters instead of all {v} parameters of YOLO11-s reduces it by a factor of {v}",
         [bud["head_trainable_params"], bud["total_params"], bud["total_params"] / bud["head_trainable_params"]]),
        ("RQ1: P precision", r"yields {v} precision even with perfect GNSS", g0["P"]["precision"][0]),
        ("RQ1: P at 5 m", r"break Eq.~(\ref{eq:budget}), and {v} at 5\,m", g5["P"]["precision"][0]),
        ("RQ1: PM", r"keeps precision at {v} but finds only {v}\% of the connected vehicles at 5\,m and {v}\% at 10\,m",
         [g5["PM"]["precision"][0], (100 * g5["PM"]["recall_connected"][0], 0), (100 * g10["PM"]["recall_connected"][0], 0)]),
        ("RQ1: C", r"raises precision to {v} and recall to {v}\% at 5\,m, {v}$\times$ PM, and still labels {v}\% of connected vehicles at 10\,m with {v} precision",
         [g5["C"]["precision"][0], (100 * g5["C"]["recall_connected"][0], 0), (ratio, 1), (100 * g10["C"]["recall_connected"][0], 0), g10["C"]["precision"][0]]),
        ("RQ1: unconnected share", r"they make up {v}\% of correct C labels", (100 * msb["C"]["correct_on_unconnected_share"][0], 0)),
        ("RQ1: identity precision", r"is {v} for C and {v} for CT", [msb["C"]["identity_precision"][0], msb["CT"]["identity_precision"][0]]),
        ("RQ1: hard share", r"About {v}\% of correct labels fall on vehicles the deployed detector missed", (100 * g5["C"]["hard_share"][0], 0)),
        ("grid: more / twice", r"C labels more connected vehicles than PM in {v} of the 75 settings and at least twice as many in {v}", [n_more, n_twice]),
        ("grid: recall exceptions", r"the two methods are within {v} points", (worst_rec_gap, 1)),
        ("grid: precision settings", r"C is at least as precise as PM in {v} settings", n_prec),
        ("grid: precision gap", r"(at most {v} points lower)", (worst_prec_gap, 1)),
        ("grid: compass", r"PM recall falls from {v} to {v} and C recall only from {v} to {v}",
         [gk(5, 0.5, 2)["PM"]["recall_connected"][0], gk(5, 0.5, 10)["PM"]["recall_connected"][0], gk(5, 0.5, 2)["C"]["recall_connected"][0], gk(5, 0.5, 10)["C"]["recall_connected"][0]]),
        ("detector: min ratio", r"C keeps at least {v}$\times$ the recall of PM in every setting", (math.floor(det_ratio * 10) / 10, 1)),
        ("detector: false proposals", r"from {v} at 0.5 to {v} at 4 false proposals per frame, where it equals PM within {v} points",
         [det["fp_rate_0.5"]["metrics"]["C"]["precision"][0], det["fp_rate_4.0"]["metrics"]["C"]["precision"][0],
          (100 * abs(det["fp_rate_4.0"]["metrics"]["C"]["precision"][0] - det["fp_rate_4.0"]["metrics"]["PM"]["precision"][0]), 1)]),
        ("detector: box noise", r"box noise of 8\% costs {v} points of precision and {v} points of recall",
         [(100 * (det["default"]["metrics"]["C"]["precision"][0] - det["box_sig_0.08"]["metrics"]["C"]["precision"][0]), 0),
          (100 * (det["default"]["metrics"]["C"]["recall_connected"][0] - det["box_sig_0.08"]["metrics"]["C"]["recall_connected"][0]), 0)]),
        ("detector: hard share", r"from {v}\% to {v}\%, which is when V2X labels are most valuable",
         [(100 * det["default"]["metrics"]["C"]["hard_share"][0], 0), (100 * det["rec_scale_0.8"]["metrics"]["C"]["hard_share"][0], 0)]),
        ("detector: camera height", r"slightly improves C, to {v} precision and {v}\% recall at 5\,m",
         [hgt["hc1.3_g5"]["metrics"]["C"]["precision"][0], (100 * hgt["hc1.3_g5"]["metrics"]["C"]["recall_connected"][0], 0)]),
        ("temporal: iid 10 m", r"at 10\,m CT even loses {v} points of recall", (100 * (C("pose_iid_g10") - C("pose_iid_g10", meth="CT")), 0)),
        ("temporal: GM 5 m", r"CT raises recall of connected vehicles from {v} to {v} and precision from {v} to {v}",
         [tg5["C"]["recall_connected"][0], tg5["CT"]["recall_connected"][0], tg5["C"]["precision"][0], tg5["CT"]["precision"][0]]),
        ("temporal: pose frames", r"accepts a pose on {v}\% of the frames instead of {v}\%", [(100 * tg5["pose"]["CT"]["frames"], 0), (100 * tg5["pose"]["C"]["frames"], 0)]),
        ("temporal: RMSE", r"lowers the RMS position error from {v}\,m (raw GNSS) to {v}\,m", [tg5["pose"]["raw"]["rms"], tg5["pose"]["CT"]["rms"]]),
        ("temporal: C RMSE", r"the corrected position ({v}\,m) is slightly worse than raw GNSS ({v}\,m)", [tg5["pose"]["C"]["rms"], msb["pose"]["raw_on_C"]["rms"]]),
        ("temporal: bias and 10 m", r"unreported 3\,m multipath bias ({v} to {v} recall) and at 10\,m ({v} to {v})",
         [C("pose_gm_bias_g5"), C("pose_gm_bias_g5", meth="CT"), C("pose_gm_g10"), C("pose_gm_g10", meth="CT")]),
        ("temporal: SM", r"is the most precise labeler ({v})", tg5["SM"]["precision"][0]),
        ("temporal: low penetration", r"it raises recall from {v} to {v} and from {v} to {v}", [C("rho_0.1"), C("rho_0.1", meth="CT"), C("rho_0.25"), C("rho_0.25", meth="CT")]),
        ("misspec: fast", r"CT still reaches {v} precision and {v} recall", [mis["ms_tau_short"]["metrics"]["CT"]["precision"][0], mis["ms_tau_short"]["metrics"]["CT"]["recall_connected"][0]]),
        ("misspec: worst case", r"it keeps {v} and {v} against {v} and {v} for C",
         [mis["ms_under_short"]["metrics"]["CT"]["precision"][0], mis["ms_under_short"]["metrics"]["CT"]["recall_connected"][0],
          mis["ms_under_short"]["metrics"]["C"]["precision"][0], mis["ms_under_short"]["metrics"]["C"]["recall_connected"][0]]),
        ("misspec: over-reported", r"neither correction improves position ({v}\,m for CT against {v}\,m raw)",
         [mis["ms_sig_over"]["metrics"]["pose"]["CT"]["rms"], mis["ms_sig_over"]["metrics"]["pose"]["raw_on_CT"]["rms"]]),
        ("latency: raw", r"from {v} to {v} and CT recall from {v} to {v}", [C("lat_dr"), C("lat_raw"), C("lat_dr", meth="CT"), C("lat_raw", meth="CT")]),
        ("latency: loss raw", r"the drop grows to {v} and {v}", [C("lat_raw_loss30"), C("lat_raw_loss30", meth="CT")]),
        ("latency: loss DR", r"30\% loss costs only {v} and {v} points ({v} and {v}",
         [(100 * (C("lat_dr") - C("lat_dr_loss30")), 0), (100 * (C("lat_dr", meth="CT") - C("lat_dr_loss30", meth="CT")), 0), C("lat_dr_loss30"), C("lat_dr_loss30", meth="CT")]),
        ("RQ2: update size", r"The head update is {v}\,MB in FP16, {v}\% of the {v}\,MB FP32 model.",
         [(bud["head_update_bytes_fp16"] / 1e6, 2), (100 * bud["head_update_bytes_fp16"] / bud["full_model_bytes_fp32"], 1),
          (bud["full_model_bytes_fp32"] / 1e6, 1)]),
        ("RQ2: participation", r"aids complete an upload in {v}\% of edge rounds, and {v}\% of uploads carry an update that waited for a later contact",
         [(100 * base["p_upload_per_aid_round"], 0), (100 * base["stale_share"], 0)]),
        ("RQ2: cross-RSU", r"{v}\% of uploads reach a different RSU from the one that served the model", (100 * base["cross_rsu_share"], 0)),
        ("RQ2: reporters", r"RSUs see {v} reporters per edge round on average and defer {v}\% of edge rounds for lack of three aids, so {v} of the 6 RSUs",
         [(base["reporters_mean"], 1), (100 * base["deferred_share"], 0), (base["active_rsus_per_cloud"], 1)]),
        ("RQ2: WAN", r"{v}\,MB of wide-area traffic against {v}\,MB for flat FL at full participation, {v}\% of it",
         [(base["wan_mb_per_cloud_round_hier"], 1), (base["wan_mb_per_cloud_round_flat"], 1), (100 * base["wan_ratio"], 1)]),
        ("RQ2: all-hop", r"aids and RSUs send {v}\,MB per cloud round, close to the {v}\,MB of flat FL", [(base["all_hop_mb_per_cloud_round_hier"], 1), (base["wan_mb_per_cloud_round_flat"], 1)]),
        ("RQ2: fixed association", r"halves participation to {v}\% and doubles deferral to {v}\%", [(100 * fixed["p_upload_per_aid_round"], 0), (100 * fixed["deferred_share"], 0)]),
        ("RQ2: windows", r"Contact windows last {v}\,s in the median and {v}\,s at the 10th percentile", [(base["window_s_median"], 0), (base["window_s_p10"], 0)]),
        ("RQ2: transfer", r"than the {v}\,s ({v}\,s at 30\% loss) a transfer needs", [(base["upload_need_s"], 1), (run(30, 6, 150, False, 0.3)["upload_need_s"], 1)]),
        ("RQ2: range", r"RSU range does ({v}\% at 100\,m, {v}\% at 250\,m)", [(100 * run(30, 6, 100)["p_upload_per_aid_round"], 0), (100 * run(30, 6, 250)["p_upload_per_aid_round"], 1)]),
        ("RQ2: 120/12", r"with 120 aids and 12 RSUs only {v}\% of RSU rounds defer and the wide-area share falls to {v}\%",
         [(100 * run(120, 12, 150)["deferred_share"], 1), (100 * run(120, 12, 150)["wan_ratio"], 1)]),
        ("RQ2: 300/24", r"with 300 aids and 24 RSUs to {v}\% and {v}\%", [(100 * run(300, 24, 150)["deferred_share"], 1), (100 * run(300, 24, 150)["wan_ratio"], 1)]),
        ("RQ2: 120/24", r"whereas 120 aids on 24 RSUs still defer {v}\% of rounds", (100 * run(120, 24, 150)["deferred_share"], 0)),
        ("RQ2: 30/12", r"raises participation to {v}\% but deferral to {v}\%", [(100 * run(30, 12, 150)["p_upload_per_aid_round"], 0), (100 * run(30, 12, 150)["deferred_share"], 0)]),
        ("RQ2: duty cycle", r"With a 25\% outdoor duty cycle, deferral rises to {v}, {v} and {v} for 30, 120 and 300 aids", [(v, 2) for v in habit25]),
        ("DP: frame-level eps", r"gives $\varepsilon={v}$, {v} and {v} at $z=0.6$, 1.0 and 1.5 with the R\'enyi accountant",
         [(dp[0.6]["eps_rdp"], 1), (dp[1.0]["eps_rdp"], 1), (dp[1.5]["eps_rdp"], 1)]),
        ("DP: PLD", r"accountant gives {v}, {v} and {v}", [(dp[0.6]["eps_pld"], 1), (dp[1.0]["eps_pld"], 1), (dp[1.5]["eps_pld"], 1)]),
        ("DP: user level 30", r"requires a noise multiplier of {v} with 30 aids, so the noise added to an RSU aggregate of five clipped updates has {v} times the norm",
         [(dpu30["z"], 1), (dpu30["noise_to_signal_norm"], 0)]),
        ("DP: user level 300", r"the factor is still {v}", (dpu300["noise_to_signal_norm"], 0)),
        ("DP: per direction", r"the noise is only about {v} times the aggregate", (dpu30["noise_to_signal_norm"] / math.sqrt(dpu["d"]), 1)),
        ("spoof: ghosts", r"lower C precision from {v} to {v} and {v}", [tg5["C"]["precision"][0], C("ghost_0.1", "precision"), C("ghost_0.3", "precision")]),
        ("spoof: shifted", r"lowers it to {v} and {v} and costs {v} and {v} points of recall",
         [C("shift_0.1", "precision"), C("shift_0.3", "precision"), (100 * (C("lat_dr") - C("shift_0.1")), 0), (100 * (C("lat_dr") - C("shift_0.3")), 0)]),
        ("spoof: wrong labels", r"({v} and {v} per 1{,}000 labels with 38\% ghosts)",
         [(C("ghost_0.3", "wrong_from_spoof_per_1k"), 0), (C("ghost_0.3", "wrong_from_spoof_per_1k", "PM"), 0)]),
        ("spoof: CT", r"it keeps {v} precision and {v} recall with 38\% ghosts and produces {v} wrong spoofed labels",
         [C("ghost_0.3", "precision", "CT"), C("ghost_0.3", meth="CT"), (C("ghost_0.3", "wrong_from_spoof_per_1k", "CT"), 0)]),
        ("gate: regressing 2 pt", r"is released with probability at most {v}\% for every safety-set size", ("le", 100 * max(ocreg(-0.02)))),
        ("gate: regressing 1 pt", r"exactly the tolerance, with at most {v}\%", ("le", 100 * max(ocreg(-0.01)))),
        ("gate: release probabilities", r"released with probability {v} at 500 instances per group, {v} at 1,000, {v} at 2,000 and {v} at 4,000",
         [(v, 2) for v in oc(0.05, 0.01)]),
        ("gate: independent", r"and {v} at 2,000 when they are independent", (oc(0.05, 0.01, "p_release_indep")[2], 2)),
        ("gate: fallback 10% churn", r"raises this to {v} for $\alpha=\tfrac12$ and {v} for $\alpha=\tfrac14$", [(ocfb[0.5], 2), (ocfb[0.25], 2)]),
        ("gate: fallback multiplicity", r"is released as an interpolant in {v}\% of rounds, and the released head regresses by more than the 1-point tolerance in {v}\% of rounds ({v}\% and {v}\% at 3 and 4 points), against {v}\% without fallback",
         [(100 * fbk[(-0.02, "prev+fallback")]["p_release"], 0), (100 * fbk[(-0.02, "prev+fallback")]["p_release_beyond_tolerance"], 1),
          (100 * fbk[(-0.03, "prev+fallback")]["p_release_beyond_tolerance"], 1), (100 * fbk[(-0.04, "prev+fallback")]["p_release_beyond_tolerance"], 1),
          (100 * fbk[(-0.02, "prev")]["p_release_beyond_tolerance"], 1)]),
        ("gate: drift", r"{v}\% of runs end with some group more than 1 point below the first release, and {v}\% when one group slowly degrades",
         [(100 * dr("plateau", "prev+fallback")["p_some_group_below_eps"], 0), (100 * dr("one_group_decay", "prev+fallback")["p_some_group_below_eps"], 0)]),
        ("gate: anchor", r"cuts these to {v}\% and {v}\%", [(100 * dr("plateau", "prev+anchor+fallback")["p_some_group_below_eps"], 1),
                                                           (100 * dr("one_group_decay", "prev+anchor+fallback")["p_some_group_below_eps"], 0)]),
        ("gate: releases", r"yields only {v} releases in 100 rounds, against {v} with fallback and anchor",
         [(dr("learning", "prev")["releases_mean"], 1), (dr("learning", "prev+anchor+fallback")["releases_mean"], 1)]),
        ("gate: safety-set optimism", r"overstate the cumulative gain of the released head by up to {v} point", ("le", optimism)),
        ("theory: doubling", r"multiplies it by {v}, {v} and {v} between 0.05 and 0.4", [(eps_g[1] / eps_g[0], 1), (eps_g[2] / eps_g[1], 1), (eps_g[3] / eps_g[2], 1)]),
        ("theory: kappa", r"changes it by less than {v}\%", ("lt", kap_var)),
        ("theory: DP d=1005", r"it is ${v}$, ${v}$ and ${v}\times10^{-3}$", [(1e3 * dp_excess(200, 0.5), 2), (1e3 * dp_excess(200, 1.0), 2), (1e3 * dp_excess(200, 2.0), 2)]),
        ("theory: DP d=105", r"it stays at or below ${v}\times10^{-3}$", ("le", 1e3 * max(dp_excess(20, z) for z in (0.5, 1.0, 2.0)))),
        ("theory: DP ratio", r"the larger model pays {v} times more", (dp_excess(200, 2.0) / dp_excess(20, 2.0), 0)),
        ("theory: clipping", r"lowered the excess loss caused by 10\% flipped labels from {v} to {v}",
         [(eps_rows[0.1]["excess"][0], 2), (next(r for r in thd if r["d_in"] == 20 and r["z"] == 0.0)["excess"][0], 3)]),
    ]
    return tables, figures, claims + dataset_claims() + device_claims(bud) + phase2_claims() + phase4_claims() + phase5_claims() + fl_claims()


DEVICE = Path(os.environ.get("FEDV2X_DEVICE_MEASUREMENTS", str(ROOT.parent / "private_evidence" / "device_measurements.json")))


def dataset_claims():
    """Image counts of the merged dataset (data/taxonomy/dataset_counts.json)."""
    ds = json.loads((ROOT / "data" / "taxonomy" / "dataset_counts.json").read_text(encoding="utf8"))
    tr, va = ds["images_by_split_and_source"]["train"], ds["images_by_split_and_source"]["val"]
    return [
        ("tab:datasets COCO", r"COCO 2017 & {v} / {v}", [tr["coco"], va["coco"]]),
        ("tab:datasets Open Images", r"Open Images V7 & {v} / {v}", [tr["oiv7"], va["oiv7"]]),
        ("tab:datasets Vistas", r"Mapillary Vistas v2 & {v} / {v}", [tr["vistas"], va["vistas"]]),
        ("tab:datasets MTSD", r"MTSD & {v} / {v}", [tr["mtsd"], va["mtsd"]]),
        ("tab:datasets Roboflow", r"Roboflow, {v} projects & {v} / {v}", [ds["roboflow_projects"], tr["roboflow"], va["roboflow"]]),
        ("methods: dataset size", r"into {v} images with 152 classes", ds["train_total"] + ds["val_total"]),
    ]


def device_claims(bud):
    """Numbers measured on the authors' device. The measurement file is not public; without it these checks are skipped."""
    if not DEVICE.exists():
        return []
    ev = json.loads(DEVICE.read_text(encoding="utf8"))
    rv, ap = ev["revalidation"]["overall"], ev["per_class_ap50_rerun"]
    ds = json.loads((ROOT / "data" / "taxonomy" / "dataset_counts.json").read_text(encoding="utf8"))
    boots, first = ev["host_boot_s"], ev["mcu_first_reading_ms"]
    bin_mb = ev["binary"]["bytes"] / 1e6
    assert ev["binary"]["contexts"] == 5, "the paper says five contexts"
    return [
        ("device: RQ4 accuracy", r"trained centrally on the {v} training images of the merged set and reaches {v} mAP50 and {v} mAP50-95 on its {v} validation images",
         [ds["train_total"], (rv["mAP50"], 3), (rv["mAP50_95"], 3), ds["val_total"]]),
        ("device: person AP50", r"person and car reach {v} AP50", (ap["person"], 2)),
        ("device: car AP50", r"person and car reach {v} AP50", (ap["car"], 2)),
        ("device: hazard AP50", r"stairs {v} and pothole {v}, but curb only {v} and crosswalk {v}",
         [(ap["stairs"], 2), (ap["pothole"], 2), (ap["curb"], 2), (ap["crosswalk"], 2)]),
        ("device: binary size", r"(five contexts, {v}\,MB)", (bin_mb, 1)),
        ("device: benchmarks", r"the model runs at {v} frames/s in the vendor benchmark, against {v} frames/s for the 80-class YOLOv8-s",
         [ev["npu_benchmark_detector_fps"]["value"], (ev["npu_benchmark_yolov8s"]["fps"], 1)]),
        ("device: service rate", r"The platform runs it at {v} frames/s to save power", (ev["camera_loop_fps"]["value"], 0)),
        ("device: safety loop and boot", r"The warning microcontroller delivers its first range reading {v}\,s after power-on, whereas the host computer takes {v} to {v}\,s to boot",
         [(max(first) / 1e3, 2), (min(boots), 1), (max(boots), 1)]),
        ("device: vision outage", r"vision stopped for {v}\,h {v}\,min", [ev["vision_outage"]["hours"], ev["vision_outage"]["minutes"]]),
        ("device: Tier 1 FOV", r"effective horizontal field of view of about {v}$^\circ$", (ev["camera_hfov_deg"], 0)),
        ("device: Tier 1 rate", r"It runs a 152-class YOLO11-s detector at {v} frames/s", (ev["camera_loop_fps"]["value"], 0)),
        ("device: conclusion rate", r"runs on a 13-TOPS NPU at {v} frames/s", ev["npu_benchmark_detector_fps"]["value"]),
        ("tab:platform detector", r"YOLO11-s, 152 classes, {v}\,M param.", (bud["total_params"] / 1e6, 2)),
        ("tab:platform accuracy", r"& {v} / {v} ({v} img.)", [(rv["mAP50"], 3), (rv["mAP50_95"], 3), ds["val_total"]]),
        ("tab:platform AP50 1", r"AP50 person / car / bus & {v} / {v} / {v}", [(ap["person"], 3), (ap["car"], 3), (ap["bus"], 3)]),
        ("tab:platform AP50 2", r"AP50 stairs / pothole / hole & {v} / {v} / {v}", [(ap["stairs"], 3), (ap["pothole"], 3), (ap["open hole"], 3)]),
        ("tab:platform AP50 3", r"AP50 curb / crosswalk / pole & {v} / {v} / {v}", [(ap["curb"], 3), (ap["crosswalk"], 3), (ap["pole"], 3)]),
        ("tab:platform binary", r"NPU binary & INT8, {v}\,MB, {v} contexts", [(bin_mb, 1), ev["binary"]["contexts"]]),
        ("tab:platform benchmark v3", r"NPU benchmark, this model & {v} frames/s", ev["npu_benchmark_detector_fps"]["value"]),
        ("tab:platform benchmark v8s", r"Same, YOLOv8-s (80 classes) & {v} frames/s, {v}\,ms", [(ev["npu_benchmark_yolov8s"]["fps"], 1), (ev["npu_benchmark_yolov8s"]["latency_ms"], 1)]),
        ("tab:platform loop", r"Camera loop (service) & {v} frames/s", (ev["camera_loop_fps"]["value"], 0)),
        ("tab:platform FOV", r"Horizontal field of view & {v}$^\circ$", (ev["camera_hfov_deg"], 0)),
        ("tab:platform first reading", r"First range reading at boot & {v}\,s ({v} trials)", [(max(first) / 1e3, 2), len(first)]),
        ("tab:platform interval", r"MCU-to-host interval & P50 {v}\,ms, P99 {v}\,ms", [(ev["mcu_interval_ms"]["p50"], 1), (ev["mcu_interval_ms"]["p99"], 1)]),
        ("tab:platform range noise", r"Forward range noise & $\sigma={v}$\,mm ({v} readings)", [(ev["range_forward"]["stdev_mm"], 2), ev["range_forward"]["n"]]),
        ("tab:platform boot", r"Host boot (systemd) & {v} to {v}\,s ({v} boots)", [(min(boots), 1), (max(boots), 1), len(boots)]),
    ]


def phase2_claims():
    """Numbers decided or measured in Phase 2 (federation, round 0, cache, taxonomy, manifest, metric check)."""
    res = ROOT / "results" / "phase2"
    if not (res / "federation_stats.json").exists():
        return []
    import yaml  # noqa: PLC0415
    st = json.loads((res / "federation_stats.json").read_text())
    fed = json.loads((res / "federation.json").read_text())
    nd = json.loads((res / "near_duplicates.json").read_text())
    lk = json.loads((res / "leakage_check.json").read_text())
    ce = json.loads((res / "cache_equivalence.json").read_text())
    xc = json.loads((res / "xcheck_ultralytics.json").read_text())
    man = json.loads((ROOT / "docs" / "dataset_manifest_summary.json").read_text())
    tax = yaml.safe_load((ROOT / "src" / "data" / "hazard_groups.yaml").read_text(encoding="utf-8"))
    r0 = json.loads((res / "round0_training.json").read_text())
    sel = json.loads(Path("E:/fedv2x_work/round0_cache/selection.json").read_text()) \
        if Path("E:/fedv2x_work/round0_cache/selection.json").exists() else {"val": [None] * 1000}
    rf_sources = sorted({k for c in fed["clients"].values() for k in c["sources"] if k.startswith("rf_")})
    ev = st["clients_evaluated_per_group"]
    assert lk["pass"] and ce["pass"] and man["pass"], "a Phase 2 check failed"
    assert ev["vehicle"] == ev["person"] == ev["curb"], "the paper states one number for vehicle, person and curb"
    assert min(st["safety_instances"].values()) >= 2000
    sim = (ROOT / "src" / "fl" / "simulate.py").read_text(encoding="utf-8")
    import re as _re  # noqa: PLC0415

    def default(flag):
        return int(_re.search(r'"--' + flag + r'", type=int, default=(\d+)', sim).group(1))
    return [
        ("federation: aids", r"{v} aids hold Mapillary Vistas images and {v} hold images of {v} Roboflow projects",
         [fed["params"]["vistas_clients"], fed["params"]["rf_clients"], len(rf_sources)]),
        ("federation: splits", r"({v}, {v} and {v} in total)", [st["images"]["train"], st["images"]["val"], st["images"]["test"]]),
        ("federation: dHash", r"Hamming distance at most {v}, over all {v} images", [nd["max_hamming"], nd["images"]]),
        ("federation: evaluated groups", r"vehicle, person and curb on {v} aids, crosswalk on {v}, pothole on {v}, and stairs, whose street-level labels are scarce, on one aid with {v} test instances",
         [ev["vehicle"], ev["crosswalk"], ev["pothole"], st["pooled_test_instances"]["stairs"]]),
        ("federation: n_min", r"holds at least {v} human-labeled instances of it", st["n_min_test_instances"]),
        ("federation: round 0", r"a 152-class head trained for {v} epochs on {v} server images that no aid holds (validation mAP50 {v} on {v} further server images)",
         [len(r0["epochs"]), r0["train_images"], (r0["epochs"][-1]["val_map50"], 2), len(sel["val"])]),
        ("federation: cache size", r"cached FP16 trunk features ({v}\,MB per image)", (ce["fp16"]["cache_bytes_per_image"] / 1e6, 2)),
        ("federation: cache equality", r"(maximum absolute difference {v} on {v} images)", [ce["fp16"]["max_abs_diff_raw_head_outputs"], ce["n_images"]]),
        ("federation: safety set", r"a safety set of {v} public images", st["safety_images"]),
        ("federation: safety floor", r"with at least {v} human-labeled instances per hazard group", 2000),
        ("federation: rounds", r"$\kappa=2$ and $T={v}$. Flat baselines run {v} rounds",
         [default("edge-rounds") // default("kappa"), default("edge-rounds")]),
        ("tab:hyper local work", r"$S={v}$ steps per edge round, batch {v}", [default("steps"), default("batch")]),
        ("tab:hyper rounds", r"$T={v}$ cloud rounds", default("edge-rounds") // default("kappa")),
        ("datasets: groups", r"to one of {v} groups, six of which", len(tax["groups"])),
        ("datasets: manifest", r"A manifest of all {v} images", man["rows"]),
        ("reproducibility: metric agreement", r"on a fixed {v}-image subset to within {v}", [xc["n_images"], ("le", xc["abs_diff_iou_rule"])]),
    ]


def phase5_claims():
    """Phase 5: range-sensor model and range-label quality on the aids' training images (E8)."""
    res = ROOT / "results" / "phase5"
    if not (res / "e8_range_labels.json").exists():
        return []
    e8 = json.loads((res / "e8_range_labels.json").read_text())
    sm = json.loads((res / "range_sensor_model.json").read_text())
    r, sup = e8["range"], e8["suppression"]
    claims = [
        ("E8 method: bench log", r"In a 60-s bench log of the range sensors all {v} steady-state forward readings were valid, with a standard deviation of {v}",
         [sm["forward"]["n_valid"], (sm["forward"]["sigma_mm"], 2)]),
        ("tab:rangelabels caption", r"Range labels on the aids' {v} training images", r["frames"]),
        ("tab:rangelabels caption frames", r"in the {v} frames with a teacher label", e8["frames_with_signal"]),
        ("E8: events", r"The emulation produces {v} range events (at most one per sensor and image); in {v} of them", [r["events"], r["misses"]]),
        ("E8: coverage", r"Regions touch {v}\% of the human-labeled curb instances, {v}\% of stairs, {v}\% of potholes and {v}\% of open holes",
         [(100 * r["dropoff_touched"][c]["share"], 0) for c in ("curb", "stairs", "pothole", "open hole")]),
        ("E8: promotion", r"Promotion is unreliable: {v}\% of the {v} promoted boxes match a human box of their class, from {v}\% for curbs to {v}\% for open holes",
         [(100 * r["promoted_precision"], 0), r["promoted"], (100 * r["promoted_per_class"]["curb"]["precision"], 0),
          (100 * r["promoted_per_class"]["open hole"]["precision"], 0)]),
        ("E8: suppression", r"In the {v} training frames with a teacher label, {v} human-labeled hazard instances carry no teacher box; a standard detection loss trains all of them as background, the source-scoped loss {v}\%",
         [e8["frames_with_signal"], sup["all"]["standard_background"], (100 * sup["all"]["scoped_background"] / sup["all"]["standard_background"], 1)]),
        ("E8: distilled/protected", r"observes their class ({v}\%, among them every crosswalk) or are protected by a proposal or a region ({v}\%)",
         [(100 * sup["all"]["scoped_out_of_scope"] / sup["all"]["standard_background"], 0), (100 * sup["all"]["scoped_protected"] / sup["all"]["standard_background"], 0)]),
        ("E8: pothole", r"The share trained as background is largest for potholes ({v}\%)",
         (100 * sup["pothole"]["scoped_background"] / sup["pothole"]["standard_background"], 0)),
    ]
    assert sup["crosswalk"].get("scoped_out_of_scope", 0) == sup["crosswalk"]["standard_background"], "every crosswalk is out of scope"
    for c in ("curb", "stairs", "pothole", "open hole"):
        t, pc = r["dropoff_touched"][c], r["promoted_per_class"][c]
        claims.append((f"tab:rangelabels {c}", r"\textit{" + c + r"} & {v} & {v} & {v} & {v}",
                       [t["instances"], (100 * t["share"], 0), pc["promoted"], (pc["precision"], 2)]))
    for g, lab in (("vehicle", "Vehicle"), ("person", "Person"), ("curb", "Curb"), ("crosswalk", "Crosswalk"),
                   ("stairs", "Stairs"), ("pothole", "Pothole"), ("all", "All")):
        v = sup[g]
        u = v["standard_background"]
        claims.append((f"tab:rangelabels {g}", lab + r" & {v} & {v} & {v} & {v}",
                       [u, (100 * v.get("scoped_background", 0) / u, 1), (100 * v.get("scoped_out_of_scope", 0) / u, 1),
                        (100 * v.get("scoped_protected", 0) / u, 1)]))
    return claims


def fl_claims():
    """Phases 3, 6, 7 and 10: E1, E7, E3, model-level edge cost (E5), client-level DP (E6) and poisoning (J4)."""
    res = ROOT / "results"
    if not (res / "phase6" / "E1_summary.json").exists():
        return []
    e1 = json.loads((res / "phase6" / "E1_summary.json").read_text())
    e7 = json.loads((res / "phase7" / "E7_summary.json").read_text())
    M, T = e1["methods"], e1["tests"]
    P = lambda m: M[m]["primary"][0]  # noqa: E731
    wan = lambda m: M[m]["comm"]["wan_bytes"]  # noqa: E731
    names = {"pretrained": "Pretrained (round 0)", "local": "Local only", "fedavg": "FedAvg", "fedprox": "FedProx",
             "scaffold": "SCAFFOLD", "fedrep": "FedRep-style", "hier_st": "HierFAVG + self-training",
             "hier_mt": "HierFAVG + mean teacher", "fedsto": "FedSTO-style", "semifl": "SemiFL-style",
             "central": "Pooled (same labels)", "fedv2x": r"\textbf{FedV2X-Aid}"}
    groups = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]
    claims = []
    for m, lab in names.items():
        v = M[m]
        claims.append((f"tab:e1 {m}", lab + r" & {v} & {v} & {v} & {v} & {v} & {v} & {v}$\pm${v} & {v} & {v}",
                       [(v["per_group"][g][0], 3) for g in groups] + [(v["primary"][0], 3), (v["primary"][1], 3),
                                                                      (v["map50"][0], 3), (v["worst_aid"][0], 3)]))
    h3, h4, h1 = T["H3_vs_fedavg"], T["H4_vs_pretrained"], T["H1_vs_local"]
    wshare = 100 * wan("fedv2x") / wan("fedavg")
    claims += [
        ("abstract: E1", r"FedV2X-Aid matches flat FedAvg in hazard recall ({v} vs.\ {v}) with {v}\% of its wide-area traffic, more than doubles local-only training ({v})",
         [(P("fedv2x"), 3), (P("fedavg"), 3), (wshare, 1), (P("local"), 3)]),
        ("abstract: E7", r"Without V2X labels or the source-scoped loss, recall falls to {v} and {v}", [(e7["ablations"]["A1"]["primary"][0], 3), (e7["ablations"]["A5"]["primary"][0], 3)]),
        ("C7: E1", r"FedV2X-Aid matches flat FedAvg ({v} against {v} mean R@P80) with {v}\% of its wide-area traffic", [(P("fedv2x"), 3), (P("fedavg"), 3), (wshare, 1)]),
        ("E1: means", r"FedV2X-Aid reaches a mean R@P80 of {v}$\pm${v} over three seeds, against {v}$\pm${v} for local-only training",
         [(P("fedv2x"), 3), (M["fedv2x"]["primary"][1], 3), (P("local"), 3), (M["local"]["primary"][1], 3)]),
        ("E1: H1", r"(median difference {v}, $p_{\mathrm{Holm}}<10^{-7}$, hypothesis H1)", (h1["median_diff"], 3)),
        ("E1: H3", r"(difference $+{v}$, 95\% interval $[{v}, {v}]$, H3) while sending {v}\% of its wide-area traffic",
         [(h3["pooled"]["difference"], 4), (h3["pooled"]["difference_ci95"][0], 4), (h3["pooled"]["difference_ci95"][1], 4), (wshare, 1)]),
        ("E1: SemiFL", r"scores highest ({v}$\pm${v})", [(P("semifl"), 3), (M["semifl"]["primary"][1], 3)]),
        ("E1: H4", r"(difference $+{v}$, 95\% interval $[{v}, {v}]$, H4)",
         [(T["pooled_vs_pretrained"]["pooled"]["difference"], 4)] + [(x, 4) for x in T["pooled_vs_pretrained"]["pooled"]["difference_ci95"]]),
        ("E1: pooled steps", r"Pooled training on the same weak labels takes about {v} steps of batch 32, against at most {v} local steps of batch 8 per aid in the federation, and ends below the round-0 head ({v})",
         [(sum(json.loads((res / "phase6" / "E1" / f"central_s{k}.json").read_text())["log"]["central_steps"] for k in range(3)) / 3, -1),
          8 * 2 * 25, (P("central"), 3)]),
        ("E1: H2", r"(H2, $p_{\mathrm{Holm}}={v}$)", (T["H2_vs_semifl"]["p_holm"], 2)),
        ("discussion: SemiFL gap", r"scores {v} higher at eleven times the wide-area traffic", (P("semifl") - P("fedv2x"), 3)),
        ("E3: WAN", r"FedV2X-Aid sends {v}\,MB over the wide-area network, against {v}\,MB for FedAvg", [(wan("fedv2x") / 1e6, 1), (wan("fedavg") / 1e6, 1)]),
        ("E3: target", r"pre-registered target of {v} mean R@P80 at the first evaluation (edge round {v}, {v} simulated hours)",
         [(e7["e3"]["target"], 3), e7["e3"]["fedv2x"]["to_target"]["edge_round"], e7["e3"]["fedv2x"]["to_target"]["hours"]]),
    ]
    assert all(e7["e3"][m]["to_target"]["reached"] == "3/3" for m in ("fedv2x", "fedavg"))
    assert all(e7["e3"][m]["to_target"]["reached"] == "0/3" for m in ("hier_st", "fedsto"))
    ab = e7["ablations"]
    rows = {"A1": "No V2X labels", "A2": r"Matched projection (PM)", "A3": r"Per-frame consensus (C)", "A4": "No range labels",
            "A5": "No source-scoped loss", "A6": "No quality weights", "A7": "Fixed RSU association", "A8": "No norm bound",
            "I1": "PM, no source scoping", "I2": "No range labels, no source scoping"}
    for k, lab in rows.items():
        claims.append((f"tab:e7 {k}", k + " & " + lab + r" & {v} &", (ab[k]["primary"][0], 3)))
    claims += [
        ("E7: A2", r"Consensus pose correction adds {v} over matched projection", (ab["A2"]["pooled_fedv2x_minus_ablation"]["difference"], 3)),
        ("E7: participation", r"leaves recall unchanged ({v} and {v} on seed 0)",
         [(e7["participation"]["P50"]["primary"], 3), (e7["participation"]["P25"]["primary"], 3)]),
        ("E8: scope worth", r"shows that the scope is worth {v} of mean recall", (ab["A5"]["pooled_fedv2x_minus_ablation"]["difference"], 3)),
    ]
    eff = json.loads((res / "phase3" / "zoo_efficiency.json").read_text())
    acc = json.loads((res / "phase3" / "zoo_accuracy.json").read_text())["models"]
    n, s = eff["yolo11n"], eff["reference"]
    claims += [
        # YOLO11-n was removed from the paper on 19 October 2026 (undertrained, 5-epoch budget); only the deployed detector is reported.
        ("RQ4: laptop cost", r"On the laptop GPU the detector takes {v}\,ms per frame in FP16 and {v}\,mJ above idle, and on its CPU {v}\,ms in INT8 ONNX Runtime, which quantization shrinks {v} times to {v}\,MB",
         [(s["gpu_fp16_ms"]["median"], 1), (s["gpu_energy_above_idle_mJ_per_frame"], 0), (s["cpu_onnxruntime"]["int8"]["median_ms"], 1),
          (s["cpu_onnxruntime"]["fp32"]["file_MB"] / s["cpu_onnxruntime"]["int8"]["file_MB"], 1), (s["cpu_onnxruntime"]["int8"]["file_MB"], 1)]),
        ("tab:devices params", r"Parameters / GMAC & {v}\,M / {v} \\", [(s["params"] / 1e6, 2), (s["gmacs_640"], 1)]),
        ("tab:devices gpu", r"GPU FP16 latency, median / p95 (ms) & {v} / {v} \\", [(s["gpu_fp16_ms"]["median"], 1), (s["gpu_fp16_ms"]["p95"], 1)]),
        ("tab:devices energy", r"GPU energy per frame above idle (mJ) & {v} \\", (s["gpu_energy_above_idle_mJ_per_frame"], 0)),
        ("tab:devices cpu", r"CPU FP32 latency, median (ms) & {v} \\", (s["cpu_onnxruntime"]["fp32"]["median_ms"], 1)),
        ("tab:devices cpu int8", r"CPU INT8 latency, median (ms) & {v} \\", (s["cpu_onnxruntime"]["int8"]["median_ms"], 1)),
        ("tab:devices size", r"ONNX size FP32 / INT8 (MB) & {v} / {v} \\",
         [(s["cpu_onnxruntime"]["fp32"]["file_MB"], 1), (s["cpu_onnxruntime"]["int8"]["file_MB"], 1)]),
        ("tab:devices map", r"mAP50 (AP50 veh. / pers.) & {v} ({v} / {v}) \\",
         [(acc["reference"]["map50"], 3), (acc["reference"]["group_ap50"]["vehicle"], 2), (acc["reference"]["group_ap50"]["person"], 2)]),
    ]
    dpr = {Path(r["file"]).stem: r for r in json.loads((res / "phase10" / "dp_accounting.json").read_text())}
    dpf = {Path(r["file"]).stem: r for r in json.loads((res / "phase10" / "full_head" / "dp_accounting.json").read_text())}
    run = lambda f: json.loads(f.read_text())  # noqa: E731
    d = {z: run(res / "phase10" / f"fedv2x-dpcls_z{z}_s0.json") for z in ("1.0", "2.0", "4.0")}
    f = {z: run(res / "phase10" / "full_head" / f"fedv2x-dp_z{z}_s0.json") for z in ("0.5", "1.0")}
    claims += [
        ("E6: dims", r"Restricting the shared update to the {v} classifier weights", d["1.0"]["log"]["dp_dims"]),
        ("E6: trade-off", r"keeps {v} of mean R@P80 at $\varepsilon={v}$, {v} at $\varepsilon={v}$ and {v} at $\varepsilon={v}$",
         [(d["1.0"]["primary_mean_r_at_p80"], 3), (dpr["fedv2x-dpcls_z1.0_s0"]["eps_rdp"], 0), (d["2.0"]["primary_mean_r_at_p80"], 3),
          (dpr["fedv2x-dpcls_z2.0_s0"]["eps_rdp"], 0), (d["4.0"]["primary_mean_r_at_p80"], 3), (dpr["fedv2x-dpcls_z4.0_s0"]["eps_rdp"], 0)]),
        ("tab:privacy full 0.5", r"Whole head (878{,}216 param.) & 0.5 & {v} & {v} & {v}",
         [(dpf["fedv2x-dp_z0.5_s0"]["eps_rdp"], 0), (f["0.5"]["primary_mean_r_at_p80"], 3), (f["0.5"]["map50_hazard_classes"], 3)]),
        ("tab:privacy full 1.0", r"Whole head & 1.0 & {v} & {v} & {v}",
         [(dpf["fedv2x-dp_z1.0_s0"]["eps_rdp"], 0), (f["1.0"]["primary_mean_r_at_p80"], 3), (f["1.0"]["map50_hazard_classes"], 3)]),
    ]
    for z in ("1.0", "2.0", "4.0"):
        claims.append((f"tab:privacy cls {z}", r"Hazard classifier" + (r" (7{,}740 param.)" if z == "1.0" else "") + f" & {z} & " + r"{v} & {v} & {v}",
                       [(dpr[f"fedv2x-dpcls_z{z}_s0"]["eps_rdp"], 0), (d[z]["primary_mean_r_at_p80"], 3), (d[z]["map50_hazard_classes"], 3)]))
    j4 = res / "phase10" / "J4"
    if (j4 / "fedv2x-att3-no_normbound_s0.json").exists():
        a1, a3, a3n = (run(j4 / f"fedv2x-{t}_s0.json") for t in ("att1", "att3", "att3-no_normbound"))
        claims += [
            ("J4: results", r"keeps mean R@P80 at {v} with one attacker and {v} with three. Without it, three attackers drive it to {v}, and the aids' local steps overflow in {v} cases",
             [(a1["primary_mean_r_at_p80"], 3), (a3["primary_mean_r_at_p80"], 3), (a3n["primary_mean_r_at_p80"], 3), a3n["log"]["skipped_steps"]]),
        ]
    return claims


def phase4_claims():
    """Phase 4: V2X labelling on real KITTI images (E2b) and the form-factor Monte-Carlo (98 deg)."""
    res = ROOT / "results" / "phase4"
    if not (res / "e2b_object_ref.json").exists():
        return []
    ob = json.loads((res / "e2b_object_ref.json").read_text())["results"]
    tr = json.loads((res / "e2b_tracking_ref.json").read_text())["results"]
    fa = json.loads((res / "e2b_failures.json").read_text())
    ff = json.loads((res / "mc_formfactor.json").read_text())["results"]

    def o(cfg, m, k="precision"):
        return ob[cfg]["metrics"][m][k][0]

    def t(cfg, m, k="precision"):
        return tr[cfg]["metrics"][m][k][0]

    def pr(src, cfg, m):
        f = o if src == "o" else t
        return [(f(cfg, m), 3), (f(cfg, m, "recall_connected"), 3)]

    def f98(h, m, k):
        return ff[f"fov98_hc{h}_g5"]["metrics"][m][k][0]
    sd_o = max(ob[c]["metrics"][m][k][1] for c in ("gt_g5", "det_g0", "det_g5", "det_g10", "det_g5_rho0.25")
               for m in ("P", "PM", "C") for k in ("precision", "recall_connected"))
    sd_t = max(tr[c]["metrics"][m][k][1] for c in ("gm_g5", "gm_g10", "iid_g5")
               for m in ("P", "PM", "C", "CT") for k in ("precision", "recall_connected"))
    sh = fa["category_shares"]
    rt = fa["runtime_ms_per_frame"]["C"]
    claims = [
        ("abstract: KITTI ratio", r"labels {v}$\times$ more vehicles than matching without pose correction at {v} precision on real KITTI images",
         [(o("det_g5", "C", "recall_connected") / o("det_g5", "PM", "recall_connected"), 1), (o("det_g5", "C"), 2)]),
        ("tab:kitti caption sd", r"standard deviations at most {v} on object frames and {v} on sequences", [("le", sd_o), ("le", sd_t)]),
        ("tab:kitti object true 5m", r"Object, true boxes, 5\,m & {v} / {v} & {v} / {v} & {v} / {v}", pr("o", "gt_g5", "P") + pr("o", "gt_g5", "PM") + pr("o", "gt_g5", "C")),
        ("tab:kitti object det 0m", r"Object, detector, 0\,m & {v} / {v} & {v} / {v} & {v} / {v}", pr("o", "det_g0", "P") + pr("o", "det_g0", "PM") + pr("o", "det_g0", "C")),
        ("tab:kitti object det 5m", r"Object, detector, 5\,m & {v} / {v} & {v} / {v} & {v} / {v}", pr("o", "det_g5", "P") + pr("o", "det_g5", "PM") + pr("o", "det_g5", "C")),
        ("tab:kitti object det 10m", r"Object, detector, 10\,m & {v} / {v} & {v} / {v} & {v} / {v}", pr("o", "det_g10", "P") + pr("o", "det_g10", "PM") + pr("o", "det_g10", "C")),
        ("tab:kitti object rho .25", r"$\rho=0.25$ & {v} / {v} & {v} / {v} & {v} / {v}", pr("o", "det_g5_rho0.25", "P") + pr("o", "det_g5_rho0.25", "PM") + pr("o", "det_g5_rho0.25", "C")),
        ("tab:kitti seq GM 5", r"Sequences, GM 5\,m & {v} / {v} & {v} / {v} & {v} / {v} & {v} / {v}", pr("t", "gm_g5", "P") + pr("t", "gm_g5", "PM") + pr("t", "gm_g5", "C") + pr("t", "gm_g5", "CT")),
        ("tab:kitti seq GM 10", r"Sequences, GM 10\,m & {v} / {v} & {v} / {v} & {v} / {v} & {v} / {v}", pr("t", "gm_g10", "P") + pr("t", "gm_g10", "PM") + pr("t", "gm_g10", "C") + pr("t", "gm_g10", "CT")),
        ("tab:kitti seq iid", r"Sequences, i.i.d.\ 5\,m & {v} / {v} & {v} / {v} & {v} / {v} & {v} / {v}", pr("t", "iid_g5", "P") + pr("t", "iid_g5", "PM") + pr("t", "iid_g5", "C") + pr("t", "iid_g5", "CT")),
        ("RQ1 KITTI: geometry", r"C keeps {v} precision at 5\,m and labels {v}\% of the connected vehicles, against {v}\% for PM",
         [(o("gt_g5", "C"), 3), (100 * o("gt_g5", "C", "recall_connected"), 1), (100 * o("gt_g5", "PM", "recall_connected"), 1)]),
        ("RQ1 KITTI: end to end", r"Real proposals lower C to {v} precision and {v}\% recall, still {v}$\times$ the recall of PM ({v} at {v} precision)",
         [(o("det_g5", "C"), 3), (100 * o("det_g5", "C", "recall_connected"), 1),
          (o("det_g5", "C", "recall_connected") / o("det_g5", "PM", "recall_connected"), 1), (o("det_g5", "PM", "recall_connected"), 3), (o("det_g5", "PM"), 3)]),
        ("RQ1 KITTI: proposal share", r"because {v}\% of the connected vehicles have a proposal", (100 * ob["det_g5"]["metrics"]["share_connected_moderate_with_proposal"], 1)),
        ("RQ1 KITTI: failures", r"{v}\% sit on a detector box that overlaps the right vehicle too loosely (IoU below 0.5), {v}\% on a false proposal, {v}\% carry the class of another vehicle and {v}\% duplicate",
         [(100 * sh["loose box"], 0), (100 * sh["false proposal"], 0), (100 * sh["wrong class"], 0), (100 * sh["duplicate"], 0)]),
        ("RQ1 KITTI: CT", r"raises recall of connected vehicles from {v} to {v} and precision from {v} to {v} under Gauss-Markov error at 5\,m",
         [(t("gm_g5", "C", "recall_connected"), 3), (t("gm_g5", "CT", "recall_connected"), 3), (t("gm_g5", "C"), 3), (t("gm_g5", "CT"), 3)]),
        ("RQ1 KITTI: position", r"lowers the position error from {v}\,m to {v}\,m, whereas per-frame consensus raises it to {v}\,m",
         [(tr["gm_g5"]["metrics"]["pose_rms_m"]["raw"], 2), (tr["gm_g5"]["metrics"]["pose_rms_m"]["CT"], 2), (tr["gm_g5"]["metrics"]["pose_rms_m"]["C"], 2)]),
        ("RQ1 KITTI: iid", r"under i.i.d.\ error CT loses recall ({v} to {v})", [(t("iid_g5", "C", "recall_connected"), 3), (t("iid_g5", "CT", "recall_connected"), 3)]),
        ("RQ1 KITTI: runtime", r"C labels a frame in {v}\,ms in the median and {v}\,ms at the 95th percentile", [(rt["median"], 1), (rt["p95"], 0)]),
        ("RQ1 98 deg", r"C reaches {v} precision and {v}\% recall at 1.0\,m, {v} and {v}\% at 1.3\,m, and {v} and {v}\% at 1.6\,m",
         [(f98(1.0, "C", "precision"), 3), (100 * f98(1.0, "C", "recall_connected"), 1), (f98(1.3, "C", "precision"), 3),
          (100 * f98(1.3, "C", "recall_connected"), 1), (f98(1.6, "C", "precision"), 3), (100 * f98(1.6, "C", "recall_connected"), 1)]),
        ("discussion: heights", r"({v} precision at 1.6\,m against {v} at 1.0\,m in the computed study at 98$^\circ$), and with KITTI's camera at 1.65\,m it keeps {v} precision",
         [(f98(1.6, "C", "precision"), 3), (f98(1.0, "C", "precision"), 3), (o("det_g5", "C"), 3)]),
    ]
    return claims


SOURCES = {  # where each checked item comes from (item label or claim-id prefix -> result or evidence file)
    "tab:det": "results/published/mc_det.json, mc_height.json", "tab:temporal": "results/published/mc_temporal.json, mc_temporal_misspec.json",
    "tab:misspec": "results/published/mc_temporal_misspec.json", "tab:robust": "results/published/mc_temporal.json",
    "tab:mobility": "results/published/mobility_sumo.json", "tab:modes": "results/published/mobility_modes.json",
    "tab:gatedrift": "results/published/gate_drift.json", "fig:mc": "results/published/v2x_pseudolabel_mc.json",
    "fig:grid": "results/published/mc_grid.json", "fig:gate": "results/published/gate_oc.json",
    "fig:theory": "results/published/theory_check.json, theory_check_dp.json",
    "abstract": "results/published/v2x_pseudolabel_mc.json, mc_grid.json, mc_temporal.json, mobility_sumo.json, yolo11_budget.json",
    "C2": "results/published/v2x_pseudolabel_mc.json", "C4": "results/published/yolo11_budget.json",
    "corollary": "results/published/v2x_pseudolabel_mc.json, yolo11_budget.json", "model budget": "results/published/yolo11_budget.json",
    "budget": "results/published/yolo11_budget.json", "related work": "results/published/yolo11_budget.json",
    "RQ1": "results/published/v2x_pseudolabel_mc.json, mc_temporal_misspec.json", "grid": "results/published/mc_grid.json",
    "detector": "results/published/mc_det.json, mc_height.json", "temporal": "results/published/mc_temporal.json, mc_temporal_misspec.json",
    "misspec": "results/published/mc_temporal_misspec.json", "latency": "results/published/mc_temporal.json",
    "RQ2": "results/published/yolo11_budget.json, mobility_sumo.json, mobility_modes.json", "RQ3": "results/published/mobility_sumo.json, mobility_modes.json",
    "discussion": "results/published/mobility_modes.json", "conclusion": "results/published/yolo11_budget.json, gate_oc.json, gate_drift.json",
    "DP": "results/published/dp_budget.json, dp_userlevel.json", "spoof": "results/published/mc_temporal.json",
    "gate": "results/published/gate_oc.json, gate_drift.json", "theory": "results/published/theory_check.json, theory_check_dp.json",
    "device": "authors' device measurements (not public; FEDV2X_DEVICE_MEASUREMENTS)",
    "tab:platform": "authors' device measurements (not public)", "tab:devices": "results/phase3/zoo_efficiency.json",
    "tab:datasets": "data/taxonomy/dataset_counts.json", "methods": "data/taxonomy/dataset_counts.json",
}


def source_of(group, item):
    if group in SOURCES:
        return SOURCES[group]
    key = item.split(":")[0].split(" ")[0] if group == "text" else group
    for k, v in SOURCES.items():
        if item.startswith(k) or key == k:
            return v
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--ledger", default=None, help="write a CSV claim ledger (item, status, source file)")
    args = ap.parse_args()
    supp = Path(args.tex).with_name(Path(args.tex).stem.replace("_journal", "") + "_supplement.tex")
    tex = prep(Path(args.tex).read_text(encoding="utf8") + ("\n" + supp.read_text(encoding="utf8") if supp.exists() else ""))
    tables, figures, claims = specs()
    rep = Report()
    for label, (rows, keys) in tables.items():
        check_table(rep, tex, label, rows, keys)
    for label, series in figures.items():
        check_figure(rep, tex, label, series, dec=2 if label == "fig:gate" else 3)
    check_text(rep, tex, claims)
    counts = rep.counts()
    lines = ["# Manuscript number check", "", f"Manuscript: `{Path(args.tex).name}`", "",
             "Each row recomputes printed values from `results/published/*.json` and compares them at the printed precision.", "",
             "Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
             "| Where | Item | Result | Detail |", "|---|---|---|---|"]
    lines += [f"| {g} | {i} | {s} | {d} |" for g, i, s, d in rep.items]
    if args.report:
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf8")
    if args.ledger:
        import csv
        with open(args.ledger, "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["where", "item", "evidence_type", "check", "detail", "source"])
            for g, i, s, d in rep.items:
                etype = "measured" if (g.startswith("tab:platform") or g.startswith("tab:devices") or i.startswith(("device", "tab:platform", "tab:devices", "tab:datasets", "methods"))) else "computed"
                w.writerow([g, i, etype, s, d, source_of(g if g != "text" else "text", i)])
            pend = re.findall(r"\\pend\{([^}]+)\}", tex[tex.index("\\begin{document}"):])
            for k in sorted(set(pend)):
                w.writerow(["manuscript", f"pending tag [{k}] x{pend.count(k)}", "pending", "n/a", "result not yet produced", "docs/experiment_registry.csv; expectations in docs/preregistered_targets.csv"])
    for g, i, s, d in rep.items:
        if s != "PASS":
            print(f"{s:8s} {g:14s} {i:38s} {d}")
    print("summary:", counts)
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
