"""PRE-REGISTERED EXPECTATIONS (formerly the orange "illustrative targets"), NOT measurements.

Since Phase 1 (16 Oct 2026) these values no longer appear in the manuscript. They are kept as the
expectations fixed before the experiments run, so that each real result can be compared with what
was expected (see docs/preregistered_targets.csv). A real result replaces nothing here; it is
reported in the paper whatever it is.

Original note: ILLUSTRATIVE TARGET VALUES for the draft paper (not measurements).

Every number defined here appears in orange in main.tex and must be replaced by the
measured result of experiments E1-E7 before submission. Derived quantities (means,
deltas, time- and bytes-to-target) are computed here so the draft stays internally
consistent; computed (non-illustrative) inputs are read from yolo11_budget.json.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
PUB = HERE.parent / "results" / "published"
budget = json.loads((PUB / "yolo11_budget.json").read_text())["nc152"]
HEAD_MB = budget["head_update_bytes_fp16"] / 1e6  # computed, not illustrative

CLASSES = ["Vehicle", "Person", "Curb", "Crosswalk", "Stairs", "Pothole"]
# E1: recall at 80 % precision (R@P80) per hazard group, CARLA federation, mean of 3 seeds
E1 = {
    "Pretrained (round 0)": dict(r=[71.2, 76.5, 41.8, 58.3, 47.6, 33.9], map50=48.7, worst=38.2),
    "Local-only":           dict(r=[75.9, 77.8, 49.5, 60.1, 51.2, 40.7], map50=52.1, worst=41.0),
    "FedAvg (flat)":        dict(r=[85.1, 81.6, 58.9, 64.0, 59.8, 50.0], map50=59.0, worst=52.4),
    "FedProx (flat)":       dict(r=[85.4, 81.9, 59.6, 64.3, 60.3, 50.9], map50=59.4, worst=53.5),
    "HierFAVG + teacher":   dict(r=[76.8, 79.6, 47.9, 62.0, 50.3, 38.1], map50=53.0, worst=44.6),
    "HierFAVG + mean teacher": dict(r=[78.9, 80.7, 50.6, 63.5, 52.8, 41.4], map50=54.6, worst=46.1),
    "FedV2X-Aid (ours)":    dict(r=[87.4, 83.0, 61.7, 65.2, 62.4, 53.6], map50=61.2, worst=57.8),
    "Centralised":          dict(r=[88.9, 84.1, 63.5, 66.4, 64.0, 55.9], map50=62.9, worst=59.6),
}
STD = {"Pretrained (round 0)": 0.0}

for name, v in E1.items():
    v["mean"] = sum(v["r"]) / len(v["r"])

ours, local, cent = E1["FedV2X-Aid (ours)"], E1["Local-only"], E1["Centralised"]
X = ours["mean"] - local["mean"]
Y = cent["mean"] - ours["mean"]

# E3: time-to-accuracy curves R(t) = R_inf - (R_inf - R0) * exp(-t / tau), t in simulated hours
R0 = E1["Pretrained (round 0)"]["mean"]
# asymptotes chosen so that R(50 h) reproduces the E1 values after the equal-compute budget
CURVES = {
    "FedV2X-Aid (PC5, 0% loss)": dict(r_inf=68.93, tau=9.0),
    # revised for the journal version: the SUMO study (mobility_sumo.json) shows that 30 % PC5 loss leaves
    # upload completion unchanged (contact windows of ~150 s vs 2.0 s transfers), and the sequential
    # Monte-Carlo study (mc_temporal.json) shows it lowers connected-vehicle label recall by ~3 %;
    # the earlier target (34 % slower) is therefore no longer consistent with computed evidence.
    "FedV2X-Aid (PC5, 30% loss)": dict(r_inf=68.75, tau=9.4),
    "FedProx (flat)": dict(r_inf=67.42, tau=14.0),
    "FedAvg (flat)": dict(r_inf=67.0, tau=15.0),
}
TARGET = 65.0
BUDGET_H = 50.0


def t_to(target, r_inf, tau):
    if r_inf <= target:
        return math.inf
    return -tau * math.log((r_inf - target) / (r_inf - R0))


def r_at(t, r_inf, tau):
    return r_inf - (r_inf - R0) * math.exp(-t / tau)


# Rounds per simulated hour and uploads per round
ROUND_HOURS_HIER = 0.5       # one cloud round = 2 edge rounds of 15 min
ROUND_HOURS_FLAT = 0.25      # one flat round = the local work of one edge round
FLAT_PARTICIPANTS = 27.3     # mean aids meeting the flat-FL round deadline (of 30); device-profile target (E5)
# computed by mobility_sumo.py (30 aids, 6 RSUs, 150 m range, no loss, mobility-driven association)
_mob = json.loads((PUB / "mobility_sumo.json").read_text())["runs"]
_def = next(r for r in _mob if r["n_aids"] == 30 and r["rsus"] == 6 and r["range_m"] == 150.0 and r["loss"] == 0.0 and not r["fixed_association"])
HIER_PARTICIPANTS = _def["p_upload_per_aid_round"] * 30   # uploads per edge round (computed)
RSUS = _def["active_rsus_per_cloud"]                       # RSUs forwarding an edge model per cloud round (computed)

t_ours = t_to(TARGET, **CURVES["FedV2X-Aid (PC5, 0% loss)"])
t_ours_loss = t_to(TARGET, **CURVES["FedV2X-Aid (PC5, 30% loss)"])
t_prox = t_to(TARGET, **CURVES["FedProx (flat)"])
t_avg = t_to(TARGET, **CURVES["FedAvg (flat)"])
rounds_ours = math.ceil(t_ours / ROUND_HOURS_HIER)
rounds_avg = math.ceil(t_avg / ROUND_HOURS_FLAT)
rounds_prox = math.ceil(t_prox / ROUND_HOURS_FLAT)
wan_ours = rounds_ours * RSUS * HEAD_MB
wan_avg = rounds_avg * FLAT_PARTICIPANTS * HEAD_MB
wan_prox = rounds_prox * FLAT_PARTICIPANTS * HEAD_MB
Z = 100 * (1 - wan_ours / wan_avg)
pc5_ours = 2 * rounds_ours * HIER_PARTICIPANTS * HEAD_MB
Z_total = 100 * (1 - (pc5_ours + wan_ours) / wan_avg)

# E4: post-compilation change in hazard mAP50 (points), per target and calibration source
E4 = {
    "A: NPU INT8": dict(public=-3.1, federated=-1.2, oracle=-0.9),
    "B: embedded GPU FP16": dict(public=-0.2, federated=-0.2, oracle=-0.2),
    "C: Pi 5 CPU INT8": dict(public=-2.4, federated=-1.0, oracle=-0.7),
}
W = max(-v["federated"] for v in E4.values())

if __name__ == "__main__":
    print(f"head update (computed) = {HEAD_MB:.3f} MB")
    for k, v in E1.items():
        print(f"{k:26s} mean R@P80 = {v['mean']:.2f}")
    print(f"X (ours - local)       = {X:.2f}")
    print(f"Y (central - ours)     = {Y:.2f}")
    for k, c in CURVES.items():
        print(f"time to {TARGET}: {k:28s} {t_to(TARGET, **c):6.2f} h   R({BUDGET_H:.0f} h) = {r_at(BUDGET_H, **c):.2f}")
    print(f"cloud rounds to target (ours) = {rounds_ours}, flat rounds FedAvg = {rounds_avg}, FedProx = {rounds_prox}")
    print(f"WAN MB to target: ours {wan_ours:.0f}, FedAvg {wan_avg:.0f}, FedProx {wan_prox:.0f}, Z = {Z:.1f} %")
    print(f"all-hop uplink to target: ours {pc5_ours + wan_ours:.0f} MB -> {Z_total:.1f} % less than FedAvg")
    print(f"per unit of local work, WAN ingress ratio at full participation = {6 / (2 * 30):.2f}; measured per cloud round = {RSUS / (2 * 30):.4f}")
    print(f"computed mobility inputs: uploads/edge round = {HIER_PARTICIPANTS:.2f}, active RSUs/cloud round = {RSUS:.2f}")
    print(f"W (max post-compilation loss, federated calibration) = {W:.1f}")
    print(f"speed-up vs FedAvg = {t_avg / t_ours:.2f}x, vs FedProx = {t_prox / t_ours:.2f}x; 30% loss slow-down = {100*(t_ours_loss/t_ours-1):.0f} %")


# ---------------------------------------------------------------------------------------------
# Journal-only targets (J-experiments), also ILLUSTRATIVE. Values appear in orange in the journal
# manuscript; they are kept here so that every target lives in one place.
# ---------------------------------------------------------------------------------------------
# J9: stronger baselines on the CARLA federation (5 seeds), same layout as E1
J9 = {
    "SCAFFOLD (flat)":          dict(r=[85.6, 82.0, 59.9, 64.5, 60.6, 51.2], map50=59.6, worst=53.9),
    "FedSTO-style SSFL":        dict(r=[79.4, 81.0, 51.3, 63.8, 53.5, 42.2], map50=55.0, worst=46.8),
    "SemiFL (detection)":       dict(r=[78.6, 80.4, 50.2, 63.1, 52.4, 41.0], map50=54.3, worst=45.5),
    "FedRep heads (personal.)": dict(r=[86.1, 82.2, 61.0, 64.6, 61.5, 52.8], map50=60.3, worst=55.2),
}
for v in J9.values():
    v["mean"] = sum(v["r"]) / len(v["r"])
# J3 downstream: vehicle R@P80 after FL under Gauss-Markov GNSS error (5 m), per-frame vs temporal consensus
J3 = dict(no_v2x=79.8, consensus=86.1, temporal=87.6)
# J2 scale: mean R@P80 and simulated hours to 65 % with 120 / 300 aids (WAN bytes are computed by mobility_sumo.py)
J2 = {120: dict(mean=69.6, hours=8.9), 300: dict(mean=70.1, hours=7.2)}
# J4 poisoning: 1, 3, 5 of 30 aids send scaled poisoned updates; mean R@P80 without / with norm bounding
J4 = {1: dict(none=67.9, bounded=68.7), 3: dict(none=63.2, bounded=67.8), 5: dict(none=58.4, bounded=66.5)}
# J1 field test: label precision / recall of consensus (C) and temporal consensus (CT) on annotated field frames,
# and RMSE of the corrected pedestrian position against RTK ground truth
J1 = dict(frames=1500, crossings=3, prec_C=0.84, rec_C=0.61, prec_CT=0.89, rec_CT=0.72, rtk_rmse_raw=4.6, rtk_rmse_CT=2.9)

if __name__ == "__main__":
    for k, v in J9.items():
        print(f"J9 {k:26s} mean R@P80 = {v['mean']:.2f}")
# E3 ablation of association: mean R@P80 and simulated hours to 65 % when each aid is fixed to a home RSU
# (participation and WAN bytes for this setting are computed by mobility_sumo.py)
FIXED_ASSOC = dict(mean=67.2, hours=17.9)
