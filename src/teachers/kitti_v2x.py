#!/usr/bin/env python3
"""E2b (Phase 4): V2X-as-teacher on real images with real 3D ground truth (KITTI object, 7,481 frames).

Message and pose emulation on real images, not a field test. For every KITTI frame:
  * every Car, Van, Truck and Tram of the label file is a potential CAM sender; it is connected with probability
    rho (default 0.5) and reports its footprint centre with N(0, 0.8 m) error per axis, its heading with N(0, 2 deg)
    error, its true length and width, and a class-prior height (mean label height of its KITTI class);
  * the receiving camera is KITTI's left colour camera (its own intrinsics, 1.65 m above the road, close to the
    eye height of a head-worn camera); the aid's position estimate carries an error of sigma_g (2-D RMS) and its
    heading an error of sigma_psi (default 5 deg), i.i.d. per frame;
  * proposals are either the frame's ground-truth 2-D vehicle boxes ("geometry-only", a perfect detector) or the
    detections of a real detector (score >= 0.05, "end-to-end");
  * labellers P, PM and C are the unchanged functions of computed/v2x_pseudolabel_mc.py, run with KITTI's camera.

A label is correct if its box overlaps a ground-truth vehicle of the message's class with IoU >= 0.5 (greedy
one-to-one, as in the computed study). Recall is measured over KITTI-"moderate" vehicles (box height >= 25 px,
occlusion <= 1, truncation <= 0.3). The conditional recall of connected moderate vehicles that have a vehicle
proposal with IoU >= 0.5 is the rate used to emulate V2X labels on 2-D street images (src/teachers/emulate.py).

    python src/teachers/kitti_v2x.py --proposals $FEDV2X_WORK/kitti/proposals_object_ref.pkl --out results/phase4/e2b_object_ref.json
"""
import os
import argparse
import json
import math
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "computed"))
import v2x_pseudolabel_mc as mc  # noqa: E402

KITTI = Path(os.environ.get("FEDV2X_WORK", "work") + "/kitti/training")
VEH = {"Car": 0, "Van": 1, "Truck": 2, "Tram": 3}
NAMES = list(VEH)
# detector classes (152-class taxonomy) that count as vehicle proposals
DET_VEH = {14, 15, 16, 17, 18, 19, 20, 24, 27}
H_CAM_KITTI = 1.65
SIG_VEH_HEAD = math.radians(2.0)


def read_labels(fid):
    objs = []
    for line in (KITTI / "label_2" / f"{fid}.txt").read_text().splitlines():
        v = line.split()
        t = v[0]
        if t == "DontCare":
            continue
        trunc, occ = float(v[1]), int(v[2])
        box = np.array(list(map(float, v[4:8])))
        h, w, l = map(float, v[8:11])
        x, y, z, ry = map(float, v[11:15])
        objs.append(dict(type=t, trunc=trunc, occ=occ, box=box, h=h, w=w, l=l, x=x, y=y, z=z, ry=ry))
    return objs


def read_p2(fid):
    for line in (KITTI / "calib" / f"{fid}.txt").read_text().splitlines():
        if line.startswith("P2:"):
            return np.array(list(map(float, line.split()[1:]))).reshape(3, 4)
    raise ValueError(fid)


def moderate(o):
    return (o["box"][3] - o["box"][1]) >= 25 and o["occ"] <= 1 and o["trunc"] <= 0.30


def to_world(o, bx):
    """KITTI camera coordinates -> the study's world frame (x forward, y left, z up; camera at the origin)."""
    fwd, left = o["z"], -(o["x"] + bx)
    yaw = math.atan2(-math.cos(o["ry"]), -math.sin(o["ry"]))
    return fwd, left, yaw


def frame_eval(fid, props_all, cfg, rng, h_prior):
    P2 = read_p2(fid)
    pred = props_all[fid]
    W, H = pred["size"]
    mc.W, mc.H = int(W), int(H)
    mc.F, mc.CX, mc.CY = float(P2[0, 0]), float(P2[0, 2]), float(P2[1, 2])
    mc.H_CAM = H_CAM_KITTI
    bx = P2[0, 3] / P2[0, 0]
    objs = read_labels(fid)
    vehs = [o for o in objs if o["type"] in VEH]
    visible = [dict(idx=i, box=o["box"], cls=VEH[o["type"]]) for i, o in enumerate(vehs)]
    mod = {i for i, o in enumerate(vehs) if moderate(o)}
    msgs = []
    for i, o in enumerate(vehs):
        if rng.random() >= cfg["rho"]:
            continue
        fwd, left, yaw = to_world(o, bx)
        msgs.append(dict(idx=i, cls=VEH[o["type"]], x=fwd + rng.normal(0, mc.SIGMA_VEH_POS),
                         y=left + rng.normal(0, mc.SIGMA_VEH_POS), yaw=yaw + rng.normal(0, SIG_VEH_HEAD),
                         l=o["l"], w=o["w"], h=h_prior[o["type"]]))
    conn = {m["idx"] for m in msgs}
    if cfg["proposals"] == "gt":
        props = [dict(box=v["box"], score=1.0, cls=v["cls"]) for v in visible]
    else:
        keep = np.isin(pred["cls"], list(DET_VEH))
        props = [dict(box=b.astype(float), score=float(s), cls=int(c))
                 for b, s, c in zip(pred["boxes"][keep], pred["scores"][keep], pred["cls"][keep])]
    pb = [p["box"] for p in props]
    sig_g, sig_psi = cfg["gnss_rms"], math.radians(cfg["heading_sigma_deg"])
    e = rng.normal(0, sig_g / math.sqrt(2), size=2) if sig_g > 0 else np.zeros(2)
    pose_hat = (float(e[0]), float(e[1]), float(rng.normal(0, sig_psi)))
    labels = {"P": [], "PM": [], "C": []}
    boxes, keep_i = mc.project_msgs(msgs, pose_hat)
    for t in mc.visible_after_occlusion(msgs, boxes, keep_i, pose_hat):
        labels["P"].append(dict(box=boxes[t], cls=msgs[keep_i[t]]["cls"], msg=msgs[keep_i[t]]["idx"]))
    for (t, k, _) in mc.match(boxes, pb, 0.3):
        labels["PM"].append(dict(box=props[k]["box"], cls=msgs[keep_i[t]]["cls"], msg=msgs[keep_i[t]]["idx"]))
    sig_rel = math.sqrt(sig_g ** 2 + 2 * mc.SIGMA_VEH_POS ** 2)
    pose_c, inl = mc.consensus_pose(msgs, props, pose_hat, sig_rel, sig_psi)
    pose_ok = pose_c is not None
    if pose_ok:
        boxes_c, keep_c = mc.project_msgs(msgs, pose_c)
        for (t, k, _) in mc.match(boxes_c, pb, 0.4):
            labels["C"].append(dict(box=props[k]["box"], cls=msgs[keep_c[t]]["cls"], msg=msgs[keep_c[t]]["idx"]))
    # proposals that overlap each vehicle (IoU >= 0.5), for the conditional recall
    has_prop = set()
    if pb:
        M = mc.iou_matrix([v["box"] for v in visible], pb) if visible else np.zeros((0, len(pb)))
        has_prop = {i for i in range(len(visible)) if M[i].max() >= 0.5}
    out = {"n_mod": len(mod), "n_mod_conn": len(mod & conn), "n_mod_conn_prop": len(mod & conn & has_prop),
           "pose": int(pose_ok), "n_msgs": len(msgs)}
    for m, labs in labels.items():
        corr = mc.evaluate(labs, visible)
        corr_set = set(corr)
        wrong = [lab for lab in labs]
        out[m] = {"emit": len(labs), "correct": len(corr), "correct_mod": len(corr_set & mod),
                  "correct_mod_conn": len(corr_set & mod & conn),
                  "correct_mod_conn_prop": len(corr_set & mod & conn & has_prop),
                  "same_identity": sum(1 for lab in labs for t in [None] if False)}
        # identity-level: correct label on the vehicle that sent the message
        ident = 0
        for lab in labs:
            j = lab["msg"]
            if j < len(visible) and mc.iou_matrix([lab["box"]], [visible[j]["box"]])[0, 0] >= 0.5:
                ident += 1
        out[m]["identity_correct"] = ident
        del wrong
    return out


def job(args):
    cfg, seed, fids, props_path, h_prior = args
    with open(props_path, "rb") as fh:
        props_all = pickle.load(fh)["pred"]
    rng = np.random.default_rng(seed)
    tot = {}
    for fid in fids:
        r = frame_eval(fid, props_all, cfg, rng, h_prior)
        for k, v in r.items():
            if isinstance(v, dict):
                d = tot.setdefault(k, {})
                for kk, vv in v.items():
                    d[kk] = d.get(kk, 0) + vv
            else:
                tot[k] = tot.get(k, 0) + v
    return cfg["name"], seed, tot


def class_height_priors():
    hs = {k: [] for k in VEH}
    for f in sorted((KITTI / "label_2").glob("*.txt")):
        for o in read_labels(f.stem):
            if o["type"] in VEH:
                hs[o["type"]].append(o["h"])
    return {k: float(np.mean(v)) for k, v in hs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    h_prior = class_height_priors()
    fids = sorted(f.stem for f in (KITTI / "label_2").glob("*.txt"))
    cfgs = []
    for prop in ("gt", "det"):
        for g in (0, 1, 3, 5, 10):
            cfgs.append(dict(name=f"{prop}_g{g}", proposals=prop, gnss_rms=float(g), rho=0.5, heading_sigma_deg=5.0))
        for r in (0.25, 1.0):
            cfgs.append(dict(name=f"{prop}_g5_rho{r}", proposals=prop, gnss_rms=5.0, rho=r, heading_sigma_deg=5.0))
    chunks = [fids[i::8] for i in range(8)]
    jobs = [(c, 1000 + 100 * s + k, ch, a.proposals, h_prior) for c in cfgs for s in range(a.seeds)
            for k, ch in enumerate(chunks)]
    with Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    by = {}
    for name, seed, tot in res:
        key = (name, seed // 100)
        agg = by.setdefault(key, {})
        for k, v in tot.items():
            if isinstance(v, dict):
                d = agg.setdefault(k, {})
                for kk, vv in v.items():
                    d[kk] = d.get(kk, 0) + vv
            else:
                agg[k] = agg.get(k, 0) + v
    results = {}
    for c in cfgs:
        per_seed = [by[(c["name"], 10 + s)] for s in range(a.seeds)]
        met = {}
        for m in ("P", "PM", "C"):
            vals = {"precision": [], "recall_connected": [], "recall_all": [], "conditional_recall": [],
                    "identity_precision": [], "labels_per_frame": []}
            for t in per_seed:
                d = t[m]
                vals["precision"].append(d["correct"] / max(d["emit"], 1))
                vals["recall_connected"].append(d["correct_mod_conn"] / max(t["n_mod_conn"], 1))
                vals["recall_all"].append(d["correct_mod"] / max(t["n_mod"], 1))
                vals["conditional_recall"].append(d["correct_mod_conn_prop"] / max(t["n_mod_conn_prop"], 1))
                vals["identity_precision"].append(d["identity_correct"] / max(d["emit"], 1))
                vals["labels_per_frame"].append(d["emit"] / len(fids))
            met[m] = {k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0] for k, v in vals.items()}
        t = per_seed[0]
        met["frames"] = len(fids)
        met["moderate_vehicles_per_frame"] = t["n_mod"] / len(fids)
        met["share_connected_moderate_with_proposal"] = float(np.mean([p["n_mod_conn_prop"] / max(p["n_mod_conn"], 1)
                                                                      for p in per_seed]))
        met["frames_with_pose_C"] = float(np.mean([p["pose"] / len(fids) for p in per_seed]))
        results[c["name"]] = {"config": c, "metrics": met}
        print(c["name"], {m: (round(met[m]["precision"][0], 3), round(met[m]["recall_connected"][0], 3),
                              round(met[m]["conditional_recall"][0], 3)) for m in ("P", "PM", "C")}, flush=True)
    out = {"proposals": a.proposals, "seeds": a.seeds, "class_height_priors_m": h_prior, "camera_height_m": H_CAM_KITTI,
           "results": results, "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
