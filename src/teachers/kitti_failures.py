#!/usr/bin/env python3
"""E2b failure cases, worked example and runtime (Phase 4, chunks 4.2 and 4.3).

Fixed rule (pre-registered in the Phase 4 report): consensus labeller C at 5 m GNSS error, 5 deg compass error,
50 % penetration, real proposals of the reference detector, one pass over all KITTI object frames in file order
with random seed 0. Every wrong C label is put in one category:
  * false proposal   - the label box overlaps no ground-truth vehicle (IoU < 0.1);
  * wrong class      - it overlaps a vehicle with IoU >= 0.5 but the message's class differs (e.g. Van vs Car);
  * loose box        - it overlaps a vehicle of the right class with 0.1 <= IoU < 0.5;
  * duplicate        - it overlaps a vehicle with IoU >= 0.5 that another label of the frame already claims.
The first 20 frames with a wrong C label are listed with their categories. The worked example is the first frame
(file order) with at least two connected moderate vehicles on which C labels at least one vehicle correctly and
PM labels none. Runtime: wall-clock time per frame of P, PM and C over the first 500 frames.

    python src/teachers/kitti_failures.py --proposals $FEDV2X_WORK/kitti/proposals_object_ref.pkl
"""
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "computed"))
sys.path.insert(0, str(ROOT / "src" / "teachers"))
import kitti_v2x as kv  # noqa: E402
import v2x_pseudolabel_mc as mc  # noqa: E402


def setup(fid, pred):
    P2 = kv.read_p2(fid)
    W, H = pred["size"]
    mc.W, mc.H = int(W), int(H)
    mc.F, mc.CX, mc.CY, mc.H_CAM = float(P2[0, 0]), float(P2[0, 2]), float(P2[1, 2]), kv.H_CAM_KITTI
    return P2[0, 3] / P2[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/phase4/e2b_failures.json"))
    a = ap.parse_args()
    props_all = pickle.load(open(a.proposals, "rb"))["pred"]
    h_prior = kv.class_height_priors()
    fids = sorted(f.stem for f in (kv.KITTI / "label_2").glob("*.txt"))
    rng = np.random.default_rng(0)
    sig_g, sig_psi = 5.0, math.radians(5.0)
    cats = {"false proposal": 0, "wrong class": 0, "loose box": 0, "duplicate": 0}
    first, example, times = [], None, {"P": [], "PM": [], "C": []}
    n_wrong = n_labels = 0
    for n, fid in enumerate(fids):
        pred = props_all[fid]
        bx = setup(fid, pred)
        objs = kv.read_labels(fid)
        vehs = [o for o in objs if o["type"] in kv.VEH]
        visible = [dict(idx=i, box=o["box"], cls=kv.VEH[o["type"]]) for i, o in enumerate(vehs)]
        msgs = []
        for i, o in enumerate(vehs):
            if rng.random() >= 0.5:
                continue
            fwd, left, yaw = kv.to_world(o, bx)
            msgs.append(dict(idx=i, cls=kv.VEH[o["type"]], x=fwd + rng.normal(0, mc.SIGMA_VEH_POS),
                             y=left + rng.normal(0, mc.SIGMA_VEH_POS), yaw=yaw + rng.normal(0, kv.SIG_VEH_HEAD),
                             l=o["l"], w=o["w"], h=h_prior[o["type"]]))
        keep = np.isin(pred["cls"], list(kv.DET_VEH))
        props = [dict(box=b.astype(float), score=float(s), cls=int(c))
                 for b, s, c in zip(pred["boxes"][keep], pred["scores"][keep], pred["cls"][keep])]
        pb = [p["box"] for p in props]
        e = rng.normal(0, sig_g / math.sqrt(2), size=2)
        pose_hat = (float(e[0]), float(e[1]), float(rng.normal(0, sig_psi)))
        t0 = time.perf_counter()
        boxes, ki = mc.project_msgs(msgs, pose_hat)
        p_labels = [dict(box=boxes[t], cls=msgs[ki[t]]["cls"]) for t in mc.visible_after_occlusion(msgs, boxes, ki, pose_hat)]
        t1 = time.perf_counter()
        pm = [dict(box=props[k]["box"], cls=msgs[ki[t]]["cls"], msg=msgs[ki[t]]["idx"]) for (t, k, _) in mc.match(boxes, pb, 0.3)]
        t2 = time.perf_counter()
        sig_rel = math.sqrt(sig_g ** 2 + 2 * mc.SIGMA_VEH_POS ** 2)
        pose_c, inl = mc.consensus_pose(msgs, props, pose_hat, sig_rel, sig_psi)
        c_labels, hyps = [], None
        if pose_c is not None:
            bc, kc = mc.project_msgs(msgs, pose_c)
            c_labels = [dict(box=props[k]["box"], cls=msgs[kc[t]]["cls"], msg=msgs[kc[t]]["idx"], iou=float(iou))
                        for (t, k, iou) in mc.match(bc, pb, 0.4)]
        t3 = time.perf_counter()
        if n < 500:
            times["P"].append(t1 - t0)
            times["PM"].append(t1 - t0 + t2 - t1)
            times["C"].append(t1 - t0 + t3 - t2)
        correct = set(mc.evaluate(c_labels, visible))
        claimed = set()
        wrong_here = []
        for lab in c_labels:
            n_labels += 1
            ious = mc.iou_matrix([lab["box"]], [v["box"] for v in visible])[0] if visible else np.zeros(0)
            j = int(np.argmax(ious)) if len(ious) else -1
            best = float(ious[j]) if j >= 0 else 0.0
            ok = j >= 0 and best >= 0.5 and visible[j]["cls"] == lab["cls"] and j not in claimed
            if ok:
                claimed.add(j)
                continue
            n_wrong += 1
            if best < 0.1:
                c = "false proposal"
            elif best >= 0.5 and visible[j]["cls"] != lab["cls"]:
                c = "wrong class"
            elif best >= 0.5:
                c = "duplicate"
            else:
                c = "loose box"
            cats[c] += 1
            wrong_here.append(c)
        if wrong_here and len(first) < 20:
            first.append({"frame": fid, "categories": wrong_here, "c_labels": len(c_labels)})
        conn_mod = [v for v in visible if v["idx"] in {m["idx"] for m in msgs} and kv.moderate(vehs[v["idx"]])]
        pm_ok = len(mc.evaluate(pm, visible))
        if example is None and len(conn_mod) >= 2 and correct and pm_ok == 0:
            example = {"frame": fid, "messages": len(msgs), "proposals": len(props), "connected_moderate": len(conn_mod),
                       "prior_error_m": round(math.hypot(*e), 2), "prior_heading_error_deg": round(math.degrees(pose_hat[2]), 2),
                       "consensus_pose_error_m": round(math.hypot(pose_c[0], pose_c[1]), 2) if pose_c else None,
                       "consensus_heading_error_deg": round(math.degrees(pose_c[2]), 2) if pose_c else None,
                       "inliers": inl, "pm_labels": len(pm), "pm_correct": pm_ok, "c_labels": len(c_labels),
                       "c_correct": len(correct), "c_label_iou_with_projection": [round(l["iou"], 3) for l in c_labels]}
    out = {"rule": "C, 5 m GNSS, 5 deg compass, 50 % penetration, reference-detector proposals, file order, seed 0",
           "labels": n_labels, "wrong": n_wrong, "categories": cats,
           "category_shares": {k: v / max(n_wrong, 1) for k, v in cats.items()}, "first_20_failure_frames": first,
           "worked_example": example,
           "runtime_ms_per_frame": {k: {"median": 1e3 * float(np.median(v)), "p95": 1e3 * float(np.quantile(v, 0.95)),
                                        "frames": len(v)} for k, v in times.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "first_20_failure_frames"}, indent=1))


if __name__ == "__main__":
    main()
