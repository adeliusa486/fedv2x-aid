"""Monte-Carlo study of V2X-as-teacher pseudo-labelling under pedestrian pose error.

Geometric simulation (not a rendered scene): vehicles are placed on the lanes of a
signalised intersection in front of a pedestrian standing at the kerb; a pinhole camera
(102 deg HFOV, 640x360 content region, 1.0 m mounting height) observes them; a parametric
detector model produces proposals (confident, low-confidence and false); a fraction of
vehicles broadcast CAM-like messages (position, heading, size, station type) with their
own small errors; the pedestrian's GNSS/compass pose estimate carries the injected error.

Three labellers are compared:
  P   projection only: the projected V2X box is the label
  PM  projection + IoU matching to detector proposals (no pose correction)
  C   consensus pose correction (RANSAC over V2X-proposal correspondences, Gaussian prior),
      then matching; C+R additionally emits projected boxes for unmatched connected
      vehicles when the pose is supported by >= 2 inliers (recovery of missed vehicles)

Outputs pseudo-label precision, recall over all visible vehicles, recall over connected
visible vehicles, and the share of correct labels that fall on vehicles the deployed
detector (score >= 0.25) missed.
"""
import argparse
import json
import math
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

W, H = 640, 360
HFOV = math.radians(102.0)
F = (W / 2) / math.tan(HFOV / 2)
CX, CY = W / 2, H / 2
H_CAM = 1.0
TAU_DET = 0.25

CLASSES = ["car", "bus", "truck", "motorcycle", "bicycle"]
CLASS_P = np.array([0.70, 0.08, 0.07, 0.09, 0.06])
DIMS = np.array([[4.6, 1.85, 1.55], [11.5, 2.55, 3.2], [8.0, 2.5, 3.2], [2.1, 0.8, 1.5], [1.8, 0.6, 1.7]])
CROSS_LANES_X = [6.0, 9.5, 13.0, 16.5]
PARALLEL_LANES_Y = [-4.0, -7.5]
SIGMA_VEH_POS = 0.8      # per-axis error of a vehicle's own reported position (m)
SIGMA_VEH_HEAD = math.radians(2.0)


def box_corners(x, y, yaw, l, w, h):
    c, s = math.cos(yaw), math.sin(yaw)
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2])
    dy = np.array([w / 2, -w / 2, -w / 2, w / 2])
    px = x + c * dx - s * dy
    py = y + s * dx + c * dy
    xs = np.concatenate([px, px])
    ys = np.concatenate([py, py])
    zs = np.concatenate([np.zeros(4), np.full(4, h)])
    return xs, ys, zs


def project(xs, ys, zs, pose):
    """Project world points with an aid pose (px, py, yaw). Returns 2D box or None."""
    px, py, yaw = pose
    c, s = math.cos(-yaw), math.sin(-yaw)
    rx, ry = xs - px, ys - py
    fwd = c * rx - s * ry
    left = s * rx + c * ry
    if np.any(fwd < 0.5):
        return None
    u = CX - F * left / fwd
    v = CY - F * (zs - H_CAM) / fwd
    return np.array([u.min(), v.min(), u.max(), v.max()])


def clip_box(b):
    if b is None:
        return None, 0.0
    area = (b[2] - b[0]) * (b[3] - b[1])
    cb = np.array([max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H)])
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        return None, 0.0
    carea = (cb[2] - cb[0]) * (cb[3] - cb[1])
    return cb, carea / max(area, 1e-9)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a)[:, None, :]
    b = np.asarray(b)[None, :, :]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    aa = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    bb = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(aa + bb - inter, 1e-9)


def sample_scene(rng, lam=7.0):
    n = rng.poisson(lam)
    vehicles = []
    tries = 0
    while len(vehicles) < n and tries < 200:
        tries += 1
        cls = rng.choice(len(CLASSES), p=CLASS_P)
        l, w, h = DIMS[cls] * rng.uniform(0.92, 1.08, size=3)
        if rng.random() < 0.6:
            lane = rng.choice(len(CROSS_LANES_X))
            x = CROSS_LANES_X[lane]
            y = rng.uniform(-35, 35)
            yaw = math.pi / 2 if lane % 2 == 0 else -math.pi / 2
            key, pos = ("c", lane), y
        else:
            lane = rng.choice(len(PARALLEL_LANES_Y))
            y = PARALLEL_LANES_Y[lane]
            x = rng.uniform(-5, 45)
            yaw = 0.0 if lane == 0 else math.pi
            key, pos = ("p", lane), x
        ok = all(not (v["key"] == key and abs(v["pos"] - pos) < (v["l"] + l) / 2 + 1.5) for v in vehicles)
        if ok:
            vehicles.append(dict(cls=cls, x=x, y=y, yaw=yaw, l=l, w=w, h=h, key=key, pos=pos))
    return vehicles


def gt_boxes(vehicles):
    """Visible amodal boxes of vehicles from the true pose, with occlusion fraction."""
    true_pose = (0.0, 0.0, 0.0)
    items = []
    for i, v in enumerate(vehicles):
        b = project(*box_corners(v["x"], v["y"], v["yaw"], v["l"], v["w"], v["h"]), true_pose)
        cb, frac = clip_box(b)
        if cb is None or frac < 0.3 or (cb[3] - cb[1]) < 10:
            continue
        rng_m = math.hypot(v["x"], v["y"])
        items.append(dict(idx=i, box=cb, cls=v["cls"], range=rng_m))
    items.sort(key=lambda d: d["range"])
    visible = []
    for it in items:
        occ = 0.0
        if visible:
            ious = iou_matrix([it["box"]], [o["box"] for o in visible])[0]
            b = it["box"]
            area = (b[2] - b[0]) * (b[3] - b[1])
            for o in visible:
                ob = o["box"]
                ix = max(0, min(b[2], ob[2]) - max(b[0], ob[0]))
                iy = max(0, min(b[3], ob[3]) - max(b[1], ob[1]))
                occ += ix * iy / area
            occ = min(occ, 1.0)
            del ious
        it["occ"] = occ
        if occ <= 0.8:
            visible.append(it)
    return visible


def detector(rng, visible):
    props = []  # dict(box, score, cls, gt)
    for g in visible:
        r, occ = g["range"], g["occ"]
        p = np.clip(0.97 - 0.006 * max(0.0, r - 10.0), 0.6, 0.97) * (1 - 0.6 * occ)
        b = g["box"]
        bw, bh = b[2] - b[0], b[3] - b[1]
        if rng.random() < p:
            sig, score = 0.04, rng.uniform(0.35, 0.95)
        elif rng.random() < 0.5:
            sig, score = 0.08, rng.uniform(0.05, 0.25)
        else:
            continue
        cx = (b[0] + b[2]) / 2 + rng.normal(0, sig * bw)
        cy = (b[1] + b[3]) / 2 + rng.normal(0, sig * bh)
        nw = bw * math.exp(rng.normal(0, 0.07 if sig < 0.05 else 0.12))
        nh = bh * math.exp(rng.normal(0, 0.07 if sig < 0.05 else 0.12))
        cls = g["cls"] if rng.random() < 0.88 else rng.choice([c for c in range(5) if c != g["cls"]])
        nb, _ = clip_box(np.array([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]))
        if nb is not None:
            props.append(dict(box=nb, score=score, cls=cls))
    for _ in range(rng.poisson(1.2)):
        rr = rng.uniform(5, 40)
        cls = rng.choice(len(CLASSES), p=CLASS_P)
        l, w, h = DIMS[cls]
        bw = F * rng.uniform(w, l) / rr
        bh = F * h / rr
        cx = rng.uniform(0, W)
        bottom = CY + F * H_CAM / rr
        nb, _ = clip_box(np.array([cx - bw / 2, bottom - bh, cx + bw / 2, bottom]))
        if nb is not None:
            props.append(dict(box=nb, score=rng.uniform(0.05, 0.4), cls=cls))
    return props


def v2x_messages(rng, vehicles, rho):
    msgs = []
    for i, v in enumerate(vehicles):
        if rng.random() >= rho:
            continue
        msgs.append(dict(idx=i, cls=v["cls"],
                         x=v["x"] + rng.normal(0, SIGMA_VEH_POS), y=v["y"] + rng.normal(0, SIGMA_VEH_POS),
                         yaw=v["yaw"] + rng.normal(0, SIGMA_VEH_HEAD), l=v["l"], w=v["w"], h=DIMS[v["cls"]][2]))
    return msgs


def project_msgs(msgs, pose):
    boxes, keep = [], []
    for j, m in enumerate(msgs):
        b = project(*box_corners(m["x"], m["y"], m["yaw"], m["l"], m["w"], m["h"]), pose)
        cb, frac = clip_box(b)
        if cb is None or frac < 0.3 or (cb[3] - cb[1]) < 10:
            continue
        boxes.append(cb)
        keep.append(j)
    return boxes, keep


def visible_after_occlusion(msgs, boxes, keep, pose):
    order = sorted(range(len(keep)), key=lambda t: math.hypot(msgs[keep[t]]["x"] - pose[0], msgs[keep[t]]["y"] - pose[1]))
    vis = []
    for t in order:
        b = boxes[t]
        area = (b[2] - b[0]) * (b[3] - b[1])
        occ = 0.0
        for s in vis:
            ob = boxes[s]
            ix = max(0, min(b[2], ob[2]) - max(b[0], ob[0]))
            iy = max(0, min(b[3], ob[3]) - max(b[1], ob[1]))
            occ += ix * iy / area
        if occ <= 0.6:
            vis.append(t)
    return set(vis)


def match(boxes_a, boxes_b, thr):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return []
    M = iou_matrix(boxes_a, boxes_b)
    r, c = linear_sum_assignment(-M)
    return [(i, k, M[i, k]) for i, k in zip(r, c) if M[i, k] >= thr]


def consensus_pose(msgs, props, pose_hat, sig_g, sig_psi):
    """RANSAC over (message, proposal) pairs with a yaw grid; returns (pose, inliers, unique)."""
    veh_props = [p for p in props]
    if not msgs or not veh_props:
        return None, 0
    pb = np.array([p["box"] for p in veh_props])
    hyps = []
    cand_count = np.zeros(len(msgs), dtype=int)
    for j, m in enumerate(msgs):
        for k, p in enumerate(veh_props):
            b = p["box"]
            bh = b[3] - b[1]
            r_size = F * m["h"] / max(bh, 1.0)
            bottom_visible = b[3] < H - 1 and b[3] - CY > 2
            r_ground = F * H_CAM / (b[3] - CY) if bottom_visible else np.inf
            if np.isfinite(r_ground) and abs(math.log(r_size / r_ground)) > 0.5:
                continue
            # range to the nearest visible point of the vehicle: ground contact if visible, else size
            r_near = r_ground if np.isfinite(r_ground) else r_size
            u = (b[0] + b[2]) / 2
            ray = -(u - CX) / F  # lateral offset per metre of depth (left positive)
            valid_any = False
            for dpsi in (-2 * sig_psi, -sig_psi, 0.0, sig_psi, 2 * sig_psi):
                yaw = pose_hat[2] + dpsi
                # footprint half-extent along the viewing ray, from the V2X-reported size and heading
                theta = yaw + math.atan(ray)
                half = 0.5 * m["l"] * abs(math.cos(m["yaw"] - theta)) + 0.5 * m["w"] * abs(math.sin(m["yaw"] - theta))
                fwd = r_near + half * math.cos(math.atan(ray))
                left = ray * fwd
                c, s = math.cos(yaw), math.sin(yaw)
                px = m["x"] - (c * fwd - s * left)
                py = m["y"] - (s * fwd + c * left)
                dx, dy = px - pose_hat[0], py - pose_hat[1]
                maha = (dx * dx + dy * dy) / max(sig_g * sig_g / 2, 0.25)
                if maha > 13.8:  # chi-square 2 dof, 99.9 %
                    continue
                valid_any = True
                hyps.append((px, py, yaw, maha + (dpsi / sig_psi) ** 2, j, k))
            if valid_any:
                cand_count[j] += 1
    if not hyps:
        return None, 0
    best, best_score, best_inl = None, -1e9, 0
    for (px, py, yaw, pen, j, k) in hyps:
        boxes, keep = project_msgs(msgs, (px, py, yaw))
        if not boxes:
            continue
        M = iou_matrix(boxes, pb)
        inl = int(np.sum(M.max(axis=1) >= 0.4))
        score = inl + 0.1 * float(np.sum(M.max(axis=1))) - 0.05 * pen
        if score > best_score:
            best, best_score, best_inl = (px, py, yaw), score, inl
    if best is None:
        return None, 0
    if best_inl >= 2:
        return best, best_inl
    if best_inl == 1 and int(np.sum(cand_count > 0)) == 1 and int(cand_count.max()) == 1:
        return best, 1
    return None, best_inl


def evaluate(labels, visible):
    """Greedy one-to-one matching of labels to visible GT vehicles (IoU >= 0.5, same class)."""
    used = set()
    correct = []
    for lab in labels:
        best, bi = 0.5, None
        for t, g in enumerate(visible):
            if t in used or g["cls"] != lab["cls"]:
                continue
            i = iou_matrix([lab["box"]], [g["box"]])[0, 0]
            if i >= best:
                best, bi = i, t
        if bi is not None:
            used.add(bi)
            correct.append(bi)
    return correct


def run(cfg, seed):
    rng = np.random.default_rng(seed)
    sig_g = cfg["gnss_rms"]
    sig_psi = math.radians(cfg["heading_sigma_deg"])
    acc = {m: dict(emit=0, correct=0, hard=0) for m in ("P", "PM", "C", "CR")}
    n_gt = n_gt_conn = 0
    conn_correct = {m: 0 for m in acc}
    frames_with_pose = 0
    for _ in range(cfg["frames"]):
        vehicles = sample_scene(rng)
        visible = gt_boxes(vehicles)
        props = detector(rng, visible)
        msgs = v2x_messages(rng, vehicles, cfg["rho"])
        conn_idx = {m["idx"] for m in msgs}
        n_gt += len(visible)
        n_gt_conn += sum(1 for g in visible if g["idx"] in conn_idx)
        # deployed detector misses: GT without a confident proposal at IoU >= 0.5
        conf = [p["box"] for p in props if p["score"] >= TAU_DET]
        missed = set()
        for t, g in enumerate(visible):
            if not conf or iou_matrix([g["box"]], conf).max() < 0.5:
                missed.add(t)
        e = rng.normal(0, sig_g / math.sqrt(2), size=2) if sig_g > 0 else np.zeros(2)
        pose_hat = (float(e[0]), float(e[1]), float(rng.normal(0, sig_psi)))
        pb = [p["box"] for p in props]

        labels = {m: [] for m in acc}
        # P: projection only
        boxes, keep = project_msgs(msgs, pose_hat)
        vis = visible_after_occlusion(msgs, boxes, keep, pose_hat)
        for t in vis:
            labels["P"].append(dict(box=boxes[t], cls=msgs[keep[t]]["cls"], msg=msgs[keep[t]]["idx"]))
        # PM: projection + matching, no pose correction
        for (t, k, _) in match(boxes, pb, 0.3):
            labels["PM"].append(dict(box=props[k]["box"], cls=msgs[keep[t]]["cls"], msg=msgs[keep[t]]["idx"]))
        # C / CR: consensus pose correction
        # relative-position uncertainty combines the aid's GNSS error and the vehicles' own
        # reported-position error (2-D RMS of the latter is sqrt(2) * SIGMA_VEH_POS)
        sig_rel = math.sqrt(sig_g ** 2 + 2 * SIGMA_VEH_POS ** 2)
        pose_c, inl = consensus_pose(msgs, props, pose_hat, sig_rel, sig_psi)
        if pose_c is not None:
            frames_with_pose += 1
            boxes_c, keep_c = project_msgs(msgs, pose_c)
            pairs = match(boxes_c, pb, 0.4)
            matched_t = set()
            for (t, k, _) in pairs:
                matched_t.add(t)
                lab = dict(box=props[k]["box"], cls=msgs[keep_c[t]]["cls"], msg=msgs[keep_c[t]]["idx"])
                labels["C"].append(lab)
                labels["CR"].append(lab)
            if inl >= 2:
                vis_c = visible_after_occlusion(msgs, boxes_c, keep_c, pose_c)
                for t in vis_c:
                    if t not in matched_t:
                        labels["CR"].append(dict(box=boxes_c[t], cls=msgs[keep_c[t]]["cls"], msg=msgs[keep_c[t]]["idx"]))
        for m in acc:
            corr = evaluate(labels[m], visible)
            acc[m]["emit"] += len(labels[m])
            acc[m]["correct"] += len(corr)
            acc[m]["hard"] += sum(1 for t in corr if t in missed)
            conn_correct[m] += sum(1 for t in corr if visible[t]["idx"] in conn_idx)
    out = {}
    for m, a in acc.items():
        out[m] = dict(
            precision=a["correct"] / max(a["emit"], 1),
            recall_all=a["correct"] / max(n_gt, 1),
            recall_connected=conn_correct[m] / max(n_gt_conn, 1),
            hard_share=a["hard"] / max(a["correct"], 1),
            labels_per_frame=a["emit"] / cfg["frames"],
        )
    out["frames_with_pose"] = frames_with_pose / cfg["frames"]
    out["gt_per_frame"] = n_gt / cfg["frames"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sweep", choices=["gnss", "rho", "heading", "all"], default="all")
    args = ap.parse_args()
    configs = []
    if args.sweep in ("gnss", "all"):
        configs += [dict(name=f"gnss_{e}", gnss_rms=float(e), rho=0.5, heading_sigma_deg=5.0) for e in (0, 1, 3, 5, 10)]
    if args.sweep in ("rho", "all"):
        configs += [dict(name=f"rho_{r}", gnss_rms=5.0, rho=r, heading_sigma_deg=5.0) for r in (0.1, 0.25, 0.5, 0.75, 1.0)]
    if args.sweep in ("heading", "all"):
        configs += [dict(name=f"head_{h}", gnss_rms=5.0, rho=0.5, heading_sigma_deg=float(h)) for h in (2, 10)]
    results = {}
    for cfg in configs:
        cfg["frames"] = args.frames
        runs = [run(cfg, 1000 + s) for s in range(args.seeds)]
        agg = {}
        for m in ("P", "PM", "C", "CR"):
            agg[m] = {}
            for key in runs[0][m]:
                vals = np.array([r[m][key] for r in runs])
                agg[m][key] = [float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0]
        agg["frames_with_pose"] = float(np.mean([r["frames_with_pose"] for r in runs]))
        agg["gt_per_frame"] = float(np.mean([r["gt_per_frame"] for r in runs]))
        results[cfg["name"]] = dict(config=cfg, metrics=agg)
        print(cfg["name"], {m: (round(agg[m]["precision"][0], 3), round(agg[m]["recall_all"][0], 3), round(agg[m]["recall_connected"][0], 3)) for m in ("P", "PM", "C", "CR")}, file=sys.stderr, flush=True)
    json.dump(results, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
