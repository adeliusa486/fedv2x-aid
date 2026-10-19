#!/usr/bin/env python3
"""E2b, temporal part (Phase 4): P, PM, C, SM and CT on the 21 KITTI tracking sequences.

Message and pose emulation on real image sequences with real 3D tracks and real GPS/IMU poses (not a field test):
  * frames are taken at 2 Hz (every fifth 10 Hz frame), as in the sequential computed study;
  * the true camera pose of every frame comes from the vehicle's GPS/IMU (OXTS, Mercator projection as in the
    KITTI devkit) and the calibration chain IMU -> Velodyne -> camera; the aid's estimate adds a first-order
    Gauss-Markov error (tau 30 s position, 60 s heading) or an i.i.d. error;
  * every Car, Van, Truck and Tram track is connected with probability rho; a connected vehicle broadcasts at
    10 Hz; the newest message received at a frame was generated 100 +- 20 ms plus a uniform CAM phase earlier;
    its position (track position interpolated to the generation time, N(0, 0.8 m) error per axis), heading
    (N(0, 2 deg)) and speed (N(0, 0.2 m/s)) are dead-reckoned to the frame time;
  * proposals are the real detections of the reference detector on each image (score >= 0.05);
  * labellers and smoother are the unchanged functions of computed/v2x_temporal_mc.py.

    python src/teachers/kitti_temporal.py --proposals $FEDV2X_WORK/kitti/proposals_tracking_ref.pkl --out results/phase4/e2b_tracking_ref.json
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
import v2x_temporal_mc as tm  # noqa: E402

K = Path(os.environ.get("FEDV2X_WORK", "work") + "/kitti/training")
VEH = {"Car": 0, "Van": 1, "Truck": 2, "Tram": 3}
DET_VEH = {14, 15, 16, 17, 18, 19, 20, 24, 27}
H_CAM = 1.65
STEP = 5          # 10 Hz -> 2 Hz
SIG_SPEED = 0.2


def calib(seq):
    c = {}
    for line in (K / "calib" / f"{seq}.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
        else:
            parts = line.split()
            if not parts:
                continue
            k, v = parts[0], " ".join(parts[1:])
        c[k.strip()] = np.array(list(map(float, v.split())))
    P2 = c["P2"].reshape(3, 4)
    R = np.eye(4); R[:3, :3] = c["R_rect"].reshape(3, 3)
    Tvc = np.eye(4); Tvc[:3] = c["Tr_velo_cam"].reshape(3, 4)
    Tiv = np.eye(4); Tiv[:3] = c["Tr_imu_velo"].reshape(3, 4)
    T_imu_to_rect = R @ Tvc @ Tiv
    return P2, np.linalg.inv(T_imu_to_rect)


def oxts_poses(seq):
    rows = [list(map(float, l.split())) for l in (K / "oxts" / f"{seq}.txt").read_text().splitlines() if l.strip()]
    er = 6378137.0
    scale = math.cos(rows[0][0] * math.pi / 180)
    poses = []
    for r in rows:
        lat, lon, alt, roll, pitch, yaw = r[:6]
        mx = scale * lon * math.pi * er / 180
        my = scale * er * math.log(math.tan((90 + lat) * math.pi / 360))
        Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        T = np.eye(4); T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = [mx, my, alt]
        poses.append(T)
    T0inv = np.linalg.inv(poses[0])
    return [T0inv @ T for T in poses]


def tracks(seq):
    by = {}
    for line in (K / "label_02" / f"{seq}.txt").read_text().splitlines():
        v = line.split()
        if v[2] not in VEH:
            continue
        f, tid = int(v[0]), int(v[1])
        by.setdefault(f, []).append(dict(tid=tid, type=v[2], trunc=float(v[3]), occ=int(v[4]),
                                         box=np.array(list(map(float, v[6:10]))), h=float(v[10]), w=float(v[11]),
                                         l=float(v[12]), loc=np.array(list(map(float, v[13:16]))), ry=float(v[16])))
    return by


def build_world(seq):
    P2, T_rect_to_imu = calib(seq)
    poses = oxts_poses(seq)
    lab = tracks(seq)
    n = len(poses)
    bx = P2[0, 3] / P2[0, 0]
    cam = []          # camera pose (x, y, yaw) per 10 Hz frame
    for f in range(n):
        T = poses[f] @ T_rect_to_imu
        c = T @ np.array([bx, 0, 0, 1.0])           # left colour camera centre (cam2 = rect origin shifted by bx)
        fwd = T[:3, :3] @ np.array([0, 0, 1.0])
        cam.append((c[0], c[1], math.atan2(fwd[1], fwd[0])))
    world = {}        # (frame, tid) -> (x, y, yaw) of the footprint centre
    for f, objs in lab.items():
        T = poses[f] @ T_rect_to_imu
        for o in objs:
            p = T @ np.array([o["loc"][0], o["loc"][1] - o["h"] / 2, o["loc"][2], 1.0])
            d = T[:3, :3] @ np.array([math.cos(o["ry"]), 0, -math.sin(o["ry"])])
            world[(f, o["tid"])] = (p[0], p[1], math.atan2(d[1], d[0]))
    return P2, cam, lab, world, n


def moderate(o):
    return (o["box"][3] - o["box"][1]) >= 25 and o["occ"] <= 1 and o["trunc"] <= 0.30


def run_seq(seq, props_all, cfg, rng, h_prior):
    P2, cam, lab, world, n = build_world(seq)
    size = props_all[f"{seq}/{0:06d}"]["size"]
    mc.W, mc.H = int(size[0]), int(size[1])
    mc.F, mc.CX, mc.CY, mc.H_CAM = float(P2[0, 0]), float(P2[0, 2]), float(P2[1, 2]), H_CAM
    frames_idx = list(range(0, n, STEP))
    tids = sorted({o["tid"] for objs in lab.values() for o in objs})
    conn = {t for t in tids if rng.random() < cfg["rho"]}
    phase = {t: rng.uniform(0, 0.1) for t in tids}
    sig_g, sig_psi = cfg["gnss_rms"], math.radians(cfg["heading_sigma_deg"])
    e, ep = tm.error_process(rng, len(frames_idx), cfg["error_model"], sig_g, sig_psi)
    frames = []
    for t, f in enumerate(frames_idx):
        objs = lab.get(f, [])
        visible = [dict(idx=o["tid"], box=o["box"], cls=VEH[o["type"]], vid=o["tid"], conn=o["tid"] in conn,
                        mod=moderate(o)) for o in objs]
        pred = props_all.get(f"{seq}/{f:06d}")
        keep = np.isin(pred["cls"], list(DET_VEH))
        props = [dict(box=b.astype(float), score=float(s), cls=int(c))
                 for b, s, c in zip(pred["boxes"][keep], pred["scores"][keep], pred["cls"][keep])]
        msgs = []
        for o in objs:
            tid = o["tid"]
            if tid not in conn:
                continue
            age = max(0.0, rng.normal(0.100, 0.020)) + phase[tid]
            fr_back = age * 10.0                                         # in 10 Hz frames
            f0 = f - int(math.ceil(fr_back))
            p_now = world[(f, tid)]
            p_old = world.get((max(f0, 0), tid), p_now)
            a = (math.ceil(fr_back) - fr_back) if f0 >= 0 and (f0, tid) in world else 1.0
            x = p_old[0] + a * (p_now[0] - p_old[0]) if f0 >= 0 else p_now[0]
            y = p_old[1] + a * (p_now[1] - p_old[1]) if f0 >= 0 else p_now[1]
            prev = world.get((f - 1, tid))
            v = math.hypot(p_now[0] - prev[0], p_now[1] - prev[1]) * 10.0 if prev else 0.0
            yaw_r = p_now[2] + rng.normal(0, mc.SIGMA_VEH_HEAD)
            x += rng.normal(0, mc.SIGMA_VEH_POS)
            y += rng.normal(0, mc.SIGMA_VEH_POS)
            v_r = v + rng.normal(0, SIG_SPEED)
            heading = math.atan2(p_now[1] - prev[1], p_now[0] - prev[0]) if prev and v > 0.5 else yaw_r
            x += v_r * age * math.cos(heading)
            y += v_r * age * math.sin(heading)
            msgs.append(dict(idx=tid, cls=VEH[o["type"]], x=x, y=y, yaw=yaw_r, l=o["l"], w=o["w"], h=h_prior[o["type"]],
                             ghost=False, shifted=False))
        cx, cy, cyaw = cam[f]
        pose_hat = (cx + float(e[t, 0]), cy + float(e[t, 1]), cyaw + float(ep[t]))
        frames.append(dict(visible=visible, props=props, msgs=msgs, pose_hat=pose_hat, true=(cx, cy, cyaw)))
    N = len(frames)
    sig_rel = math.sqrt(sig_g ** 2 + 2 * mc.SIGMA_VEH_POS ** 2)
    obs = {}
    for t, fr in enumerate(frames):
        pc, inl = tm.consensus(fr["msgs"], fr["props"], fr["pose_hat"], sig_rel, sig_psi)
        fr["pc"], fr["inl"] = pc, inl
        if pc is not None and inl >= 2:
            ph = fr["pose_hat"]
            obs[t] = np.array([ph[0] - pc[0], ph[1] - pc[1], ph[2] - pc[2]])
    for it in range(tm.N_ITER):
        xs, Ps = tm.smooth(obs, N, sig_g, sig_psi)
        obs_new = {}
        for t, fr in enumerate(frames):
            ph = fr["pose_hat"]
            corr = (ph[0] - xs[t, 0], ph[1] - xs[t, 1], ph[2] - xs[t, 2])
            prior_rms = math.sqrt(Ps[t, 0, 0] + Ps[t, 1, 1])
            sig_rel2 = math.sqrt(Ps[t, 0, 0] + Ps[t, 1, 1] + 2 * mc.SIGMA_VEH_POS ** 2)
            sig_psi2 = max(math.sqrt(Ps[t, 2, 2]), math.radians(0.5))
            pct, inl2 = tm.consensus(fr["msgs"], fr["props"], corr, sig_rel2, sig_psi2, single_ok=prior_rms < tm.TIGHT_PRIOR)
            if pct is not None and inl2 >= 2:
                obs_new[t] = np.array([ph[0] - pct[0], ph[1] - pct[1], ph[2] - pct[2]])
            fr["pct"], fr["corr"] = (pct if pct is not None else fr["pc"]), corr
        if it < tm.N_ITER - 1:
            obs = obs_new
    acc = {m: dict(emit=0, correct=0, conn_mod=0, conn_mod_prop=0) for m in ("P", "PM", "C", "SM", "CT")}
    n_mod_conn = n_mod = n_mod_conn_prop = 0
    perr = {"raw": [], "C": [], "CT": []}
    for fr in frames:
        visible, props, msgs, ph = fr["visible"], fr["props"], fr["msgs"], fr["pose_hat"]
        pb = [p["box"] for p in props]
        n_mod += sum(1 for g in visible if g["mod"])
        n_mod_conn += sum(1 for g in visible if g["mod"] and g["conn"])
        has_prop = set()
        if pb and visible:
            M = mc.iou_matrix([g["box"] for g in visible], pb)
            has_prop = {i for i in range(len(visible)) if M[i].max() >= 0.5}
        n_mod_conn_prop += sum(1 for i, g in enumerate(visible) if g["mod"] and g["conn"] and i in has_prop)
        labels = {m: [] for m in acc}
        boxes, kp = mc.project_msgs(msgs, ph)
        for i in mc.visible_after_occlusion(msgs, boxes, kp, ph):
            labels["P"].append(dict(box=boxes[i], cls=msgs[kp[i]]["cls"]))
        for (i, k, _) in mc.match(boxes, pb, 0.3):
            labels["PM"].append(dict(box=props[k]["box"], cls=msgs[kp[i]]["cls"]))
        if fr["pc"] is not None:
            bc, kc = mc.project_msgs(msgs, fr["pc"])
            for (i, k, _) in mc.match(bc, pb, 0.4):
                labels["C"].append(dict(box=props[k]["box"], cls=msgs[kc[i]]["cls"]))
        bs, ks = mc.project_msgs(msgs, fr["corr"])
        for (i, k, _) in mc.match(bs, pb, 0.4):
            labels["SM"].append(dict(box=props[k]["box"], cls=msgs[ks[i]]["cls"]))
        if fr["pct"] is not None:
            bt, kt = mc.project_msgs(msgs, fr["pct"])
            for (i, k, _) in mc.match(bt, pb, 0.4):
                labels["CT"].append(dict(box=props[k]["box"], cls=msgs[kt[i]]["cls"]))
        tx, ty, _ = fr["true"]
        perr["raw"].append(math.hypot(ph[0] - tx, ph[1] - ty))
        if fr["pc"] is not None:
            perr["C"].append(math.hypot(fr["pc"][0] - tx, fr["pc"][1] - ty))
        if fr["pct"] is not None:
            perr["CT"].append(math.hypot(fr["pct"][0] - tx, fr["pct"][1] - ty))
        for m in acc:
            corr = mc.evaluate(labels[m], visible)
            acc[m]["emit"] += len(labels[m])
            acc[m]["correct"] += len(corr)
            acc[m]["conn_mod"] += sum(1 for i in corr if visible[i]["conn"] and visible[i]["mod"])
            acc[m]["conn_mod_prop"] += sum(1 for i in corr if visible[i]["conn"] and visible[i]["mod"] and i in has_prop)
    return acc, n_mod, (n_mod_conn, n_mod_conn_prop), N, {k: (float(np.sum(np.square(v))), len(v)) for k, v in perr.items()}


def job(args):
    cfg, seed, seqs, props_path, h_prior, reps = args
    with open(props_path, "rb") as fh:
        props_all = pickle.load(fh)["pred"]
    rng = np.random.default_rng(seed)
    tot = {m: dict(emit=0, correct=0, conn_mod=0, conn_mod_prop=0) for m in ("P", "PM", "C", "SM", "CT")}
    n_mod = n_mod_conn = n_mod_conn_prop = frames = 0
    pe = {"raw": [0.0, 0], "C": [0.0, 0], "CT": [0.0, 0]}
    for _ in range(reps):
        for seq in seqs:
            acc, nm, nmc, N, perr = run_seq(seq, props_all, cfg, rng, h_prior)
            for m in tot:
                for k in tot[m]:
                    tot[m][k] += acc[m][k]
            n_mod += nm
            n_mod_conn += nmc[0]
            n_mod_conn_prop += nmc[1]
            frames += N
            for k, (ss, c) in perr.items():
                pe[k][0] += ss
                pe[k][1] += c
    return cfg["name"], seed, tot, n_mod, (n_mod_conn, n_mod_conn_prop), frames, pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10, help="error-process realisations per sequence and seed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    t0 = time.time()
    hs = {k: [] for k in VEH}
    for f in sorted((K / "label_2").glob("*.txt")):
        for line in f.read_text().splitlines():
            v = line.split()
            if v[0] in VEH:
                hs[v[0]].append(float(v[8]))
    h_prior = {k: float(np.mean(v)) for k, v in hs.items()}
    seqs = sorted(p.stem for p in (K / "label_02").glob("*.txt"))
    cfgs = [dict(name=f"gm_g{g}", error_model="gm", gnss_rms=float(g), rho=0.5, heading_sigma_deg=5.0) for g in (3, 5, 10)]
    cfgs.append(dict(name="iid_g5", error_model="iid", gnss_rms=5.0, rho=0.5, heading_sigma_deg=5.0))
    if a.only:
        cfgs = [c for c in cfgs if c["name"] in a.only]
    jobs = [(c, 2000 + 10 * s + k, seqs[k::2], a.proposals, h_prior, a.reps) for c in cfgs for s in range(a.seeds) for k in range(2)]
    with Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    by = {}
    for name, seed, tot, nm, nmc, fr, pe in res:
        key = (name, (seed - 2000) // 10)
        d = by.setdefault(key, {"tot": {m: dict(emit=0, correct=0, conn_mod=0, conn_mod_prop=0) for m in tot}, "nm": 0, "nmc": 0, "nmcp": 0, "fr": 0,
                                "pe": {k: [0.0, 0] for k in pe}})
        for m in tot:
            for k in tot[m]:
                d["tot"][m][k] += tot[m][k]
        d["nm"] += nm; d["nmc"] += nmc[0]; d["nmcp"] += nmc[1]; d["fr"] += fr
        for k in pe:
            d["pe"][k][0] += pe[k][0]; d["pe"][k][1] += pe[k][1]
    out = {}
    for c in cfgs:
        ds = [by[(c["name"], s)] for s in range(a.seeds)]
        met = {}
        for m in ("P", "PM", "C", "SM", "CT"):
            prec = [d["tot"][m]["correct"] / max(d["tot"][m]["emit"], 1) for d in ds]
            rec = [d["tot"][m]["conn_mod"] / max(d["nmc"], 1) for d in ds]
            cond = [d["tot"][m]["conn_mod_prop"] / max(d["nmcp"], 1) for d in ds]
            met[m] = {"precision": [float(np.mean(prec)), float(np.std(prec, ddof=1))],
                      "recall_connected": [float(np.mean(rec)), float(np.std(rec, ddof=1))],
                      "conditional_recall": [float(np.mean(cond)), float(np.std(cond, ddof=1))]}
        met["share_connected_moderate_with_proposal"] = float(np.mean([d["nmcp"] / max(d["nmc"], 1) for d in ds]))
        met["pose_rms_m"] = {k: float(math.sqrt(sum(d["pe"][k][0] for d in ds) / max(sum(d["pe"][k][1] for d in ds), 1)))
                             for k in ("raw", "C", "CT")}
        met["pose_frames_share"] = {k: sum(d["pe"][k][1] for d in ds) / sum(d["fr"] for d in ds) for k in ("C", "CT")}
        met["frames_per_seed"] = ds[0]["fr"]
        out[c["name"]] = {"config": c, "metrics": met}
        print(c["name"], {m: (round(met[m]["precision"][0], 3), round(met[m]["recall_connected"][0], 3)) for m in ("PM", "C", "SM", "CT")},
              met["pose_rms_m"], flush=True)
    Path(a.out).write_text(json.dumps({"proposals": a.proposals, "sequences": len(seqs), "seeds": a.seeds, "reps": a.reps,
                                       "class_height_priors_m": h_prior, "results": out,
                                       "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
