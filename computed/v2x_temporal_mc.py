"""Sequential Monte-Carlo study of V2X-as-teacher: correlated pose error, temporal
smoothing of the consensus correction, message latency/loss, and spoofed messages.

Extends the static study (v2x_pseudolabel_mc.py, reused for camera, detector, matching
and scoring) from independent frames to 2 Hz sequences of a pedestrian waiting at the
kerb while vehicles drive through the intersection at constant speed:

  * vehicles keep their identity, so connected vehicles keep broadcasting;
  * CAMs are generated at 10 Hz, arrive after 100 +/- 20 ms, and are lost i.i.d. on PC5;
    the receiver uses the newest message and either uses its position as reported or
    dead-reckons it to the frame time with the reported speed and heading;
  * pedestrian GNSS and compass errors are i.i.d. per frame, first-order Gauss-Markov
    (tau = 30 s position, 60 s heading), or Gauss-Markov plus a constant multipath bias;
  * an insider attacker can add ghost vehicles (messages with no physical vehicle) or
    make connected vehicles report positions shifted along their lane.

Labellers: P (projection), PM (projection + matching), C (per-frame consensus, Alg. 1),
SM (smoothed pose + matching) and CT (temporal consensus: consensus observations are
fused by a Kalman filter and RTS smoother over the sequence, and every frame is then
re-labelled with the smoothed pose as a tight prior; single-vehicle frames are accepted
when that prior is tight). Labels are mined offline while charging, so smoothing may
use future frames.

Usage: python v2x_temporal_mc.py --study pose|rho|latency|spoof|all --seqs 15 --seeds 3 --workers 16 > out.json
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

import v2x_pseudolabel_mc as mc

DT = 0.5                 # frame period (2 Hz)
SEQ_FRAMES = 240         # 120 s per sequence
WARMUP = 12.0            # s of traffic simulated before the first frame
CAM_PERIOD = 0.1         # 10 Hz
MAX_AGE = 1.0            # messages older than this are discarded
SIG_SPEED = 0.2          # m/s error of the reported speed
TAU_POS, TAU_PSI = 30.0, 60.0
R_POS, R_PSI = 2.0, math.radians(4.5)   # consensus-observation noise: robust residual sigma of C (see --study calib)
N_ITER = 2               # smooth -> relabel iterations
TIGHT_PRIOR = 2.0        # m: 2-D RMS of the smoothed prior below which single matches are accepted

# lanes: (key, fixed coordinate, travel direction yaw, start, end, speed mean)
LANES = [("c0", 6.0, math.pi / 2, -35.0, 35.0, 10.0), ("c1", 9.5, -math.pi / 2, 35.0, -35.0, 10.0),
         ("c2", 13.0, math.pi / 2, -35.0, 35.0, 10.0), ("c3", 16.5, -math.pi / 2, 35.0, -35.0, 10.0),
         ("p0", -4.0, 0.0, -5.0, 45.0, 8.0), ("p1", -7.5, math.pi, 45.0, -5.0, 8.0)]
# densities (vehicles per metre) matching the static study: Poisson mean 7, 60 % on crossing lanes
DENS = {"c": 0.6 * 7.0 / (4 * 70.0), "p": 0.4 * 7.0 / (2 * 50.0)}


class Traffic:
    """Constant-speed vehicles on the six lanes, Poisson arrivals with a minimum gap."""

    def __init__(self, rng, rho, ghost_rate, shift_frac):
        self.rng, self.rho, self.ghost_rate, self.shift_frac = rng, rho, ghost_rate, shift_frac
        self.veh = []
        self.next_id = 0
        self.t = -WARMUP
        self.next_arrival = {}
        for li, lane in enumerate(LANES):
            self.next_arrival[li] = self.t + rng.exponential(1.0 / self.rate(lane))
            self.next_arrival[("g", li)] = self.t + rng.exponential(1.0 / max(self.rate(lane) * ghost_rate, 1e-9)) if ghost_rate > 0 else np.inf

    def rate(self, lane):
        return DENS[lane[0][0]] * lane[5]

    def spawn(self, li, ghost):
        key, fixed, yaw, start, end, vmean = LANES[li]
        cls = int(self.rng.choice(len(mc.CLASSES), p=mc.CLASS_P))
        l, w, h = mc.DIMS[cls] * self.rng.uniform(0.92, 1.08, size=3)
        v = max(3.0, self.rng.normal(vmean, 2.0))
        same = [o for o in self.veh if o["lane"] == li and o["ghost"] == ghost]
        if same:
            last = min(same, key=lambda o: o["s"])
            if last["s"] < (last["l"] + l) / 2 + 1.5 + 2.0:
                return  # entry blocked: arrival dropped
            v = min(v, last["v"])  # no overtaking
        conn = ghost or (self.rng.random() < self.rho)
        shift = 0.0
        if conn and not ghost and self.rng.random() < self.shift_frac:
            shift = self.rng.uniform(3.0, 8.0) * (1 if self.rng.random() < 0.5 else -1)
        self.veh.append(dict(id=self.next_id, lane=li, cls=cls, l=l, w=w, h=h, v=v, s=0.0, conn=conn,
                             ghost=ghost, shift=shift, phase=self.rng.uniform(0, CAM_PERIOD)))
        self.next_id += 1

    def step(self, t_new):
        while True:
            pending = [(tt, k) for k, tt in self.next_arrival.items() if tt <= t_new]
            if not pending:
                break
            tt, k = min(pending, key=lambda a: a[0])
            dt = tt - self.t
            self.advance(dt)
            self.t = tt
            if isinstance(k, tuple):
                li = k[1]
                self.spawn(li, ghost=True)
                self.next_arrival[k] = tt + self.rng.exponential(1.0 / (self.rate(LANES[li]) * self.ghost_rate))
            else:
                self.spawn(k, ghost=False)
                self.next_arrival[k] = tt + self.rng.exponential(1.0 / self.rate(LANES[k]))
        self.advance(t_new - self.t)
        self.t = t_new

    def advance(self, dt):
        for o in self.veh:
            o["s"] += o["v"] * dt
        self.veh = [o for o in self.veh if o["s"] <= abs(LANES[o["lane"]][4] - LANES[o["lane"]][3])]

    @staticmethod
    def pose_at(o, s):
        key, fixed, yaw, start, end, _ = LANES[o["lane"]]
        d = start + s * (1 if end > start else -1)
        if key[0] == "c":
            return fixed, d, yaw
        return d, fixed, yaw


def messages(rng, traffic, t, loss, latency, extrapolate):
    """Newest received CAM of every broadcasting vehicle at frame time t."""
    msgs = []
    for o in traffic.veh:
        if not o["conn"]:
            continue
        # walk back over generation instants until one arrived (not lost, delivered before t)
        g = math.floor((t - o["phase"]) / CAM_PERIOD) * CAM_PERIOD + o["phase"]
        chosen = None
        for _ in range(int(MAX_AGE / CAM_PERIOD) + 1):
            d = max(0.0, rng.normal(0.100, 0.020)) if latency else 0.0
            if g + d <= t and rng.random() >= loss:
                chosen = g
                break
            g -= CAM_PERIOD
        if chosen is None or t - chosen > MAX_AGE:
            continue
        age = t - chosen
        s_g = o["s"] - o["v"] * age
        if s_g < 0:
            continue
        x, y, yaw = Traffic.pose_at(o, s_g + o["shift"])
        x += rng.normal(0, mc.SIGMA_VEH_POS)
        y += rng.normal(0, mc.SIGMA_VEH_POS)
        yaw_r = yaw + rng.normal(0, mc.SIGMA_VEH_HEAD)
        if extrapolate:
            v_r = o["v"] + rng.normal(0, SIG_SPEED)
            x += v_r * age * math.cos(yaw_r)
            y += v_r * age * math.sin(yaw_r)
        msgs.append(dict(idx=o["id"], cls=o["cls"], x=x, y=y, yaw=yaw_r, l=o["l"], w=o["w"], h=mc.DIMS[o["cls"]][2],
                         ghost=o["ghost"], shifted=o["shift"] != 0.0))
    return msgs


def consensus(msgs, props, pose_hat, sig_rel, sig_psi, single_ok=False):
    """Algorithm 1 (as in v2x_pseudolabel_mc.consensus_pose) with an optional tight-prior rule:
    when single_ok is set, one inlier suffices (the prior, not a second vehicle, rules out
    ambiguous poses)."""
    if not msgs or not props:
        return None, 0
    pb = np.array([p["box"] for p in props])
    hyps = []
    cand_count = np.zeros(len(msgs), dtype=int)
    var_floor = max(sig_rel * sig_rel / 2, 0.25)
    for j, m in enumerate(msgs):
        for k, p in enumerate(props):
            b = p["box"]
            bh = b[3] - b[1]
            r_size = mc.F * m["h"] / max(bh, 1.0)
            bottom_visible = b[3] < mc.H - 1 and b[3] - mc.CY > 2
            r_ground = mc.F * mc.H_CAM / (b[3] - mc.CY) if bottom_visible else np.inf
            if np.isfinite(r_ground) and abs(math.log(r_size / r_ground)) > 0.5:
                continue
            r_near = r_ground if np.isfinite(r_ground) else r_size
            u = (b[0] + b[2]) / 2
            ray = -(u - mc.CX) / mc.F
            valid_any = False
            for dpsi in (-2 * sig_psi, -sig_psi, 0.0, sig_psi, 2 * sig_psi):
                yaw = pose_hat[2] + dpsi
                theta = yaw + math.atan(ray)
                half = 0.5 * m["l"] * abs(math.cos(m["yaw"] - theta)) + 0.5 * m["w"] * abs(math.sin(m["yaw"] - theta))
                fwd = r_near + half * math.cos(math.atan(ray))
                left = ray * fwd
                c, s = math.cos(yaw), math.sin(yaw)
                px = m["x"] - (c * fwd - s * left)
                py = m["y"] - (s * fwd + c * left)
                dx, dy = px - pose_hat[0], py - pose_hat[1]
                maha = (dx * dx + dy * dy) / var_floor
                if maha > 13.8:
                    continue
                valid_any = True
                hyps.append((px, py, yaw, maha + (dpsi / sig_psi) ** 2))
            if valid_any:
                cand_count[j] += 1
    if not hyps:
        return None, 0
    best, best_score, best_inl = None, -1e9, 0
    for (px, py, yaw, pen) in hyps:
        boxes, keep = mc.project_msgs(msgs, (px, py, yaw))
        if not boxes:
            continue
        M = mc.iou_matrix(boxes, pb)
        inl = int(np.sum(M.max(axis=1) >= 0.4))
        score = inl + 0.1 * float(np.sum(M.max(axis=1))) - 0.05 * pen
        if score > best_score:
            best, best_score, best_inl = (px, py, yaw), score, inl
    if best is None:
        return None, 0
    if best_inl >= 2:
        return best, best_inl
    if best_inl == 1 and (single_ok or (int(np.sum(cand_count > 0)) == 1 and int(cand_count.max()) == 1)):
        return best, 1
    return None, best_inl


def evaluate_pairs(labels, visible):
    """Same greedy matching as v2x_pseudolabel_mc.evaluate, but returns (label index, GT index) pairs."""
    used, pairs = set(), []
    for li, lab in enumerate(labels):
        best, bi = 0.5, None
        for t, g in enumerate(visible):
            if t in used or g["cls"] != lab["cls"]:
                continue
            i = mc.iou_matrix([lab["box"]], [g["box"]])[0, 0]
            if i >= best:
                best, bi = i, t
        if bi is not None:
            used.add(bi)
            pairs.append((li, bi))
    return pairs


def error_process(rng, n, model, sig_g, sig_psi, tau_pos=TAU_POS, tau_psi=TAU_PSI):
    """Pedestrian GNSS (2-D, sig_g = 2-D RMS) and compass errors for one sequence.
    tau_pos / tau_psi are the TRUE correlation times; the smoother always assumes TAU_POS / TAU_PSI,
    so passing other values tests a misspecified error model."""
    s_ax = sig_g / math.sqrt(2)
    if model == "iid":
        e = rng.normal(0, s_ax, size=(n, 2)) if sig_g > 0 else np.zeros((n, 2))
        ep = rng.normal(0, sig_psi, size=n)
        return e, ep
    a, ap = math.exp(-DT / tau_pos), math.exp(-DT / tau_psi)
    e = np.zeros((n, 2))
    ep = np.zeros(n)
    e[0] = rng.normal(0, s_ax, size=2)
    ep[0] = rng.normal(0, sig_psi)
    for t in range(1, n):
        e[t] = a * e[t - 1] + rng.normal(0, s_ax * math.sqrt(1 - a * a), size=2)
        ep[t] = ap * ep[t - 1] + rng.normal(0, sig_psi * math.sqrt(1 - ap * ap))
    if model == "gm_bias":
        ang = rng.uniform(0, 2 * math.pi)
        e += 3.0 * np.array([math.cos(ang), math.sin(ang)])  # multipath bias, not reflected in the reported sigma
    return e, ep


def smooth(obs, n, sig_g, sig_psi):
    """Kalman filter + RTS smoother of the 3-D error state (ex, ey, epsi), Gauss-Markov model.
    obs: dict frame -> z (3,) observation of the error. Innovation gating rejects outliers."""
    a = np.array([math.exp(-DT / TAU_POS)] * 2 + [math.exp(-DT / TAU_PSI)])
    s2 = np.array([sig_g ** 2 / 2] * 2 + [sig_psi ** 2])
    Q = np.diag(s2 * (1 - a * a))
    A = np.diag(a)
    R = np.diag([R_POS ** 2, R_POS ** 2, R_PSI ** 2])
    xf = np.zeros((n, 3)); Pf = np.zeros((n, 3, 3)); xp = np.zeros((n, 3)); Pp = np.zeros((n, 3, 3))
    x, P = np.zeros(3), np.diag(s2)
    for t in range(n):
        if t > 0:
            x, P = A @ x, A @ P @ A.T + Q
        xp[t], Pp[t] = x, P
        if t in obs:
            S = P + R
            nu = obs[t] - x
            if float(nu @ np.linalg.solve(S, nu)) <= 11.34:  # chi2_3 99 %
                K = P @ np.linalg.inv(S)
                x = x + K @ nu
                P = (np.eye(3) - K) @ P
        xf[t], Pf[t] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for t in range(n - 2, -1, -1):
        G = Pf[t] @ A.T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xf[t] + G @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + G @ (Ps[t + 1] - Pp[t + 1]) @ G.T
    return xs, Ps


def run(cfg, seed):
    rng = np.random.default_rng(seed)
    mc.H_CAM = 1.0
    sig_g = cfg["gnss_rms"]
    sig_psi = math.radians(cfg["heading_sigma_deg"])
    methods = ("P", "PM", "C", "SM", "CT")
    acc = {m: dict(emit=0, correct=0, hard=0, conn=0, spoof_emit=0, spoof_wrong=0, on_unconn=0, other_id=0) for m in methods}
    n_gt = n_gt_conn = 0
    pose_err = {"raw": [], "C": [], "CT": [], "raw_on_C": [], "raw_on_CT": []}
    # the receiver reports sig_g; the true error may differ (gnss_true) and may decorrelate faster or
    # slower than the smoother assumes (tau_pos_true, tau_psi_true); defaults reproduce the main study
    sig_true = cfg.get("gnss_true", sig_g)
    tau_p_true, tau_psi_true = cfg.get("tau_pos_true", TAU_POS), cfg.get("tau_psi_true", TAU_PSI)
    for _ in range(cfg["seqs"]):
        traffic = Traffic(rng, cfg["rho"], cfg.get("ghost_rate", 0.0), cfg.get("shift_frac", 0.0))
        e, ep = error_process(rng, SEQ_FRAMES, cfg["error_model"], sig_true, sig_psi, tau_p_true, tau_psi_true)
        frames = []
        for t in range(SEQ_FRAMES):
            traffic.step(t * DT)
            phys = [o for o in traffic.veh if not o["ghost"]]
            vehicles = []
            for o in phys:
                x, y, yaw = Traffic.pose_at(o, o["s"])
                vehicles.append(dict(cls=o["cls"], x=x, y=y, yaw=yaw, l=o["l"], w=o["w"], h=o["h"], id=o["id"], conn=o["conn"]))
            visible = mc.gt_boxes(vehicles)
            for g in visible:  # gt_boxes indexes into the list; carry the persistent id
                g["vid"] = vehicles[g["idx"]]["id"]
                g["conn"] = vehicles[g["idx"]]["conn"]
            props = mc.detector(rng, visible)
            msgs = messages(rng, traffic, t * DT, cfg.get("loss", 0.0), cfg.get("latency", True), cfg.get("extrapolate", True))
            pose_hat = (float(e[t, 0]), float(e[t, 1]), float(ep[t]))
            frames.append(dict(visible=visible, props=props, msgs=msgs, pose_hat=pose_hat))
        sig_rel = math.sqrt(sig_g ** 2 + 2 * mc.SIGMA_VEH_POS ** 2)
        # pass 1: per-frame consensus (method C) and observations for the smoother
        obs = {}
        for t, fr in enumerate(frames):
            pc, inl = consensus(fr["msgs"], fr["props"], fr["pose_hat"], sig_rel, sig_psi)
            fr["pc"], fr["inl"] = pc, inl
            if pc is not None and inl >= 2:
                ph = fr["pose_hat"]
                obs[t] = np.array([ph[0] - pc[0], ph[1] - pc[1], ph[2] - pc[2]])
        # passes 2..: smooth the observations, re-run consensus with the smoothed prior, and
        # use the new consensus poses as observations for the next smoothing pass
        for it in range(N_ITER):
            xs, Ps = smooth(obs, SEQ_FRAMES, sig_g, sig_psi)
            obs_new = {}
            for t, fr in enumerate(frames):
                ph = fr["pose_hat"]
                corr = (ph[0] - xs[t, 0], ph[1] - xs[t, 1], ph[2] - xs[t, 2])
                prior_rms = math.sqrt(Ps[t, 0, 0] + Ps[t, 1, 1])
                sig_rel2 = math.sqrt(Ps[t, 0, 0] + Ps[t, 1, 1] + 2 * mc.SIGMA_VEH_POS ** 2)
                sig_psi2 = max(math.sqrt(Ps[t, 2, 2]), math.radians(0.5))
                pct, inl2 = consensus(fr["msgs"], fr["props"], corr, sig_rel2, sig_psi2, single_ok=prior_rms < TIGHT_PRIOR)
                if pct is not None and inl2 >= 2:
                    obs_new[t] = np.array([ph[0] - pct[0], ph[1] - pct[1], ph[2] - pct[2]])
                fr["pct"], fr["corr"] = (pct if pct is not None else fr["pc"]), corr
            if it < N_ITER - 1:
                obs = obs_new
        for t, fr in enumerate(frames):
            visible, props, msgs, ph = fr["visible"], fr["props"], fr["msgs"], fr["pose_hat"]
            pb = [p["box"] for p in props]
            n_gt += len(visible)
            n_gt_conn += sum(1 for g in visible if g["conn"])
            conf = [p["box"] for p in props if p["score"] >= mc.TAU_DET]
            missed = {i for i, g in enumerate(visible) if not conf or mc.iou_matrix([g["box"]], conf).max() < 0.5}
            labels = {m: [] for m in methods}
            boxes, keep = mc.project_msgs(msgs, ph)
            for i in mc.visible_after_occlusion(msgs, boxes, keep, ph):
                labels["P"].append(dict(box=boxes[i], cls=msgs[keep[i]]["cls"], m=msgs[keep[i]]))
            for (i, k, _) in mc.match(boxes, pb, 0.3):
                labels["PM"].append(dict(box=props[k]["box"], cls=msgs[keep[i]]["cls"], m=msgs[keep[i]]))
            if fr["pc"] is not None:
                bc, kc = mc.project_msgs(msgs, fr["pc"])
                for (i, k, _) in mc.match(bc, pb, 0.4):
                    labels["C"].append(dict(box=props[k]["box"], cls=msgs[kc[i]]["cls"], m=msgs[kc[i]]))
            # smoothed prior (SM) and temporal consensus (CT) from the last pass
            corr, pct = fr["corr"], fr["pct"]
            bs, ks = mc.project_msgs(msgs, corr)
            for (i, k, _) in mc.match(bs, pb, 0.4):
                labels["SM"].append(dict(box=props[k]["box"], cls=msgs[ks[i]]["cls"], m=msgs[ks[i]]))
            if pct is not None:
                bt, kt = mc.project_msgs(msgs, pct)
                for (i, k, _) in mc.match(bt, pb, 0.4):
                    labels["CT"].append(dict(box=props[k]["box"], cls=msgs[kt[i]]["cls"], m=msgs[kt[i]]))
            pose_err["raw"].append(math.hypot(ph[0], ph[1]))
            if fr["pc"] is not None:
                pose_err["C"].append(math.hypot(fr["pc"][0], fr["pc"][1]))
                pose_err["raw_on_C"].append(math.hypot(ph[0], ph[1]))
            if pct is not None:
                pose_err["CT"].append(math.hypot(pct[0], pct[1]))
                pose_err["raw_on_CT"].append(math.hypot(ph[0], ph[1]))
            for m in methods:
                corr_idx = mc.evaluate(labels[m], visible)
                acc[m]["emit"] += len(labels[m])
                acc[m]["correct"] += len(corr_idx)
                acc[m]["hard"] += sum(1 for i in corr_idx if i in missed)
                acc[m]["conn"] += sum(1 for i in corr_idx if visible[i]["conn"])
                # correct labels (box and class right) that sit on an unconnected vehicle, or on a
                # vehicle other than the one that sent the message
                for li, gi in evaluate_pairs(labels[m], visible):
                    if not visible[gi]["conn"]:
                        acc[m]["on_unconn"] += 1
                    if labels[m][li]["m"]["idx"] != visible[gi]["vid"]:
                        acc[m]["other_id"] += 1
                sp = [lab for lab in labels[m] if lab["m"]["ghost"] or lab["m"]["shifted"]]
                acc[m]["spoof_emit"] += len(sp)
                if sp:
                    ok = mc.evaluate(sp, visible)
                    acc[m]["spoof_wrong"] += len(sp) - len(ok)
    n_frames = cfg["seqs"] * SEQ_FRAMES
    out = {}
    for m, a in acc.items():
        out[m] = dict(precision=a["correct"] / max(a["emit"], 1), recall_all=a["correct"] / max(n_gt, 1),
                      recall_connected=a["conn"] / max(n_gt_conn, 1), hard_share=a["hard"] / max(a["correct"], 1),
                      labels_per_frame=a["emit"] / n_frames, spoof_label_share=a["spoof_emit"] / max(a["emit"], 1),
                      wrong_from_spoof_per_1k=1000 * a["spoof_wrong"] / max(a["emit"], 1),
                      correct_on_unconnected_share=a["on_unconn"] / max(a["correct"], 1),
                      correct_other_vehicle_share=a["other_id"] / max(a["correct"], 1),
                      identity_precision=(a["correct"] - a["other_id"]) / max(a["emit"], 1))
    out["pose"] = {k: dict(rms=float(np.sqrt(np.mean(np.square(v)))) if v else float("nan"), frames=len(v) / n_frames) for k, v in pose_err.items()}
    out["gt_per_frame"] = n_gt / n_frames
    return out


def calib(seed, n=40):
    """Error of the per-frame consensus pose (inl >= 2) at the default setting, for R_POS / R_PSI."""
    rng = np.random.default_rng(seed)
    ex, epsi = [], []
    for _ in range(n):
        traffic = Traffic(rng, 0.5, 0.0, 0.0)
        e, ep = error_process(rng, SEQ_FRAMES, "gm", 5.0, math.radians(5))
        sig_rel = math.sqrt(25 + 2 * mc.SIGMA_VEH_POS ** 2)
        for t in range(0, SEQ_FRAMES, 4):
            traffic.step(t * DT)
            phys = [o for o in traffic.veh if not o["ghost"]]
            vehicles = [dict(zip(("x", "y", "yaw"), Traffic.pose_at(o, o["s"])), cls=o["cls"], l=o["l"], w=o["w"], h=o["h"]) for o in phys]
            visible = mc.gt_boxes(vehicles)
            props = mc.detector(rng, visible)
            msgs = messages(rng, traffic, t * DT, 0.0, True, True)
            pc, inl = consensus(msgs, props, (e[t, 0], e[t, 1], ep[t]), sig_rel, math.radians(5))
            if pc is not None and inl >= 2:
                ex += [pc[0], pc[1]]
                epsi.append(pc[2])
    ex, epsi = np.array(ex), np.array(epsi)
    mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
    return dict(n=len(epsi), pos_rms=float(np.sqrt(np.mean(ex ** 2))), pos_robust_sigma=float(mad(ex)),
                psi_rms_deg=float(np.degrees(np.sqrt(np.mean(epsi ** 2)))), psi_robust_sigma_deg=float(np.degrees(mad(epsi))))


def configs(study, seqs):
    base = dict(gnss_rms=5.0, rho=0.5, heading_sigma_deg=5.0, error_model="gm", latency=True, extrapolate=True, loss=0.0)
    out = []

    def add(name, **kw):
        c = dict(base)
        c.update(kw)
        c["name"], c["seqs"] = name, seqs
        out.append(c)

    if study in ("pose", "all"):
        for model in ("iid", "gm", "gm_bias"):
            for g in (3.0, 5.0, 10.0):
                add(f"pose_{model}_g{g:g}", error_model=model, gnss_rms=g)
    if study in ("rho", "all"):
        for r in (0.1, 0.25, 0.5, 1.0):
            add(f"rho_{r}", rho=r)
    if study in ("latency", "all"):
        add("lat_none", latency=False, extrapolate=False)
        add("lat_raw", latency=True, extrapolate=False)
        add("lat_dr", latency=True, extrapolate=True)
        add("lat_raw_loss30", latency=True, extrapolate=False, loss=0.3)
        add("lat_dr_loss30", latency=True, extrapolate=True, loss=0.3)
    if study in ("spoof", "all"):
        for gr in (0.1, 0.3):
            add(f"ghost_{gr}", ghost_rate=gr)
        for sf in (0.1, 0.3):
            add(f"shift_{sf}", shift_frac=sf)
    if study == "misspec":
        # smoother keeps tau = 30 s / 60 s and the reported sigma_g = 5 m; the truth differs
        add("ms_base")                                                    # model matches truth (= pose_gm_g5)
        add("ms_tau_short", tau_pos_true=10.0, tau_psi_true=20.0)         # error decorrelates 3x faster
        add("ms_tau_long", tau_pos_true=120.0, tau_psi_true=240.0)        # 4x slower
        add("ms_sig_under", gnss_true=8.0)                                # receiver under-reports its error
        add("ms_sig_over", gnss_true=3.0)                                 # receiver over-reports its error
        add("ms_under_short", gnss_true=8.0, tau_pos_true=10.0, tau_psi_true=20.0)
    return out


def job(a):
    cfg, seed = a
    return cfg["name"], run(cfg, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["pose", "rho", "latency", "spoof", "all", "calib", "misspec"], default="all")
    ap.add_argument("--seqs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.study == "calib":
        json.dump(calib(7), sys.stdout, indent=1)
        return
    cfgs = configs(args.study, args.seqs)
    jobs = [(c, 2000 + s) for c in cfgs for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    by = {}
    for name, r in res:
        by.setdefault(name, []).append(r)
    results = {}
    for c in cfgs:
        runs = by[c["name"]]
        agg = {}
        for m in ("P", "PM", "C", "SM", "CT"):
            agg[m] = {k: [float(np.mean([r[m][k] for r in runs])), float(np.std([r[m][k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0] for k in runs[0][m]}
        agg["pose"] = {k: {kk: float(np.nanmean([r["pose"][k][kk] for r in runs])) for kk in ("rms", "frames")} for k in runs[0]["pose"]}
        agg["gt_per_frame"] = float(np.mean([r["gt_per_frame"] for r in runs]))
        results[c["name"]] = dict(config=c, metrics=agg)
        print(c["name"], {m: (round(agg[m]["precision"][0], 3), round(agg[m]["recall_connected"][0], 3)) for m in ("P", "PM", "C", "SM", "CT")}, {k: round(v["rms"], 2) for k, v in agg["pose"].items()}, file=sys.stderr, flush=True)
    json.dump(dict(study=args.study, seqs=args.seqs, seq_frames=SEQ_FRAMES, seeds=args.seeds, results=results), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
