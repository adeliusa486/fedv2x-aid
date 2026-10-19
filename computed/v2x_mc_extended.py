"""Extended geometric Monte-Carlo study of V2X-as-teacher (E2(a) sweeps).

Reuses v2x_pseudolabel_mc.py unchanged and adds three studies:
  grid    full factorial: GNSS error x V2X penetration x compass error
  height  camera height 1.0 m (handle) vs 1.3 m (chest mount)
  det     sensitivity to the parametric detector: recall scale, false proposals,
          box noise and class confusion (one factor at a time around the default)

Usage: python v2x_mc_extended.py --study grid|height|det --frames 3000 --seeds 3 --workers 16 > out.json
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

import v2x_pseudolabel_mc as mc

DEFAULT_DET = dict(rec_scale=1.0, fp_rate=1.2, box_sig=0.04, confusion=0.12)
_ORIG_DETECTOR = mc.detector


def make_detector(rec_scale, fp_rate, box_sig, confusion):
    """Parametric detector of v2x_pseudolabel_mc.detector with exposed knobs (defaults reproduce it)."""
    CLASSES, CLASS_P, DIMS, F, W, H, CY = mc.CLASSES, mc.CLASS_P, mc.DIMS, mc.F, mc.W, mc.H, mc.CY

    def detector(rng, visible):
        props = []
        for g in visible:
            r, occ = g["range"], g["occ"]
            p = np.clip(0.97 - 0.006 * max(0.0, r - 10.0), 0.6, 0.97) * (1 - 0.6 * occ) * rec_scale
            b = g["box"]
            bw, bh = b[2] - b[0], b[3] - b[1]
            if rng.random() < p:
                sig, score, lsig = box_sig, rng.uniform(0.35, 0.95), 0.07 * box_sig / 0.04
            elif rng.random() < 0.5:
                sig, score, lsig = 2 * box_sig, rng.uniform(0.05, 0.25), 0.12 * box_sig / 0.04
            else:
                continue
            cx = (b[0] + b[2]) / 2 + rng.normal(0, sig * bw)
            cy = (b[1] + b[3]) / 2 + rng.normal(0, sig * bh)
            nw = bw * math.exp(rng.normal(0, lsig))
            nh = bh * math.exp(rng.normal(0, lsig))
            cls = g["cls"] if rng.random() < 1 - confusion else rng.choice([c for c in range(5) if c != g["cls"]])
            nb, _ = mc.clip_box(np.array([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]))
            if nb is not None:
                props.append(dict(box=nb, score=score, cls=cls))
        for _ in range(rng.poisson(fp_rate)):
            rr = rng.uniform(5, 40)
            cls = rng.choice(len(CLASSES), p=CLASS_P)
            l, w, h = DIMS[cls]
            bw = F * rng.uniform(w, l) / rr
            bh = F * h / rr
            cx = rng.uniform(0, W)
            bottom = CY + F * mc.H_CAM / rr
            nb, _ = mc.clip_box(np.array([cx - bw / 2, bottom - bh, cx + bw / 2, bottom]))
            if nb is not None:
                props.append(dict(box=nb, score=rng.uniform(0.05, 0.4), cls=cls))
        return props

    return detector


def job(args):
    cfg, seed = args
    mc.H_CAM = cfg.get("h_cam", 1.0)
    det = cfg.get("det")
    mc.detector = make_detector(**det) if det is not None else _ORIG_DETECTOR
    return cfg["name"], seed, mc.run(cfg, seed)


def configs(study, frames):
    out = []
    if study == "grid":
        for g in (0, 1, 3, 5, 10):
            for r in (0.1, 0.25, 0.5, 0.75, 1.0):
                for h in (2, 5, 10):
                    out.append(dict(name=f"g{g}_r{r}_h{h}", gnss_rms=float(g), rho=r, heading_sigma_deg=float(h)))
    elif study == "height":
        for hc in (1.0, 1.3):
            for g in (0, 3, 5, 10):
                out.append(dict(name=f"hc{hc}_g{g}", gnss_rms=float(g), rho=0.5, heading_sigma_deg=5.0, h_cam=hc))
    elif study == "det":
        sweeps = dict(rec_scale=(0.8, 0.9, 1.0), fp_rate=(0.5, 1.2, 2.5, 4.0), box_sig=(0.02, 0.04, 0.08), confusion=(0.05, 0.12, 0.25))
        seen = set()
        for k, vals in sweeps.items():
            for v in vals:
                det = dict(DEFAULT_DET)
                det[k] = v
                key = tuple(sorted(det.items()))
                if key in seen:
                    continue
                seen.add(key)
                name = "default" if det == DEFAULT_DET else f"{k}_{v}"
                out.append(dict(name=name, gnss_rms=5.0, rho=0.5, heading_sigma_deg=5.0, det=det, factor=k, value=v))
    for c in out:
        c["frames"] = frames
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["grid", "height", "det"], required=True)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfgs = configs(args.study, args.frames)
    jobs = [(c, 1000 + s) for c in cfgs for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    by = {}
    for name, seed, r in res:
        by.setdefault(name, []).append(r)
    results = {}
    for c in cfgs:
        runs = by[c["name"]]
        agg = {}
        for m in ("P", "PM", "C", "CR"):
            agg[m] = {}
            for key in runs[0][m]:
                vals = np.array([r[m][key] for r in runs])
                agg[m][key] = [float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0]
        agg["frames_with_pose"] = float(np.mean([r["frames_with_pose"] for r in runs]))
        agg["gt_per_frame"] = float(np.mean([r["gt_per_frame"] for r in runs]))
        results[c["name"]] = dict(config=c, metrics=agg)
        print(c["name"], {m: (round(agg[m]["precision"][0], 3), round(agg[m]["recall_connected"][0], 3)) for m in ("P", "PM", "C")}, file=sys.stderr, flush=True)
    json.dump(dict(study=args.study, frames=args.frames, seeds=args.seeds, results=results), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
