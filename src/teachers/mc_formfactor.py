#!/usr/bin/env python3
"""Phase 4.1: the static V2X Monte-Carlo study at the measured camera field of view and three wearable heights.

Re-uses computed/v2x_pseudolabel_mc.py and the job runner of computed/v2x_mc_extended.py unchanged; only the
camera is changed: horizontal field of view 98 deg (measured on the camera in its
1536 x 864 mode) instead of 102 deg, and heights 1.0 m (hand-held or belt), 1.3 m (chest) and 1.6 m (head-worn).
Same seeds (1000, 1001, 1002) and frames (3,000) as the published height study.

    python src/teachers/mc_formfactor.py --workers 8 > results/phase4/mc_formfactor.json
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "computed"))
import v2x_mc_extended as ext  # noqa: E402
import v2x_pseudolabel_mc as mc  # noqa: E402


def job(args):
    cfg, seed = args
    hfov = math.radians(cfg["hfov_deg"])
    mc.HFOV = hfov
    mc.F = (mc.W / 2) / math.tan(hfov / 2)
    return ext.job((cfg, seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfgs = []
    for hfov in (102.0, 98.0):
        for hc in (1.0, 1.3, 1.6):
            if hfov == 102.0 and hc != 1.0:
                continue                       # 102 deg at 1.0 m is the published reference (checked below)
            for g in (0, 3, 5, 10):
                cfgs.append(dict(name=f"fov{int(hfov)}_hc{hc}_g{g}", gnss_rms=float(g), rho=0.5, heading_sigma_deg=5.0,
                                 h_cam=hc, hfov_deg=hfov, frames=a.frames))
    jobs = [(c, 1000 + s) for c in cfgs for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    by = {}
    for name, seed, r in res:
        by.setdefault(name, []).append(r)
    out = {}
    for c in cfgs:
        runs = by[c["name"]]
        agg = {}
        for m in ("P", "PM", "C", "CR"):
            agg[m] = {k: [float(np.mean([r[m][k] for r in runs])), float(np.std([r[m][k] for r in runs], ddof=1))]
                      for k in runs[0][m]}
        agg["frames_with_pose"] = float(np.mean([r["frames_with_pose"] for r in runs]))
        agg["gt_per_frame"] = float(np.mean([r["gt_per_frame"] for r in runs]))
        out[c["name"]] = {"config": c, "metrics": agg}
        print(c["name"], {m: (round(agg[m]["precision"][0], 3), round(agg[m]["recall_connected"][0], 3))
                          for m in ("PM", "C")}, file=sys.stderr, flush=True)
    json.dump({"frames": a.frames, "seeds": a.seeds, "results": out}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
