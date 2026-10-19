#!/usr/bin/env python3
"""Phase 10: client-level (eps, delta) of the DP runs, read from their result files.

Each RSU aggregate is a Gaussian mechanism with noise multiplier z on norm-bounded updates. An aid is composed over
the RSU aggregations it actually joined (logged as "participations"); no amplification by subsampling is assumed.
RDP of the Gaussian mechanism: alpha / (2 z^2) per release; conversion eps = min_alpha [T alpha / (2 z^2) +
log(1/delta) / (alpha - 1)]. A second, closed-form check uses the optimum alpha of the same bound.

    python src/privacy/accountant.py results/phase10/fedv2x-dp_z1.0.json --delta 1e-5
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def eps_rdp(T, z, delta, alphas=np.concatenate([np.linspace(1.01, 10, 900), np.arange(11, 513)])):
    return float(np.min(T * alphas / (2 * z * z) + math.log(1 / delta) / (alphas - 1)))


def eps_closed(T, z, delta):
    """Same bound at its optimum alpha = 1 + z sqrt(2 log(1/delta) / T): eps = T/(2z^2) + sqrt(2 T log(1/delta)) / z."""
    return T / (2 * z * z) + math.sqrt(2 * T * math.log(1 / delta)) / z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for f in a.results:
        r = json.load(open(f))
        log = r["log"]
        z = log["args"]["dp_z"]
        T = max(log["participations"].values())
        rows.append({"file": Path(f).name, "z": z, "max_participations": T, "delta": a.delta,
                     "eps_rdp": round(eps_rdp(T, z, a.delta), 2), "eps_closed_form": round(eps_closed(T, z, a.delta), 2)})
        print(rows[-1])
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
