#!/usr/bin/env python3
"""Run experiment E7 (ablations A1-A8, interactions I1-I2) and the participation study of Phase 7, resumably
(docs/preregistration_E7.md). A9 (no RSU tier) is the FedAvg row of E1; full participation is the FedV2X-Aid row.

    python src/fl/run_e7.py                      # every configuration, seeds 0-2
    python src/fl/run_e7.py --configs A5 I1 --seeds 0
"""
import os
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEACH = os.environ.get("FEDV2X_WORK", "work") + "/teachers"
CONFIGS = {  # ID: (result tag, simulator arguments)
    "A1": ("fedv2x-no_v2x", ["--ablate", "no_v2x"]),
    "A2": ("fedv2x-pm", ["--teacher-labels", f"{TEACH}/E7_pm"]),
    "A3": ("fedv2x-c", ["--teacher-labels", f"{TEACH}/E7_c"]),
    "A4": ("fedv2x-no_range", ["--ablate", "no_range"]),
    "A5": ("fedv2x-no_scope", ["--ablate", "no_scope"]),
    "A6": ("fedv2x-no_quality", ["--ablate", "no_quality"]),
    "A7": ("fedv2x-fixed_rsu", ["--ablate", "fixed_rsu"]),
    "A8": ("fedv2x-no_normbound", ["--ablate", "no_normbound"]),
    "I1": ("fedv2x-pm-no_scope", ["--teacher-labels", f"{TEACH}/E7_pm", "--ablate", "no_scope"]),
    "I2": ("fedv2x-no_range-no_scope", ["--ablate", "no_range+no_scope"]),
    "P50": ("fedv2x-part0.50", ["--participation", "0.5"]),
    "P25": ("fedv2x-part0.25", ["--participation", "0.25"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--out", default=str(ROOT / "results/phase7/E7"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        for c in a.configs:
            tag, args = CONFIGS[c]
            if (out / f"{tag}_s{seed}.json").exists():
                continue
            t0 = time.time()
            cmd = [sys.executable, str(ROOT / "src/fl/simulate.py"), "--method", "fedv2x", "--seed", str(seed),
                   "--tag", tag, "--out", str(out)] + args
            with open(out / f"{tag}_s{seed}.log", "w") as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(f"{c} {tag}_s{seed}: exit {r.returncode}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
