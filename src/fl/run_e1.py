#!/usr/bin/env python3
"""Run experiment E1 (all 12 methods x 5 seeds) or a subset, resumably: a run whose result file exists is skipped.

    python src/fl/run_e1.py                      # everything
    python src/fl/run_e1.py --methods fedv2x fedavg --seeds 0
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.fl.simulate import METHODS  # noqa: E402

ORDER = ["pretrained", "fedv2x", "local", "fedavg", "hier_st", "fedprox", "scaffold", "fedrep", "hier_mt", "fedsto",
         "semifl", "central"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(ROOT / "results/phase6/E1"))
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    assert set(a.methods) <= set(METHODS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:                      # seed-major: every method gets a first seed early
        for m in a.methods:
            tag = f"{m}_s{seed}"
            if (out / f"{tag}.json").exists():
                continue
            t0 = time.time()
            cmd = [sys.executable, str(ROOT / "src/fl/simulate.py"), "--method", m, "--seed", str(seed), "--out", str(out)] + a.extra
            with open(out / f"{tag}.log", "w") as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(f"{tag}: exit {r.returncode}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
