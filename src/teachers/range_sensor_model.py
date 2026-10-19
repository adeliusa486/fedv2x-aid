#!/usr/bin/env python3
"""Range-sensor model from a bench log of the range sensors (Phase 5, chunk 5.1; experiment E8).

Reads the raw 60-s stream of the range-sensor board (data/range_sensor/bench_stream_60s.csv: forward and downward readings in mm with validity flags) and derives, with the first 2 s
after the port opened excluded (the bench README's failure log F5), the quantities the teacher emulation needs:
  * validity rate of each sensor (share of rows with a valid reading);
  * noise of the forward sensor (standard deviation in mm at a fixed bench target);
  * the probability that a drop-off of height h is detected with the threshold delta_d = 0.10 m and that a flat
    floor raises a false drop-off event, under a Gaussian noise model with the measured standard deviation.
The downward sensor produced no valid reading in this log (absent from its bus), so its model is taken from the
forward sensor of the same type; this is stated wherever the model is used.

    python src/teachers/range_sensor_model.py   ->  results/phase5/range_sensor_model.json
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/range_sensor/bench_stream_60s.csv"


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    rows = list(csv.DictReader(open(RAW, encoding="utf-8")))
    t0 = int(rows[0]["host_ns"])
    steady = [r for r in rows if (int(r["host_ns"]) - t0) / 1e9 >= 2.0]
    fwd_ok = [r for r in steady if r["fwd_ok"] == "1"]
    dn_ok = [r for r in steady if r["down_ok"] == "1"]
    fwd_mm = np.array([float(r["fwd_mm"]) for r in fwd_ok])
    sigma_mm = float(fwd_mm.std(ddof=1)) if len(fwd_mm) > 1 else float("nan")
    delta_d = 0.10
    detect = {f"{h:.2f}": phi((h - delta_d) / (sigma_mm / 1000)) for h in (0.08, 0.10, 0.12, 0.15, 0.20)}
    false_event = 1 - phi(delta_d / (sigma_mm / 1000))
    out = {
        "source": str(RAW.relative_to(ROOT)), "rows": len(rows), "steady_rows": len(steady),
        "forward": {"valid_rate": len(fwd_ok) / len(steady), "n_valid": len(fwd_ok), "mean_mm": float(fwd_mm.mean()),
                    "sigma_mm": sigma_mm, "min_mm": float(fwd_mm.min()), "max_mm": float(fwd_mm.max()),
                    "target": "bench object, distance not independently measured"},
        "downward": {"valid_rate": len(dn_ok) / len(steady), "n_valid": len(dn_ok),
                     "note": "absent from its bus during the bench log; modelled with the forward sensor's parameters"},
        "emulation": {"p_valid": len(fwd_ok) / len(steady), "delta_d_m": delta_d,
                      "p_detect_drop_of_height_m": detect, "p_false_dropoff_flat_floor": false_event,
                      "note": "with sigma ~1 mm the noise is negligible against 0.10 m; emulation errors are geometric"},
    }
    dst = ROOT / "results/phase5/range_sensor_model.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
