#!/usr/bin/env python3
"""Per-round aid-RSU association traces for the real-image federation (Phase 2 design, used in Phases 6-7).

Re-uses the SUMO mobility study unchanged (computed/mobility_sumo.py: the same three grid towns, walkers, RSU
placement by farthest-point sampling and contact-window rules) and records, for the main configuration
(30 aids, 2 RSUs per town = 6 RSUs, 150 m range, no PC5 loss, mobility-driven association), which RSU each aid
downloads from and uploads to in every edge round, or that it has no usable contact. One trace per seed.

    python src/fl/contact_trace.py --workdir $FEDV2X_WORK/sumo  ->  results/phase2/contact_traces.json
"""
import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "computed"))
import mobility_sumo as ms  # noqa: E402


def trace(seg_pool, rsu_per_town, n_aids, loss, rate_bps, rounds, seed):
    """Same association rule as mobility_sumo.federation (mobility-driven, 80 % of rounds in the home town)."""
    rng = np.random.default_rng(seed)
    towns = sorted(seg_pool)
    need = 8 * ms.HEAD_BYTES / (rate_bps * (1 - loss))
    home = [towns[i % len(towns)] for i in range(n_aids)]
    rid_index = {(t, e): i for i, (t, e) in enumerate((t, e) for t in towns for e in range(rsu_per_town))}
    out = []
    for r in range(rounds):
        row = []
        for k in range(n_aids):
            town = home[k] if rng.random() < 0.8 else rng.choice([t for t in towns if t != home[k]])
            seg = seg_pool[town][int(rng.integers(len(seg_pool[town])))]
            cands = []
            for e, wins in enumerate(seg):
                for (a, b) in wins:
                    if b - a >= need:
                        cands.append((a, b, (town, e)))
            if not cands:
                row.append(None)
                continue
            cands.sort()
            row.append([rid_index[cands[0][2]], rid_index[cands[-1][2]]])   # [download RSU, upload RSU]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.environ.get("FEDV2X_WORK", "work") + "/sumo")
    ap.add_argument("--walkers", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    Path(a.workdir).mkdir(parents=True, exist_ok=True)
    traces, tls = {}, {}
    for i, (name, t) in enumerate(ms.TOWNS.items()):
        fcd, tls_xy = ms.build_town(a.workdir, name, t["grid"], t["length"], a.walkers, 11 + i)
        traces[name] = ms.read_traces(fcd)
        tls[name] = tls_xy
    pool = {name: ms.segments(traces[name], ms.farthest_points(tls[name], 2, 0), 150.0) for name in ms.TOWNS}
    res = {"config": {"aids": 30, "rsus": 6, "range_m": 150.0, "loss": 0.0, "rate_bps": 10e6,
                      "rounds": a.rounds, "source": "computed/mobility_sumo.py (towns, segments, association rule)"},
           "segments_per_town": {k: len(v) for k, v in pool.items()}, "traces": {}}
    for s in range(a.seeds):
        tr = trace(pool, 2, 30, 0.0, 10e6, a.rounds, s)
        part = np.mean([[x is not None for x in row] for row in tr])
        rep = np.zeros((a.rounds, 6))
        for r, row in enumerate(tr):
            for x in row:
                if x is not None:
                    rep[r, x[1]] += 1
        res["traces"][str(s)] = tr
        res.setdefault("summary", {})[str(s)] = {"p_upload": float(part), "reporters_mean": float(rep.mean()),
                                                 "deferred_share": float((rep < ms.MIN_AIDS).mean())}
    (ROOT / "results/phase2/contact_traces.json").write_text(json.dumps(res))
    print(json.dumps(res["summary"], indent=1), json.dumps(res["segments_per_town"]))


if __name__ == "__main__":
    main()
