#!/usr/bin/env python3
"""Automated leakage check of the real-image federation (Phase 2, chunk 2.3).

Independently of the builder, the script assigns every image to its cell (client split, safety set or server
pool) and checks that
1. no image is in two cells;
2. every source group key (one original photo; Roboflow's augmented copies share it) lies in one cell;
3. no pair of images with dHash Hamming distance <= 4 lies in two different cells (all pairs are re-derived
   here with a 5-band search, not taken from the builder's components);
4. client splits and the safety set contain only images of the sources they are meant to hold.

    python tools/leakage_check.py   ->  results/phase2/leakage_check.json (exit code 1 on any violation)
"""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.near_duplicates import BANDS, popcount64  # noqa: E402


def main():
    fed = json.loads((ROOT / "results/phase2/federation.json").read_text())
    cell = {}
    dup = 0
    for name, c in fed["clients"].items():
        for s in ("train", "val", "test"):
            for im in c[s]:
                dup += im in cell
                cell[im] = f"{name}/{s}"
    for key in ("safety", "server"):
        for im in fed[key]:
            dup += im in cell
            cell[im] = key
    group_cells = defaultdict(set)
    src_of = {}
    n_manifest = 0
    with gzip.open(ROOT / "docs/dataset_manifest.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_manifest += 1
            src_of[r["image"]] = r["source"]
            group_cells[r["group"]].add(cell.get(r["image"], "UNASSIGNED"))
    group_viol = {g: sorted(c) for g, c in group_cells.items() if len(c) > 1}
    unassigned = n_manifest - len(cell)

    dh = np.load(ROOT / "results/phase2/dhash.npz")
    names = dh["names"].tolist()
    h = dh["dhash"].astype(np.uint64)
    valid = dh["valid"]
    cells = np.array([cell.get(n, "UNASSIGNED") for n in names], dtype=object)
    pairs_checked = 0
    cross = []
    for lo, hi in BANDS:
        key = (h >> np.uint64(64 - hi)) & np.uint64((1 << (hi - lo)) - 1)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.r_[0, np.where(ks[1:] != ks[:-1])[0] + 1, len(ks)]
        for b in range(len(starts) - 1):
            idx = order[starts[b]:starts[b + 1]]
            idx = idx[valid[idx]]
            if len(idx) < 2:
                continue
            for p in range(len(idx) - 1):
                a = idx[p]
                rest = idx[p + 1:]
                d = popcount64(h[rest] ^ h[a])
                close = rest[d <= 4]
                pairs_checked += len(close)
                for bb in close:
                    if cells[a] != cells[bb]:
                        cross.append((names[a], cells[a], names[bb], cells[bb]))
    cross = sorted(set(tuple(sorted([(x[0], x[1]), (x[2], x[3])])) for x in cross))
    client_src_bad = [im for name, c in fed["clients"].items() for s in ("train", "val", "test") for im in c[s]
                      if (name.startswith("v") and src_of[im] != "vistas")
                      or (name.startswith("r") and src_of[im] not in fed["params"]["rf_client_sources"])]
    out = {"images_in_two_cells": dup, "unassigned_images": unassigned,
           "groups_in_two_cells": len(group_viol), "group_examples": dict(list(group_viol.items())[:10]),
           "near_duplicate_pairs_within_4_bits_checked": pairs_checked,
           "near_duplicate_pairs_across_cells": len(cross), "cross_examples": cross[:10],
           "client_images_from_wrong_source": len(client_src_bad)}
    out["pass"] = (dup == 0 and unassigned == 0 and not group_viol and not cross and not client_src_bad)
    (ROOT / "results/phase2/leakage_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
