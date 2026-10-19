#!/usr/bin/env python3
"""64-bit difference hashes (dHash) of every merged image and near-duplicate components (Phase 2, chunk 2.3).

dHash: the image is converted to grey, resized to 9x8 (bilinear) and each bit says whether a pixel is brighter
than its right neighbour (the same hash as the dataset build, which used Hamming distance <= 3).
Here two images are near duplicates if their hashes differ in at most ``--max-dist`` bits (default 4).
Candidate pairs come from 5 bands of 13/13/13/13/12 bits: by the pigeonhole principle, two hashes within
4 bits share at least one band exactly, so the band search finds every pair within distance 4.

Near duplicates are joined into connected components (union-find); the federation builder assigns whole
components to one partition cell, and tools/leakage_check.py verifies that no component crosses cells.

    python src/data/near_duplicates.py    ->  results/phase2/dhash.npz, results/phase2/near_duplicates.json
"""
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BANDS = [(0, 13), (13, 26), (26, 39), (39, 52), (52, 64)]


def dhash(path):
    try:
        with Image.open(path) as img:
            if img.format == "JPEG":
                img.draft("L", (64, 64))
            g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    except Exception:
        return -1
    bits = (g[:, :8] > g[:, 1:]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _chunk(paths):
    return [dhash(p) for p in paths]


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def components(h: np.ndarray, max_dist=4):
    """Union-find over all pairs within ``max_dist`` bits. Returns (component id per image, number of pairs)."""
    n = len(h)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = set()
    for lo, hi in BANDS:
        key = (h >> np.uint64(64 - hi)) & np.uint64((1 << (hi - lo)) - 1)
        buckets = defaultdict(list)
        for i, k in enumerate(key.tolist()):
            buckets[k].append(i)
        for idx in buckets.values():
            if len(idx) < 2:
                continue
            idx = np.array(idx)
            for a_pos in range(len(idx) - 1):
                a = idx[a_pos]
                rest = idx[a_pos + 1:]
                d = popcount64(h[rest] ^ h[a])
                for b in rest[d <= max_dist]:
                    pairs.add((int(a), int(b)))
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    comp = np.array([find(i) for i in range(n)])
    return comp, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("FEDV2X_DATA", "data/merged"))
    ap.add_argument("--max-dist", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "results/phase2/dhash.npz"))
    a = ap.parse_args()
    t0 = time.time()
    names, paths = [], []
    for split in ("train", "val"):
        for e in sorted(os.scandir(Path(a.root) / "images" / split), key=lambda e: e.name):
            names.append(e.name)
            paths.append(e.path)
    chunks = [paths[i:i + 500] for i in range(0, len(paths), 500)]
    hashes = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(_chunk, chunks)):
            hashes += r
            if (k + 1) % 50 == 0:
                print(f"  {len(hashes)} hashed, {time.time() - t0:.0f} s", flush=True)
    bad = [names[i] for i, v in enumerate(hashes) if v < 0]
    h = np.array([v if v >= 0 else 0 for v in hashes], dtype=np.uint64)
    valid = np.array([v >= 0 for v in hashes])
    comp = np.arange(len(h))
    comp_valid, npairs = components(h[valid], a.max_dist)
    vidx = np.where(valid)[0]
    comp[vidx] = vidx[comp_valid]
    np.savez_compressed(a.out, names=np.array(names), dhash=h, component=comp, valid=valid)
    sizes = np.bincount(np.unique(comp, return_inverse=True)[1])
    summary = {"images": len(names), "unreadable": bad, "max_hamming": a.max_dist, "pairs_within_distance": npairs,
               "components": int(len(sizes)), "images_in_multi_image_components": int(sizes[sizes > 1].sum()),
               "largest_component": int(sizes.max()), "seconds": round(time.time() - t0, 1)}
    Path(a.out).with_name("near_duplicates.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
