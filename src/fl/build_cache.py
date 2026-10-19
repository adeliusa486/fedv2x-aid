#!/usr/bin/env python3
"""Feature cache of the real-image federation (Phases 2/6).

For every client split (train/val/test), the safety set and a labelled server subset (for the SemiFL-style
baseline), the frozen trunk (official COCO-pretrained YOLO11-s, identical to the round-0 trunk) is run once in
FP16 on letterboxed 640 x 640 images, and its three outputs are stored (src/fl/feature_cache.py). Next to each
shard, labels.pkl holds per image, in letterbox pixels (xyxy):
  * human: human boxes (labels_human/) with classes  -> evaluation only
  * world: merged boxes (human + pseudo, labels/)      -> teacher emulation only (which objects exist)
Duplicate label rows are removed as in Ultralytics.

    python src/fl/build_cache.py --out $FEDV2X_WORK/cache
"""
import os
import argparse
import json
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.labels import read_yolo, to_xyxy  # noqa: E402
from src.fl.feature_cache import FeatureShard, letterbox, load_detector, to_tensor, trunk_forward  # noqa: E402

DATA = Path(os.environ.get("FEDV2X_DATA", "data/merged"))


def load_item(args):
    name, split = args
    img = cv2.imread(str(DATA / "images" / split / name))
    h0, w0 = img.shape[:2]
    lb, gain, pad = letterbox(img, 640)
    stem = name.rsplit(".", 1)[0]
    out = {}
    for kind, folder in (("human", "labels_human"), ("world", "labels")):
        b, c = to_xyxy(read_yolo(DATA / folder / split / f"{stem}.txt"), w0, h0)
        b = b * gain
        b[:, [0, 2]] += pad[0]
        b[:, [1, 3]] += pad[1]
        out[kind] = (b.astype(np.float32), c.astype(np.int16))
    return lb, gain, pad, (w0, h0), out


def build(model, names, split_of, folder, batch=32, workers=12):
    x0 = to_tensor([np.zeros((640, 640, 3), np.uint8)], half=True)
    shapes = {l: tuple(f.shape[1:]) for l, f in zip(("p3", "p4", "p5"), trunk_forward(model, x0))}
    shard = FeatureShard(folder, n=len(names), shapes=shapes, mode="w")
    labels = []
    with ThreadPoolExecutor(workers) as ex:
        items = ex.map(load_item, [(n, split_of[n]) for n in names])
        buf = []
        i0 = 0
        for it in items:
            buf.append(it)
            if len(buf) == batch:
                feats = trunk_forward(model, to_tensor([b[0] for b in buf], half=True))
                shard.put(i0, feats, names[i0:i0 + len(buf)], [b[1] for b in buf], [list(b[2]) for b in buf])
                labels += [{"size": b[3], **b[4]} for b in buf]
                i0 += len(buf)
                buf = []
        if buf:
            feats = trunk_forward(model, to_tensor([b[0] for b in buf], half=True))
            shard.put(i0, feats, names[i0:i0 + len(buf)], [b[1] for b in buf], [list(b[2]) for b in buf])
            labels += [{"size": b[3], **b[4]} for b in buf]
    shard.close()
    with open(Path(folder) / "labels.pkl", "wb") as f:
        pickle.dump(labels, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--out", default=os.environ.get("FEDV2X_WORK", "work") + "/cache")
    ap.add_argument("--server-n", type=int, default=2000)
    a = ap.parse_args()
    fed = json.loads((ROOT / "results/phase2/federation.json").read_text())
    import csv
    import gzip
    split_of = {}
    with gzip.open(ROOT / "docs/dataset_manifest.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split_of[r["image"]] = r["split"]
    model = load_detector(a.weights).half()
    jobs = []
    for name, c in fed["clients"].items():
        for s in ("train", "val", "test"):
            jobs.append((Path(a.out) / "clients" / name / s, c[s]))
    jobs.append((Path(a.out) / "safety", fed["safety"]))
    street_server = sorted(im for im in fed["server"] if im.startswith(("vistas__", "rf_")))
    server_sub = sorted(random.Random(2).sample(street_server, min(a.server_n, len(street_server))))
    jobs.append((Path(a.out) / "server_labelled", server_sub))
    t0 = time.time()
    total = 0
    for folder, names in jobs:
        if (folder / "labels.pkl").exists() and json.loads((folder / "index.json").read_text())["n"] == len(names):
            continue
        build(model, names, split_of, folder)
        total += len(names)
        print(f"{folder}: {len(names)} images, {time.time() - t0:.0f} s", flush=True)
    meta = {"weights": a.weights, "precision": "fp16", "imgsz": 640, "letterbox": "square, pad 114",
            "server_labelled_subset": {"n": len(server_sub), "seed": 2, "from": "street-level server-pool images"},
            "images": total, "seconds": round(time.time() - t0, 1)}
    (Path(a.out) / "cache_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    with torch.no_grad():
        main()
