#!/usr/bin/env python3
"""Feature-cache feasibility and equivalence test (Phase 2, chunk 2.5).

On 1,000 street-level images (fixed seed) this script
1. runs the full detector and, separately, the frozen trunk -> FP16 cache on disk -> head path;
2. reports the maximum absolute difference of the raw head outputs (box-distribution and class logits)
   and of the decoded class probabilities and boxes, for two precision settings:
   * "fp16": trunk and head in FP16, as Ultralytics runs on the GPU; the cache stores the FP16 trunk
     outputs exactly. Pass criterion: max abs difference of the raw head outputs < 1e-3.
   * "fp32": trunk and head in FP32, cache in FP16 (the storage rounding is the only difference).
     Reported for information;
3. measures the cache size per image, the trunk throughput and the cache read speed (random order).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.fl.feature_cache import FeatureShard, head_raw, letterbox, load_detector, to_tensor, trunk_forward  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.environ.get("FEDV2X_DATA", "data/merged") + "/images/train")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--work", default=os.environ.get("FEDV2X_WORK", "work") + "/cache_test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase2" / "cache_equivalence.json"))
    a = ap.parse_args()
    names = sorted(e.name for e in os.scandir(a.images) if e.name.startswith(("vistas__", "mtsd__", "rf_")))
    pick = sorted(np.random.default_rng(20261017).choice(len(names), a.n, replace=False).tolist())
    names = [names[i] for i in pick]
    model = load_detector(a.weights)
    res = {"weights": a.weights, "n_images": a.n, "seed": 20261017}
    for prec in ("fp16", "fp32"):
        half = prec == "fp16"
        m = model.half() if half else model.float()
        x0 = to_tensor([letterbox(np.zeros((480, 640, 3), np.uint8))[0]], half=half)
        shapes = {l: tuple(f.shape[1:]) for l, f in zip(("p3", "p4", "p5"), trunk_forward(m, x0))}
        shard = FeatureShard(Path(a.work) / prec, n=a.n, shapes=shapes, mode="w")
        full_out, t_trunk, t_read_img = [], 0.0, 0.0
        for i0 in range(0, a.n, a.batch):
            t = time.time()
            batch = [letterbox(cv2.imread(os.path.join(a.images, nm))) for nm in names[i0:i0 + a.batch]]
            t_read_img += time.time() - t
            x = to_tensor([b[0] for b in batch], half=half)
            with torch.no_grad():
                torch.cuda.synchronize()
                t = time.time()
                feats = trunk_forward(m, x)
                torch.cuda.synchronize()
                t_trunk += time.time() - t
                ref = head_raw(m, feats)
                y_full, preds_full = m(x)          # the complete model, independently of the split
            shard.put(i0, feats, names[i0:i0 + a.batch], [b[1] for b in batch], [b[2] for b in batch])
            full_out.append({"boxes": preds_full["boxes"].float().cpu(), "scores": preds_full["scores"].float().cpu(),
                             "y": y_full.float().cpu(), "split_boxes": ref["boxes"].float().cpu(),
                             "split_scores": ref["scores"].float().cpu()})
        shard.close()
        del shard
        # read back from disk in random order and run the head
        rd = FeatureShard(Path(a.work) / prec)
        order = np.random.default_rng(1).permutation(a.n)
        t = time.time()
        for i0 in range(0, a.n, a.batch):
            rd.get(order[i0:i0 + a.batch], dtype=torch.float16 if half else torch.float32)
        torch.cuda.synchronize()
        t_read = time.time() - t
        d_raw = d_prob = d_box = d_split = 0.0
        detect = m.model[-1]
        for b, i0 in enumerate(range(0, a.n, a.batch)):
            idx = np.arange(i0, min(i0 + a.batch, a.n))
            feats = rd.get(idx, dtype=torch.float16 if half else torch.float32)
            with torch.no_grad():
                out = head_raw(m, feats)
                y = detect._inference(out)
            f = full_out[b]
            d_raw = max(d_raw, (out["boxes"].float().cpu() - f["boxes"]).abs().max().item(),
                        (out["scores"].float().cpu() - f["scores"]).abs().max().item())
            d_split = max(d_split, (f["split_boxes"] - f["boxes"]).abs().max().item(),
                          (f["split_scores"] - f["scores"]).abs().max().item())
            nc = f["scores"].shape[1]
            d_prob = max(d_prob, (y[:, 4:4 + nc].float().cpu() - f["y"][:, 4:4 + nc]).abs().max().item())
            d_box = max(d_box, (y[:, :4].float().cpu() - f["y"][:, :4]).abs().max().item())
        size = sum(os.path.getsize(Path(a.work) / prec / f"{l}.npy") for l in ("p3", "p4", "p5")) / a.n
        res[prec] = {"max_abs_diff_raw_head_outputs": d_raw, "max_abs_diff_class_prob": d_prob,
                     "max_abs_diff_box_px": d_box, "max_abs_diff_split_vs_full_in_memory": d_split,
                     "cache_bytes_per_image": size, "shapes": {k: list(v) for k, v in shapes.items()},
                     "trunk_images_per_s": a.n / t_trunk, "image_decode_letterbox_images_per_s": a.n / t_read_img,
                     "cache_read_random_images_per_s": a.n / t_read,
                     "cache_read_random_MB_per_s": a.n * size / t_read / 1e6,
                     "note_read": "read right after writing; the OS file cache may hold part of the data"}
    res["pass"] = res["fp16"]["max_abs_diff_raw_head_outputs"] < 1e-3
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
