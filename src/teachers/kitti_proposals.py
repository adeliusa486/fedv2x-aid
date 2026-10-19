#!/usr/bin/env python3
"""Detector proposals on KITTI images (Phase 4, E2b): every detection with score >= 0.05 after NMS (IoU 0.7),
at the deployment resolution (640 px, letterbox), in original image pixels.

    python src/teachers/kitti_proposals.py --model <weights.pt> --tag ref --images $FEDV2X_WORK/kitti/training/image_2
    python src/teachers/kitti_proposals.py --model <weights.pt> --tag ref --images $FEDV2X_WORK/kitti/training/image_02 --tracking
"""
import os
import argparse
import pickle
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--images", default=os.environ.get("FEDV2X_WORK", "work") + "/kitti/training/image_2")
    ap.add_argument("--tracking", action="store_true", help="images are sequence folders <seq>/<frame>.png")
    ap.add_argument("--out", default=os.environ.get("FEDV2X_WORK", "work") + "/kitti")
    ap.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    root = Path(a.images)
    files = sorted(root.glob("*/*.png")) if a.tracking else sorted(root.glob("*.png"))
    out = {}
    t0 = time.time()
    for i0 in range(0, len(files), 64):
        chunk = files[i0:i0 + 64]
        for f, r in zip(chunk, model.predict([str(x) for x in chunk], imgsz=640, conf=a.conf, iou=0.7, max_det=300,
                                             half=True, verbose=False, stream=False)):
            key = f"{f.parent.name}/{f.stem}" if a.tracking else f.stem
            b = r.boxes
            out[key] = {"boxes": b.xyxy.cpu().numpy().astype(np.float32), "scores": b.conf.cpu().numpy().astype(np.float32),
                        "cls": b.cls.cpu().numpy().astype(np.int16), "size": r.orig_shape[::-1]}
    name = f"proposals_{'tracking' if a.tracking else 'object'}_{a.tag}.pkl"
    with open(Path(a.out) / name, "wb") as fh:
        pickle.dump({"model": a.model, "conf": a.conf, "imgsz": 640, "iou": 0.7, "images": len(files),
                     "seconds": round(time.time() - t0, 1), "pred": out}, fh)
    print(f"{len(files)} images, {time.time() - t0:.0f} s -> {name}")


if __name__ == "__main__":
    main()
