"""YOLO label files as FedV2X-Aid reads them.

Rows are parsed to float32 and exact duplicate rows are removed, as Ultralytics does when it verifies a
dataset (``verify_image_label``: ``np.unique(lb, axis=0)``); the merged label files contain some duplicate
rows (for example Open Images boxes listed twice), and counting them would inflate the ground truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_yolo(path: str | Path, dedup=True) -> np.ndarray:
    """(N, 5) float32 array [class, cx, cy, w, h] (normalised); empty (0, 5) if the file is missing or empty."""
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 5), np.float32)
    rows = [r.split()[:5] for r in p.read_text().splitlines() if len(r.split()) >= 5]
    lb = np.array(rows, dtype=np.float32).reshape(-1, 5)
    if dedup and len(lb):
        _, i = np.unique(lb, axis=0, return_index=True)
        if len(i) < len(lb):
            lb = lb[np.sort(i)]
    return lb


def to_xyxy(lb: np.ndarray, w: float, h: float):
    """Pixel boxes (N, 4) and integer classes (N,) from normalised YOLO rows."""
    c = lb[:, 0].astype(int)
    cx, cy, bw, bh = (lb[:, 1] * w, lb[:, 2] * h, lb[:, 3] * w, lb[:, 4] * h)
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float64)
    return boxes.reshape(-1, 4), c
