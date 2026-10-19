"""Frozen-trunk feature cache for head-only federated learning (YOLO11 detectors, Ultralytics 8.4).

The detector is split at the input of its Detect layer: the *trunk* (backbone and neck, layers 0..N-2) is
frozen and run once per image; its three outputs (P3/8, P4/16 and P5/32 feature maps) are stored in FP16;
the *head* (the Detect layer) is trained on the stored features. Images are letterboxed to a square
``imgsz`` canvas (grey 114 padding, as in Ultralytics), so every cached image has the same shape.

Storage: one folder per shard with three C-ordered float16 memmaps ``p3.f16``, ``p4.f16``, ``p5.f16`` of
shape (N, C_l, H_l, W_l), and ``index.json`` (image names, letterbox gain and padding, shapes).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

LEVELS = ("p3", "p4", "p5")


def load_detector(weights: str, device="cuda"):
    from ultralytics import YOLO
    model = YOLO(weights).model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def letterbox(img: np.ndarray, size=640):
    """Resize the long side to ``size`` and pad to a square canvas (value 114). Returns (canvas, gain, (padw, padh))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def to_tensor(imgs_bgr: list[np.ndarray], device="cuda", half=False):
    x = np.stack([im[:, :, ::-1].transpose(2, 0, 1) for im in imgs_bgr])  # BGR->RGB, HWC->CHW
    x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    x = x.half() if half else x.float()
    return x / 255.0


@torch.no_grad()
def trunk_forward(model, x: torch.Tensor) -> list[torch.Tensor]:
    """Run layers 0..N-2 of an Ultralytics DetectionModel; return the inputs of its Detect layer."""
    y = []
    for m in model.model[:-1]:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in model.save else None)
    detect = model.model[-1]
    return [y[j] for j in detect.f]


def head_raw(model, feats: list[torch.Tensor]) -> dict:
    """Raw head outputs (box distribution logits and class logits), as used by the training loss."""
    detect = model.model[-1]
    return detect.forward_head(list(feats), **detect.one2many)


class FeatureShard:
    """Write or read one cache shard."""

    def __init__(self, folder: str | Path, n: int | None = None, shapes: dict | None = None, mode="r"):
        self.folder = Path(folder)
        if mode == "w":
            self.folder.mkdir(parents=True, exist_ok=True)
            self.index = {"n": n, "shapes": shapes, "images": [None] * n, "gain": [None] * n, "pad": [None] * n}
            self.maps = {l: np.lib.format.open_memmap(self.folder / f"{l}.npy", mode="w+", dtype=np.float16,
                                                        shape=(n, *shapes[l])) for l in LEVELS}
        else:
            self.index = json.loads((self.folder / "index.json").read_text())
            self.maps = {l: np.load(self.folder / f"{l}.npy", mmap_mode="r") for l in LEVELS}

    def put(self, i0: int, feats: list[torch.Tensor], names, gains, pads):
        for l, f in zip(LEVELS, feats):
            self.maps[l][i0:i0 + len(names)] = f.detach().to(torch.float16).cpu().numpy()
        for k, (nm, g, p) in enumerate(zip(names, gains, pads)):
            self.index["images"][i0 + k], self.index["gain"][i0 + k], self.index["pad"][i0 + k] = nm, g, p

    def close(self):
        for m in self.maps.values():
            if hasattr(m, "flush"):
                m.flush()
        (self.folder / "index.json").write_text(json.dumps(self.index))

    def get(self, idx, device="cuda", dtype=torch.float32) -> list[torch.Tensor]:
        idx = np.asarray(idx)
        order = np.argsort(idx)                       # read in file order, return in request order
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        out = []
        for l in LEVELS:
            a = np.ascontiguousarray(self.maps[l][idx[order]])[inv]
            out.append(torch.from_numpy(a).to(device, non_blocking=True).to(dtype))
        return out

    def __len__(self):
        return self.index["n"]
