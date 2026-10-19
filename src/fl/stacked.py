"""K copies of a fused YOLO11 Detect head evaluated in one pass with grouped convolutions.

Federated simulation trains many clients' heads per round. Instead of looping over clients, the K heads are
stacked: client k's features occupy channel block k of the input, and every convolution becomes a grouped
convolution (groups = K, or K * C for depthwise ones) whose weight block k is client k's weight. The clients'
parameters are separate tensor slices, so SGD with momentum and weight decay acts on each client exactly as a
separate optimiser would (tests/test_stacked.py checks equality with K independent heads).

Parameter order and flat vectors follow ``src.fl.heads.trainable`` of the fused template head, so stacked rows,
template heads and aggregated vectors are interchangeable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fl.heads import trainable


def _layers(level: int):
    """(prefix, kind) of the conv layers of one level, in forward order. kind: 'conv' (act), 'dw' (act), 'lin'."""
    box = [(f"cv2.{level}.0.conv", "conv"), (f"cv2.{level}.1.conv", "conv"), (f"cv2.{level}.2", "lin")]
    cls = [(f"cv3.{level}.0.0.conv", "dw"), (f"cv3.{level}.0.1.conv", "conv"), (f"cv3.{level}.1.0.conv", "dw"),
           (f"cv3.{level}.1.1.conv", "conv"), (f"cv3.{level}.2", "lin")]
    return box, cls


class StackedHead(nn.Module):
    def __init__(self, template: nn.Module, K: int):
        super().__init__()
        self.K, self.nl, self.nc, self.reg_max = K, template.nl, template.nc, template.reg_max
        self.names = [n for n, _ in trainable(template)]
        self.shapes = {n: p.shape for n, p in trainable(template)}
        self.params = nn.ParameterDict({n.replace(".", "_"): nn.Parameter(
            p.detach().unsqueeze(0).repeat(K, *[1] * p.dim()).clone()) for n, p in trainable(template)})
        self.sizes = [int(torch.tensor(self.shapes[n]).prod()) for n in self.names]

    def p(self, name):
        return self.params[name.replace(".", "_")]

    # ---- flat vectors (K, D) in template order
    def get_vecs(self) -> torch.Tensor:
        return torch.cat([self.p(n).detach().reshape(self.K, -1) for n in self.names], 1)

    def vecs(self) -> torch.Tensor:
        """Flat (K, D) parameter view that keeps the autograd graph (for proximal or orthogonality terms)."""
        return torch.cat([self.p(n).reshape(self.K, -1) for n in self.names], 1)

    def set_vecs(self, vecs: torch.Tensor):
        i = 0
        with torch.no_grad():
            for n, s in zip(self.names, self.sizes):
                self.p(n).copy_(vecs[:, i:i + s].view(self.p(n).shape))
                i += s

    def _conv(self, x, prefix, kind):
        w, b = self.p(prefix + ".weight"), self.p(prefix + ".bias")
        K = self.K
        W = w.reshape(K * w.shape[1], *w.shape[2:])
        groups = K * w.shape[1] if kind == "dw" else K
        y = F.conv2d(x, W.to(x.dtype), b.reshape(-1).to(x.dtype), padding=w.shape[-1] // 2, groups=groups)
        return F.silu(y) if kind in ("conv", "dw") else y

    def forward(self, feats: list[torch.Tensor]) -> dict:
        """feats[l]: (K, b, C_l, H_l, W_l). Returns raw outputs for K*b images, client-major:
        {"boxes": (K*b, 64, A), "scores": (K*b, nc, A), "feats": list of (K*b, C, H, W)-shaped placeholders}."""
        K = self.K
        b = feats[0].shape[1]
        boxes, scores, shapes = [], [], []
        for l, f in enumerate(feats):
            _, _, C, H, W = f.shape
            x = f.permute(1, 0, 2, 3, 4).reshape(b, K * C, H, W)
            box_layers, cls_layers = _layers(l)
            y = x
            for pre, kind in box_layers:
                y = self._conv(y, pre, kind)
            z = x
            for pre, kind in cls_layers:
                z = self._conv(z, pre, kind)
            boxes.append(y.reshape(b, K, 4 * self.reg_max, H * W).permute(1, 0, 2, 3).reshape(K * b, -1, H * W))
            scores.append(z.reshape(b, K, self.nc, H * W).permute(1, 0, 2, 3).reshape(K * b, -1, H * W))
            shapes.append(f.new_empty((K * b, 1, H, W)))
        return {"boxes": torch.cat(boxes, -1), "scores": torch.cat(scores, -1), "feats": shapes}
