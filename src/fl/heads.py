"""Detect-head utilities for head-only federated learning on cached trunk features."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torchvision  # noqa: F401  Ultralytics uses torchvision's NMS kernel only if torchvision is imported (else a slow loop)
from ultralytics.nn.modules import Conv
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import fuse_conv_and_bn


def fused_head(detect: torch.nn.Module, device="cuda") -> torch.nn.Module:
    """Copy of a Detect module with BatchNorm folded into the convolutions (frozen BN; per-example gradients are
    well defined, as DP-SGD requires). The DFL projection stays fixed."""
    h = copy.deepcopy(detect).float().to(device)
    for m in h.modules():
        if isinstance(m, Conv) and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, "bn")
            m.forward = m.forward_fuse
    for n, p in h.named_parameters():
        p.requires_grad_(not n.startswith("dfl"))
    h.train()
    return h


def load_round0(path, device="cuda") -> torch.nn.Module:
    """Fused round-0 head from src/fl/train_round0_cache.py ({"head": module}) or from an Ultralytics checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "head" in ck:
        h = ck["head"].float().to(device)
        for n, p in h.named_parameters():
            p.requires_grad_(not n.startswith("dfl"))
        h.train()
        return h
    return fused_head((ck.get("ema") or ck["model"]).float().model[-1], device)


def trainable(head) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in head.named_parameters() if p.requires_grad]


def get_vec(head) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for _, p in trainable(head)])


def set_vec(head, vec: torch.Tensor):
    i = 0
    with torch.no_grad():
        for _, p in trainable(head):
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p))
            i += n


def raw(head, feats):
    return head.forward_head(list(feats), **head.one2many)


@torch.no_grad()
def predict(head, feats, conf=0.001, iou=0.7, max_det=300):
    """Detections per image: list of (N, 6) tensors [x1, y1, x2, y2, score, class] in letterbox pixels."""
    out = raw(head, feats)
    y = head._inference(out)                     # (B, 4 + nc, A): xywh boxes and class probabilities
    return non_max_suppression(y, conf_thres=conf, iou_thres=iou, max_det=max_det, max_time_img=10.0)  # never truncate


@torch.no_grad()
def predict_batched(head, feats, conf=0.5, iou=0.7, max_det=300):
    """Same detections as ``predict`` (best class per anchor, class-aware NMS, at most ``max_det`` per image, sorted
    by score) with one NMS call for the whole batch instead of one per image; boxes of different images and classes
    are separated by float64 offsets. Used for self-labels, where it is several times faster than ``predict``; the
    evaluation keeps ``predict`` (Ultralytics' routine)."""
    from torchvision.ops import nms
    from ultralytics.utils.ops import xywh2xyxy
    y = head._inference(raw(head, feats)).transpose(1, 2)        # (B, A, 4 + nc)
    score, cls = y[..., 4:].max(-1)
    b, a = (score > conf).nonzero(as_tuple=True)
    box = xywh2xyxy(y[b, a, :4])
    s, c = score[b, a], cls[b, a]
    off = (b.double() * head.nc + c.double()) * 8192.0
    keep = nms(box.double() + off[:, None], s.double(), iou)       # indices sorted by decreasing score
    det = torch.cat([box, s[:, None], c[:, None].float()], 1)[keep]
    bk = b[keep]
    out = []
    for i in range(y.shape[0]):
        out.append(det[bk == i][:max_det])
    return out


@torch.no_grad()
def predict_with_probs(head, feats, classes, conf=0.05, iou=0.7, max_det=300):
    """Like predict, plus the probabilities of ``classes`` at each kept detection (for set-valued promotion)."""
    out = raw(head, feats)
    y = head._inference(out)
    dets, idxs = non_max_suppression(y, conf_thres=conf, iou_thres=iou, max_det=max_det, return_idxs=True,
                                     max_time_img=10.0)  # Ultralytics' default time limit truncates crowded images
    res = []
    for b in range(len(dets)):
        idx = idxs[b].view(-1).long()                       # anchor index of each kept detection
        res.append((dets[b], y[b, 4:, :][:, idx].T[:, classes]))
    return res
