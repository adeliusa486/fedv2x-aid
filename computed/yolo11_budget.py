"""Exact parameter, MAC and update-size budget for YOLO11-s with a custom class count.

Re-implements the Ultralytics YOLO11 module definitions (Conv, C3k2, C3k, Bottleneck,
SPPF, C2PSA, Detect) closely enough to reproduce the published YOLO11s parameter count
(9,458,752 for 80 classes), then reports the split between the frozen trunk
(backbone + neck) and the trainable detection head for nc = 152.
"""
import json
import math
import sys


def make_divisible(x, d=8):
    return int(math.ceil(x / d) * d)


class Counter:
    def __init__(self):
        self.params = 0
        self.macs = 0
        self.bn_channels = 0
        self.conv_layers = 0

    def conv(self, c1, c2, k=1, s=1, g=1, hw_out=0, bn=True, bias=False):
        p = c1 * c2 * k * k // g + (c2 if bias else 0)
        if bn:
            p += 2 * c2  # BN weight + bias (running stats are buffers)
            self.bn_channels += c2
        self.params += p
        self.macs += (c1 * c2 * k * k // g) * hw_out
        self.conv_layers += 1
        return p


def bottleneck(C, c1, c2, e=0.5, k=3, hw=0):
    c_ = int(c2 * e)
    C.conv(c1, c_, k, hw_out=hw)
    C.conv(c_, c2, k, hw_out=hw)


def c3k(C, c1, c2, n=2, e=0.5, k=3, hw=0):
    c_ = int(c2 * e)
    C.conv(c1, c_, 1, hw_out=hw)
    C.conv(c1, c_, 1, hw_out=hw)
    C.conv(2 * c_, c2, 1, hw_out=hw)
    for _ in range(n):
        bottleneck(C, c_, c_, e=1.0, k=k, hw=hw)


def c3k2(C, c1, c2, n=1, use_c3k=False, e=0.5, hw=0):
    c = int(c2 * e)
    C.conv(c1, 2 * c, 1, hw_out=hw)
    C.conv((2 + n) * c, c2, 1, hw_out=hw)
    for _ in range(n):
        if use_c3k:
            c3k(C, c, c, n=2, hw=hw)
        else:
            bottleneck(C, c, c, e=0.5, hw=hw)


def sppf(C, c1, c2, hw=0):
    c_ = c1 // 2
    C.conv(c1, c_, 1, hw_out=hw)
    C.conv(c_ * 4, c2, 1, hw_out=hw)


def c2psa(C, c1, c2, n=1, e=0.5, hw=0):
    c = int(c1 * e)
    C.conv(c1, 2 * c, 1, hw_out=hw)
    C.conv(2 * c, c1, 1, hw_out=hw)
    for _ in range(n):
        heads = c // 64
        head_dim = c // heads
        key_dim = int(head_dim * 0.5)
        h = c + key_dim * heads * 2
        C.conv(c, h, 1, hw_out=hw)          # qkv
        C.conv(c, c, 1, hw_out=hw)          # proj
        C.conv(c, c, 3, g=c, hw_out=hw)     # positional encoding (depthwise)
        C.conv(c, 2 * c, 1, hw_out=hw)      # ffn
        C.conv(2 * c, c, 1, hw_out=hw)
        # attention matmuls (QK^T and AV) at P5 add < 1% of trunk MACs; not counted


def detect(C, nc, ch, hws, reg_max=16):
    c2 = max(16, ch[0] // 4, reg_max * 4)
    c3 = max(ch[0], min(nc, 100))
    for x, hw in zip(ch, hws):
        C.conv(x, c2, 3, hw_out=hw)
        C.conv(c2, c2, 3, hw_out=hw)
        C.conv(c2, 4 * reg_max, 1, hw_out=hw, bn=False, bias=True)
        C.conv(x, x, 3, g=x, hw_out=hw)      # DWConv
        C.conv(x, c3, 1, hw_out=hw)
        C.conv(c3, c3, 3, g=c3, hw_out=hw)   # DWConv
        C.conv(c3, c3, 1, hw_out=hw)
        C.conv(c3, nc, 1, hw_out=hw, bn=False, bias=True)
    C.params += reg_max  # DFL conv weight (frozen)


def yolo11(nc, scale=(0.50, 0.50, 1024), imgsz=640):
    depth, width, max_ch = scale

    def ch(c):
        return make_divisible(min(c, max_ch) * width, 8)

    def rep(n):
        return max(round(n * depth), 1) if n > 1 else n

    s = imgsz
    hw = {1: (s // 2) ** 2, 2: (s // 4) ** 2, 3: (s // 8) ** 2, 4: (s // 16) ** 2, 5: (s // 32) ** 2}
    B, N, H = Counter(), Counter(), Counter()
    # backbone
    B.conv(3, ch(64), 3, 2, hw_out=hw[1])
    B.conv(ch(64), ch(128), 3, 2, hw_out=hw[2])
    c3k2(B, ch(128), ch(256), rep(2), False, 0.25, hw=hw[2])
    B.conv(ch(256), ch(256), 3, 2, hw_out=hw[3])
    c3k2(B, ch(256), ch(512), rep(2), False, 0.25, hw=hw[3])
    p3 = ch(512)
    B.conv(ch(512), ch(512), 3, 2, hw_out=hw[4])
    c3k2(B, ch(512), ch(512), rep(2), True, 0.5, hw=hw[4])
    p4 = ch(512)
    B.conv(ch(512), ch(1024), 3, 2, hw_out=hw[5])
    c3k2(B, ch(1024), ch(1024), rep(2), True, 0.5, hw=hw[5])
    sppf(B, ch(1024), ch(1024), hw=hw[5])
    c2psa(B, ch(1024), ch(1024), rep(2), hw=hw[5])
    p5 = ch(1024)
    # neck
    c3k2(N, p5 + p4, ch(512), rep(2), False, 0.5, hw=hw[4])
    n13 = ch(512)
    c3k2(N, n13 + p3, ch(256), rep(2), False, 0.5, hw=hw[3])
    n16 = ch(256)
    N.conv(n16, ch(256), 3, 2, hw_out=hw[4])
    c3k2(N, ch(256) + n13, ch(512), rep(2), False, 0.5, hw=hw[4])
    n19 = ch(512)
    N.conv(n19, ch(512), 3, 2, hw_out=hw[5])
    c3k2(N, ch(512) + p5, ch(1024), rep(2), True, 0.5, hw=hw[5])
    n22 = ch(1024)
    # head
    detect(H, nc, (n16, n19, n22), (hw[3], hw[4], hw[5]))
    return B, N, H, (n16, n19, n22)


def report(nc):
    B, N, H, ch = yolo11(nc)
    total = B.params + N.params + H.params
    trunk = B.params + N.params
    out = {
        "nc": nc,
        "total_params": total,
        "trunk_params": trunk,
        "head_params": H.params,
        "head_trainable_params": H.params - 16,
        "head_fraction": H.params / total,
        "head_channels_in": ch,
        "head_gmacs_640": H.macs / 1e9,
        "trunk_gmacs_640": (B.macs + N.macs) / 1e9,
        "head_update_bytes_fp16": (H.params - 16) * 2,
        "head_update_bytes_fp32": (H.params - 16) * 4,
        "full_model_bytes_fp32": total * 4,
        "bn_channels_total": B.bn_channels + N.bn_channels + H.bn_channels,
        "conv_layers_total": B.conv_layers + N.conv_layers + H.conv_layers,
    }
    # federated calibration payload: per-channel mean and second moment (fp32) for every
    # BN-normalised conv output, plus a 256-bin fp32 histogram per conv output tensor
    out["calib_stats_bytes"] = out["bn_channels_total"] * 2 * 4 + out["conv_layers_total"] * 256 * 4
    return out


if __name__ == "__main__":
    ref = report(80)
    assert ref["total_params"] == 9458752, ref["total_params"]  # Ultralytics YOLO11s summary
    res = {"nc80": ref, "nc152": report(152)}
    json.dump(res, sys.stdout, indent=2)
    print()
