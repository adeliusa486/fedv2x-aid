"""The stacked (grouped-convolution) head equals K independent fused heads (GPU environment only)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fl.heads import fused_head, get_vec, raw, set_vec  # noqa: E402
from src.fl.loss import ScopedDetectionLoss  # noqa: E402
from src.fl.stacked import StackedHead  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.allow_tf32 = False      # exact float32 arithmetic, so that grouped and plain kernels agree
torch.backends.cuda.matmul.allow_tf32 = False


def make_template(nc=152):
    from ultralytics.nn.tasks import DetectionModel
    m = DetectionModel("yolo11s.yaml", nc=nc, verbose=False)
    return fused_head(m.model[-1], DEV)


def test_stacked_forward_and_gradients_equal_independent_heads():
    torch.manual_seed(0)
    t = make_template()
    K, b = 3, 2
    heads = [fused_head(t, DEV) for _ in range(K)]
    for k, h in enumerate(heads):                      # make the K heads different
        set_vec(h, get_vec(h) + 0.01 * k * torch.randn_like(get_vec(h)))
    st = StackedHead(t, K).to(DEV)
    st.set_vecs(torch.stack([get_vec(h) for h in heads]))
    feats = [torch.randn(K, b, c, s, s, device=DEV) for c, s in ((128, 80), (256, 40), (512, 20))]
    out = st(feats)
    for k, h in enumerate(heads):
        ref = raw(h, [f[k] for f in feats])
        assert torch.allclose(out["boxes"][k * b:(k + 1) * b], ref["boxes"], atol=1e-5, rtol=1e-5)
        assert torch.allclose(out["scores"][k * b:(k + 1) * b], ref["scores"], atol=1e-5, rtol=1e-5)
    # gradients of the summed per-client losses equal each head's own loss gradient
    crit = ScopedDetectionLoss(152, [8, 16, 32], device=DEV)
    gt = torch.tensor([[[100., 100., 200., 220.], [300., 300., 360., 400.]]], device=DEV).repeat(K * b, 1, 1)
    batch = {"gt_boxes": gt, "gt_cls": torch.tensor([[14, 0]], device=DEV).repeat(K * b, 1),
             "gt_w": torch.ones(K * b, 2, device=DEV), "gt_mask": torch.ones(K * b, 2, dtype=torch.bool, device=DEV)}
    client = torch.arange(K, device=DEV).repeat_interleave(b)
    loss, _ = crit(st(feats), batch, scoped=False, client=client, n_clients=K)
    loss.backward()
    g_st = torch.cat([st.p(n).grad.reshape(K, -1) for n in st.names], 1)
    for k, h in enumerate(heads):
        sub = {key: v[k * b:(k + 1) * b] for key, v in batch.items()}
        lk, _ = crit(raw(h, [f[k] for f in feats]), sub, scoped=False)
        h.zero_grad()
        lk.backward()
        g = torch.cat([p.grad.reshape(-1) for n, p in h.named_parameters() if p.requires_grad])
        assert torch.allclose(g_st[k], g, atol=1e-5, rtol=1e-4)
