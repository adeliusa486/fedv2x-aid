#!/usr/bin/env python3
"""Round-0 head trained on cached trunk features (Phase 2/6; replaces the Ultralytics run of train_round0.py).

The Ultralytics training of the round-0 head (src/fl/train_round0.py) reached only 50 to 70 images/s on this
machine and failed twice in its data loader. Round 0 is therefore trained with the same machinery as the
federation: frozen official COCO-pretrained YOLO11-s trunk, FP16 trunk features cached once, and the
152-class Detect head (BatchNorm folded, COCO weights for every layer except the 152-class output convolutions)
trained with the standard detection loss on the server's merged labels.

Data: 24,000 images drawn at random (seed 5) from the 58,000 server training images selected by
train_round0.py, and 1,000 of its 2,000 validation images (seed 6); read from the 640-px copies made by
src/data/resize_copy.py. Optimisation: SGD (momentum 0.937, Nesterov, weight decay 5e-4), batch 32, 10 epochs,
learning rate 0.01 with one warm-up epoch and linear decay to 1e-4, FP16 autocast, seed 0.

    python src/fl/train_round0_cache.py --stage cache
    python src/fl/train_round0_cache.py --stage train
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

COPY = Path(os.environ.get("FEDV2X_WORK", "work") + "/round0_data")
OUT = Path(os.environ.get("FEDV2X_WORK", "work") + "/round0_cache")
COCO = "yolo11s.pt"
SHARD = 1000


def load_item(path):
    img = cv2.imread(str(path))
    h0, w0 = img.shape[:2]
    lb, gain, pad = letterbox(img, 640)
    lab = Path(str(path).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")
    b, c = to_xyxy(read_yolo(lab), w0, h0)
    b = b * gain
    b[:, [0, 2]] += pad[0]
    b[:, [1, 3]] += pad[1]
    return lb, gain, pad, (b.astype(np.float32), c.astype(np.int16))


def stage_cache(a):
    model = load_detector(COCO).half()
    train = [p for p in (COPY / "train.txt").read_text().split("\n") if p.strip()]
    val = [p for p in (COPY / "val.txt").read_text().split("\n") if p.strip()]
    sel = {"train": sorted(random.Random(5).sample(train, a.n_train)), "val": sorted(random.Random(6).sample(val, a.n_val))}
    x0 = to_tensor([np.zeros((640, 640, 3), np.uint8)], half=True)
    shapes = {l: tuple(f.shape[1:]) for l, f in zip(("p3", "p4", "p5"), trunk_forward(model, x0))}
    t0 = time.time()
    for split, paths in sel.items():
        for s0 in range(0, len(paths), SHARD):
            folder = OUT / split / f"shard{s0 // SHARD:03d}"
            names = paths[s0:s0 + SHARD]
            if (folder / "labels.pkl").exists():
                continue
            shard = FeatureShard(folder, n=len(names), shapes=shapes, mode="w")
            labels = []
            with ThreadPoolExecutor(12) as ex:
                items = list(ex.map(load_item, names))
            for i0 in range(0, len(items), 32):
                chunk = items[i0:i0 + 32]
                with torch.no_grad():
                    feats = trunk_forward(model, to_tensor([c[0] for c in chunk], half=True))
                shard.put(i0, feats, [Path(p).name for p in names[i0:i0 + 32]], [c[1] for c in chunk],
                          [list(c[2]) for c in chunk])
                labels += [c[3] for c in chunk]
            shard.close()
            with open(folder / "labels.pkl", "wb") as fh:
                pickle.dump(labels, fh)
            print(f"{folder}: {len(names)} images, {time.time() - t0:.0f} s", flush=True)
    (OUT / "selection.json").write_text(json.dumps({k: v for k, v in sel.items()}, indent=0))


def template_head():
    from ultralytics.nn.tasks import DetectionModel
    from src.fl.heads import fused_head
    m = DetectionModel("yolo11s.yaml", nc=152, verbose=False)
    ck = torch.load(COCO, map_location="cpu", weights_only=False)
    src = ck["model"].float().state_dict()
    dst = m.state_dict()
    loaded = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    m.load_state_dict(loaded, strict=False)
    return fused_head(m.model[-1]), len(loaded), len(dst)


def batches(split, bs, rng):
    shards = sorted((OUT / split).glob("shard*"))
    rng.shuffle(shards)
    for folder in shards:
        feats = [torch.from_numpy(np.load(folder / f"{l}.npy")) for l in ("p3", "p4", "p5")]
        labels = pickle.load(open(folder / "labels.pkl", "rb"))
        order = rng.permutation(len(labels))
        for i0 in range(0, len(order), bs):
            idx = np.sort(order[i0:i0 + bs])
            yield [f[torch.from_numpy(idx)] for f in feats], [labels[i] for i in idx]


def to_batch(labels, dev="cuda"):
    from src.fl.simulate import Labels
    frames = [{"boxes": b, "cls": c.astype(int), "w": np.ones(len(c), np.float32), "attested": range(152)} for b, c in labels]
    return Labels(frames, 152).batch(torch.arange(len(frames)), dev)


@torch.no_grad()
def evaluate(head, dev="cuda"):
    from src.fl.heads import predict
    from src.metrics.detection import DetectionAccumulator
    acc = DetectionAccumulator(rule="iou")
    head.eval()
    for feats, labels in batches("val", 32, np.random.default_rng(0)):
        dets = predict(head, [f.to(dev).float() for f in feats])
        for d, (gb, gc) in zip(dets, labels):
            d = d.cpu().numpy()
            acc.add(d[:, :4], d[:, 4], d[:, 5].astype(int), gb, gc.astype(int))
    head.train()
    return acc.map()


def stage_train(a):
    from src.fl.loss import ScopedDetectionLoss
    from src.fl.stacked import StackedHead
    from src.fl.heads import get_vec, set_vec
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    head, n_loaded, n_total = template_head()
    st = StackedHead(head, 1).cuda()
    crit = ScopedDetectionLoss(152, head.stride.tolist())
    opt = torch.optim.SGD(st.parameters(), lr=a.lr, momentum=0.937, nesterov=True, weight_decay=5e-4)
    steps_per_epoch = (a_n := sum(len(pickle.load(open(f / "labels.pkl", "rb"))) for f in (OUT / "train").glob("shard*"))) // a.batch + 1
    total = steps_per_epoch * a.epochs
    log = {"loaded_coco_tensors": n_loaded, "model_tensors": n_total, "train_images": a_n, "epochs": []}
    step = 0
    t0 = time.time()
    for ep in range(a.epochs):
        losses = []
        for feats, labels in batches("train", a.batch, rng):
            frac = step / total
            lr = a.lr * (step / steps_per_epoch if step < steps_per_epoch else (1 - frac) * (1 - 0.01) + 0.01)
            for g in opt.param_groups:
                g["lr"] = lr
            f = [x.unsqueeze(0).cuda(non_blocking=True).float() for x in feats]
            batch = to_batch(labels)
            with torch.autocast("cuda", dtype=torch.float16):
                preds = st(f)
            loss, parts = crit(preds, batch, scoped=False)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.parameters(), 10.0)
            opt.step()
            losses.append(float(loss))
            step += 1
        set_vec(head, st.get_vecs()[0])
        m50, ncls = evaluate(head)
        log["epochs"].append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_map50": m50, "val_classes": ncls,
                              "seconds": round(time.time() - t0, 1)})
        print(log["epochs"][-1], flush=True)
    head.cpu()
    torch.save({"head": head, "nc": 152, "trunk": COCO, "vector": get_vec(head).cpu(), "log": log},
               OUT.parent / "round0" / "round0_head.pt")
    (ROOT / "results/phase2/round0_training.json").write_text(json.dumps(log, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "train"], required=True)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    a = ap.parse_args()
    {"cache": stage_cache, "train": stage_train}[a.stage](a)


if __name__ == "__main__":
    main()
