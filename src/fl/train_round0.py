#!/usr/bin/env python3
"""Round-0 model of the federation (Phase 2, chunk 2.5): official COCO-pretrained YOLO11-s trunk, frozen; a
152-class Detect head trained on the server pool only (public data that no client holds and that is disjoint
import os
from the safety set; see results/phase2/federation.json).

Training data (pre-registered): every server-pool image of Mapillary Vistas and of the Roboflow projects, plus a
uniform random sample (seed 0) of the server-pool images of COCO, Open Images and MTSD, to 60,000 images in
total; 2,000 of them (random, seed 1) are held out for validation. Labels: the merged labels (human +
pseudo-labels from public teacher detectors), as for the reference detector.
Optimisation: Ultralytics 8.4.155 defaults except freeze=23 (all layers but the Detect head), 10 epochs,
batch 32, 640 px, mosaic 0 (single-image augmentation), seed 0, deterministic=False (in deterministic mode the
run reached only about 50 images/s with the GPU mostly idle). Images are read from a copy resized
once to 640 px on the long side (src/data/resize_copy.py): decoding the full-size street photos every epoch limited
training to about 45 images/s.

The script then verifies that the trunk of the trained model is bit-identical to the official trunk (all
backbone and neck tensors, BatchNorm statistics included), which the feature cache relies on.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = os.environ.get("FEDV2X_DATA", "data/merged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--work", default=os.environ.get("FEDV2X_WORK", "work") + "/round0")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--copy", default=os.environ.get("FEDV2X_WORK", "work") + "/round0_data",
                    help="resized copy made by src/data/resize_copy.py (640 px long side); '' reads the originals")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    fed = json.loads((ROOT / "results/phase2/federation.json").read_text())
    server = fed["server"]
    street = [im for im in server if im.split("__")[0] == "vistas" or im.startswith("rf_")]
    other = sorted(set(server) - set(street))
    random.Random(0).shuffle(other)
    chosen = sorted(street + other[:max(0, a.n - len(street))])
    val = set(random.Random(1).sample(chosen, a.n_val))
    split_of = {}
    import csv
    import gzip
    with gzip.open(ROOT / "docs/dataset_manifest.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split_of[r["image"]] = r["split"]
    lines = {"train": [], "val": []}
    for im in chosen:
        lines["val" if im in val else "train"].append(f"{DATA}/images/{split_of[im]}/{im}")
    for k, v in lines.items():
        (work / f"{k}.txt").write_text("\n".join(v) + "\n")
    names = (Path(DATA) / os.environ.get("FEDV2X_DATA_YAML", "dataset.yaml")).read_text(encoding="utf-8-sig").split("names:", 1)[1]
    src = Path(a.copy) if a.copy else work
    if a.copy:   # the resized copy must hold exactly the selected images
        for k in ("train", "val"):
            got = [Path(x).stem for x in (src / f"{k}.txt").read_text().split("\n") if x.strip()]
            assert got == [Path(x).stem for x in lines[k]], f"resized copy does not match the {k} selection"
    (work / "data.yaml").write_text(f"train: {(src / 'train.txt').as_posix()}\nval: {(src / 'val.txt').as_posix()}\n"
                                    f"names:{names}")
    (work / "selection.json").write_text(json.dumps({"train": len(lines["train"]), "val": len(lines["val"]),
                                                     "street_level": len(street), "n_target": a.n}, indent=1))

    # never write a label cache into the read-only dataset folder
    import ultralytics.data.dataset as uds
    _save = uds.save_dataset_cache_file

    def save_guard(prefix, path, x, version):
        if str(Path(DATA).resolve()) in str(Path(path).resolve()):
            x["version"] = version          # what the original function stamps before saving
            print(f"label cache not written to the read-only dataset ({path})")
            return
        _save(prefix, path, x, version)
    uds.save_dataset_cache_file = save_guard

    from ultralytics import YOLO
    last = work / "train/weights/last.pt"
    if a.resume and last.exists():   # continue an interrupted run with its stored settings
        YOLO(str(last)).train(resume=True)
    else:
        YOLO("yolo11s.yaml").load(a.weights).train(data=str(work / "data.yaml"), epochs=a.epochs, batch=a.batch, imgsz=640, freeze=23, mosaic=0.0,
                close_mosaic=0, workers=a.workers, seed=0, deterministic=False, plots=False, project=str(work),
                name="train", exist_ok=True, amp=True, val=True, patience=100)

    # trunk identity check
    import torch
    best = Path(work / "train/weights/best.pt")
    trained = torch.load(best, map_location="cpu", weights_only=False)["model"].float().state_dict()
    official = torch.load(a.weights, map_location="cpu", weights_only=False)["model"].float().state_dict()
    trunk_keys = [k for k in official if not k.startswith("model.23.")]
    diffs = {k: float((trained[k].float() - official[k].float()).abs().max()) for k in trunk_keys
             if official[k].dtype.is_floating_point or official[k].dtype == torch.int64}
    out = {"best": str(best), "trunk_tensors": len(trunk_keys), "max_abs_trunk_diff": max(diffs.values()),
           "trunk_identical": max(diffs.values()) == 0.0}
    (work / "trunk_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["trunk_identical"] else 1)


if __name__ == "__main__":
    main()
