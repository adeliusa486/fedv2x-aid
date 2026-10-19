#!/usr/bin/env python3
"""Phase 3, chunk 3.1: edge model zoo under one fixed, equal training budget.

Candidates: YOLO11-n, YOLO11-s, YOLO26-n and YOLO26-s, each fine-tuned from its official COCO weights on the same
fixed subset of the merged training split (12,000 images, seed 3, proportional to the sources) with the same
settings (5 epochs, batch 16, 640 px, single-image augmentation, seed 0), and validated on the same fixed subset
of the merged validation split (2,000 images, seed 4). Labels are the merged labels (human and pseudo), as for the
reference detector, which is validated on the same subset as a reference (it was trained with a much larger
budget and is not budget-matched). Images are read from copies resized once to 640 px (src/data/resize_copy.py).

Per model: mAP50 and mAP50-95 over the classes present (Ultralytics), AP50 per hazard group (classes of a group
merged, src/metrics; score-greedy matching) and per hazard class.

    python src/edge/zoo.py --stage data      # subsets and resized copies
    python src/edge/zoo.py --stage train --model yolo11n
    python src/edge/zoo.py --stage val       # all trained candidates and the reference detector
"""
import os
import argparse
import csv
import gzip
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = os.environ.get("FEDV2X_DATA", "data/merged")
WORK = Path(os.environ.get("FEDV2X_WORK", "work") + "/zoo")
WEIGHTS = {"yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt", "yolo26n": "yolo26n.pt",
           "yolo26s": "yolo26s.pt"}
REFERENCE = os.environ.get("FEDV2X_REFERENCE_WEIGHTS", "reference_detector.pt")


def stage_data(a):
    rows = []
    with gzip.open(ROOT / "docs/dataset_manifest.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["split"], r["source"], r["image"]))
    by = {"train": {}, "val": {}}
    for split, src, im in rows:
        by[split].setdefault(src, []).append(im)
    picks = {}
    for split, n, seed in (("train", a.n_train, 3), ("val", a.n_val, 4)):
        total = sum(len(v) for v in by[split].values())
        rng = random.Random(seed)
        chosen = []
        for src in sorted(by[split]):
            lst = sorted(by[split][src])
            k = round(n * len(lst) / total)
            chosen += rng.sample(lst, min(k, len(lst)))
        picks[split] = sorted(chosen)
        lst_file = WORK / f"{split}_src.txt"
        WORK.mkdir(parents=True, exist_ok=True)
        lst_file.write_text("\n".join(f"{DATA}/images/{split}/{im}" for im in picks[split]) + "\n")
        subprocess.run([sys.executable, str(ROOT / "src/data/resize_copy.py"), "--list", str(lst_file), "--out",
                        str(WORK / "data"), "--split", split, "--workers", "12"], check=True)
    names = (Path(DATA) / os.environ.get("FEDV2X_DATA_YAML", "dataset.yaml")).read_text(encoding="utf-8-sig").split("names:", 1)[1]
    (WORK / "data.yaml").write_text(f"train: {(WORK / 'data/train.txt').as_posix()}\nval: {(WORK / 'data/val.txt').as_posix()}\n"
                                    f"names:{names}")
    (WORK / "subsets.json").write_text(json.dumps({k: len(v) for k, v in picks.items()}, indent=1))
    print({k: len(v) for k, v in picks.items()})


def stage_train(a):
    from ultralytics import YOLO
    m = YOLO(WEIGHTS[a.model])
    m.train(data=str(WORK / "data.yaml"), epochs=a.epochs, batch=16, imgsz=640, mosaic=0.0, close_mosaic=0,
            workers=8, seed=0, deterministic=False, plots=False, val=False, amp=True, project=str(WORK / "runs"),
            name=a.model, exist_ok=True)


def stage_val(a):
    import numpy as np
    import yaml
    from PIL import Image
    from ultralytics import YOLO
    from src.data.labels import read_yolo, to_xyxy
    from src.metrics.detection import DetectionAccumulator
    names = [n.strip() for n in (ROOT / "data/taxonomy/class_names_152.txt")
             .read_text(encoding="utf-8").splitlines() if n.strip()]
    tax = yaml.safe_load((ROOT / "src/data/hazard_groups.yaml").read_text(encoding="utf-8"))
    groups = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]
    group_of = {names.index(c): g for g in groups for c in tax["groups"][g]["classes"]}
    models = {k: WORK / "runs" / k / "weights" / "last.pt" for k in WEIGHTS}
    models["reference"] = Path(REFERENCE)
    val_imgs = [p for p in (WORK / "data/val.txt").read_text().split("\n") if p.strip()]
    out = {}
    for name, w in models.items():
        if not Path(w).exists():
            continue
        r = YOLO(str(w)).val(data=str(WORK / "data.yaml"), split="val", imgsz=640, batch=16, conf=0.001, iou=0.7,
                             save_json=True, plots=False, project=str(WORK / "val"), name=name, exist_ok=True,
                             verbose=False)
        preds = json.loads((Path(r.save_dir) / "predictions.json").read_text())
        by = {}
        for p in preds:
            by.setdefault(str(p["image_id"]), []).append(p)
        acc = DetectionAccumulator(group_of=group_of)
        for ip in val_imgs:
            stem = Path(ip).stem
            with Image.open(ip) as im:
                wdt, hgt = im.size
            gb, gc = to_xyxy(read_yolo(Path(ip.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")), wdt, hgt)
            pr = by.get(stem, [])
            pb = np.array([[q["bbox"][0], q["bbox"][1], q["bbox"][0] + q["bbox"][2], q["bbox"][1] + q["bbox"][3]] for q in pr]).reshape(-1, 4)
            acc.add(pb, np.array([q["score"] for q in pr]), np.array([q["category_id"] - 1 for q in pr], dtype=int), gb, gc)
        per_class = {names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.all_ap[:, 0])}
        out[name] = {"weights": str(w), "map50": float(r.box.map50), "map50_95": float(r.box.map),
                     "group_ap50": {g: acc.ap(g) for g in groups},
                     "hazard_class_ap50": {c: per_class.get(c) for g in groups for c in tax["groups"][g]["classes"]},
                     "budget_matched": name != "reference"}
        print(name, round(out[name]["map50"], 4), {g: round(v, 3) for g, v in out[name]["group_ap50"].items()}, flush=True)
    dst = ROOT / "results/phase3/zoo_accuracy.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"subsets": json.loads((WORK / "subsets.json").read_text()), "epochs": a.epochs,
                               "models": out}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "train", "val"], required=True)
    ap.add_argument("--model", choices=list(WEIGHTS))
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    {"data": stage_data, "train": stage_train, "val": stage_val}[a.stage](a)


if __name__ == "__main__":
    main()
