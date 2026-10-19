#!/usr/bin/env python3
"""Cross-check of src/metrics against Ultralytics' validator (Phase 2, chunk 2.4).

1. Draw a fixed 1,000-image subset of the validation split of the merged set (numpy default_rng(20261016)) and copy
   its images and merged labels to a scratch folder (so that Ultralytics never writes a label cache into
   the read-only dataset).
2. Validate the reference detector on the subset with Ultralytics (conf 0.001, NMS IoU 0.7, 640 px) and
   save its predictions (save_json).
3. Recompute AP50 per class and mAP50 from the same predictions and labels with src/metrics
   (rule="iou", Ultralytics' matching rule, and rule="score", the COCO rule used in the paper's metrics).

Pass: |mAP50(src/metrics, rule=iou) - mAP50(Ultralytics)| < 0.001.
"""
import os
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.labels import read_yolo, to_xyxy  # noqa: E402
from src.metrics.detection import DetectionAccumulator  # noqa: E402


def read_labels(p, w, h):
    return to_xyxy(read_yolo(p), w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("FEDV2X_DATA", "data/merged"))
    ap.add_argument("--model", default=os.environ.get("FEDV2X_REFERENCE_WEIGHTS", "reference_detector.pt"))
    ap.add_argument("--work", default=os.environ.get("FEDV2X_WORK", "work") + "/xcheck1000")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase2" / "xcheck_ultralytics.json"))
    a = ap.parse_args()
    src, work = Path(a.data_root), Path(a.work)
    names = sorted(p.name for p in (src / "images" / "val").iterdir())
    pick = sorted(np.random.default_rng(20261016).choice(len(names), a.n, replace=False).tolist())
    subset = [names[i] for i in pick]
    for sub in ("images/val", "labels/val"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    for n in subset:
        stem = n.rsplit(".", 1)[0]
        if not (work / "images/val" / n).exists():
            shutil.copy2(src / "images/val" / n, work / "images/val" / n)
        lab = src / "labels/val" / f"{stem}.txt"
        if lab.exists():
            shutil.copy2(lab, work / "labels/val" / f"{stem}.txt")
    yaml_names = (src / os.environ.get("FEDV2X_DATA_YAML", "dataset.yaml")).read_text(encoding="utf-8-sig").split("names:", 1)[1]
    (work / "data.yaml").write_text(f"path: {work.as_posix()}\ntrain: images/val\nval: images/val\nnames:{yaml_names}")
    (work / "subset.txt").write_text("\n".join(subset) + "\n")

    from ultralytics import YOLO
    model = YOLO(a.model)
    res = model.val(data=str(work / "data.yaml"), split="val", imgsz=640, batch=16, conf=0.001, iou=0.7,
                    save_json=True, plots=False, project=str(work / "runs"), name="val", exist_ok=True, verbose=False)
    ul_map50 = float(res.box.map50)
    ul_per_class = {int(c): float(v) for c, v in zip(res.box.ap_class_index, res.box.all_ap[:, 0])}
    preds = json.loads((Path(res.save_dir) / "predictions.json").read_text())
    by_img = {}
    for p in preds:
        by_img.setdefault(str(p["image_id"]), []).append(p)

    accs = {"iou": DetectionAccumulator(rule="iou"), "score": DetectionAccumulator(rule="score")}
    for n in subset:
        stem = n.rsplit(".", 1)[0]
        with Image.open(work / "images/val" / n) as im:
            w, h = im.size
        gb, gc = read_labels(work / "labels/val" / f"{stem}.txt", w, h)
        pr = by_img.get(stem, [])
        pb = np.array([[q["bbox"][0], q["bbox"][1], q["bbox"][0] + q["bbox"][2], q["bbox"][1] + q["bbox"][3]]
                       for q in pr]).reshape(-1, 4)
        ps = np.array([q["score"] for q in pr])
        # Ultralytics writes category_id = class + 1 for non-COCO datasets (class_map = 1..nc)
        pc = np.array([int(q["category_id"]) - 1 for q in pr], dtype=int)
        for acc in accs.values():
            acc.add(pb, ps, pc, gb, gc, image_id=stem)
    out = {"subset_seed": 20261016, "n_images": len(subset), "model": a.model, "ultralytics_map50": ul_map50,
           "n_predictions": len(preds)}
    for rule, acc in accs.items():
        m, ncls = acc.map()
        out[f"ours_map50_{rule}"] = m
        out[f"classes_{rule}"] = ncls
        per = {c: acc.ap(c) for c in ul_per_class}
        out[f"max_abs_class_diff_{rule}"] = max(abs(per[c] - ul_per_class[c]) for c in ul_per_class)
    out["abs_diff_iou_rule"] = abs(out["ours_map50_iou"] - ul_map50)
    out["pass"] = out["abs_diff_iou_rule"] < 0.001
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
