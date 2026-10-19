#!/usr/bin/env python3
"""Experiment E8 (Phase 5, chunk 5.3): quality of the emulated range labels and exposure to false suppression.

From the teacher labels of E1 (seed 0; the range emulation does not depend on the seed because p_valid = 1) and the
human labels of the aids' training images (never used for training):

1. Range labels per drop-off class: training instances, the share touched by a range region (set-valued
   supervision), promoted boxes and their precision (IoU >= 0.5 with a human box of the same class, counted only on
   images whose source attests the class).
2. False suppression. In the training frames that carry a teacher signal (the buffers), every human-labelled
   hazard instance without a teacher box of its class (IoU >= 0.5) would be trained as background by a standard
   detection loss. Under the source-scoped loss it is trained as background only if its class is attested by a
   label source of the frame and its centre lies in no detector proposal (score >= 0.05) and no range region
   (the anchors of A^- in Eq. (local)); otherwise it is distilled towards the received head or covered by the
   set-valued loss. Reported per hazard group.

    python src/teachers/e8_range_quality.py
"""
import os
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.teachers.emulate import iou_matrix  # noqa: E402

CACHE = Path(os.environ.get("FEDV2X_WORK", "work") + "/cache")
GROUPS = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]


def inside(pts, boxes):
    if len(boxes) == 0 or len(pts) == 0:
        return np.zeros(len(pts), bool)
    x, y = pts[:, 0:1], pts[:, 1:2]
    return ((x >= boxes[None, :, 0]) & (x <= boxes[None, :, 2]) & (y >= boxes[None, :, 1]) & (y <= boxes[None, :, 3])).any(1)


def main():
    names = [n.strip() for n in (ROOT / "data/taxonomy/class_names_152.txt")
             .read_text(encoding="utf-8").splitlines() if n.strip()]
    tax = yaml.safe_load((ROOT / "src/data/hazard_groups.yaml").read_text(encoding="utf-8"))
    group_of = {names.index(c): g for g in GROUPS for c in tax["groups"][g]["classes"]}
    att = yaml.safe_load((ROOT / "docs/attested_classes.yaml").read_text(encoding="utf-8"))["sources"]
    by_src = {k: {names.index(c) for c in v["classes"]} for k, v in att.items()}
    teach = pickle.load(open(os.environ.get("FEDV2X_WORK", "work") + "/teachers/E1/seed0.pkl", "rb"))
    stats = json.loads((ROOT / "results/phase5/teacher_stats_E1_stats_seed0.json").read_text())
    sup = {g: defaultdict(int) for g in GROUPS}
    frames_signal = 0
    for aid, frames in sorted(teach.items()):
        folder = CACHE / "clients" / aid / "train"
        labels = pickle.load(open(folder / "labels.pkl", "rb"))
        images = json.loads((folder / "index.json").read_text())["images"]
        for i, f in frames.items():
            if not f["signal"]:
                continue
            frames_signal += 1
            hb, hc = labels[i]["human"]
            hc = hc.astype(int)
            src_cls = by_src[images[i].split("__")[0]]
            props, regs = f["props"], f["regions"]
            for b, c in zip(hb, hc):
                g = group_of.get(int(c))
                if g is None or int(c) not in src_cls:
                    continue
                s = sup[g]
                s["instances"] += 1
                m = f["cls"] == c
                if m.any() and iou_matrix(b[None], f["boxes"][m]).max() >= 0.5:
                    s["teacher_box"] += 1
                    continue
                s["standard_background"] += 1                    # standard loss: unlabelled = background
                ctr = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]])
                if int(c) not in f["attested"]:
                    s["scoped_out_of_scope"] += 1                # distilled: no source of the frame observes it
                elif inside(ctr, props)[0] or inside(ctr, regs)[0]:
                    s["scoped_protected"] += 1                   # in scope, but covered by a proposal or region
                else:
                    s["scoped_background"] += 1
    out = {"frames_with_signal": frames_signal, "suppression": {}, "range": {}}
    tot = defaultdict(int)
    for g in GROUPS:
        s = dict(sup[g])
        for k, v in s.items():
            tot[k] += v
        n = max(s.get("standard_background", 0), 1)
        s["scoped_background_share_of_unlabelled"] = s.get("scoped_background", 0) / n
        out["suppression"][g] = s
    tot = dict(tot)
    tot["scoped_background_share_of_unlabelled"] = tot["scoped_background"] / max(tot["standard_background"], 1)
    out["suppression"]["all"] = tot
    r = stats["range"]
    out["range"] = {"events": r["events"], "invalid": r["invalid"], "misses": r["misses"], "promoted": r["promoted"],
                    "promoted_precision": r["promoted_precision"],
                    "dropoff_touched": {c: {"touched": v[0], "instances": v[1], "share": v[0] / max(v[1], 1)}
                                        for c, v in r["dropoff_instances_touched_by_a_region"].items()},
                    "promoted_per_class": {c: {"correct": v[0], "promoted": v[1], "precision": v[2]}
                                           for c, v in r["promoted_per_class_on_attesting_images"].items()},
                    "frames": stats["frames"], "frames_with_signal": stats["frames_with_signal"]}
    dst = ROOT / "results/phase5/e8_range_labels.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["suppression"], indent=1))
    print({k: v for k, v in out["range"].items() if k not in ("promoted_per_class",)})


if __name__ == "__main__":
    main()
