#!/usr/bin/env python3
"""Manifest of the merged image set: one row per image.

For every image of images/train and images/val the manifest records the source, the source's own
split and image ID (both parsed from the file name, e.g. vistas__train__<id>.jpg), a group key (the
original image: Roboflow exports carry several augmented copies of one photo, <stem>_jpg.rf.<hash>),
the image size, SHA-256 hashes of the human-only label file (labels_human/) and of the merged label
file (labels/ = human + pseudo-labels), the number of human and pseudo boxes per class, and the
licence of the source.

Pseudo boxes are those listed in the dataset's pseudo_labels.csv; human MTSD boxes whose class the
merged dataset pipeline corrected from "traffic sign" to "stop sign" are listed in mtsd_stop_relabel.csv. For every
image the script verifies the identity merged = human - relabelled(out) + relabelled(in) + pseudo, class by class.

    python src/data/build_manifest.py --root os.environ.get("FEDV2X_DATA", "data/merged")

writes docs/dataset_manifest.csv.gz and docs/dataset_manifest_summary.json, and compares the counts
with data/taxonomy/{source_counts,class_counts}.csv (exit code 1 on any mismatch).
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "data" / "taxonomy"
RF_GROUP = re.compile(r"^(?P<g>.+?)(_(jpe?g|png|bmp|webp))?(\.rf\.[0-9a-f]+)?$")

# Licences of the sources. Roboflow licences are those recorded by the dataset build
# (datasets/pipeline_code/roboflow_sources.yaml) from the Roboflow Universe API at download time.
LICENCE = {
    "coco": "COCO 2017: annotations CC BY 4.0; images under the Flickr terms of use",
    "oiv7": "Open Images V7: annotations CC BY 4.0; images listed as CC BY 2.0",
    "oiv7x": "Open Images V7: annotations CC BY 4.0; images listed as CC BY 2.0",
    "vistas": "Mapillary Vistas v2: CC BY-NC-SA 4.0 (research, non-commercial)",
    "mtsd": "Mapillary Traffic Sign Dataset: CC BY-NC-SA 4.0 (research, non-commercial)",
    "rf_exit": "Roboflow Universe emergency-exit-signs v4: Public Domain",
}
RF_PROJECTS = {
    "rf_aid": "kartezyencorp/aid-for-the-blind v13", "rf_atm": "mehant-kammakomati/atm-dataset v10",
    "rf_blind725": "obstacles-for-blind-zjnnn/obstacles-for-blind v5",
    "rf_bump1": "speed-bump-detection/speed-bump-detection-se0eh v15",
    "rf_bump2": "detection-system/humps-bumps-potholes-detection v8", "rf_escooter": "kdigital/electric-scooter-cd7hw v1",
    "rf_esera": "esera/bollards-crosswalk-stairs v1", "rf_etg": "etg-ik2gp/electric_scooter v5",
    "rf_exit": "emergency-exit-signs/emergency-exit-signs v4",
    "rf_obstacle": "visually-impaired-obstacle-detection-uxdze/obstacle-detection-yeuzf v11",
    "rf_openhole": "northeastern-4sfxe/construction-safety-open-hole-excavation-detection v5",
    "rf_pillar": "adli/pillar v4", "rf_risk": "pbl5mu/risk-detection-1 v2", "rf_sideguide": "scottsdale/sideguide v1",
    "rf_starvision": "malzag3-uic-edu/starvision v29", "rf_strollers1": "furnitureselectronics/ultimite_strollers_detection v1",
    "rf_strollers2": "thales-a5kye/stroller_final v3", "rf_tactile": "susam/tactile-pavement v2",
    "rf_underground": "labelimg-djgsu/underground-pnj5a v1", "rf_wetfloor": "lena-f7w17/wet-floor-detection1 v2",
}
for _s, _p in RF_PROJECTS.items():
    LICENCE.setdefault(_s, f"Roboflow Universe {_p}: CC BY 4.0")

FIELDS = ["image", "split", "source", "orig_split", "orig_id", "group", "width", "height",
          "sha256_human", "sha256_merged", "n_human", "n_pseudo", "human_counts", "pseudo_counts", "relabel"]


def parse_name(name):
    stem = name.rsplit(".", 1)[0]
    source, orig_split, orig_id = stem.split("__", 2)
    if source.startswith("rf_"):
        group = RF_GROUP.match(orig_id).group("g")
    else:
        group = orig_id
    return source, orig_split, orig_id, f"{source}:{group}"


def class_counts(data):
    c = Counter()
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            c[int(line.split()[0])] += 1
    return c


def fmt(counter):
    return ";".join(f"{k}:{counter[k]}" for k in sorted(counter))


PSEUDO, RELABEL = {}, {}   # (split, image) -> Counter, filled in main()


def one(args):
    root, split, name = args
    stem = name.rsplit(".", 1)[0]
    try:
        with Image.open(root / "images" / split / name) as im:
            w, h = im.size
    except Exception as exc:  # recorded, never silently skipped
        w, h = -1, -1
        print(f"unreadable image {split}/{name}: {exc}", file=sys.stderr)
    rows = {}
    for kind, folder in (("human", "labels_human"), ("merged", "labels")):
        p = root / folder / split / f"{stem}.txt"
        data = p.read_bytes() if p.exists() else b""
        rows[kind] = (hashlib.sha256(data).hexdigest() if p.exists() else "", class_counts(data))
    hum, mer = rows["human"][1], rows["merged"][1]
    pseudo = PSEUDO.get((split, name), Counter())
    rel = RELABEL.get((split, name), Counter())          # (old, new) -> n
    expect = Counter(hum)
    for (old, new), n in rel.items():
        expect[old] -= n
        expect[new] += n
    expect.update(pseudo)
    bad = sorted(k for k in set(expect) | set(mer) if expect.get(k, 0) != mer.get(k, 0))
    source, orig_split, orig_id, group = parse_name(name)
    return {"image": name, "split": split, "source": source, "orig_split": orig_split, "orig_id": orig_id,
            "group": group, "width": w, "height": h, "sha256_human": rows["human"][0],
            "sha256_merged": rows["merged"][0], "n_human": sum(hum.values()), "n_pseudo": sum(pseudo.values()),
            "human_counts": fmt(hum), "pseudo_counts": fmt(pseudo),
            "relabel": ";".join(f"{o}>{nw}:{n}" for (o, nw), n in sorted(rel.items())),
            "_bad": bad, "_hum": hum, "_pse": pseudo, "_net": Counter({k: mer.get(k, 0) - hum.get(k, 0)
                                                                   for k in set(mer) | set(hum)})}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("FEDV2X_DATA", "data/merged"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "dataset_manifest.csv.gz"))
    ap.add_argument("--summary", default=str(ROOT / "docs" / "dataset_manifest_summary.json"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = Path(a.root)
    t0 = time.time()
    for r in csv.DictReader(open(root / "pseudo_labels.csv", encoding="utf-8")):
        PSEUDO.setdefault((r["split"], r["image"]), Counter())[int(r["class"])] += 1
    for r in csv.DictReader(open(root / "mtsd_stop_relabel.csv", encoding="utf-8")):
        img = r["label_file"].rsplit(".", 1)[0] + ".jpg"
        RELABEL.setdefault((r["split"], img), Counter())[(int(r["old_class"]), int(r["new_class"]))] += 1
    jobs = []
    for split in ("train", "val"):
        names = sorted(e.name for e in os.scandir(root / "images" / split) if e.is_file())
        jobs += [(root, split, n) for n in names]
    print(f"{len(jobs)} images listed in {time.time() - t0:.0f} s", flush=True)

    rows, per_src = [], Counter()
    hum_tot, pse_tot, net_tot = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    bad_images, missing_human, missing_merged, unreadable = [], 0, 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=256)):
            if r["_bad"]:
                bad_images.append(r["image"])
            missing_human += r["sha256_human"] == ""
            missing_merged += r["sha256_merged"] == ""
            unreadable += r["width"] < 0
            per_src[(r["split"], r["source"])] += 1
            hum_tot[r["split"]].update(r["_hum"])
            pse_tot[r["split"]].update(r["_pse"])
            for k, v in r["_net"].items():
                net_tot[r["split"]][k] += v
            for k in ("_bad", "_hum", "_pse", "_net"):
                del r[k]
            rows.append(r)
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1} images, {time.time() - t0:.0f} s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    import io
    # mtime=0 and a fixed file name in the gzip header make the output byte-for-byte reproducible
    with open(a.out, "wb") as raw,             gzip.GzipFile(filename="dataset_manifest.csv", mode="wb", fileobj=raw, mtime=0) as gz,             io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)

    # ---- comparison with the dataset records
    checks = {}
    ref = {(r["split"], r["source"]): int(r["images"]) for r in csv.DictReader(open(EVIDENCE / "source_counts.csv"))}
    diff = {f"{k[0]}/{k[1]}": (per_src.get(k, 0), ref.get(k, 0)) for k in set(ref) | set(per_src)
            if per_src.get(k, 0) != ref.get(k, 0)}
    checks["source_counts_equal"] = not diff
    checks["source_count_differences"] = diff
    n_split = Counter()
    for (s, _), n in per_src.items():
        n_split[s] += n
    checks["train_images"], checks["val_images"] = n_split["train"], n_split["val"]
    checks["split_totals_ok"] = n_split["train"] == 235951 and n_split["val"] == 27906
    rf = sorted({src for (_, src) in per_src if src.startswith("rf_")})
    checks["roboflow_projects"] = len(rf)
    cls_ref = list(csv.DictReader(open(EVIDENCE / "class_counts.csv")))
    cdiff = []
    for r in cls_ref:
        c = int(r["id"])
        got = (hum_tot["train"][c], net_tot["train"][c])
        want = (int(r["train_boxes_human"]), int(r["train_boxes_pseudo"]))
        if got != want:
            cdiff.append({"class": r["class"], "manifest_human_pseudo": got, "class_counts_csv": want})
        if hum_tot["train"][c] + net_tot["train"][c] != int(r["train_boxes"]):
            cdiff.append({"class": r["class"], "train_boxes_total_mismatch": True})
        if hum_tot["val"][c] + net_tot["val"][c] != int(r["val_boxes"]):
            cdiff.append({"class": r["class"], "val_boxes_total_mismatch": True})
    checks["class_counts_equal"] = not cdiff
    checks["class_count_differences"] = cdiff[:50]
    checks["images_violating_merged_identity"] = len(bad_images)
    checks["pseudo_boxes"] = {s_: sum(pse_tot[s_].values()) for s_ in pse_tot}
    checks["relabelled_boxes"] = sum(sum(v.values()) for v in RELABEL.values())
    checks["note"] = ("class_counts.csv counts pseudo boxes as merged minus human per class, so the 573 MTSD "
                      "boxes relabelled from traffic sign to stop sign appear there as +stop sign / -traffic sign")
    checks["missing_human_label_files"] = missing_human
    checks["missing_merged_label_files"] = missing_merged
    checks["unreadable_images"] = unreadable
    checks["rows_complete"] = all(r["source"] and r["orig_split"] and r["orig_id"] and r["group"] for r in rows) \
        and all(src in LICENCE for (_, src) in per_src)
    ok = (checks["source_counts_equal"] and checks["split_totals_ok"] and checks["roboflow_projects"] == 20
          and checks["class_counts_equal"] and not bad_images and checks["rows_complete"])

    sha = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    summary = {
        "generated_by": "src/data/build_manifest.py", "root": str(root), "manifest": Path(a.out).name,
        "manifest_sha256": sha, "rows": len(rows), "seconds": round(time.time() - t0, 1),
        "images_per_split_and_source": {f"{s}/{src}": n for (s, src), n in sorted(per_src.items())},
        "groups": len({r["group"] for r in rows}),
        "groups_per_source": dict(sorted(Counter(g.split(":", 1)[0] for g in {r["group"] for r in rows}).items())),
        "human_boxes_per_class": {s: {str(c): hum_tot[s][c] for c in sorted(hum_tot[s])} for s in hum_tot},
        "pseudo_boxes_per_class": {s: {str(c): pse_tot[s][c] for c in sorted(pse_tot[s])} for s in pse_tot},
        "merged_minus_human_per_class": {s: {str(c): net_tot[s][c] for c in sorted(net_tot[s]) if net_tot[s][c]} for s in net_tot},
        "licence_per_source": {src: LICENCE[src] for src in sorted({src for (_, src) in per_src})},
        "checks": checks, "pass": ok,
    }
    Path(a.summary).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in checks.items() if "differences" not in k}, indent=1))
    print("PASS" if ok else "FAIL", f"{len(rows)} rows, {time.time() - t0:.0f} s, sha256 {sha[:16]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
