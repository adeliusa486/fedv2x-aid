#!/usr/bin/env python3
"""Classes attested by each source's human labels (Phase 2, chunk 2.2).

A class is attested by a source if at least one human box of that class exists in the source's images
(labels_human/, read from docs/dataset_manifest.csv.gz). Counts are human boxes after the source's mapping
onto the 152-class taxonomy; pseudo-labels never count. The script also lists, per source, the hazard
groups of src/data/hazard_groups.yaml that its human labels cover and their instance counts.

    python src/data/attested_classes.py      ->  docs/attested_classes.yaml
"""
import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    names = [n.strip() for n in (ROOT / "data/taxonomy/class_names_152.txt")
             .read_text(encoding="utf-8").splitlines() if n.strip()]
    tax = yaml.safe_load((ROOT / "src/data/hazard_groups.yaml").read_text(encoding="utf-8"))
    group_of = {c: g for g, s in tax["groups"].items() for c in s["classes"]}
    hum, imgs, imgs_with = defaultdict(Counter), Counter(), defaultdict(Counter)
    with gzip.open(ROOT / "docs/dataset_manifest.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            imgs[r["source"]] += 1
            if r["human_counts"]:
                for kv in r["human_counts"].split(";"):
                    c, n = kv.split(":")
                    hum[r["source"]][names[int(c)]] += int(n)
                    imgs_with[r["source"]][names[int(c)]] += 1
    out = {"generated_by": "src/data/attested_classes.py",
           "rule": "a class is attested by a source if its human labels contain at least one box of that class",
           "sources": {}}
    for src in sorted(imgs):
        cls = sorted(hum[src], key=lambda c: names.index(c))
        groups = Counter()
        for c in cls:
            groups[group_of[c]] += hum[src][c]
        out["sources"][src] = {
            "images": imgs[src], "classes": cls,
            "human_boxes": {c: hum[src][c] for c in cls},
            "images_with_class": {c: imgs_with[src][c] for c in cls},
            "evaluated_groups": {g: groups[g] for g, s in tax["groups"].items() if s["evaluated"] and groups[g]},
        }
    (ROOT / "docs/attested_classes.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
                                                    encoding="utf-8")
    for src, s in out["sources"].items():
        print(f"{src:14s} {s['images']:7d} img  {len(s['classes']):3d} classes  hazard groups: {s['evaluated_groups']}")


if __name__ == "__main__":
    main()
