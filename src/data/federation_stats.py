#!/usr/bin/env python3
"""Per-client statistics of the real-image federation (Phase 2, chunk 2.3).

For every client: images per split, human hazard instances per group and split, sources, the Dirichlet
mixture q_k, the evaluated groups (at least ``--n-min`` human test instances), and non-IID measures: the
Jensen-Shannon divergence (base 2) between the client's training group distribution and the pooled one, and
the effective number of groups exp(H). Writes docs/federation_clients.csv and results/phase2/federation_stats.json.
"""
import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]


def js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=20)
    a = ap.parse_args()
    names = [n.strip() for n in (ROOT / "data/taxonomy/class_names_152.txt")
             .read_text(encoding="utf-8").splitlines() if n.strip()]
    tax = yaml.safe_load((ROOT / "src/data/hazard_groups.yaml").read_text(encoding="utf-8"))
    gid = {names.index(c): g for g in GROUPS for c in tax["groups"][g]["classes"]}
    inst = {}
    with gzip.open(ROOT / "docs/dataset_manifest.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = Counter()
            if r["human_counts"]:
                for kv in r["human_counts"].split(";"):
                    k, n = kv.split(":")
                    if int(k) in gid:
                        c[gid[int(k)]] += int(n)
            inst[r["image"]] = c
    fed = json.loads((ROOT / "results/phase2/federation.json").read_text())
    rows, pooled = [], Counter()
    for name, c in fed["clients"].items():
        for im in c["train"]:
            pooled.update(inst[im])
    pooled_vec = [pooled[g] for g in GROUPS]
    evaluated_by_group = Counter()
    pooled_test = Counter()
    for name, c in fed["clients"].items():
        row = {"client": name, "sources": ";".join(f"{k}:{v}" for k, v in sorted(c["sources"].items())),
               "q": ";".join(f"{k}:{v}" for k, v in c["q"].items())}
        tr = Counter()
        for s in ("train", "val", "test"):
            cnt = Counter()
            for im in c[s]:
                cnt.update(inst[im])
            row[f"{s}_images"] = len(c[s])
            for g in GROUPS:
                row[f"{s}_{g}"] = cnt[g]
            if s == "train":
                tr = cnt
            if s == "test":
                ev = [g for g in GROUPS if cnt[g] >= a.n_min]
                row["evaluated_groups"] = ";".join(ev)
                evaluated_by_group.update(ev)
                pooled_test.update(cnt)
        vec = np.array([tr[g] for g in GROUPS], float)
        row["js_divergence_to_pool"] = round(js(vec + 1e-12, np.array(pooled_vec, float) + 1e-12), 4)
        p = vec / vec.sum() if vec.sum() else vec
        nz = p[p > 0]
        row["effective_groups"] = round(float(np.exp(-(nz * np.log(nz)).sum())), 3) if len(nz) else 0
        rows.append(row)
    with open(ROOT / "docs/federation_clients.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "clients": len(rows), "n_min_test_instances": a.n_min,
        "images": {s: sum(r[f"{s}_images"] for r in rows) for s in ("train", "val", "test")},
        "pooled_test_instances": {g: pooled_test[g] for g in GROUPS},
        "pooled_train_instances": {g: pooled[g] for g in GROUPS},
        "clients_evaluated_per_group": {g: evaluated_by_group[g] for g in GROUPS},
        "clients_without_evaluated_group": [r["client"] for r in rows if not r["evaluated_groups"]],
        "min_test_instances_on_evaluated_groups": min(
            r[f"test_{g}"] for r in rows for g in r["evaluated_groups"].split(";") if g),
        "js_divergence_mean": round(float(np.mean([r["js_divergence_to_pool"] for r in rows])), 4),
        "js_divergence_range": [min(r["js_divergence_to_pool"] for r in rows),
                                max(r["js_divergence_to_pool"] for r in rows)],
        "effective_groups_mean": round(float(np.mean([r["effective_groups"] for r in rows])), 3),
        "safety_instances": fed["safety_instances"], "safety_images": len(fed["safety"]),
        "server_images": len(fed["server"]),
    }
    (ROOT / "results/phase2/federation_stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
