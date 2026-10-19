#!/usr/bin/env python3
"""Client partition, splits, safety set and server pool of the real-image federation (Phase 2, chunk 2.3).

Unit of assignment: an *atom*, the connected component of images that share a source group key (one original
photo, e.g. Roboflow's augmented copies) or are near duplicates (64-bit dHash, Hamming distance <= 4; see
near_duplicates.py). Whole atoms go to one cell: a client's train, validation or test split, the safety set,
or the server pool. Hazard instances are human boxes of the six evaluated groups (labels_human/); pseudo-labels
never count.

Design (pre-registered in docs/federation_design.md):
* 30 clients: 22 from Mapillary Vistas and 8 from the Roboflow projects whose human labels cover a hazard
  group (rf_risk, rf_bump2, rf_obstacle, rf_openhole, rf_aid, rf_esera, rf_starvision). MTSD's human labels
  cover none of the six groups, so MTSD images cannot be scored and go to the server pool.
* Client pools hold atoms with at least one human hazard instance. The primary group of an atom is its rarest
  hazard group (by instance frequency in its pool). Each client draws a mixture q_k ~ Dirichlet(0.5 * 1) over the
  groups of its pool (Hsu et al., 2019) and fills its quota of 510 images by sampling atoms of each primary group
  in proportion to q_k (quota of an exhausted group is redistributed in proportion to q_k).
* Each client's atoms are split 300 / 60 / 150 images (train / validation / test), stratified by primary group.
* Safety set, reserved before the clients are drawn: hazard atoms chosen at random, rarest group first, until
  every hazard group has at least 2,000 human instances (street-level atoms; Open Images for stairs, because
  street-level stairs labels are too scarce).
* Server pool (round-0 training data): every remaining atom, including atoms of more than 20 images (dHash
  chains), which are never given to a client or to the safety set.
"""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]
RF_CLIENT_SOURCES = ["rf_risk", "rf_bump2", "rf_obstacle", "rf_openhole", "rf_aid", "rf_esera", "rf_starvision"]


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def load(root=ROOT):
    names = [n.strip() for n in (root / "data/taxonomy/class_names_152.txt")
             .read_text(encoding="utf-8").splitlines() if n.strip()]
    tax = yaml.safe_load((root / "src/data/hazard_groups.yaml").read_text(encoding="utf-8"))
    gid = {}
    for g in GROUPS:
        for c in tax["groups"][g]["classes"]:
            gid[names.index(c)] = g
    rows = []
    with gzip.open(root / "docs/dataset_manifest.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            inst = Counter()
            if r["human_counts"]:
                for kv in r["human_counts"].split(";"):
                    c, n = kv.split(":")
                    if int(c) in gid:
                        inst[gid[int(c)]] += int(n)
            rows.append({"image": r["image"], "split": r["split"], "source": r["source"], "group": r["group"],
                         "inst": inst})
    dh = np.load(root / "results/phase2/dhash.npz")
    pos = {n: i for i, n in enumerate(dh["names"].tolist())}
    comp = dh["component"]
    return rows, pos, comp


def build_atoms(rows, pos, comp):
    n = len(rows)
    uf = UF(n)
    first = {}
    for i, r in enumerate(rows):
        if r["group"] in first:
            uf.union(i, first[r["group"]])
        else:
            first[r["group"]] = i
    comp_first = {}
    for i, r in enumerate(rows):
        c = int(comp[pos[r["image"]]])
        if c in comp_first:
            uf.union(i, comp_first[c])
        else:
            comp_first[c] = i
    atoms = defaultdict(list)
    for i in range(n):
        atoms[uf.find(i)].append(i)
    out = []
    for members in atoms.values():
        inst = Counter()
        for i in members:
            inst.update(rows[i]["inst"])
        srcs = Counter(rows[i]["source"] for i in members)
        out.append({"members": sorted(members), "inst": inst, "sources": srcs,
                    "source": srcs.most_common(1)[0][0], "size": len(members)})
    return out


def primary_groups(atoms, idx):
    freq = Counter()
    for a in idx:
        freq.update(atoms[a]["inst"])
    prim = {}
    for a in idx:
        present = [g for g in GROUPS if atoms[a]["inst"][g] > 0]
        prim[a] = min(present, key=lambda g: (freq[g], GROUPS.index(g)))
    return prim, freq


def assign_clients(atoms, pool, n_clients, quota, alpha, rng, prefix):
    prim, freq = primary_groups(atoms, pool)
    groups = [g for g in GROUPS if any(prim[a] == g for a in pool)]
    by_group = {g: [a for a in pool if prim[a] == g] for g in groups}
    for g in groups:
        rng.shuffle(by_group[g])
    clients = {}
    for k in range(n_clients):
        q = rng.dirichlet(alpha * np.ones(len(groups)))
        want = {g: q[i] * quota for i, g in enumerate(groups)}
        chosen, total = [], 0
        while total < quota:
            avail = [g for g in groups if by_group[g]]
            if not avail:
                break
            w = np.array([want[g] for g in avail])
            if w.sum() <= 0:
                w = np.array([q[groups.index(g)] for g in avail]) + 1e-12
            g = avail[int(rng.choice(len(avail), p=w / w.sum()))]
            a = by_group[g].pop()
            chosen.append(a)
            total += atoms[a]["size"]
            want[g] = max(0.0, want[g] - atoms[a]["size"])
        clients[f"{prefix}{k:02d}"] = {"atoms": chosen, "q": dict(zip(groups, q.round(4).tolist())),
                                       "primary": {a: prim[a] for a in chosen}}
    return clients, by_group


def split_client(atoms, c, sizes, rng):
    """Stratified split of a client's atoms into train/val/test by primary group (image counts ~ sizes)."""
    tot = sum(sizes.values())
    out = {s: [] for s in sizes}
    by_g = defaultdict(list)
    for a in c["atoms"]:
        by_g[c["primary"][a]].append(a)
    for g, lst in by_g.items():
        rng.shuffle(lst)
        n_img = sum(atoms[a]["size"] for a in lst)
        target = {s: sizes[s] / tot * n_img for s in sizes}
        got = {s: 0 for s in sizes}
        for a in lst:  # give each atom to the split furthest below its target share
            s = max(sizes, key=lambda s: (target[s] - got[s], -list(sizes).index(s)))
            out[s].append(a)
            got[s] += atoms[a]["size"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--vistas-clients", type=int, default=22)
    ap.add_argument("--rf-clients", type=int, default=8)
    ap.add_argument("--quota", type=int, default=510)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--safety-min", type=int, default=2000)
    ap.add_argument("--max-atom", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "results/phase2/federation.json"))
    a = ap.parse_args()
    sizes = {"train": 300, "val": 60, "test": 150}
    rng = np.random.default_rng(a.seed)
    rows, pos, comp = load()
    atoms = build_atoms(rows, pos, comp)
    # Atoms larger than --max-atom images are chains of different photos with similar coarse layouts that the
    # 64-bit dHash links transitively (the largest has 7,574 images). They stay together in the server pool,
    # which is conservative: they cannot leak, they are only unavailable to clients and to the safety set.
    hazard = [i for i, t in enumerate(atoms) if sum(t["inst"][g] for g in GROUPS) > 0 and t["size"] <= a.max_atom]
    vistas_pool = [i for i in hazard if set(atoms[i]["sources"]) == {"vistas"}]
    rf_pool = [i for i in hazard if set(atoms[i]["sources"]) <= set(RF_CLIENT_SOURCES)]
    # ---- safety set first (rarest group first): street-level hazard atoms, Open Images for stairs
    oiv = [i for i in range(len(atoms)) if set(atoms[i]["sources"]) <= {"oiv7", "oiv7x"}
           and atoms[i]["size"] <= a.max_atom and atoms[i]["inst"]["stairs"] > 0]
    street = vistas_pool + rf_pool
    total = Counter()
    for i in street:
        total.update(atoms[i]["inst"])
    safety, have, used = [], Counter(), set()
    for g in sorted(GROUPS, key=lambda g: total[g]):
        cand_lists = (oiv,) if g == "stairs" else (street,)
        for cand in cand_lists:
            cand = [i for i in cand if atoms[i]["inst"][g] > 0 and i not in used]
            rng.shuffle(cand)
            for i in cand:
                if have[g] >= a.safety_min:
                    break
                safety.append(i)
                used.add(i)
                have.update(atoms[i]["inst"])
    vistas_pool = [i for i in vistas_pool if i not in used]
    rf_pool = [i for i in rf_pool if i not in used]
    # ---- clients from the remaining pools
    v_clients, v_left = assign_clients(atoms, vistas_pool, a.vistas_clients, a.quota, a.alpha, rng, "v")
    r_clients, r_left = assign_clients(atoms, rf_pool, a.rf_clients, a.quota, a.alpha, rng, "r")
    clients = {**v_clients, **r_clients}
    fed = {"params": vars(a) | {"split_sizes": sizes, "rf_client_sources": RF_CLIENT_SOURCES,
                                "groups": GROUPS}, "clients": {}}
    for name, c in clients.items():
        sp = split_client(atoms, c, sizes, rng)
        fed["clients"][name] = {"q": c["q"], "sources": dict(Counter(atoms[x]["source"] for x in c["atoms"])),
                                **{s: sorted(rows[m]["image"] for x in sp[s] for m in atoms[x]["members"])
                                   for s in sizes}}
        used.update(c["atoms"])
        fed["clients"][name]["images"] = sum(atoms[x]["size"] for x in c["atoms"])
    fed["safety"] = sorted(rows[m]["image"] for x in safety for m in atoms[x]["members"])
    fed["safety_instances"] = {g: have[g] for g in GROUPS}
    fed["server"] = sorted(rows[m]["image"] for i in range(len(atoms)) if i not in used for m in atoms[i]["members"])
    fed["atoms"] = {"total": len(atoms), "multi_image": sum(t["size"] > 1 for t in atoms),
                    "vistas_pool": len(vistas_pool), "rf_pool": len(rf_pool)}
    Path(a.out).write_text(json.dumps(fed))
    print(json.dumps({k: v for k, v in fed.items() if k in ("params", "safety_instances", "atoms")}, indent=1))
    print("server images", len(fed["server"]), "safety images", len(fed["safety"]))


if __name__ == "__main__":
    main()
