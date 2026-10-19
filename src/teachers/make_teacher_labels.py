#!/usr/bin/env python3
"""Emulated teacher labels of every aid's training images (Phases 5-6; method in src/teachers/emulate.py).

Step 1 (once per round-0 model, GPU): proposals of the round-0 model on the cached trunk features of every aid's
training images (score >= 0.05, NMS IoU 0.7), with the probabilities of the range-teacher classes at each
detection -> $FEDV2X_WORK/teachers/proposals_<model tag>.pkl.
Step 2 (per seed, CPU, parallel over aids): for every training image
  * V2X labels from the world vehicles of the image (merged labels), the proposals and the labeller rates measured
    on KITTI (E2b): conditional recall ``--r-cond`` and precision ``--precision``;
  * range-sensor regions and promoted boxes (validity ``--p-valid`` from the bench log, E8);
  * the attested class sets C_s of the frame's label sources (CAM classes for V2X; drop-off or obstacle classes
    for range regions and promoted boxes).
Output: $FEDV2X_WORK/teachers/<tag>/seed<seed>.pkl and results/phase5/teacher_stats_<tag>_seed<seed>.json
(label counts and quality against the human labels).

    python src/teachers/make_teacher_labels.py --seed 0 --r-cond 0.9 --precision 0.9 --tag E1
"""
import os
import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.teachers.emulate import iou_matrix, range_events, v2x_labels  # noqa: E402

CACHE = Path(os.environ.get("FEDV2X_WORK", "work") + "/cache")
TEACH = Path(os.environ.get("FEDV2X_WORK", "work") + "/teachers")


def class_sets():
    names = [n.strip() for n in (ROOT / "data/taxonomy/class_names_152.txt")
             .read_text(encoding="utf-8").splitlines() if n.strip()]
    tax = yaml.safe_load((ROOT / "src/data/hazard_groups.yaml").read_text(encoding="utf-8"))
    cam = {names.index(c) for c in tax["teachers"]["v2x_cam"]}
    drop = [names.index(c) for c in tax["teachers"]["range_dropoff"]]
    obst = [names.index(c) for c in tax["teachers"]["range_obstacle"]]
    return names, cam, drop, obst


def proposals(round0, out):
    import torch
    from src.fl.heads import load_round0, predict_with_probs
    from src.fl.simulate import load_full
    _, _, drop, obst = class_sets()
    fed = json.loads((ROOT / "results/phase2/federation.json").read_text())
    head = load_round0(round0)
    head.eval()
    res = {}
    for aid in sorted(fed["clients"]):
        feats = load_full(CACHE / "clients" / aid / "train")
        dets = []
        for i0 in range(0, len(feats[0]), 32):
            fb = [f[i0:i0 + 32].cuda().float() for f in feats]
            with torch.no_grad():
                for d, probs in predict_with_probs(head, fb, drop + obst, conf=0.05):
                    dets.append((d.cpu().numpy().astype(np.float32), probs.cpu().numpy().astype(np.float16)))
        res[aid] = dets
        print(aid, len(dets), flush=True)
    with open(out, "wb") as fh:
        pickle.dump({"round0": round0, "conf": 0.05, "iou": 0.7, "dets": res}, fh)


def attested(images, names):
    """Per image, the classes that the human labels of its source attest (the source is the prefix of the image
    name), as in the simulator's evaluation."""
    att = yaml.safe_load((ROOT / "docs/attested_classes.yaml").read_text(encoding="utf-8"))["sources"]
    by_src = {k: {names.index(c) for c in v["classes"]} for k, v in att.items()}
    return [by_src[im.split("__")[0]] for im in images]


def emulate_aid(args):
    aid, dets, seed, a = args
    names, cam, drop, obst = class_sets()
    folder = CACHE / "clients" / aid / "train"
    labels = pickle.load(open(folder / "labels.pkl", "rb"))
    index = json.loads((folder / "index.json").read_text())
    att_img = attested(index["images"], names)
    rng = np.random.default_rng([10_000 + seed, int(aid[1:]) + (100 if aid[0] == "r" else 0)])
    frames = {}
    v2x_tot = {k: 0 for k in ("vehicles", "connected", "with_proposal", "correct", "wrong", "wrong_wanted")}
    rng_tot = {k: 0 for k in ("events", "invalid", "misses", "promoted")}
    prom = [0, 0]
    prom_cls = {}           # class -> [promoted boxes matching a human box, promoted], on aids whose labels attest it
    reg_hits = {c: [0, 0] for c in drop}
    for i, (d, probs) in enumerate(dets):
        lab = labels[i]
        wb, wc = lab["world"]
        gain, pad = index["gain"][i], index["pad"][i]
        vb, vc, vst = v2x_labels(wb, wc, d[:, :4], cam, rng, a["penetration"], a["r_cond"], a["precision"])
        for k in v2x_tot:
            v2x_tot[k] += vst[k]
        if a["no_range"]:
            regions, promoted, rst = [], [], {k: 0 for k in rng_tot}
        else:
            regions, promoted, rst = range_events(wb, wc, d[:, :4], d[:, 5].astype(int), d[:, 4], probs.astype(np.float32),
                                                  gain, pad, lab["size"], drop, obst, rng, p_valid=a["p_valid"])
        for k in rng_tot:
            rng_tot[k] += rst[k]
        hb, hc = lab["human"]
        hc = hc.astype(int)
        for p in promoted:
            prom[1] += 1
            pc = prom_cls.setdefault(int(p[1]), [0, 0]) if int(p[1]) in att_img[i] else [0, 0]
            pc[1] += 1
            m = hc == p[1]
            if m.any() and iou_matrix(p[0][None], hb[m]).max() >= 0.5:
                prom[0] += 1
                pc[0] += 1
        for c in drop:
            m = hc == c
            reg_hits[c][1] += int(m.sum())
            if m.any():
                for R, _ in regions:
                    reg_hits[c][0] += int((iou_matrix(hb[m], R[None])[:, 0] > 0).sum())
        boxes = [vb] + [np.array([p[0]]) for p in promoted]
        cls = [vc] + [np.array([p[1]]) for p in promoted]
        w = [np.ones(len(vc), np.float32)] + [np.array([p[2]], np.float32) for p in promoted]
        att_v2x = set(cam) if len(vc) else set()
        att_rng = set()
        for _, nm in regions:
            att_rng |= set(drop if nm == "dropoff" else obst)
        for p in promoted:
            att_rng |= set(drop if p[1] in drop else obst)
        frames[i] = {"boxes": np.concatenate(boxes).astype(np.float32).reshape(-1, 4),
                     "cls": np.concatenate(cls).astype(int), "w": np.concatenate(w).astype(np.float32),
                     "src": ["v2x"] * len(vc) + ["range"] * len(promoted), "props": d[:, :4].astype(np.float32),
                     "regions": np.array([R for R, _ in regions], np.float32).reshape(-1, 4),
                     "region_sets": [set(drop if nm == "dropoff" else obst) for _, nm in regions],
                     "attested": att_v2x | att_rng, "attested_v2x": att_v2x, "attested_range": att_rng,
                     "signal": bool(len(vc) or regions or promoted)}
    return aid, frames, v2x_tot, rng_tot, prom, reg_hits, prom_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="E1")
    ap.add_argument("--round0", default=os.environ.get("FEDV2X_WORK", "work") + "/round0/round0_head.pt")
    ap.add_argument("--model-tag", default="round0")
    ap.add_argument("--penetration", type=float, default=0.5)
    ap.add_argument("--r-cond", type=float, required=True)
    ap.add_argument("--precision", type=float, required=True)
    ap.add_argument("--p-valid", type=float, default=1.0)
    ap.add_argument("--no-range", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    names, cam, drop, obst = class_sets()
    pfile = TEACH / f"proposals_{a.model_tag}.pkl"
    TEACH.mkdir(parents=True, exist_ok=True)
    if not pfile.exists():
        proposals(a.round0, pfile)
    dets = pickle.load(open(pfile, "rb"))["dets"]
    opts = {"penetration": a.penetration, "r_cond": a.r_cond, "precision": a.precision, "p_valid": a.p_valid,
            "no_range": a.no_range}
    out, stats = {}, {"frames": 0, "frames_with_signal": 0}
    v2x_tot = {k: 0 for k in ("vehicles", "connected", "with_proposal", "correct", "wrong", "wrong_wanted")}
    rng_tot = {k: 0 for k in ("events", "invalid", "misses", "promoted")}
    prom = [0, 0]
    prom_cls = {}
    reg_hits = {c: [0, 0] for c in drop}
    with ProcessPoolExecutor(a.workers) as ex:
        for aid, frames, vt, rt, pr, rh, pcl in ex.map(emulate_aid, [(aid, dets[aid], a.seed, opts) for aid in sorted(dets)]):
            out[aid] = frames
            for k in v2x_tot:
                v2x_tot[k] += vt[k]
            for k in rng_tot:
                rng_tot[k] += rt[k]
            prom[0] += pr[0]; prom[1] += pr[1]
            for c in drop:
                reg_hits[c][0] += rh[c][0]; reg_hits[c][1] += rh[c][1]
            for c, (h, t) in pcl.items():
                pc = prom_cls.setdefault(c, [0, 0])
                pc[0] += h; pc[1] += t
            stats["frames"] += len(frames)
            stats["frames_with_signal"] += sum(f["signal"] for f in frames.values())
    stats["v2x"] = dict(v2x_tot, precision_emulated=v2x_tot["correct"] / max(v2x_tot["correct"] + v2x_tot["wrong"], 1))
    stats["range"] = dict(rng_tot, promoted_precision=prom[0] / max(prom[1], 1), promoted=prom[1],
                          dropoff_instances_touched_by_a_region={names[c]: v for c, v in reg_hits.items()},
                          promoted_per_class_on_attesting_images={names[c]: [h, t, h / t] for c, (h, t) in sorted(prom_cls.items(), key=lambda x: -x[1][1])})
    stats["params"] = vars(a)
    dst = TEACH / a.tag
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / f"seed{a.seed}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    res = ROOT / "results" / "phase5" / f"teacher_stats_{a.tag}_seed{a.seed}.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "params"}, indent=1, default=str))


if __name__ == "__main__":
    main()
