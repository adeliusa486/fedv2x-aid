#!/usr/bin/env python3
"""Real-image federation simulator of FedV2X-Aid and its baselines (E1; also E7 ablations and E3 curves).

Every method trains the 152-class Detect head (BatchNorm folded) of the round-0 model on cached FP16 trunk
features of the 30 aids (src/fl/build_cache.py). All aids of a round are trained in one pass with a stacked
head (src/fl/stacked.py), which is numerically equivalent to training them one by one (tests/test_stacked.py).
The protocol is pre-registered in docs/preregistration_E1.md.

    python src/fl/simulate.py --method fedv2x --seed 0 --out results/phase6/E1
"""
from __future__ import annotations

import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.fl.feature_cache import FeatureShard  # noqa: E402
from src.fl.heads import fused_head, get_vec, load_round0, predict, predict_batched, set_vec, trainable  # noqa: E402
from src.fl.loss import ScopedDetectionLoss  # noqa: E402
from src.fl.stacked import StackedHead  # noqa: E402

GROUPS = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]
CACHE = Path(os.environ.get("FEDV2X_WORK", "work") + "/cache")
ABLATIONS = {"no_v2x", "no_range", "no_scope", "no_quality", "fixed_rsu", "no_normbound"}
METHODS = ["pretrained", "local", "fedavg", "fedprox", "scaffold", "fedrep", "hier_st", "hier_mt", "fedsto",
           "semifl", "central", "fedv2x", "server"]
HIER = {"fedv2x", "hier_st", "hier_mt"}
SELF = {"hier_st", "hier_mt", "fedsto", "semifl"}


# ----------------------------------------------------------------------------------------------- data
class Rows:
    """Rows of a memory-mapped FP16 feature array, selected through an index map; returns torch tensors.

    Memory-mapped files share the operating system's file cache, so several simulator processes can use one cached
    copy of the features (cold random reads are slow on Windows, hence ``warm`` reads every file sequentially once).
    """

    def __init__(self, mm, idx=None):
        self.mm = mm
        self.idx = np.arange(len(mm)) if idx is None else np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        if isinstance(k, torch.Tensor):
            k = k.numpy()
        rows = self.idx[k]
        if np.ndim(rows) == 0:
            return torch.from_numpy(np.ascontiguousarray(self.mm[int(rows)]))
        order = np.argsort(rows, kind="stable")
        out = np.ascontiguousarray(self.mm[rows[order]])
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return torch.from_numpy(out[inv])


def warm(folder: Path):
    for l in ("p3", "p4", "p5"):
        with open(folder / f"{l}.npy", "rb") as fh:
            while fh.read(64 << 20):
                pass


def load_full(folder: Path) -> list[Rows]:
    """The three FP16 feature arrays of a shard, memory-mapped (shared file cache)."""
    return [Rows(np.load(folder / f"{l}.npy", mmap_mode="r")) for l in ("p3", "p4", "p5")]


class ClientData:
    """Training buffer of one aid (FP16 features of its buffer frames), memory-mapped or, with ``ram``, read into
    host memory (self-training baselines label every training image in every round)."""

    def __init__(self, name, folder: Path, idx: np.ndarray, ram=False):
        self.name, self.idx = name, np.asarray(idx, dtype=np.int64)
        self.feats = [Rows(np.load(folder / f"{l}.npy", mmap_mode=None if ram else "r"), self.idx)
                      for l in ("p3", "p4", "p5")]
        self.n = len(idx)


def pad_stack(lists, width, fill=0.0, dtype=torch.float32):
    n = len(lists)
    m = max([len(x) for x in lists] + [1])
    out = torch.full((n, m, width) if width else (n, m), fill, dtype=dtype)
    mask = torch.zeros(n, m, dtype=torch.bool)
    for i, x in enumerate(lists):
        if len(x):
            out[i, :len(x)] = torch.as_tensor(np.asarray(x), dtype=dtype)
            mask[i, :len(x)] = True
    return out, mask


class Labels:
    """Padded per-frame label tensors of one aid's buffer (kept on the host; batches are moved to the GPU)."""

    def __init__(self, frames: list[dict], nc: int):
        self.gt_boxes, self.gt_mask = pad_stack([f["boxes"] for f in frames], 4)
        self.gt_cls, _ = pad_stack([f["cls"] for f in frames], 0, 0, torch.long)
        self.gt_w, _ = pad_stack([f["w"] for f in frames], 0, 0.0)
        self.attested = torch.zeros(len(frames), nc, dtype=torch.bool)
        for i, f in enumerate(frames):
            self.attested[i, list(f["attested"])] = True
        self.prop_boxes, self.prop_mask = pad_stack([f.get("props", np.zeros((0, 4))) for f in frames], 4)
        self.reg_boxes, self.reg_mask = pad_stack([f.get("regions", np.zeros((0, 4))) for f in frames], 4)
        rs = [f.get("region_sets", []) for f in frames]
        R = self.reg_boxes.shape[1]
        self.reg_sets = torch.zeros(len(frames), R, nc, dtype=torch.bool)
        for i, sets in enumerate(rs):
            for r, s in enumerate(sets):
                self.reg_sets[i, r, list(s)] = True

    def batch(self, idx, device="cuda"):
        def g(t):
            return t[idx].to(device, non_blocking=True)
        return {"gt_boxes": g(self.gt_boxes), "gt_cls": g(self.gt_cls), "gt_w": g(self.gt_w),
                "gt_mask": g(self.gt_mask), "attested": g(self.attested), "prop_boxes": g(self.prop_boxes),
                "prop_mask": g(self.prop_mask), "reg_boxes": g(self.reg_boxes), "reg_sets": g(self.reg_sets),
                "reg_mask": g(self.reg_mask)}


def merge_batches(bs: list[dict]) -> dict:
    """Concatenate per-client batches, padding the label dimension (dim 1) with zeros / False."""
    out = {}
    for k in bs[0]:
        ts = [b[k] for b in bs]
        if ts[0].dim() >= 2 and k != "attested":
            m = max(t.shape[1] for t in ts)
            padded = []
            for t in ts:
                if t.shape[1] < m:
                    z = torch.zeros((t.shape[0], m) + tuple(t.shape[2:]), dtype=t.dtype, device=t.device)
                    z[:, :t.shape[1]] = t
                    t = z
                padded.append(t)
            ts = padded
        out[k] = torch.cat(ts, 0)
    return out


# ----------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_heads(heads: dict, feats_of, labels: dict, group_of: dict, allowed: dict, device="cuda", bs=32,
               splits=("val", "test")):
    """Predictions of each aid's head on its own split; returns accumulators per (split) with aid tags.

    Each image is scored only on the hazard classes that the human labels of its own source attest
    (``allowed[aid][split][j]``; aids of Roboflow images mix several projects): predictions and labels of other
    classes are dropped, so that a correct detection of, e.g., a pothole in an image whose source never labels
    potholes is not counted as a false positive."""
    from src.metrics.detection import DetectionAccumulator
    accs = {"val": DetectionAccumulator(group_of=group_of), "test": DetectionAccumulator(group_of=group_of),
            "val_cls": DetectionAccumulator(), "test_cls": DetectionAccumulator()}
    for name, head in heads.items():
        head.eval()
        for split in splits:
            full, lab = feats_of(name, split), labels[name][split]
            n = len(full[0])
            for i0 in range(0, n, bs):
                idx = np.arange(i0, min(i0 + bs, n))
                feats = [f[i0:i0 + len(idx)].to(device).float() for f in full]
                dets = predict(head, feats)
                for j, d in zip(idx, dets):
                    ok = allowed[name][split][j]
                    d = d.cpu().numpy()
                    gb, gc = lab[j]["human"]
                    gc = gc.astype(int)
                    keep_p = np.isin(d[:, 5].astype(int), ok)
                    keep_g = np.isin(gc, ok)
                    d, gb, gc = d[keep_p], gb[keep_g], gc[keep_g]
                    accs[split].add(d[:, :4], d[:, 4], d[:, 5].astype(int), gb, gc, image_id=(name, j), aid=name)
                    hz = np.isin(d[:, 5].astype(int), list(group_of))
                    gz = np.isin(gc, list(group_of))
                    accs[split + "_cls"].add(d[hz, :4], d[hz, 4], d[hz, 5].astype(int), gb[gz], gc[gz],
                                             image_id=(name, j), aid=name)
        head.train()
    return accs


def summarise(accs, aids, attested_groups: dict, n_min=20):
    from src.metrics.detection import mean_and_worst_aid, recall_at_precision
    rp = recall_at_precision(accs["val"], accs["test"], GROUPS, 0.8, aids=aids)
    per_group = {g: rp["per_label"][g] for g in GROUPS}
    primary = float(np.mean([per_group[g]["test_recall"] for g in GROUPS]))
    five = float(np.mean([per_group[g]["test_recall"] for g in GROUPS if g != "stairs"]))
    per_aid = {f"{a}|{g}": v for (a, g), v in rp["per_aid"].items() if g in attested_groups.get(a, ())}
    pa = {(a, g): v for (a, g), v in rp["per_aid"].items() if g in attested_groups.get(a, ())}
    aid_means, mean_aid, worst, who = mean_and_worst_aid(pa, set(GROUPS), min_instances=n_min)
    m50, ncls = accs["test_cls"].map()
    return {"primary_mean_r_at_p80": primary, "mean_r_at_p80_without_stairs": five, "per_group": per_group,
            "per_aid_group": per_aid, "aid_means": aid_means, "mean_aid_recall": mean_aid,
            "worst_aid_recall": worst, "worst_aid": who, "map50_hazard_classes": m50, "hazard_classes_scored": ncls}


# ----------------------------------------------------------------------------------------------- main loop
class Sim:
    def __init__(self, a):
        self.a = a
        self.dev = "cuda"
        torch.manual_seed(a.seed)
        self.rng = np.random.default_rng(1000 + a.seed)
        self.fed = json.loads((ROOT / "results/phase2/federation.json").read_text())
        self.aids = sorted(self.fed["clients"])[: a.max_aids] if a.max_aids else sorted(self.fed["clients"])
        tax = yaml.safe_load((ROOT / "src/data/hazard_groups.yaml").read_text(encoding="utf-8"))
        names = [n.strip() for n in (ROOT / "data/taxonomy/class_names_152.txt")
                 .read_text(encoding="utf-8").splitlines() if n.strip()]
        self.names = names
        self.group_of = {names.index(c): g for g in GROUPS for c in tax["groups"][g]["classes"]}
        att = yaml.safe_load((ROOT / "docs/attested_classes.yaml").read_text(encoding="utf-8"))["sources"]
        self.attested_groups, self.allowed = {}, {}
        ok_src = {k: np.array(sorted({names.index(c) for c in v["classes"]} & set(self.group_of)), dtype=int)
                  for k, v in att.items()}
        for a_ in self.aids:                    # an image's source is the prefix of its name (e.g. rf_risk__...)
            c_ = self.fed["clients"][a_]
            self.attested_groups[a_] = set().union(*(att[k]["evaluated_groups"] for k in c_["sources"]))
            self.allowed[a_] = {sp: [ok_src[im.split("__")[0]] for im in c_[sp]] for sp in ("val", "test")}
        self.template = load_round0(a.round0, self.dev)
        self.w0 = get_vec(self.template).clone()
        self.nc = self.template.nc
        self.crit = ScopedDetectionLoss(self.nc, self.template.stride.tolist(), device=self.dev,
                                        beta=a.beta, lambda_t=a.lambda_t)
        if a.warm:
            for n in self.aids:
                for s_ in ("train", "val"):
                    warm(CACHE / "clients" / n / s_)
        self.val_feats = {n: load_full(CACHE / "clients" / n / "val") for n in self.aids}
        self.labels = {n: {s: pickle.load(open(CACHE / "clients" / n / s / "labels.pkl", "rb"))
                           for s in ("val", "test")} for n in self.aids}
        self.teacher = pickle.load(open(Path(a.teacher_labels) / f"seed{a.seed}.pkl", "rb")) \
            if a.method not in SELF and a.method != "pretrained" else None
        self.trace = json.loads((ROOT / "results/phase2/contact_traces.json").read_text())["traces"][str(a.seed)]
        self.abl = set() if a.ablate == "none" else set(a.ablate.split("+"))
        assert self.abl <= ABLATIONS, f"unknown ablation(s): {self.abl - ABLATIONS}"
        self.log = {"method": a.method, "seed": a.seed, "args": vars(a), "curve": [], "comm": {}}

    # ---- buffers
    def build_buffers(self):
        self.data, self.lab = {}, {}
        for n in self.aids:
            if self.a.method in SELF:
                idx = np.arange(len(self.fed["clients"][n]["train"]))
            else:
                frames = self.teacher[n]
                idx = np.array(sorted(i for i, f in frames.items() if f["signal"]), dtype=int)
            self.data[n] = ClientData(n, CACHE / "clients" / n / "train", idx, ram=self.a.method in SELF)
            if self.a.method not in SELF:
                fr = [self.teacher[n][int(i)] for i in idx]
                if "no_v2x" in self.abl:
                    fr = [self.drop(f, "v2x") for f in fr]
                if "no_range" in self.abl:
                    fr = [self.drop(f, "range") for f in fr]
                self.lab[n] = Labels(fr, self.nc)
                self.mass = getattr(self, "mass", {})
                self.mass[n] = float(sum(f["w"].sum() for f in fr) + 0.5 * sum(len(f.get("regions", [])) for f in fr))

    @staticmethod
    def drop(f, which):
        g = dict(f)
        keep = np.array([s != which for s in f["src"]], dtype=bool) if len(f["src"]) else np.zeros(0, bool)
        g["boxes"], g["cls"], g["w"], g["src"] = f["boxes"][keep], f["cls"][keep], f["w"][keep], \
            [s for s, k in zip(f["src"], keep) if k]
        if which == "range":
            g["regions"], g["region_sets"] = np.zeros((0, 4), np.float32), []
        g["attested"] = set(f["attested_v2x"]) if which == "range" else set(f["attested_range"])
        return g

    # ---- self labels
    @torch.no_grad()
    def self_labels(self, head_vecs: torch.Tensor, aids: list[str], thr=0.5):
        """Pseudo-labels (score >= thr) of each aid's starting head on its training images."""
        out = {}
        jobs = [(n, i0) for n in aids for i0 in range(0, self.data[n].n, 64)]
        pool = ThreadPoolExecutor(1)                    # features of the next batch are read while the GPU works

        def gather(j):
            n, i0 = jobs[j]
            return [f[i0:i0 + 64] for f in self.data[n].feats]
        nxt = pool.submit(gather, 0) if jobs else None
        j = 0
        for k, n in enumerate(aids):
            set_vec(self.template, head_vecs[k])
            d = self.data[n]
            frames = []
            for i0 in range(0, d.n, 64):
                host = nxt.result()
                j += 1
                if j < len(jobs):
                    nxt = pool.submit(gather, j)
                feats = [f.to(self.dev).float() for f in host]
                for det in predict_batched(self.template, feats, conf=thr):
                    det = det.cpu().numpy()
                    frames.append({"boxes": det[:, :4], "cls": det[:, 5].astype(int),
                                   "w": np.ones(len(det), np.float32), "attested": range(self.nc)})
            out[n] = Labels(frames, self.nc)
        pool.shutdown()
        return out

    # ---- local training of K aids at once
    def local_train(self, aids, start_vecs, labels, steps, scoped, extra=None):
        K = len(aids)
        st = StackedHead(self.template, K).to(self.dev)
        st.set_vecs(start_vecs)
        params = list(st.parameters())
        opt = torch.optim.SGD(params, lr=self.a.lr, momentum=0.9, weight_decay=5e-4)
        teacher = None
        if scoped and self.a.beta > 0:
            teacher = StackedHead(self.template, K).to(self.dev)
            teacher.set_vecs(start_vecs)
            teacher.requires_grad_(False)
        client = torch.arange(K, device=self.dev).repeat_interleave(self.a.batch)
        w_start = start_vecs.clone()
        # batch indices drawn in the same order as one draw per step and aid; the features of step s+1 are
        # gathered on a host thread while the GPU works on step s
        plan = [[np.sort(self.rng.choice(self.data[n].n, self.a.batch, replace=self.data[n].n < self.a.batch))
                 for n in aids] for _ in range(steps)]

        def gather(idxs):
            return [torch.stack([self.data[n].feats[l][ix] for n, ix in zip(aids, idxs)]) for l in range(3)]
        pool = ThreadPoolExecutor(1)
        nxt = pool.submit(gather, plan[0])
        for s in range(steps):
            idxs = plan[s]
            host = nxt.result()
            if s + 1 < steps:
                nxt = pool.submit(gather, plan[s + 1])
            feats = [f.to(self.dev).float() for f in host]
            batch = merge_batches([labels[n].batch(torch.from_numpy(ix), self.dev) for n, ix in zip(aids, idxs)])
            with torch.autocast("cuda", dtype=torch.float16):
                preds = st(feats)
                tl = teacher(feats)["scores"] if teacher is not None else None
            loss, parts = self.crit(preds, batch, tl, scoped=scoped, client=client, n_clients=K)
            if extra is not None:
                loss = loss + extra(st, w_start)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.a.method == "scaffold" and self.scaf is not None:
                c_glob, c_loc = self.scaf
                i = 0
                for n_, size in zip(st.names, st.sizes):
                    g = st.p(n_).grad
                    corr = (c_glob[None, i:i + size] - c_loc[:, i:i + size]).view(g.shape)
                    g.add_(corr)
                    i += size
            if self.a.method == "fedrep":
                self.fedrep_mask(st, s, steps)
            if not self.finite(params):
                continue                         # FP16 overflow: skip the step, as a gradient scaler would
            opt.step()
        pool.shutdown()
        return st.get_vecs()

    def finite(self, params):
        """True if every gradient is finite; otherwise the gradients are cleared and the step is counted."""
        g = [q.grad for q in params if q.grad is not None]
        if g and bool(torch.isfinite(torch.stack([x.float().abs().sum() for x in g]).sum())):
            return True
        for q in params:
            q.grad = None
        self.log["skipped_steps"] = self.log.get("skipped_steps", 0) + 1
        return False

    # ---- FedRep: personal = final class convolutions cv3.l.2
    def fedrep_mask(self, st, s, steps):
        personal_phase = s < max(1, steps // 4)
        for n_ in st.names:
            personal = n_.startswith("cv3.") and n_.split(".")[2] == "2"
            if personal != personal_phase:
                st.p(n_).grad = None                 # SGD skips it: no momentum or weight-decay update

    def personal_slices(self):
        sl, i = [], 0
        for n_, p in trainable(self.template):
            size = p.numel()
            if n_.startswith("cv3.") and n_.split(".")[2] == "2":
                sl.append((i, i + size))
            i += size
        return sl

    def train_chunks(self, aids, start_vecs, labels, steps, scoped, extra=None):
        out = []
        for i0 in range(0, len(aids), self.a.chunk):
            ch = aids[i0:i0 + self.a.chunk]
            sv = start_vecs[i0:i0 + self.a.chunk]
            if self.a.method == "scaffold":
                c_glob, c_loc_all = self.scaf_state
                self.scaf = (c_glob, torch.stack([c_loc_all[n] for n in ch]))
            out.append(self.local_train(ch, sv, labels, steps, scoped, extra))
        return torch.cat(out, 0)

    # ---- methods
    def run(self):
        a = self.a
        t0 = time.time()
        method = a.method
        self.scaf = None
        if a.val_only:                               # pilot reference: round-0 head on the validation splits
            self.curve(0, {n: self.w0 for n in self.aids})
        if method == "pretrained":
            return self.finish({n: self.w0 for n in self.aids}, t0)
        self.build_buffers()
        scoped = method not in SELF and "no_scope" not in self.abl
        D = self.w0.numel()
        bytes_head = D * 2                                               # FP16 update
        comm = {"wan_bytes": 0, "uplink_bytes": 0, "download_bytes": 0, "uploads": 0, "deferred_rsu_rounds": 0,
                "rsu_rounds": 0}
        self.comm = comm                                                 # cumulative, copied into curve points
        extra = None
        if method == "fedprox":
            mu = a.mu

            def extra(st, w_start):
                return 0.5 * mu * ((st.vecs() - w_start) ** 2).sum()
        if method == "fedsto":
            def extra(st, w_start):
                pen = 0.0
                for n_ in st.names:
                    if n_.endswith("weight") and st.p(n_).shape[-1] == 1 and st.p(n_).dim() == 5:
                        w = st.p(n_).flatten(2)                           # (K, out, in)
                        g = w @ w.transpose(1, 2)
                        eye = torch.eye(g.shape[-1], device=g.device)
                        pen = pen + ((g - eye) ** 2).sum()
                return a.orth * pen
        if method == "central":
            return self.run_central(t0)
        if method == "server":                       # E9 control: server supervised steps only, no aids
            g = self.w0.clone()
            for r in range(a.edge_rounds):
                g = self.server_step(g)
                if (r + 1) % a.kappa == 0:
                    self.curve(r + 1, {n: g for n in self.aids})
            return self.finish({n: g for n in self.aids}, t0, comm)
        if method == "local":
            vecs = self.w0.unsqueeze(0).repeat(len(self.aids), 1)
            for r in range(a.edge_rounds):
                vecs = self.train_chunks(self.aids, vecs, self.lab, a.steps, scoped, extra)
                if (r + 1) % a.kappa == 0:
                    self.curve(r + 1, {n: vecs[i] for i, n in enumerate(self.aids)})
            return self.finish({n: vecs[i] for i, n in enumerate(self.aids)}, t0, comm)
        if method in HIER:
            return self.run_hier(t0, scoped, comm, bytes_head)
        # ---- flat methods: FedAvg, FedProx, SCAFFOLD, FedRep, FedSTO, SemiFL
        g = self.w0.clone()
        personal = {n: self.w0.clone() for n in self.aids} if method == "fedrep" else None
        psl = self.personal_slices() if method == "fedrep" else []
        if method == "scaffold":
            self.scaf_state = (torch.zeros(D, device=self.dev), {n: torch.zeros(D, device=self.dev) for n in self.aids})
        weights = {n: (self.data[n].n if method in SELF else self.data[n].n) for n in self.aids}
        for r in range(a.edge_rounds):
            if method == "semifl":
                g = self.server_step(g)
            if method == "fedrep":
                start = torch.stack([self.with_personal(g, personal[n], psl) for n in self.aids])
            else:
                start = g.unsqueeze(0).repeat(len(self.aids), 1)
            labels = self.self_labels(start, self.aids) if method in SELF else self.lab
            new = self.train_chunks(self.aids, start, labels, a.steps, scoped, extra)
            w = torch.tensor([weights[n] for n in self.aids], device=self.dev, dtype=torch.float32)
            if method == "scaffold":
                c_glob, c_loc = self.scaf_state
                deltas = []
                for i, n in enumerate(self.aids):
                    c_new = c_loc[n] - c_glob + (g - new[i]) / (a.steps * a.lr / (1 - 0.9))   # option II; effective step of momentum-0.9 SGD
                    deltas.append(c_new - c_loc[n])
                    c_loc[n] = c_new
                c_glob = c_glob + torch.stack(deltas).mean(0)
                self.scaf_state = (c_glob, c_loc)
            g = (w[:, None] * new).sum(0) / w.sum()
            if method == "fedrep":
                for i, n in enumerate(self.aids):
                    personal[n] = new[i].clone()
            comm["uplink_bytes"] += len(self.aids) * bytes_head
            comm["wan_bytes"] += len(self.aids) * bytes_head
            comm["download_bytes"] += len(self.aids) * bytes_head
            comm["uploads"] += len(self.aids)
            if (r + 1) % a.kappa == 0:
                heads = {n: (self.with_personal(g, personal[n], psl) if method == "fedrep" else g) for n in self.aids}
                self.curve(r + 1, heads)
        heads = {n: (self.with_personal(g, personal[n], psl) if method == "fedrep" else g) for n in self.aids}
        return self.finish(heads, t0, comm)

    @staticmethod
    def with_personal(g, p, slices):
        v = g.clone()
        for i0, i1 in slices:
            v[i0:i1] = p[i0:i1]
        return v

    def run_hier(self, t0, scoped, comm, bytes_head):
        a, method = self.a, self.a.method
        n_rsu = 6
        g = self.w0.clone()
        edge = [g.clone() for _ in range(n_rsu)]
        acc = [[] for _ in range(n_rsu)]                      # pending (vec, mass) reports per RSU
        agg_mass = [0.0] * n_rsu
        active = set()
        gamma = None
        ema = self.w0.clone()
        if a.dp_z > 0:
            assert method == "fedv2x" and "no_normbound" not in self.abl, "client-level DP needs the norm bound"
            self.dp_gen = torch.Generator(device=self.dev).manual_seed(2000 + a.seed)
            dp_mask = self.dp_mask()                # aids release only the hazard-class classifier rows
            gamma = a.dp_clip                       # fixed clip: a clip computed from the aids' updates would leak
            self.log["dp_dims"] = int(dp_mask.sum())
        for r in range(a.edge_rounds):
            row = self.trace[r]
            part = [(i, n) for i, n in enumerate(self.aids) if row[i] is not None]
            if a.participation < 1.0:
                part = [(i, n) for i, n in part if self.rng.random() < a.participation]
            aids = [n for _, n in part]
            if aids:
                start = torch.stack([edge[row[i][0]] for i, _ in part])
                if method == "hier_st":
                    labels = self.self_labels(start, aids)
                elif method == "hier_mt":
                    labels = self.self_labels(ema.unsqueeze(0).repeat(len(aids), 1), aids)
                else:
                    labels = self.lab
                new = self.train_chunks(aids, start, labels, a.steps, scoped)
                delta = new - start
                if a.attackers:                       # J4: the first K aids reverse their update and scale it tenfold
                    bad = torch.tensor([n in self.aids[:a.attackers] for n in aids], device=self.dev)
                    delta = torch.where(bad[:, None], -10.0 * delta, delta)
                    new = start + delta
                if a.dp_z > 0:
                    delta = delta * dp_mask
                    new = start + delta
                if method == "fedv2x" and "no_normbound" not in self.abl:
                    norms = delta.norm(dim=1)
                    if gamma is None:                 # 90th percentile of benign norms (the paper's Gamma)
                        benign = norms[~bad] if a.attackers else norms
                        gamma = float(torch.quantile(benign, 0.9))
                        self.log["gamma"] = gamma
                    scale = torch.clamp(gamma / (norms + 1e-12), max=1.0)
                    new = start + delta * scale[:, None]
                for k, (i, n) in enumerate(part):
                    self.log.setdefault("participations", {})[n] = self.log.get("participations", {}).get(n, 0) + 1
                    up = row[i][1] if "fixed_rsu" not in self.abl else self.home_rsu(i)
                    m = self.mass.get(n, 1.0) if method == "fedv2x" and "no_quality" not in self.abl else \
                        float(self.data[n].n)
                    if a.dp_z > 0:
                        m = 1.0                         # uniform weights bound each aid's influence by the clip
                    acc[up].append((new[k], m))
                    comm["uplink_bytes"] += bytes_head
                    comm["download_bytes"] += bytes_head
                    comm["uploads"] += 1
            for e in range(n_rsu):
                comm["rsu_rounds"] += 1
                if len(acc[e]) >= 3:
                    ms = torch.tensor([m for _, m in acc[e]], device=self.dev)
                    vs = torch.stack([v for v, _ in acc[e]])
                    edge[e] = (ms[:, None] * vs).sum(0) / ms.sum()
                    if a.dp_z > 0:                    # client-level Gaussian mechanism on the clipped RSU aggregate
                        std = a.dp_z * gamma * float(ms.max()) / float(ms.sum())
                        edge[e] = edge[e] + std * dp_mask * torch.randn(edge[e].shape, device=self.dev,
                                                                          generator=self.dp_gen)
                    agg_mass[e] += float(ms.sum())
                    active.add(e)
                    acc[e] = []
                else:
                    comm["deferred_rsu_rounds"] += 1
            if (r + 1) % a.kappa == 0:
                if active:
                    M = torch.tensor([agg_mass[e] for e in sorted(active)], device=self.dev)
                    V = torch.stack([edge[e] for e in sorted(active)])
                    g = (M[:, None] * V).sum(0) / M.sum()
                    comm["wan_bytes"] += len(active) * bytes_head
                if a.server_anchor:                   # E9: the cloud anchors g on its labelled data, no extra WAN
                    for _ in range(a.kappa):
                        g = self.server_step(g)
                if method == "hier_mt":
                    ema = 0.9 * ema + 0.1 * g
                edge = [g.clone() for _ in range(n_rsu)]
                acc = [[] for _ in range(n_rsu)]
                agg_mass = [0.0] * n_rsu
                active = set()
                self.curve(r + 1, {n: g for n in self.aids})
        return self.finish({n: g for n in self.aids}, t0, comm)

    def dp_mask(self):
        """1 on the classifier rows (weights and biases of cv3.*.2) of the hazard classes, 0 elsewhere."""
        hz = torch.zeros(self.nc, dtype=torch.bool)
        hz[list(self.group_of)] = True
        mask = []
        for n_, p in trainable(self.template):
            m = torch.zeros(p.shape, dtype=torch.bool)
            if n_.startswith("cv3.") and n_.split(".")[2] == "2":
                m[hz] = True
            mask.append(m.reshape(-1))
        return torch.cat(mask).float().to(self.dev)

    def home_rsu(self, i):
        return i % 6

    def server_step(self, g):
        """SemiFL-style alternate training: S supervised steps on the labelled server subset."""
        if not hasattr(self, "server"):
            lab = pickle.load(open(CACHE / "server_labelled" / "labels.pkl", "rb"))
            self.data["__server__"] = ClientData("__server__", CACHE / "server_labelled", np.arange(len(lab)))
            frames = [{"boxes": l["world"][0], "cls": l["world"][1].astype(int), "w": np.ones(len(l["world"][1]), np.float32),
                       "attested": range(self.nc)} for l in lab]
            self.server = Labels(frames, self.nc)
        out = self.local_train(["__server__"], g.unsqueeze(0), {"__server__": self.server}, self.a.steps, scoped=False)
        return out[0]

    def run_central(self, t0):
        a = self.a
        pooled = [(n, i) for n in self.aids for i in range(self.data[n].n)]
        epochs = np.mean([a.edge_rounds * a.steps * a.batch / max(self.data[n].n, 1) for n in self.aids])
        steps = int(round(epochs * len(pooled) / a.central_batch))
        self.log["central_steps"] = steps
        st = StackedHead(self.template, 1).to(self.dev)
        st.set_vecs(self.w0.unsqueeze(0))
        teacher = StackedHead(self.template, 1).to(self.dev)
        teacher.set_vecs(self.w0.unsqueeze(0))
        teacher.requires_grad_(False)
        opt = torch.optim.SGD(st.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
        for s in range(steps):
            pick = self.rng.choice(len(pooled), a.central_batch, replace=False)
            sel = sorted(pooled[j] for j in pick)
            feats = [torch.stack([self.data[n].feats[l][i] for n, i in sel]).unsqueeze(0).to(self.dev).float()
                     for l in range(3)]
            batch = merge_batches([self.lab[n].batch(torch.tensor([i]), self.dev) for n, i in sel])
            with torch.autocast("cuda", dtype=torch.float16):
                preds = st(feats)
                tl = teacher(feats)["scores"]
            loss, _ = self.crit(preds, batch, tl, scoped=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.finite(list(st.parameters())):
                opt.step()
            if (s + 1) % max(1, steps // 25) == 0:
                teacher.set_vecs(st.get_vecs())                      # distillation target: the last checkpoint
                self.curve(s + 1, {n: st.get_vecs()[0] for n in self.aids})
        g = st.get_vecs()[0]
        return self.finish({n: g for n in self.aids}, t0, {})

    # ---- evaluation and output
    def heads_for(self, vecs: dict):
        heads, cache = {}, {}
        for n, v in vecs.items():
            key = id(v)
            if key not in cache:
                h = fused_head(self.template, self.dev)
                set_vec(h, v)
                cache[key] = h
            heads[n] = cache[key]
        return heads

    def curve(self, r, vecs):
        """Validation curve point (thresholds chosen on the validation data itself; for time-to-target only).
        Evaluated at every ``--curves``-th call (one call per cloud round; 0 disables curves)."""
        self._curve_calls = getattr(self, "_curve_calls", 0) + 1
        if not self.a.curves or self._curve_calls % self.a.curves:
            return
        from src.metrics.detection import recall_at_precision
        accs = eval_heads(self.heads_for(vecs), lambda n, s: self.val_feats[n], self.labels, self.group_of,
                          self.allowed, self.dev, splits=("val",))
        rp = recall_at_precision(accs["val"], accs["val"], GROUPS, 0.8)
        self.log["curve"].append({"round": r, "val_mean_r_at_p80": float(np.mean(
            [rp["per_label"][g]["val_recall"] for g in GROUPS])), "time_s": round(time.time() - self.t_start, 1),
            "comm": dict(getattr(self, "comm", {}))})

    def finish(self, vecs, t0, comm=None):
        if self.a.val_only:                          # pilot: validation curve only, the test split is never read
            return {"method": self.a.method, "seed": self.a.seed, "val_only": True, "comm": comm or {},
                    "seconds": round(time.time() - t0, 1), "log": self.log}, vecs
        def feats_of(n, s):
            return self.val_feats[n] if s == "val" else load_full(CACHE / "clients" / n / "test")
        accs = eval_heads(self.heads_for(vecs), feats_of, self.labels, self.group_of, self.allowed, self.dev)
        res = summarise(accs, self.aids, self.attested_groups)
        res.update({"method": self.a.method, "seed": self.a.seed, "ablate": self.a.ablate, "comm": comm or {},
                    "seconds": round(time.time() - t0, 1), "log": self.log})
        return res, vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round0", default=os.environ.get("FEDV2X_WORK", "work") + "/round0/round0_head.pt")
    ap.add_argument("--teacher-labels", default=os.environ.get("FEDV2X_WORK", "work") + "/teachers/E1")
    ap.add_argument("--edge-rounds", type=int, default=50)
    ap.add_argument("--kappa", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--orth", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lambda-t", type=float, default=0.5)
    ap.add_argument("--central-batch", type=int, default=32)
    ap.add_argument("--participation", type=float, default=1.0)
    ap.add_argument("--dp-z", type=float, default=0.0, help="noise multiplier of client-level DP at the RSU (fedv2x)")
    ap.add_argument("--attackers", type=int, default=0, help="J4: number of poisoning aids (the first K aids)")
    ap.add_argument("--dp-clip", type=float, default=0.1, help="fixed L2 clip of an aid's released update under DP")
    ap.add_argument("--ablate", default="none", help="none, or one or more of ABLATIONS joined by '+'")
    ap.add_argument("--tag", default=None, help="name of the result file (default: method[-ablate])")
    ap.add_argument("--chunk", type=int, default=10, help="aids trained together in one stacked head (GPU memory)")
    ap.add_argument("--curves", type=int, default=5, help="validation curve every N cloud rounds (0: none)")
    ap.add_argument("--max-aids", type=int, default=0, help="debugging only: use the first N aids")
    ap.add_argument("--warm", type=int, default=1, help="read every feature file once so that it is in the file cache")
    ap.add_argument("--out", default=str(ROOT / "results/phase6/E1"))
    ap.add_argument("--server-anchor", action="store_true", help="E9: kappa server steps per cloud round (fedv2x)")
    ap.add_argument("--val-only", action="store_true", help="pilot: report validation curves only, never read test")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = (a.tag or f"{a.method}" + ("" if a.ablate == "none" else f"-{a.ablate}")) + f"_s{a.seed}"
    sim = Sim(a)
    sim.t_start = time.time()
    res, vecs = sim.run()
    try:
        res["code_version"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                             text=True).stdout.strip()
    except Exception:
        res["code_version"] = "unknown"
    res["config_hash"] = hashlib.sha256(json.dumps(vars(a), sort_keys=True).encode()).hexdigest()[:16]
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    heads_dir = Path(os.environ.get("FEDV2X_WORK", "work") + "/heads") / out.name
    heads_dir.mkdir(parents=True, exist_ok=True)
    uniq = {}
    for n, v in vecs.items():
        uniq.setdefault(id(v), (v, []))[1].append(n)
    torch.save({"vectors": [v.detach().cpu() for v, _ in uniq.values()],
                "aids": [names for _, names in uniq.values()]}, heads_dir / f"{tag}.pt")
    if a.val_only:
        print(tag, [(c["round"], round(c["val_mean_r_at_p80"], 4)) for c in res["log"]["curve"]])
        return
    print(f"{tag}: primary {res['primary_mean_r_at_p80']:.4f}, mAP50 {res['map50_hazard_classes']:.4f}, "
          f"worst aid {res['worst_aid_recall']:.4f}, {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
