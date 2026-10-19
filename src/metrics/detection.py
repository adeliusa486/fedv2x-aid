"""Detection metrics of FedV2X-Aid: box matching, AP50, recall at a precision target (R@P80).

Conventions (fixed in Phase 2, used by every learning experiment):

* Boxes are (x1, y1, x2, y2) in pixels. A prediction is a true positive at IoU threshold t if it is
  matched to a ground-truth box of the same class (or hazard group) with IoU >= t.
* Two matching rules are implemented. ``rule="score"`` (default, COCO/VOC convention): predictions are
  visited in descending score order and each takes the unmatched ground truth with the highest IoU;
  duplicates of an already matched object are false positives. ``rule="iou"`` reproduces Ultralytics'
  validator (pairs above the threshold are matched in descending IoU order, each prediction and each
  ground truth at most once); it is used only to cross-check AP50 against Ultralytics.
* AP is computed as in Ultralytics 8.4 (and COCO): the precision envelope, with precision 0 beyond the
  highest recall reached, is sampled at 101 recall points and integrated with the trapezoid rule. A perfect
  detector therefore scores 0.995.
* R@P (recall at a precision target) is evaluated at a score threshold chosen on validation data: the
  lowest threshold at which validation precision is at least the target (which maximises validation
  recall subject to the target). The threshold is then applied unchanged to the test data. If no
  threshold reaches the target on validation data, the threshold is +inf and test recall is 0.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-16


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N, 4) and b (M, 4), xyxy."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, iou_thr=0.5, rule="score"):
    """Match the predictions of one image to its ground truth.

    Returns (tp, gt_matched): tp[i] is True if prediction i is a true positive; gt_matched[j] is the
    index of the prediction matched to ground truth j, or -1.
    """
    pb = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    ps = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    pc = np.asarray(pred_cls).reshape(-1)
    gb = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    gc = np.asarray(gt_cls).reshape(-1)
    tp = np.zeros(len(pb), dtype=bool)
    gt_matched = -np.ones(len(gb), dtype=int)
    if len(pb) == 0 or len(gb) == 0:
        return tp, gt_matched
    iou = box_iou(pb, gb) * (pc[:, None] == gc[None, :])
    if rule == "score":
        taken = np.zeros(len(gb), dtype=bool)
        for i in np.argsort(-ps, kind="stable"):
            cand = np.where((iou[i] >= iou_thr) & ~taken)[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                taken[j] = True
                tp[i] = True
                gt_matched[j] = i
    elif rule == "iou":
        # Ultralytics DetectionValidator.match_predictions at one threshold, step for step: (gt, pred) pairs
        # above the threshold, sorted by descending IoU, de-duplicated first by prediction and then by ground
        # truth (np.unique re-sorts the pairs between the two steps, so the order of the steps matters).
        iou_gp = iou.T
        matches = np.array(np.nonzero(iou_gp >= iou_thr)).T  # (gt, pred) pairs
        if len(matches):
            if len(matches) > 1:
                matches = matches[iou_gp[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            tp[matches[:, 1]] = True
            gt_matched[matches[:, 0]] = matches[:, 1]
    else:
        raise ValueError(rule)
    return tp, gt_matched


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP from a recall/precision curve ordered by descending score (101-point, trapezoid)."""
    # precision is 0 beyond the highest recall reached (COCO; Ultralytics >= 8.4 uses the same sentinels)
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


class DetectionAccumulator:
    """Collects per-prediction (score, class, tp, image, aid) records and ground-truth counts.

    ``group_of`` optionally maps a detector class to an evaluation label (e.g. a hazard group); classes
    mapped to None are ignored on both sides. With a mapping, predictions of different classes of one
    group are de-duplicated per image by greedy NMS (IoU ``dedup_iou``, highest score kept), so that one
    object detected as both "car" and "truck" is one vehicle detection, not a vehicle and a false positive.
    """

    def __init__(self, iou_thr=0.5, rule="score", group_of=None, dedup_iou=0.7):
        self.iou_thr, self.rule, self.group_of, self.dedup_iou = iou_thr, rule, group_of, dedup_iou
        self.scores, self.labels, self.tps, self.images, self.aids = [], [], [], [], []
        self.n_gt = {}            # label -> count
        self.n_gt_aid = {}        # (aid, label) -> count
        self.n_images = 0

    def _map(self, boxes, scores, cls):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        cls = np.asarray(cls).reshape(-1)
        scores = None if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1)
        if self.group_of is None:
            return boxes, scores, cls
        lab = np.array([self.group_of.get(int(c)) for c in cls], dtype=object)
        keep = np.array([v is not None for v in lab], dtype=bool)
        boxes, lab = boxes[keep], lab[keep]
        scores = None if scores is None else scores[keep]
        return boxes, scores, lab

    def _dedup(self, boxes, scores, lab):
        if self.group_of is None or len(boxes) < 2:
            return boxes, scores, lab
        keep = []
        for g in dict.fromkeys(lab.tolist()):
            idx = np.where(lab == g)[0]
            idx = idx[np.argsort(-scores[idx], kind="stable")]
            iou = box_iou(boxes[idx], boxes[idx])
            alive = np.ones(len(idx), dtype=bool)
            for a in range(len(idx)):
                if alive[a]:
                    keep.append(idx[a])
                    alive[a + 1:] &= iou[a, a + 1:] < self.dedup_iou
        keep = np.array(sorted(keep))
        return boxes[keep], scores[keep], lab[keep]

    def add(self, pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, image_id=None, aid=None):
        pb, ps, pl = self._map(pred_boxes, pred_scores, pred_cls)
        pb, ps, pl = self._dedup(pb, ps, pl)
        gb, _, gl = self._map(gt_boxes, None, gt_cls)
        tp, _ = match_image(pb, ps, pl, gb, gl, self.iou_thr, self.rule)
        img = self.n_images if image_id is None else image_id
        self.n_images += 1
        self.scores.extend(ps.tolist())
        self.labels.extend(pl.tolist())
        self.tps.extend(tp.tolist())
        self.images.extend([img] * len(ps))
        self.aids.extend([aid] * len(ps))
        for g in gl.tolist():
            self.n_gt[g] = self.n_gt.get(g, 0) + 1
            self.n_gt_aid[(aid, g)] = self.n_gt_aid.get((aid, g), 0) + 1

    # ---- curves
    def _arrays(self):
        """Typed arrays of all records, rebuilt only after new records were added (queries stay vectorised)."""
        n = len(self.scores)
        if getattr(self, "_cache_n", -1) != n:
            self._lab_codes = {l: i for i, l in enumerate(dict.fromkeys(self.labels))}
            self._aid_codes = {a: i for i, a in enumerate(dict.fromkeys(self.aids))}
            self._s = np.asarray(self.scores, dtype=np.float64)
            self._t = np.asarray(self.tps, dtype=bool)
            self._l = np.fromiter((self._lab_codes[l] for l in self.labels), dtype=np.int64, count=n)
            self._a = np.fromiter((self._aid_codes[a] for a in self.aids), dtype=np.int64, count=n)
            self._sel = {}
            self._cache_n = n
        return self._s, self._t, self._l, self._a

    def _records(self, label, aid=None):
        s_all, t_all, l_all, a_all = self._arrays()
        key = (label, aid)
        if key not in self._sel:
            lc = self._lab_codes.get(label, -1)
            m = l_all == lc
            if aid is not None:
                m &= a_all == self._aid_codes.get(aid, -1)
            s, t = s_all[m], t_all[m]
            o = np.argsort(-s, kind="stable")
            self._sel[key] = (s[o], t[o])
        return self._sel[key]

    def pr_curve(self, label, aid=None):
        """Scores (descending), recall and precision after each prediction, and the ground-truth count."""
        s, t = self._records(label, aid)
        n = self.n_gt.get(label, 0) if aid is None else self.n_gt_aid.get((aid, label), 0)
        tpc = np.cumsum(t)
        fpc = np.cumsum(~t)
        return s, tpc / (n + EPS), tpc / (tpc + fpc + EPS), n

    def ap(self, label, aid=None):
        """AP at the accumulator's IoU threshold; NaN if the label has no ground truth."""
        s, r, p, n = self.pr_curve(label, aid)
        if n == 0:
            return float("nan")
        if len(s) == 0:
            return 0.0
        return compute_ap(r, p)

    def map(self, labels=None):
        """Mean AP over labels with ground truth (Ultralytics' mAP50 when rule='iou', iou_thr=0.5)."""
        labels = sorted(self.n_gt, key=str) if labels is None else labels
        aps = [self.ap(l) for l in labels if self.n_gt.get(l, 0) > 0]
        return float(np.mean(aps)) if aps else float("nan"), len(aps)

    def recall_at(self, label, threshold, aid=None):
        """(recall, precision, n_gt) of predictions with score >= threshold."""
        s, t = self._records(label, aid)
        n = self.n_gt.get(label, 0) if aid is None else self.n_gt_aid.get((aid, label), 0)
        keep = s >= threshold
        tp = int(t[keep].sum())
        npred = int(keep.sum())
        rec = tp / n if n else float("nan")
        prec = tp / npred if npred else float("nan")
        return rec, prec, n


def threshold_at_precision(acc: DetectionAccumulator, label, target=0.8):
    """Lowest score threshold at which precision >= target on ``acc`` (validation data).

    Precision is evaluated at every distinct score (all predictions with score >= s are kept). Returns
    +inf if the target is never reached or there is no ground truth.
    """
    s, t = acc._records(label)
    n = acc.n_gt.get(label, 0)
    if n == 0 or len(s) == 0:
        return float("inf")
    tpc = np.cumsum(t)
    k = np.arange(1, len(s) + 1)
    prec = tpc / k
    # evaluate only at the last prediction of each distinct score (ties are kept or dropped together)
    last = np.r_[s[1:] != s[:-1], True]
    ok = last & (prec >= target - 1e-12)
    if not ok.any():
        return float("inf")
    return float(s[np.where(ok)[0].max()])


def recall_at_precision(val_acc, test_acc, labels, target=0.8, aids=None):
    """R@P with thresholds chosen on validation data.

    Returns a dict: per label {threshold, val_recall, val_precision, test_recall, test_precision, n_test},
    and, if ``aids`` is given, per (aid, label) test recall at the same threshold.
    """
    out = {"per_label": {}, "per_aid": {}}
    for l in labels:
        thr = threshold_at_precision(val_acc, l, target)
        vr, vp, _ = val_acc.recall_at(l, thr)
        tr, tp_, n = test_acc.recall_at(l, thr)
        out["per_label"][l] = {"threshold": thr, "val_recall": vr, "val_precision": vp,
                               "test_recall": (0.0 if (thr == float("inf") and n) else tr),
                               "test_precision": tp_, "n_test": n}
        if aids is not None:
            for a in aids:
                r, p, na = test_acc.recall_at(l, thr, aid=a)
                if na:
                    out["per_aid"][(a, l)] = {"recall": (0.0 if thr == float("inf") else r), "n": na}
    return out


def mean_and_worst_aid(per_aid: dict, labels, min_instances=1):
    """Per-aid mean recall over the labels evaluated on that aid, their mean over aids, and the worst aid.

    ``per_aid`` maps (aid, label) -> {"recall", "n"}; a label is evaluated on an aid if it has at least
    ``min_instances`` test instances there.
    """
    by_aid = {}
    for (a, l), v in per_aid.items():
        if l in labels and v["n"] >= min_instances:
            by_aid.setdefault(a, []).append(v["recall"])
    aid_means = {a: float(np.mean(v)) for a, v in by_aid.items()}
    if not aid_means:
        return aid_means, float("nan"), float("nan"), None
    worst = min(aid_means, key=lambda a: (aid_means[a], str(a)))
    return aid_means, float(np.mean(list(aid_means.values()))), aid_means[worst], worst
