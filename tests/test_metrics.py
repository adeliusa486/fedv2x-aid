"""Hand-computed fixtures for src/metrics (Phase 2, chunk 2.4). Run: python -m pytest -q tests/test_metrics.py"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.detection import (DetectionAccumulator, box_iou, compute_ap, match_image,  # noqa: E402
                                   mean_and_worst_aid, recall_at_precision, threshold_at_precision)
from src.metrics.stats import (cluster_bootstrap, cluster_bootstrap_pooled, holm,  # noqa: E402
                               paired_cluster_bootstrap, wilcoxon_paired)

A = [0, 0, 10, 10]
B = [20, 20, 30, 30]


def test_box_iou_by_hand():
    # 10x10 boxes overlapping in a 5x10 strip: 50 / (100 + 100 - 50) = 1/3
    assert box_iou([A], [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)
    assert box_iou([A], [B])[0, 0] == 0.0
    assert box_iou(np.zeros((0, 4)), [A]).shape == (0, 1)


def test_duplicates_and_unmatched_labels():
    # two predictions on A (duplicate), none on B (unmatched label)
    tp, gm = match_image([A, A], [0.9, 0.8], [0, 0], [A, B], [0, 0])
    assert tp.tolist() == [True, False]          # the duplicate is a false positive
    assert gm.tolist() == [0, -1]                # B is unmatched


def test_class_must_agree():
    tp, gm = match_image([A], [0.9], [1], [A], [0])
    assert tp.tolist() == [False] and gm.tolist() == [-1]


def test_matching_rules_differ_as_documented():
    # G2 overlaps A. p1 (score 0.9) fits A exactly; p2 (score 0.8) overlaps A (IoU 0.905) and G2 (IoU 0.739).
    g2, p1, p2 = [2, 0, 12, 10], [0, 0, 10, 10], [0.5, 0, 10.5, 10]
    tp_s, gm_s = match_image([p1, p2], [0.9, 0.8], [0, 0], [A, g2], [0, 0], rule="score")
    tp_i, gm_i = match_image([p1, p2], [0.9, 0.8], [0, 0], [A, g2], [0, 0], rule="iou")
    assert tp_s.tolist() == [True, True] and gm_s.tolist() == [0, 1]   # p2 takes the free G2
    # Ultralytics: each prediction keeps only its best-IoU label (A for both); A goes to the lower prediction
    # index, so p2 is a false positive although G2 is free.
    assert tp_i.tolist() == [True, False] and gm_i.tolist() == [0, -1]


def test_empty_images():
    tp, gm = match_image(np.zeros((0, 4)), [], [], np.zeros((0, 4)), [])
    assert len(tp) == 0 and len(gm) == 0
    acc = DetectionAccumulator()
    acc.add([A], [0.7], [0], np.zeros((0, 4)), [])   # a prediction on an image without labels: FP
    acc.add(np.zeros((0, 4)), [], [], np.zeros((0, 4)), [])
    acc.add([A], [0.9], [0], [A], [0])
    s, r, p, n = acc.pr_curve(0)
    assert n == 1 and s.tolist() == [0.9, 0.7]
    assert r.tolist() == pytest.approx([1.0, 1.0]) and p.tolist() == pytest.approx([1.0, 0.5])


def test_perfect_detector_scores_0995():
    # Ultralytics' integration gives 0.995 for a perfect detector
    acc = DetectionAccumulator(rule="iou")
    acc.add([A, B], [0.9, 0.8], [0, 0], [A, B], [0, 0])
    assert acc.ap(0) == pytest.approx(0.995)


def test_ap_by_hand():
    # Sorted predictions: 0.9 TP, 0.8 FP (duplicate), 0.7 FP (background), 0.6 TP; 3 labels (B never found).
    acc = DetectionAccumulator()
    acc.add([A, A, [50, 50, 60, 60]], [0.9, 0.8, 0.7], [0, 0, 0], [A, B], [0, 0])
    acc.add([A], [0.6], [0], [A], [0])
    s, r, p, n = acc.pr_curve(0)
    assert n == 3
    assert r.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 2 / 3])
    assert p.tolist() == pytest.approx([1, 0.5, 1 / 3, 0.5])
    # precision envelope by hand: 1 up to recall 1/3, 0.5 up to the highest recall 2/3, 0 beyond it.
    # On the 101-point grid: 34 points at 1 (0.00..0.33), 33 at 0.5 (0.34..0.66), 34 at 0 (0.67..1.00);
    # trapezoid = 0.01 * (34 + 16.5 - (1 + 0) / 2) = 0.500.
    assert acc.ap(0) == pytest.approx(0.5, abs=1e-12)


def test_absent_class_excluded_from_map():
    acc = DetectionAccumulator()
    acc.add([A, B], [0.9, 0.9], [0, 2], [A], [0])   # class 2 predicted but never labelled
    assert math.isnan(acc.ap(2))
    m, n = acc.map()
    assert n == 1 and m == pytest.approx(0.995)
    # a labelled class without predictions has AP 0 and counts
    acc.add(np.zeros((0, 4)), [], [], [B], [1])
    m, n = acc.map()
    assert n == 2 and m == pytest.approx(0.995 / 2)


def _inject(acc, label, scores, tps, n_gt, aid=None):
    acc.scores += list(scores)
    acc.labels += [label] * len(scores)
    acc.tps += list(tps)
    acc.images += list(range(len(scores)))
    acc.aids += [aid] * len(scores)
    acc.n_gt[label] = acc.n_gt.get(label, 0) + n_gt
    acc.n_gt_aid[(aid, label)] = acc.n_gt_aid.get((aid, label), 0) + n_gt


def test_recall_at_precision_threshold_from_validation():
    val, test = DetectionAccumulator(), DetectionAccumulator()
    # precision after each: 1, 1, .667, .75, .8, .667, .571, .625 -> lowest threshold with P>=.8 is 0.7
    _inject(val, "v", [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4], [1, 1, 0, 1, 1, 0, 0, 1], 6)
    _inject(test, "v", [0.9, 0.75, 0.71, 0.69], [1, 1, 0, 1], 4)
    assert threshold_at_precision(val, "v", 0.8) == 0.7
    out = recall_at_precision(val, test, ["v"], 0.8)["per_label"]["v"]
    assert out["val_recall"] == pytest.approx(4 / 6) and out["val_precision"] == pytest.approx(0.8)
    assert out["test_recall"] == pytest.approx(0.5) and out["test_precision"] == pytest.approx(2 / 3)


def test_recall_at_precision_unreachable_and_ties():
    val, test = DetectionAccumulator(), DetectionAccumulator()
    _inject(val, "c", [0.9, 0.8], [0, 1], 2)            # precision never reaches 0.8
    _inject(test, "c", [0.95], [1], 1)
    assert threshold_at_precision(val, "c", 0.8) == float("inf")
    assert recall_at_precision(val, test, ["c"])["per_label"]["c"]["test_recall"] == 0.0
    tie = DetectionAccumulator()
    _inject(tie, "t", [0.9, 0.5, 0.5], [1, 1, 0], 2)     # tied scores are kept or dropped together
    assert threshold_at_precision(tie, "t", 0.8) == 0.9


def test_group_mapping_deduplicates_classes_of_one_group():
    acc = DetectionAccumulator(group_of={14: "vehicle", 16: "vehicle", 0: "person"})
    acc.add([A, A], [0.9, 0.4], [14, 16], [A], [16])     # car 0.9 and truck 0.4 on one truck
    s, r, p, n = acc.pr_curve("vehicle")
    assert n == 1 and s.tolist() == [0.9] and r[-1] == 1.0 and p[-1] == 1.0
    acc.add([A], [0.8], [99], [A], [99])                 # unmapped classes are ignored on both sides
    assert set(acc.n_gt) == {"vehicle"}


def test_per_aid_mean_and_worst():
    per_aid = {("a1", "g1"): {"recall": 0.8, "n": 10}, ("a1", "g2"): {"recall": 0.4, "n": 10},
               ("a2", "g1"): {"recall": 0.5, "n": 10}, ("a2", "g2"): {"recall": 0.9, "n": 2}}
    means, mean, worst, who = mean_and_worst_aid(per_aid, {"g1", "g2"}, min_instances=5)
    assert means == pytest.approx({"a1": 0.6, "a2": 0.5})
    assert mean == pytest.approx(0.55) and worst == pytest.approx(0.5) and who == "a2"


def test_holm_by_hand():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"] == (pytest.approx(0.03), True)
    assert out["c"][0] == pytest.approx(0.06) and not out["c"][1]
    assert out["b"][0] == pytest.approx(0.06) and not out["b"][1]   # monotone: max(0.06, 0.04)


def test_wilcoxon_exact_small_sample():
    a = {i: float(i) for i in range(1, 9)}
    b = {i: 0.0 for i in range(1, 9)}
    stat, p, med, n = wilcoxon_paired(a, b)
    assert n == 8 and med == 4.5
    assert p == pytest.approx(2 / 2 ** 8)          # all eight differences positive: exact two-sided p
    assert wilcoxon_paired(b, b)[1] == 1.0


def test_cluster_bootstrap():
    est, lo, hi = cluster_bootstrap({k: 0.5 for k in range(30)}, n_boot=200)
    assert est == lo == hi == 0.5
    vals = {k: float(k % 2) for k in range(30)}
    est, lo, hi = cluster_bootstrap(vals, n_boot=2000, seed=1)
    assert est == pytest.approx(0.5) and lo < 0.5 < hi
    assert cluster_bootstrap(vals, n_boot=2000, seed=1) == (est, lo, hi)   # deterministic
    d_est, d_lo, d_hi = paired_cluster_bootstrap({k: v + 0.1 for k, v in vals.items()}, vals, n_boot=500)
    assert d_est == pytest.approx(0.1) and d_lo == pytest.approx(0.1) and d_hi == pytest.approx(0.1)
    p_est, p_lo, p_hi = cluster_bootstrap_pooled({"a": (1, 2), "b": (3, 4)}, n_boot=500)
    assert p_est == pytest.approx(4 / 6) and 0.5 <= p_lo <= p_est <= p_hi <= 0.75
