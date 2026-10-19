"""Teacher emulation on 2-D street images (method fixed in Phase 2; rates measured in Phases 4 and 5).

None of the street images comes with V2X messages or range readings, so both teachers are emulated from the
objects that exist in the image (the merged labels, "world" boxes) and from the real proposals of the
round-0 detector on that image (score >= tau_low = 0.05 after NMS).

V2X teacher (CAM/BSM senders only: src/data/hazard_groups.yaml, teachers.v2x_cam)
  1. Each world vehicle of a CAM class is connected with probability ``penetration`` (default 0.5).
  2. A connected vehicle can only be labelled if a proposal (any class) overlaps it with IoU >= 0.5, because the
     consensus labeller keeps the detector's box and takes only the class from the message.
  3. Given such a proposal, the vehicle is labelled with probability ``r_cond`` on that proposal box with the
     vehicle's class; ``r_cond`` is the conditional recall of the labeller measured on real images (E2b, Phase 4).
  4. Wrong labels: for each correct label, with probability (1 - p)/p a wrong label is added on a proposal that
     overlaps no world vehicle (IoU < 0.5), with the class of a connected vehicle of the frame; p is the label
     precision measured in E2b. If no such proposal exists, no wrong label is added (reported).
  Label weight q = 1 (the consensus quality weight cannot be emulated without geometry; documented).

Range teacher (two sensors of a wearable: forward protrusion, downward drop-off)
  * Footprints are fixed regions of the 640 x 640 letterboxed content: downward sensor
    x in [0.35, 0.65] W, y in [0.80, 1.00] H; forward sensor x in [0.40, 0.60] W, y in [0.35, 0.75] H.
  * Drop-off event: a world box of a drop-off class (curb, stairs, pothole, open hole) covers at least
    ``min_cover`` (10 %) of the downward footprint. Protrusion event: a world box of an obstacle class covers at
    least 10 % of the forward footprint and is near (box height >= 0.3 H, a proxy for range below d_max = 3 m).
  * A sensor reading is valid with probability ``p_valid`` (from a bench log of the range sensors, E8); an
    invalid reading produces no event.
  * An event is a *miss* (a training signal) if no detection of the current model with score >= 0.25 and an
    admissible class covers at least half of the region. Only misses are emitted, as regions R with the
    admissible class set S_R, plus, if a low-confidence proposal (0.05 <= score < 0.25) overlaps R, a promoted
    box label on that proposal with class argmax_{c in S_R} p_c and weight 0.5.
"""
from __future__ import annotations

import numpy as np

FOOT_DOWN = (0.35, 0.80, 0.65, 1.00)
FOOT_FWD = (0.40, 0.35, 0.60, 0.75)


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    aa = (a[:, 2:] - a[:, :2]).clip(0).prod(-1)
    bb = (b[:, 2:] - b[:, :2]).clip(0).prod(-1)
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def cover(boxes, region):
    """Share of ``region`` covered by each box."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    r = np.asarray(region, float)
    tl = np.maximum(boxes[:, :2], r[:2])
    br = np.minimum(boxes[:, 2:], r[2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    return inter / max((r[2] - r[0]) * (r[3] - r[1]), 1e-9)


def content_region(frac, gain, pad, size):
    """Footprint in letterbox pixels: fractions of the image content (not of the padding)."""
    w0, h0 = size
    W, H = w0 * gain, h0 * gain
    return np.array([pad[0] + frac[0] * W, pad[1] + frac[1] * H, pad[0] + frac[2] * W, pad[1] + frac[3] * H])


def v2x_labels(world_b, world_c, prop_b, cam_classes, rng, penetration=0.5, r_cond=1.0, precision=1.0):
    """Returns (boxes, classes, stats). world_*: objects in the frame; prop_b: proposal boxes (score >= 0.05)."""
    veh = [i for i, c in enumerate(world_c) if int(c) in cam_classes]
    connected = [i for i in veh if rng.random() < penetration]
    boxes, cls = [], []
    st = {"vehicles": len(veh), "connected": len(connected), "with_proposal": 0, "correct": 0, "wrong": 0,
          "wrong_wanted": 0}
    if len(prop_b) == 0 or not connected:
        return np.zeros((0, 4), np.float32), np.zeros(0, int), st
    iou = iou_matrix(np.asarray(world_b)[connected], prop_b)
    used = set()
    for row, i in enumerate(connected):
        j = int(np.argmax(iou[row]))
        if iou[row, j] < 0.5:
            continue
        st["with_proposal"] += 1
        if j in used or rng.random() >= r_cond:
            continue
        used.add(j)
        boxes.append(prop_b[j])
        cls.append(int(world_c[i]))
        st["correct"] += 1
    if st["correct"] and precision < 1.0:
        free = [j for j in range(len(prop_b))
                if j not in used and (len(veh) == 0 or iou_matrix(prop_b[j:j + 1], np.asarray(world_b)[veh]).max() < 0.5)]
        p_wrong = (1 - precision) / precision
        for _ in range(st["correct"]):
            if rng.random() < p_wrong:
                st["wrong_wanted"] += 1
                if free:
                    j = free.pop(int(rng.integers(len(free))))
                    boxes.append(prop_b[j])
                    cls.append(int(world_c[connected[int(rng.integers(len(connected)))]]))
                    st["wrong"] += 1
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(cls, int), st


def range_events(world_b, world_c, det_b, det_c, det_s, prop_probs, gain, pad, size, dropoff, obstacle, rng,
                 p_valid=1.0, min_cover=0.10, near_h=0.30, deploy_thr=0.25, tau_low=0.05):
    """Range-teacher regions and promoted boxes for one frame.

    det_b/det_c/det_s: detections of the current model (score >= tau_low) with classes and scores;
    prop_probs: (N, len(dropoff) + len(obstacle)) probabilities of the admissible classes at each detection.
    Returns (regions [(box, set_name)], promoted [(box, class, weight)], stats).
    """
    world_b = np.asarray(world_b, float).reshape(-1, 4)
    world_c = np.asarray(world_c).reshape(-1)
    H = size[1] * gain
    regions, promoted = [], []
    st = {"events": 0, "invalid": 0, "misses": 0, "promoted": 0}
    for name, frac, classes in (("dropoff", FOOT_DOWN, dropoff), ("obstacle", FOOT_FWD, obstacle)):
        R = content_region(frac, gain, pad, size)
        m = np.isin(world_c, list(classes))
        if not m.any():
            continue
        cv = cover(world_b[m], R)
        ok = cv >= min_cover
        if name == "obstacle":
            ok &= (world_b[m][:, 3] - world_b[m][:, 1]) >= near_h * H
        if not ok.any():
            continue
        st["events"] += 1
        if rng.random() >= p_valid:
            st["invalid"] += 1
            continue
        dm = np.isin(det_c, list(classes)) & (det_s >= deploy_thr)
        if dm.any() and (cover(det_b[dm], R) >= 0.5).any():
            continue                                         # the deployed model already sees it: no miss
        st["misses"] += 1
        regions.append((R.astype(np.float32), name))
        low = (det_s >= tau_low) & (det_s < deploy_thr)
        if low.any():
            ov = cover(det_b[low], R) > 0
            if ov.any():
                cand = np.where(low)[0][ov]
                k = cand[np.argmax(det_s[cand])]
                off = 0 if name == "dropoff" else len(dropoff)
                probs = prop_probs[k, off:off + len(classes)]
                promoted.append((det_b[k].astype(np.float32), int(list(classes)[int(np.argmax(probs))]), 0.5))
                st["promoted"] += 1
    return regions, promoted, st
