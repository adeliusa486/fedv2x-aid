"""V2X labelling edge cases (Phase 4, chunk 4.3): empty frames, unmatched messages, duplicate proposals.

Uses the unchanged labeller functions of computed/v2x_pseudolabel_mc.py with a KITTI-like camera.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "computed"))
import v2x_pseudolabel_mc as mc  # noqa: E402


def camera():
    mc.W, mc.H, mc.F, mc.CX, mc.CY, mc.H_CAM = 1242, 375, 721.5, 609.6, 172.9, 1.65


def msg(x, y, idx=0, cls=0):
    return dict(idx=idx, cls=cls, x=x, y=y, yaw=0.0, l=4.2, w=1.8, h=1.53)


def test_empty_frame_gives_no_pose_and_no_label():
    camera()
    assert mc.consensus_pose([], [], (0.0, 0.0, 0.0), 5.0, math.radians(5)) == (None, 0)
    boxes, keep = mc.project_msgs([], (0.0, 0.0, 0.0))
    assert boxes == [] and keep == [] and mc.match(boxes, [], 0.4) == []


def test_unmatched_message_gives_no_label():
    camera()
    m = [msg(15.0, 0.0)]
    boxes, keep = mc.project_msgs(m, (0.0, 0.0, 0.0))
    assert len(boxes) == 1
    far = [np.array([10.0, 10.0, 40.0, 40.0])]                 # a proposal far from the projected vehicle
    assert mc.match(boxes, far, 0.4) == []
    pose, inl = mc.consensus_pose(m, [dict(box=far[0], score=0.9, cls=0)], (0.0, 0.0, 0.0), 1.2, math.radians(5))
    assert pose is None or inl == 0


def test_duplicate_proposals_give_one_label():
    camera()
    m = [msg(15.0, 0.0)]
    boxes, _ = mc.project_msgs(m, (0.0, 0.0, 0.0))
    b = boxes[0]
    dup = [b.copy(), b + np.array([2.0, 1.0, 2.0, 1.0])]        # two proposals on the same vehicle
    pairs = mc.match(boxes, dup, 0.4)
    assert len(pairs) == 1                                      # one-to-one (Hungarian) matching
    visible = [dict(idx=0, box=b, cls=0)]
    labels = [dict(box=dup[0], cls=0), dict(box=dup[1], cls=0)]
    assert len(mc.evaluate(labels, visible)) == 1               # the second label is not a second correct label


def test_consensus_recovers_shifted_pose():
    camera()
    true = [msg(15.0, 3.0, 0), msg(22.0, -4.0, 1), msg(30.0, 1.0, 2)]
    props = []
    for m in true:
        b, _ = mc.project_msgs([m], (0.0, 0.0, 0.0))
        props.append(dict(box=b[0], score=0.9, cls=0))
    prior = (2.0, -1.5, math.radians(3.0))                      # biased position and heading prior
    pose, inl = mc.consensus_pose(true, props, prior, 3.0, math.radians(5))
    assert pose is not None and inl >= 2
    assert math.hypot(pose[0], pose[1]) < math.hypot(prior[0], prior[1])
