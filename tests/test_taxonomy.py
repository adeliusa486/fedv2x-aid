"""Validation of the hazard taxonomy (Phase 2, chunk 2.2)."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TAX = yaml.safe_load((ROOT / "src" / "data" / "hazard_groups.yaml").read_text(encoding="utf-8"))
NAMES = (ROOT / "data" / "taxonomy" / "class_names_152.txt").read_text(
    encoding="utf-8").split("\n")
NAMES = [n.strip() for n in NAMES if n.strip()]
PAPER_GROUPS = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]


def test_class_list_is_the_152_class_taxonomy():
    assert len(NAMES) == 152 and len(set(NAMES)) == 152


def test_every_class_in_exactly_one_group_or_excluded():
    seen = []
    for g, spec in TAX["groups"].items():
        assert spec["classes"], g
        assert spec.get("reason"), g
        seen += spec["classes"]
    excluded = [e["class"] for e in TAX.get("excluded") or []]
    for e in TAX.get("excluded") or []:
        assert e.get("reason")
    assert len(seen + excluded) == len(set(seen + excluded)), "a class appears twice"
    assert set(seen + excluded) == set(NAMES), set(NAMES) ^ set(seen + excluded)


def test_paper_groups_present_and_evaluated():
    evaluated = [g for g, s in TAX["groups"].items() if s["evaluated"]]
    assert evaluated == PAPER_GROUPS


@pytest.mark.parametrize("scope", ["v2x_cam", "range_dropoff", "range_obstacle"])
def test_teacher_scopes_use_known_classes(scope):
    assert set(TAX["teachers"][scope]) <= set(NAMES)


def test_v2x_scope_is_vehicles_without_bicycles():
    veh = set(TAX["groups"]["vehicle"]["classes"])
    assert set(TAX["teachers"]["v2x_cam"]) <= veh and "bicycle" not in TAX["teachers"]["v2x_cam"]


def test_dropoff_scope_covers_the_three_dropoff_groups():
    d = set(TAX["teachers"]["range_dropoff"])
    for g in ("curb", "stairs", "pothole"):
        assert set(TAX["groups"][g]["classes"]) <= d


def test_attested_classes_generated():
    p = ROOT / "docs" / "attested_classes.yaml"
    if not p.exists():
        pytest.skip("run src/data/attested_classes.py first")
    att = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert att["generated_by"] == "src/data/attested_classes.py"
    for src, spec in att["sources"].items():
        assert set(spec["classes"]) <= set(NAMES), src
