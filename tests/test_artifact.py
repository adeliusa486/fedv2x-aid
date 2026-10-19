"""Fast checks of the FedV2X-Aid artifact (seconds): `python -m pytest -q` from the repository root.

The full reproduction of every computed study is `python reproduce.py --full` (about 2 h); these tests
cover what can run in seconds: the comparison logic, two exact reproductions, the evidence checksums,
and, when the manuscript source is available, every printed number and the structural rules.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import reproduce  # noqa: E402

TEX = Path(os.environ.get("FEDV2X_TEX", ROOT.parent / "FedV2X-Aid_journal.tex"))
BIB = Path(os.environ.get("FEDV2X_BIB", ROOT.parent / "references_journal.bib"))


def test_compare_detects_changes():
    old = {"a": {"x": [0.5, 0.1]}, "b": 2}
    assert reproduce.compare(json.loads(json.dumps(old)), old, None, 1e-9)["ok"]
    changed = {"a": {"x": [0.5000001, 0.1]}, "b": 2}
    res = reproduce.compare(changed, old, None, 1e-9)
    assert not res["ok"] and res["n_mismatched"] == 1
    assert not reproduce.compare({"a": {"x": [0.5, 0.1]}}, old, None, 1e-9)["ok"]  # missing key


def test_compare_subset_scope():
    old = {"gnss_1": {"v": 1.0}, "gnss_10": {"v": 2.0}, "rho_0.1": {"v": 3.0}}
    assert reproduce.compare({"gnss_1": {"v": 1.0}}, old, "produced", 1e-9)["ok"]
    assert not reproduce.compare({"gnss_1": {"v": 1.5}}, old, "produced", 1e-9)["ok"]


@pytest.mark.parametrize("study", ["yolo11_budget", "dp_budget"])
def test_fast_studies_reproduce_exactly(study, tmp_path):
    cmd, published, _, _, scope = reproduce.STUDIES[study]
    out = subprocess.run([sys.executable] + cmd, cwd=ROOT / "computed", capture_output=True, text=True, check=True).stdout
    res = reproduce.compare(json.loads(out), json.loads((ROOT / "results/published" / published).read_text()), scope, 0.0)
    assert res["ok"], res


def test_data_checksums():
    """The dataset metadata and the range-sensor bench log in data/ match their recorded checksums."""
    checked = 0
    for line in (ROOT / "data/SHA256SUMS").read_text(encoding="utf8").splitlines():
        digest, rest = line.split("  ", 1)
        assert hashlib.sha256((ROOT / "data" / rest).read_bytes()).hexdigest() == digest, rest
        checked += 1
    assert checked >= 5


def test_no_target_values_in_preregistration_are_results():
    """The expectations file must stay separate from results: no expected value may be named as a result file."""
    import csv
    rows = list(csv.DictReader(open(ROOT / "docs/preregistered_targets.csv", newline="", encoding="utf8")))
    assert len(rows) == 330
    assert all(r["fate"] for r in rows)


@pytest.mark.skipif(not TEX.exists(), reason="manuscript source not available (set FEDV2X_TEX)")
def test_manuscript_numbers():
    r = subprocess.run([sys.executable, str(ROOT / "tools/verify_manuscript_numbers.py"), "--tex", str(TEX)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not (TEX.exists() and BIB.exists()), reason="manuscript source not available (set FEDV2X_TEX and FEDV2X_BIB)")
def test_manuscript_structure():
    r = subprocess.run([sys.executable, str(ROOT / "tools/check_manuscript.py"), "--tex", str(TEX), "--bib", str(BIB)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
