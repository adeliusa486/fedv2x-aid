"""Re-run the computed studies of the FedV2X-Aid paper and compare them with the archived outputs.

Every computed (non-measured) number in the paper comes from a script in computed/. This runner
re-executes those scripts with the exact arguments used for the paper, stores the new outputs in
results/reproduced/<timestamp>/, and compares every numeric leaf of each new JSON file with the
archived file in results/published/.

Usage (from the repository root, with the venv of requirements-computed.txt active):
    python reproduce.py --list              # show the studies and their expected runtimes
    python reproduce.py --quick             # about 6 min: budget, DP accounting, static Monte-Carlo GNSS sweep
    python reproduce.py --full              # about 2 h on an i9-13900H: every computed study in the paper
    python reproduce.py --only mc_grid gate_oc

Exit status is 0 only if every requested study reproduces its archived output within --tol
(default 1e-9 absolute on every numeric value; the scripts are seeded, so differences are expected
to be exactly zero).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPUTED = ROOT / "computed"
PUBLISHED = ROOT / "results" / "published"
WORKERS = str(max(1, min(16, (os.cpu_count() or 4) - 2)))

# name: (arguments after the script, archived file, minutes measured in the Phase 1 run on an i9-13900H, in quick set,
#        keys compared: None = whole file, or a list of top-level paths that the run produces)
STUDIES = {
    "yolo11_budget": (["yolo11_budget.py"], "yolo11_budget.json", 0.1, True, None),
    "dp_budget": (["dp_budget.py"], "dp_budget.json", 0.1, True, None),
    "dp_userlevel": (["dp_userlevel.py"], "dp_userlevel.json", 2.5, True, None),
    "mc_static_gnss": (["v2x_pseudolabel_mc.py", "--frames", "3000", "--seeds", "3", "--sweep", "gnss"],
                       "v2x_pseudolabel_mc.json", 3.7, True, "produced"),
    "mc_static": (["v2x_pseudolabel_mc.py", "--frames", "3000", "--seeds", "3"], "v2x_pseudolabel_mc.json", 14.9, False, None),
    "mc_grid": (["v2x_mc_extended.py", "--study", "grid", "--workers", WORKERS], "mc_grid.json", 35.5, False, None),
    "mc_height": (["v2x_mc_extended.py", "--study", "height", "--workers", WORKERS], "mc_height.json", 3.4, False, None),
    "mc_det": (["v2x_mc_extended.py", "--study", "det", "--workers", WORKERS], "mc_det.json", 4.0, False, None),
    "mc_temporal": (["v2x_temporal_mc.py", "--study", "all", "--seqs", "15", "--seeds", "3", "--workers", WORKERS],
                    "mc_temporal.json", 18.3, False, None),
    "mc_temporal_misspec": (["v2x_temporal_mc.py", "--study", "misspec", "--seqs", "15", "--seeds", "3", "--workers", WORKERS],
                            "mc_temporal_misspec.json", 5.4, False, None),
    "mobility_sumo": (["mobility_sumo.py", "--workdir", "{workdir}"], "mobility_sumo.json", 3.0, False, None),
    "mobility_modes": (["mobility_sumo.py", "--study", "modes", "--workdir", "{workdir}"], "mobility_modes.json", 3.7, False, None),
    "gate_oc": (["gate_oc.py"], "gate_oc.json", 21.3, False, None),
    "gate_drift": (["gate_drift.py"], "gate_drift.json", 3.1, False, None),
    # theory_check.json also holds an older, superseded dp sweep (see results/README.md); only eps and kappa are compared
    "theory_check": (["theory_check.py", "eps", "kappa"], "theory_check.json", 4.3, False, ["sweeps/eps", "sweeps/kappa"]),
    "theory_check_dp": (["theory_check.py", "dp"], "theory_check_dp.json", 8.1, False, None),
}


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare(new, old, scope, tol):
    """Compare two JSON documents leaf by leaf. scope: None (all keys of both), 'produced' (only keys
    present in the new output, for subset runs) or a list of path prefixes."""
    a, b = dict(flatten(new)), dict(flatten(old))
    if scope is None:
        keys_new, keys_old = set(a), set(b)
    elif scope == "produced":
        tops = {_top(k) for k in a}
        keys_new = set(a)
        keys_old = {k for k in b if _top(k) in tops}
    else:
        def inside(k):
            return any(k == "/" + p or k.startswith("/" + p + "/") or k.startswith("/" + p + "[") for p in scope)
        keys_new = {k for k in a if inside(k)}
        keys_old = {k for k in b if inside(k)}
    missing, extra = sorted(keys_old - keys_new), sorted(keys_new - keys_old)
    max_diff, worst, mismatched = 0.0, None, []
    for k in sorted(keys_new & keys_old):
        x, y = a[k], b[k]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if x != y:
                mismatched.append(k)
            continue
        d = abs(float(x) - float(y))
        if d > max_diff:
            max_diff, worst = d, k
        if d > tol:
            mismatched.append(k)
    ok = not missing and not extra and not mismatched
    return dict(ok=ok, compared=len(keys_new & keys_old), max_abs_diff=max_diff, worst_key=worst,
                missing=missing[:20], extra=extra[:20], mismatched=mismatched[:20], n_mismatched=len(mismatched))


def _top(key):
    """Top-level JSON key of a flattened path such as '/gnss_1/metrics/C/precision[0]' -> 'gnss_1'."""
    return key.split("/")[1].split("[")[0] if key.startswith("/") else key


def environment():
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    try:
        sumo = subprocess.run(["sumo", "--version"], capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
    except Exception:
        sumo = None
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
                numpy=ver("numpy"), scipy=ver("scipy"), dp_accounting=ver("dp_accounting"), sumo=sumo, workers=int(WORKERS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--only", nargs="+", choices=sorted(STUDIES))
    g.add_argument("--list", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", default=None, help="output directory (default results/reproduced/<timestamp>)")
    ap.add_argument("--workdir", default=None, help="scratch directory for SUMO files (default: a temporary directory)")
    args = ap.parse_args()

    if args.list:
        for name, (cmd, pub, minutes, quick, _) in STUDIES.items():
            print(f"{name:22s} {minutes:5.1f} min  {'quick ' if quick else '      '} -> results/published/{pub}   python {' '.join(cmd)}")
        return 0

    names = [n for n, s in STUDIES.items() if s[3]] if args.quick else (args.only or [n for n in STUDIES if n != "mc_static_gnss"])
    out = Path(args.out) if args.out else ROOT / "results" / "reproduced" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    workdir = args.workdir or tempfile.mkdtemp(prefix="fedv2x_sumo_")
    env_info = environment()
    (out / "environment.json").write_text(json.dumps(env_info, indent=1))
    print(f"output: {out}\nstudies: {', '.join(names)}\nexpected time: about {sum(STUDIES[n][2] for n in names):.0f} min", flush=True)

    report = {}
    for name in names:
        cmd, pub, minutes, _, scope = STUDIES[name]
        cmd = [c.replace("{workdir}", os.path.join(workdir, name)) for c in cmd]
        t0 = time.time()
        print(f"[{datetime.now():%H:%M:%S}] {name}: python {' '.join(cmd)}", flush=True)
        with open(out / f"{name}.json", "w") as fo, open(out / f"{name}.log", "w") as fe:
            rc = subprocess.run([sys.executable] + cmd, cwd=COMPUTED, stdout=fo, stderr=fe).returncode
        secs = time.time() - t0
        if rc != 0:
            report[name] = dict(ok=False, error=f"exit code {rc}, see {name}.log", seconds=secs)
            print(f"    FAILED (exit {rc})", flush=True)
            continue
        new = json.loads((out / f"{name}.json").read_text())
        old = json.loads((PUBLISHED / pub).read_text())
        res = compare(new, old, scope, args.tol)
        res.update(seconds=round(secs, 1), published=pub)
        report[name] = res
        print(f"    {'PASS' if res['ok'] else 'FAIL'}  {res['compared']} values compared, max |diff| = {res['max_abs_diff']:.3g}, {secs/60:.1f} min", flush=True)

    (out / "comparison.json").write_text(json.dumps(dict(environment=env_info, tol=args.tol, studies=report), indent=1))
    lines = ["# Reproduction report", "", f"Date: {datetime.now():%Y-%m-%d %H:%M}", f"Tolerance: {args.tol:g} absolute", "",
             "| Study | Result | Values compared | Max abs diff | Minutes | Archived file |", "|---|---|---:|---:|---:|---|"]
    for name, r in report.items():
        if "error" in r:
            lines.append(f"| {name} | ERROR: {r['error']} | | | {r['seconds']/60:.1f} | |")
        else:
            lines.append(f"| {name} | {'PASS' if r['ok'] else 'FAIL'} | {r['compared']} | {r['max_abs_diff']:.3g} | {r['seconds']/60:.1f} | `{r['published']}` |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    ok = all(r.get("ok") for r in report.values())
    print(f"\n{'ALL PASS' if ok else 'SOME STUDIES FAILED'}; report: {out / 'comparison.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
