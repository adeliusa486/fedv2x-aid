"""Structural checks of the FedV2X-Aid manuscript before it is shared.

- no orange target (\\est) and no draft watermark is left;
- every pending tag \\pend{ID} names an experiment of docs/experiment_registry.csv;
- every figure and table label is referenced in the text, and every reference has a label;
- every cited key is in the .bib file and the .bib file holds only cited entries;
- every prose paragraph is one physical line (the authors' house rule);
- the abstract is under 250 words (Discover Internet of Things).

Usage: python tools/check_manuscript.py --tex ../FedV2X-Aid_journal.tex --bib ../references_journal.bib
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--bib", required=True)
    args = ap.parse_args()
    tex = Path(args.tex).read_text(encoding="utf8")
    bib = Path(args.bib).read_text(encoding="utf8")
    body = tex[tex.index("\\begin{document}"):]
    problems, notes = [], []

    if "\\est{" in tex or "estcolor" in tex:
        problems.append("orange target markup (\\est or estcolor) still present")
    if "shipout/foreground" in tex or "AddToHook" in tex:
        problems.append("draft watermark hook still present")

    registry = {row["id"] for row in csv.DictReader(open(ROOT / "docs" / "experiment_registry.csv", newline="", encoding="utf8"))}
    pend = re.findall(r"\\pend\{([^}]+)\}", body)
    unknown = sorted(set(pend) - registry)
    if unknown:
        problems.append(f"pending tags without a registry entry: {unknown}")
    counts = {k: pend.count(k) for k in sorted(set(pend))}
    notes.append(f"pending tags: {sum(counts.values())} ({', '.join(f'{k} x{v}' for k, v in counts.items())})")

    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", body))
    for lab in sorted(labels):
        if lab.startswith(("fig:", "tab:")) and lab not in refs:
            problems.append(f"{lab} is never referenced")
    supp = Path(args.tex).with_name(Path(args.tex).stem.replace("_journal", "") + "_supplement.tex")
    if supp.exists():                           # labels of the supplement, referenced through xr
        labels |= set(re.findall(r"\\label\{([^}]+)\}", supp.read_text(encoding="utf8")))
    missing_labels = sorted(refs - labels)
    if missing_labels:
        problems.append(f"references without label: {missing_labels}")

    cited = set()
    for group in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", body):
        cited |= {k.strip() for k in group.split(",")}
    entries = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    if cited - entries:
        problems.append(f"cited but not in the .bib: {sorted(cited - entries)}")
    if entries - cited:
        problems.append(f"in the .bib but never cited: {sorted(entries - cited)}")
    notes.append(f"citations: {len(cited)} keys cited, {len(entries)} .bib entries")

    # One physical line per paragraph: two consecutive prose lines outside verbatim-like environments.
    skip_env = ("tabular", "tikzpicture", "axis", "groupplot", "algorithmic", "equation", "align", "aligned", "itemize", "enumerate", "figure", "sidewaysfigure", "table", "abstract", "IEEEkeywords")
    depth, prev_prose, wraps = 0, False, []
    for no, line in enumerate(body.splitlines(), tex[:tex.index("\\begin{document}")].count("\n") + 1):
        s = line.strip()
        depth += sum(s.count(f"\\begin{{{e}") for e in skip_env) - sum(s.count(f"\\end{{{e}") for e in skip_env)
        prose = depth == 0 and s != "" and not s.startswith(("\\", "%", "$", "&", "}", "{"))
        if prose and prev_prose:
            wraps.append(no)
        prev_prose = prose
    if wraps:
        problems.append(f"hard-wrapped paragraph lines at: {wraps[:15]}")

    if "\\begin{abstract}" in tex:                # IEEEtran / article
        a = tex[tex.index("\\begin{abstract}") + len("\\begin{abstract}"):tex.index("\\end{abstract}")]
    else:                                         # sn-jnl: \abstract{...} on one line
        a = re.search(r"^\\abstract\{(.*)\}\s*$", tex, re.M).group(1)
    for x, y in (("$\\times$", "x"), ("{,}", ","), ("\\,", " "), ("\\%", "%"), ("--", "-")):
        a = a.replace(x, y)
    words = len(re.findall(r"\S+", a))
    notes.append(f"abstract: {words} words")
    if words >= 250:
        problems.append(f"abstract has {words} words (limit: under 250)")

    for n in notes:
        print("info ", n)
    for p in problems:
        print("FAIL ", p)
    print("all structural checks passed" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
