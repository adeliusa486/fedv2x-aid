#!/usr/bin/env python3
"""LaTeX rows of the E1 results table, generated from results/phase6/E1_summary.json (src/fl/analyze_e1.py) so that
no number is typed by hand; the number checker reads the same file.

    python src/fl/e1_table.py            # prints the table body
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]
NAMES = {"pretrained": "Pretrained (round 0)", "local": "Local only", "fedavg": "FedAvg", "fedprox": "FedProx",
         "scaffold": "SCAFFOLD", "fedrep": "FedRep-style", "hier_st": "HierFAVG + self-training",
         "hier_mt": "HierFAVG + mean teacher", "fedsto": "FedSTO-style", "semifl": "SemiFL-style",
         "central": "Pooled (same labels)", "fedv2x": "FedV2X-Aid"}
ORDER = list(NAMES)


def rows(summary):
    out = []
    for m in ORDER:
        if m not in summary["methods"]:
            continue
        v = summary["methods"][m]
        cells = [f"{v['per_group'][g][0]:.3f}" for g in GROUPS]
        cells.append(f"{v['primary'][0]:.3f}$\\pm${v['primary'][1]:.3f}")
        cells += [f"{v['map50'][0]:.3f}", f"{v['worst_aid'][0]:.3f}"]
        name = NAMES[m] if m != "fedv2x" else "\\textbf{" + NAMES[m] + "}"
        out.append(name + " & " + " & ".join(cells) + " \\\\")
    return out


def main():
    f = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/phase6/E1_summary.json"
    print("\n".join(rows(json.loads(f.read_text()))))


if __name__ == "__main__":
    main()
