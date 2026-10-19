"""List every orange target (\\est{...}) of a FedV2X-Aid manuscript version with its experiment, phase and fate.

Used once in Phase 1 to move all illustrative targets out of the manuscript into
docs/preregistered_targets.csv, so that each later result can be compared with the expectation that
was fixed before the experiment ran.

Usage:
    python tools/extract_targets.py --tex <manuscript.tex> --out docs/preregistered_targets.csv
"""
import argparse
import csv
import re
from pathlib import Path

# Where a target sits -> (experiment ID, phase that produces the real value, fate in the manuscript)
BY_LABEL = {
    "tab:main": ("E1/J9 federated hazard recall", 6, "replace with measured result"),
    "fig:mc": ("J3 downstream vehicle recall (panel d)", 6, "replace with measured result"),
    "tab:mobility": ("E3/J2 recall and time to target in the mobility study", 7, "replace with measured result"),
    "fig:tta": ("E3 time to accuracy", 7, "replace with measured result"),
    "fig:compile": ("E4 post-compilation loss", 8, "replace with measured result"),
    "tab:devices": ("E5 device profile", 9, "replace with measured result"),
    "tab:privacy": ("E6 privacy attacks and DP utility", 10, "replace with measured result"),
    "tab:poison": ("J4 poisoned updates", 10, "replace with measured result"),
    "tab:ablation": ("E7 ablation", 7, "replace with measured result"),
    "tab:field": ("J1 field check", None, "deleted: no field data, no approval"),
    "tab:datasets": ("datasets table (Riyadh set, field recordings)", None, "deleted: data do not exist"),
}
BY_SECTION = {
    "sec:intro": ("J1 field-check status sentence", None, "deleted"),
    "sec:rq1": ("E1/J3/J9 hazard recall in text", 6, "replace with measured result"),
    "sec:rq2": ("E3 time to accuracy in text", 7, "replace with measured result"),
    "sec:rq3": ("E4/E5 compilation and device cost in text", 8, "replace with measured result"),
    "sec:rq4": ("E6 privacy attacks in text", 10, "replace with measured result"),
    "sec:rq5": ("J4 poisoning in text", 10, "replace with measured result"),
    "sec:rqgate": ("E7 ablation in text", 7, "replace with measured result"),
    "sec:field": ("J1 field check in text", None, "deleted: no field data, no approval"),
    "sec:discussion": ("E1/E5 values quoted in the discussion", 6, "replace with measured result"),
}
MACROS = {
    "resX": ("E1 gain over local training (abstract)", 6), "resY": ("E1 gap to centralised (abstract)", 6),
    "resZ": ("E3 WAN saving to target", 7), "resW": ("E4 max post-compilation loss", 8),
    "resPSNR": ("E6 gradient-inversion PSNR", 10), "resAUC": ("E6 membership-inference AUC", 10),
    "resDPcost": ("E6 recall cost of DP", 10), "resRiyN": ("Riyadh image count", None),
    "resRiyA": ("Riyadh recall, round 0", None), "resRiyB": ("Riyadh recall, FedV2X-Aid", None),
}


def braced(s, start):
    k, depth = start, 1
    while k < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        k += 1
    return s[start:k - 1], k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tex = Path(args.tex).read_text(encoding="utf8")
    begin_doc = tex.find("\\begin{document}")
    line_of = lambda pos: tex.count("\n", 0, pos) + 1  # noqa: E731

    rows = []
    # Macro definitions in the preamble
    for m in re.finditer(r"\\newcommand\{\\(res\w+)\}\{\\est\{(.*?)\}\}", tex):
        name, val = m.group(1), m.group(2)
        exp, phase = MACROS.get(name, (name, None))
        fate = "deleted: data do not exist" if phase is None else "replace with measured result"
        rows.append(dict(line=line_of(m.start()), where=f"macro \\{name}", experiment=exp, phase=phase or "", fate=fate, value=val, context="preamble definition"))
    # Every \est{...} in the body
    sections = [(m.start(), m.group(1)) for m in re.finditer(r"\\label\{(sec:[^}]+)\}", tex)]
    envs = []
    for m in re.finditer(r"\\begin\{(table\*?|figure\*?)\}(.*?)\\end\{\1\}", tex, re.S):
        lab = re.search(r"\\label\{([^}]+)\}", m.group(2))
        envs.append((m.start(), m.end(), lab.group(1) if lab else "?"))
    for m in re.finditer(r"\\est\{", tex):
        if m.start() < begin_doc:
            continue
        val, end = braced(tex, m.end())
        env = next((e for e in envs if e[0] <= m.start() < e[1]), None)
        sec = max((s for s in sections if s[0] <= m.start()), default=(0, "front"))[1]
        if env and env[2] in BY_LABEL:
            where = env[2]
            exp, phase, fate = BY_LABEL[where]
        elif sec in BY_SECTION:
            where = sec
            exp, phase, fate = BY_SECTION[sec]
        else:
            where = env[2] if env else sec
            exp, phase, fate = ("status or placeholder text", "", "rewritten in Phase 1")
        ctx = re.sub(r"\s+", " ", tex[max(0, m.start() - 70):min(len(tex), end + 50)]).replace("\n", " ")
        rows.append(dict(line=line_of(m.start()), where=where, experiment=exp, phase=phase if phase is not None else "", fate=fate, value=val, context=ctx))
    # Illustrative plot data that are not wrapped in \est (coordinates and curve parameters)
    for label in ("fig:mc", "fig:tta", "fig:compile"):
        e = next((x for x in envs if x[2] == label), None)
        if e is None:
            continue
        body = tex[e[0]:e[1]]
        if label == "fig:mc":
            body = body[body.find("(d) Vehicle R@P80"):]
        for pm in re.finditer(r"\\addplot(\[[^\]]*\])?\s*(coordinates\s*\{.*?\}|\{.*?\})\s*;", body, re.S):
            exp, phase, fate = BY_LABEL[label]
            rows.append(dict(line=line_of(e[0] + body.find(pm.group(0)) + (tex[e[0]:e[1]].find("(d) Vehicle R@P80") if label == "fig:mc" else 0)),
                             where=label, experiment=exp, phase=phase, fate=fate, value=re.sub(r"\s+", " ", pm.group(2))[:160], context="illustrative plot data"))
    rows.sort(key=lambda r: r["line"])
    for i, r in enumerate(rows, 1):
        r["id"] = f"T{i:03d}"
    with open(args.out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "line", "where", "experiment", "phase", "fate", "value", "context"])
        w.writeheader()
        w.writerows(rows)
    n_est = sum(1 for r in rows if r["context"] not in ("preamble definition", "illustrative plot data"))
    print(f"{len(rows)} rows: {n_est} \\est occurrences in the body, "
          f"{sum(1 for r in rows if r['context'] == 'preamble definition')} macro definitions, "
          f"{sum(1 for r in rows if r['context'] == 'illustrative plot data')} illustrative plot series")
    by = {}
    for r in rows:
        by.setdefault((r["experiment"], r["phase"], r["fate"]), 0)
        by[(r["experiment"], r["phase"], r["fate"])] += 1
    for (e, p, f), n in sorted(by.items(), key=lambda x: (str(x[0][1]), x[0][0])):
        print(f"  {n:4d}  phase {p!s:>4}  {e}  -> {f}")


if __name__ == "__main__":
    main()
