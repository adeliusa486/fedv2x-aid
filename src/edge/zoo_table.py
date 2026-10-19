#!/usr/bin/env python3
"""Phase 3, chunk 3.4: decision table of the edge model zoo from the measured results only.

Merges results/phase3/zoo_accuracy.json (validation subset), results/phase3/zoo_efficiency.json and the NPU compile
records ($FEDV2X_WORK/zoo/npu/<name>_compile.json; number of contexts parsed from results/phase3/npu_compile_<name>.log)
into results/phase3/zoo_decision.json and prints the table in Markdown.

    python src/edge/zoo_table.py
"""
import os
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NPU = Path(os.environ.get("FEDV2X_WORK", "work") + "/zoo/npu")
ORDER = ["yolo11n", "yolo26n", "yolo11s", "yolo26s", "reference"]
GROUPS = ["vehicle", "person", "curb", "crosswalk", "stairs", "pothole"]


def contexts(name):
    log = ROOT / "results" / "phase3" / f"npu_compile_{name}.log"
    if not log.exists():
        return None
    idx = [int(m) for m in re.findall(rf"{re.escape(name)}_context_(\d+)", log.read_text(encoding="utf-8", errors="replace"))]
    return max(idx) + 1 if idx else None


def main():
    res = ROOT / "results" / "phase3"
    acc = json.loads((res / "zoo_accuracy.json").read_text())["models"] if (res / "zoo_accuracy.json").exists() else {}
    eff = json.loads((res / "zoo_efficiency.json").read_text()) if (res / "zoo_efficiency.json").exists() else {}
    table = {}
    for m in ORDER:
        row = {}
        if m in acc:
            row.update({"map50": acc[m]["map50"], "map50_95": acc[m]["map50_95"], "group_ap50": acc[m]["group_ap50"],
                        "budget_matched": acc[m]["budget_matched"]})
        if m in eff:
            e = eff[m]
            row.update({"params_M": e["params"] / 1e6, "gmacs_640": e["gmacs_640"], "nms_free": e["end2end_nms_free"],
                        "gpu_fp16_ms": e["gpu_fp16_ms"]["median"], "gpu_fp16_p95_ms": e["gpu_fp16_ms"]["p95"],
                        "gpu_nms_ms": e["gpu_nms_ms"]["median"],
                        "gpu_mJ_above_idle": e["gpu_energy_above_idle_mJ_per_frame"]})
            if "cpu_onnxruntime" in e:
                c = e["cpu_onnxruntime"]
                row.update({"cpu_fp32_ms": c["fp32"]["median_ms"], "cpu_int8_ms": c["int8"]["median_ms"],
                            "cpu_int8_p95_ms": c["int8"]["p95_ms"], "int8_MB": c["int8"]["file_MB"]})
        f = NPU / f"{m}_compile.json"
        if f.exists():
            n = json.loads(f.read_text())
            row["npu"] = {"ok": n["ok"], "seconds": n["seconds"], "binary_MB": n.get("hef_bytes", 0) / 1e6,
                          "sha256": n.get("hef_sha256"), "contexts": contexts(m), "error": n.get("error")}
        if row:
            table[m] = row
    (res / "zoo_decision.json").write_text(json.dumps(table, indent=1))
    hdr = "| Model | mAP50 | mAP50-95 | " + " | ".join(f"AP50 {g}" for g in GROUPS) + \
          " | Params (M) | GMACs | GPU FP16 ms | NMS ms | GPU mJ/frame | CPU FP32 ms | CPU INT8 ms | NPU contexts | NPU MB |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))

    def f(x, d=3):
        return "--" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    for m, r in table.items():
        g = r.get("group_ap50", {})
        n = r.get("npu", {})
        print(f"| {m} | {f(r.get('map50'))} | {f(r.get('map50_95'))} | " + " | ".join(f(g.get(k)) for k in GROUPS)
              + f" | {f(r.get('params_M'), 2)} | {f(r.get('gmacs_640'), 1)} | {f(r.get('gpu_fp16_ms'), 2)} | {f(r.get('gpu_nms_ms'), 2)}"
              + f" | {f(r.get('gpu_mJ_above_idle'), 0)} | {f(r.get('cpu_fp32_ms'), 1)} | {f(r.get('cpu_int8_ms'), 1)}"
              + f" | {f(n.get('contexts')) if n.get('ok') else ('fails' if n else '--')} | {f(n.get('binary_MB'), 1) if n.get('ok') else '--'} |")


if __name__ == "__main__":
    main()
