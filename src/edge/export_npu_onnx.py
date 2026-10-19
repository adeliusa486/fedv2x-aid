#!/usr/bin/env python3
"""Phase 3, chunk 3.3 (step 1): export the model-zoo candidates to ONNX (opset 11, static 1x3x640x640, simplified)
for the NPU vendor's dataflow compiler, which then cuts each graph at the head's raw output convolutions
(src/edge/npu_compile.py).

    python src/edge/export_npu_onnx.py --models yolo11n yolo11s yolo26n yolo26s reference
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.edge.efficiency import weights_of  # noqa: E402
from src.edge.zoo import WORK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    from ultralytics import YOLO
    out = WORK / "onnx_npu"
    out.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        dst = out / f"{name}.onnx"
        if dst.exists():
            continue
        src = Path(weights_of(name))
        tmp = out / f"{name}.pt"                  # export next to a copy, so that nothing is written beside the weights
        shutil.copyfile(src, tmp)
        p = YOLO(str(tmp)).export(format="onnx", imgsz=640, opset=11, simplify=True, dynamic=False)
        Path(p).replace(dst)
        tmp.unlink()
        print(name, dst, dst.stat().st_size)


if __name__ == "__main__":
    main()
