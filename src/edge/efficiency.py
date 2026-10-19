#!/usr/bin/env python3
"""Phase 3, chunk 3.2: efficiency of the edge model zoo.

For each candidate (and the reference detector):
  * parameters and GMACs at 640 x 640 (Ultralytics' profiler: GFLOPs / 2);
  * GPU latency of the network at batch 1 in FP16 (median, 95th percentile over 500 frames after 50 warm-up
    frames), plus NMS time for detectors that need it, and GPU energy per frame from NVML's energy counter over
    the same 500 frames (total, and above the idle power measured for 5 s before);
  * CPU latency with ONNX Runtime (4 intra-op threads, a proxy for a quad-core edge CPU; measured on the laptop CPU,
    not on an edge device) in FP32 and in static INT8 (QDQ, per-channel, 64 calibration images from the training
    subset), median and 95th percentile over 500 frames after 50 warm-up frames.
Real letterboxed validation images are used as inputs.

    python src/edge/efficiency.py --models yolo11n yolo11s yolo26n yolo26s reference
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.edge.zoo import REFERENCE, WORK  # noqa: E402
from src.fl.feature_cache import letterbox  # noqa: E402


def weights_of(name):
    """Weights of a candidate; the reference detector is copied to the work folder first, because an export writes
    next to the weights file and the dataset folder is read-only."""
    if name == "reference":
        dst = WORK / "reference" / "reference.pt"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REFERENCE, dst)
        return str(dst)
    return str(WORK / "runs" / name / "weights" / "last.pt")


def inputs(n=64):
    paths = [p for p in (WORK / "data/val.txt").read_text().split("\n") if p.strip()][:n]
    return [np.ascontiguousarray(letterbox(cv2.imread(p), 640)[0][:, :, ::-1].transpose(2, 0, 1)) for p in paths]


def gpu_profile(name, frames=500, warm=50):
    import pynvml
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    y = YOLO(weights_of(name))
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    info = (None, get_num_params(y.model), None, get_flops(y.model, 640))  # model_info returns None when not verbose
    model = y.model.cuda().half().eval()
    end2end = getattr(model.model[-1], "end2end", False)
    xs = [torch.from_numpy(x).cuda().half().unsqueeze(0) / 255 for x in inputs()]
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    with torch.no_grad():
        for i in range(warm):
            model(xs[i % len(xs)])
        torch.cuda.synchronize()
        time.sleep(1.0)
        p_idle = np.mean([pynvml.nvmlDeviceGetPowerUsage(h) / 1e3 for _ in range(50) if not time.sleep(0.1)])
        lat, nms = [], []
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        t_start = time.perf_counter()
        for i in range(frames):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(xs[i % len(xs)])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if not end2end:
                non_max_suppression(out[0] if isinstance(out, (list, tuple)) else out, conf_thres=0.25, iou_thres=0.7)
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            lat.append(t1 - t0)
            nms.append(t2 - t1)
        dt = time.perf_counter() - t_start
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    e_frame = (e1 - e0) / 1e3 / frames
    return {"params": int(info[1]), "gmacs_640": float(info[3]) / 2, "end2end_nms_free": bool(end2end),
            "gpu_fp16_ms": {"median": 1e3 * float(np.median(lat)), "p95": 1e3 * float(np.quantile(lat, 0.95))},
            "gpu_nms_ms": {"median": 1e3 * float(np.median(nms)), "p95": 1e3 * float(np.quantile(nms, 0.95))},
            "gpu_energy_mJ_per_frame": 1e3 * e_frame, "gpu_idle_W": float(p_idle),
            "gpu_energy_above_idle_mJ_per_frame": 1e3 * (e_frame - p_idle * dt / frames), "frames": frames}


def cpu_profile(name, frames=500, warm=50, threads=4):
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from ultralytics import YOLO
    out_dir = WORK / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / f"{name}.onnx"
    if not fp32.exists():
        p = YOLO(weights_of(name)).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
        Path(p).replace(fp32)
    int8 = out_dir / f"{name}_int8.onnx"
    xs = [x[None].astype(np.float32) / 255 for x in inputs(64)]
    if not int8.exists():
        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter(xs)
                self.name = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0].name

            def get_next(self):
                x = next(self.it, None)
                return None if x is None else {self.name: x}
        quantize_static(str(fp32), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    res = {}
    for tag, f in (("fp32", fp32), ("int8", int8)):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
        nm = sess.get_inputs()[0].name
        for i in range(warm):
            sess.run(None, {nm: xs[i % len(xs)]})
        lat = []
        for i in range(frames):
            t0 = time.perf_counter()
            sess.run(None, {nm: xs[i % len(xs)]})
            lat.append(time.perf_counter() - t0)
        res[tag] = {"median_ms": 1e3 * float(np.median(lat)), "p95_ms": 1e3 * float(np.quantile(lat, 0.95)),
                    "file_MB": f.stat().st_size / 1e6}
    return {"cpu_threads": threads, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["yolo11n", "yolo11s", "yolo26n", "yolo26s", "reference"])
    ap.add_argument("--skip-cpu", action="store_true")
    a = ap.parse_args()
    dst = ROOT / "results/phase3/zoo_efficiency.json"
    out = json.loads(dst.read_text()) if dst.exists() else {}
    for m in a.models:
        r = gpu_profile(m)
        if not a.skip_cpu:
            r["cpu_onnxruntime"] = cpu_profile(m)
        out[m] = r
        print(m, json.dumps(r), flush=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
