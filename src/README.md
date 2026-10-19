# Source code of the pending experiments

This folder receives the code of the experiments that are still pending, phase by phase:

| Module (planned) | Experiments | Phase |
|---|---|---|
| `data/` manifests, hazard taxonomy, leakage-free splits | all learning experiments | 2 |
| `metrics/` R@P80, AP50, worst-aid recall, clustered bootstrap, Wilcoxon + Holm (with unit tests) | E1, E7, E2b, E8 | 2 |
| `edge/` model zoo, latency and energy harness, NPU and ONNX Runtime compilation | E4, E5, model choice | 3, 8, 9 |
| `teachers/` V2X consensus on KITTI, range sensor model, label emulation | E2b, E8 | 4, 5 |
| `fl/` feature cache, hierarchical RSU federation driven by SUMO contacts, baselines | E1, E3, E7 | 6, 7 |
| `privacy/` DP-SGD on real heads, gradient inversion, LiRA, poisoning | E6, J4 | 10 |
| `gate/` safety gate on real releases | gate | 11 |
