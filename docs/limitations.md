# Limitations

These are the limitations stated in the paper.

- **No accuracy gain.** With the emulated V2X and range labels the federation preserves, but does not raise, the recall of the round-0 head (pre-registered hypothesis H4), and no method detects curbs or crosswalks at 80% precision (H2).
- **Emulated teachers.** Learning results use real street images with emulated V2X and range teachers. There is no field test with vehicles and no study with blind or low-vision users.
- **Synthetic mobility.** The mobility study uses SUMO grid towns.
- **Energy.** On-device power and energy were not measured. Energy is reported for a development laptop only.
- **Privacy.** Client-level DP at a meaningful budget costs most of the recall with 30 aids. Frame-level DP-SGD is accounted but not trained.
- **Data not redistributed.** The merged-set images (88 GB) and the source datasets are not included. They are available from their providers under their own licenses.
- **Device measurements.** The measurements of the authors' device are not part of this repository. The number checker skips them when the measurement file is absent.
