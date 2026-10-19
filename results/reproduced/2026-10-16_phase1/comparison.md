# Reproduction report

Date: 2026-10-16 21:28
Tolerance: 1e-09 absolute

| Study | Result | Values compared | Max abs diff | Minutes | Archived file |
|---|---|---:|---:|---:|---|
| yolo11_budget | PASS | 34 | 0 | 0.0 | `yolo11_budget.json` |
| dp_budget | PASS | 17 | 0 | 0.0 | `dp_budget.json` |
| dp_userlevel | PASS | 86 | 0 | 2.0 | `dp_userlevel.json` |
| mc_static | PASS | 564 | 0 | 14.9 | `v2x_pseudolabel_mc.json` |
| mc_grid | PASS | 3528 | 0 | 35.5 | `mc_grid.json` |
| mc_height | PASS | 387 | 0 | 3.4 | `mc_height.json` |
| mc_det | PASS | 533 | 0 | 4.0 | `mc_det.json` |
| mc_temporal | FAIL | 1900 | 0 | 18.3 | `mc_temporal.json` |
| mc_temporal_misspec | PASS | 733 | 0 | 5.4 | `mc_temporal_misspec.json` |
| mobility_sumo | PASS | 606 | 0 | 3.0 | `mobility_sumo.json` |
| mobility_modes | PASS | 585 | 0 | 3.7 | `mobility_modes.json` |
| gate_oc | PASS | 807 | 0 | 21.3 | `gate_oc.json` |
| gate_drift | PASS | 155 | 0 | 3.1 | `gate_drift.json` |
| theory_check | PASS | 88 | 0 | 4.3 | `theory_check.json` |
| theory_check_dp | PASS | 79 | 0 | 8.1 | `theory_check_dp.json` |

## Addendum (16 October 2026, after the run)

`mc_temporal` FAIL is a missing-field difference, not a value difference: all 1,900 values of the archived `mc_temporal.json` were reproduced exactly (largest absolute difference 0), but the released `v2x_temporal_mc.py` also writes 748 further fields (identity-level precision, shares of correct labels on unconnected and on other vehicles, same-frame raw-GNSS RMS) that the archived file, written by an earlier version of the script, lacked. The archive in `results/published/` was replaced by this run's complete output. An independent second run with 12 instead of 16 workers (`mc_temporal_second_run_12_workers.json`) reproduces all 2,648 values exactly, so the output does not depend on the worker count. No number printed in the paper changed (`tools/verify_manuscript_numbers.py`: 221 of 221 pass after the replacement).

Total: 15 of 15 studies reproduce every archived value exactly; 10,102 values compared in this run.
