# Results

- `published/`: the archived outputs that the paper reports. `tools/verify_manuscript_numbers.py` checks the paper against these files.
- `reproduced/<run>/`: re-runs made with `reproduce.py`, each with every output, its log, `environment.json` and `comparison.md` (values compared and largest absolute difference per study).
- `logs/`: logs of the original runs.

## Notes on individual files

- `published/theory_check.json` also holds an older `dp` sweep in which the noise-free row was not clipped; the paper uses `published/theory_check_dp.json`, where every row is clipped so that only the noise varies. `reproduce.py` therefore compares only the `eps` and `kappa` sweeps of `theory_check.json`.
- `published/mc_temporal.json` was replaced on 16 October 2026 (Phase 1). The previous file had been written by an earlier version of `v2x_temporal_mc.py`. The Phase 1 reproduction reproduced all 1,900 of its values exactly, but the released script also writes 748 further fields (identity-level precision, the shares of correct labels on unconnected and on other vehicles, and same-frame raw-GNSS RMS for each setting), so the strict comparison reported missing fields. The archive is now the released script's complete output; an independent second run with 12 instead of 16 workers (`reproduced/2026-10-16_phase1/mc_temporal_second_run_12_workers.json`) reproduces all 2,648 values exactly. The previous file is kept in the history of the paper's own repository (`analysis/results/mc_temporal.json`, commit `93f1a70`). No number in the paper changed.
