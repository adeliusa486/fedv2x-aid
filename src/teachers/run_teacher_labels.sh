#!/usr/bin/env bash
# Teacher labels of E1 (seeds 0-4) and of the E7 variants A2/A3 (seeds 0-2), emulated with the conditional recall
# and precision measured on KITTI sequences (results/phase4/e2b_tracking_ref_g5_conditional.json, Gauss-Markov
# error at 5 m): CT (E1) 0.669 / 0.841, C (A3) 0.609 / 0.818, PM (A2) 0.235 / 0.733. Penetration 0.5, p_valid 1.0.
set -e
cd "$(dirname "$0")/../.."
PY=.venv-gpu/Scripts/python.exe
for s in 0 1 2 3 4; do $PY src/teachers/make_teacher_labels.py --seed $s --tag E1 --r-cond 0.669 --precision 0.841; done
for s in 0 1 2; do $PY src/teachers/make_teacher_labels.py --seed $s --tag E7_c --r-cond 0.609 --precision 0.818; done
for s in 0 1 2; do $PY src/teachers/make_teacher_labels.py --seed $s --tag E7_pm --r-cond 0.235 --precision 0.733; done
