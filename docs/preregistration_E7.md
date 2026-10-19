# Pre-registration of experiment E7 (ablations) and of the Phase 7 participation study

Fixed on 17 October 2026 before any E7 run. Protocol, data, metrics and statistics are those of `docs/preregistration_E1.md`; seeds 0, 1, 2 (three seeds, as stated in the paper for experiments other than E1). Every run differs from FedV2X-Aid in exactly the listed factor(s).

## How each mechanism is removed

| ID | Mechanism removed | Implementation |
|---|---|---|
| A1 | V2X labels | `--ablate no_v2x`: V2X boxes dropped from every frame; range labels kept |
| A2 | consensus pose correction | V2X labels emulated with the rates of matching without correction (PM) measured on KITTI (E2b, end to end, 5 m) instead of the temporal-consensus rates |
| A3 | temporal consensus | V2X labels emulated with the per-frame consensus (C) rates measured on KITTI instead of the temporal-consensus (CT) rates |
| A4 | range labels | `--ablate no_range`: range regions and promoted boxes dropped |
| A5 | source-scoped masks | `--ablate no_scope`: standard detection loss (every unlabelled anchor-class pair is background, no distillation, no set loss) |
| A6 | quality (label-mass) weights | `--ablate no_quality`: RSU and cloud weights by the number of buffered frames |
| A7 | mobility-driven association | `--ablate fixed_rsu`: every aid always uploads to one home RSU (deferral and staleness follow from the SUMO contacts of the home RSU) |
| A8 | norm bounding | `--ablate no_normbound` |
| A9 | the RSU tier | the FedAvg row of E1 (flat aggregation, same labels and local work) |
| -- | federated calibration, safety gate | not part of the FP16 head evaluation; measured in Phases 8 and 11 |

## Pre-specified interactions

* **I1 = A2 x A5**: noisier V2X labels (PM rates) combined with the standard loss. Question: does the source-scoped loss protect more when the V2X labels are noisier?
* **I2 = A4 x A5**: no range labels combined with the standard loss. Question: is the benefit of scoping larger when the frames carry fewer label sources?
The interaction effect is (FedV2X-Aid - A) - (B - AB) in the primary metric, with a 95 % bootstrap interval over aids (seed-averaged per-aid recalls).

## Statistics

Each ablation is compared with FedV2X-Aid (seeds 0 to 2) by the paired Wilcoxon test over aids with Holm correction across A1 to A8 and the two interactions (10 tests), and by the bootstrap interval of the difference in the primary metric. A mechanism counts as supported when removing it lowers the primary metric with a Holm-adjusted p < 0.05; an increase is reported as such.

## Participation (Phase 7.3)

FedV2X-Aid with each SUMO contact kept with probability 0.5 and 0.25 (`--participation`), seeds 0 to 2, against full participation (E1 seeds 0 to 2): primary metric, time and bytes to the target, deferral share.

## Communication and time to target (Phase 7.2)

From the E1 logs: wide-area bytes, total uplink, downloads per run; validation curves every 5 cloud rounds; the target is 90 % of the final validation mean R@P80 of FedAvg (fixed here, before the runs), and the time and bytes to reach it are interpolated between curve points.
