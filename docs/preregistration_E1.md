# Pre-registration of experiment E1 (and of the protocol shared by E7, E3 time-to-target)

Fixed on 17 October 2026, before any federated run, and committed to the artifact repository before the first run. Changes after this point are allowed only as documented deviations in the phase report (what, why, before or after seeing which result).

## 1. Question

Does FedV2X-Aid, whose aids learn a 152-class hazard detector's head from emulated V2X and range-sensor teachers and aggregate at roadside units under pedestrian mobility, reach higher hazard recall on real street images than (i) each aid learning alone, (ii) self-training federated baselines that use no teacher, and does (iii) the RSU tier cost recall relative to flat federated averaging with the same labels?

## 2. Data, model and evaluation units (from Phase 2)

* Federation: 30 aids (22 Mapillary Vistas, 8 Roboflow) of `results/phase2/federation.json`; per aid about 300 training, 60 validation and 150 test images; atoms never cross cells (`results/phase2/leakage_check.json`: pass).
* Model: official COCO-pretrained YOLO11-s trunk, frozen; 152-class Detect head (BatchNorm folded), round 0 trained on the server pool (`docs/round0_decision.md`). Training reads cached FP16 trunk features (`src/fl/build_cache.py`; equivalence test `results/phase2/cache_equivalence.json`).
* Hazard groups: vehicle, person, curb, crosswalk, stairs, pothole (`src/data/hazard_groups.yaml`). Ground truth: human labels of the aid's test images, only for groups attested by the aid's source. A group is *evaluated on an aid* if its test split has at least 20 instances.
* No human label of any aid image is used for training or for tuning a training run.

## 3. Hypotheses

* **H1** FedV2X-Aid has a higher primary metric than Local-only.
* **H2** FedV2X-Aid has a higher mean R@P80 over the groups that no self-training baseline supervises independently of its own predictions (pothole, curb, stairs: the drop-off groups of the range teacher) than each of the four self-training baselines (HierFAVG + self-training, HierFAVG + mean teacher, FedSTO-style, SemiFL-style).
* **H3** (non-inferiority) FedV2X-Aid's primary metric is not lower than FedAvg's (flat, same labels) by more than 1 point (lower 95 % bootstrap bound of the difference above -0.01).
* **H4** FedV2X-Aid improves the primary metric over the round-0 model (Pretrained).
* Exploratory, no claim: the gap to Centralised; FedProx, SCAFFOLD and FedRep-style relative to FedAvg; per-group differences; mAP50 over hazard classes; worst-aid recall.

## 4. Metrics

* **Primary metric:** mean over the six hazard groups of the pooled test R@P80 (recall at 80 % precision; for each group the score threshold is the lowest threshold at which the pooled *validation* precision of that model reaches 0.80; it is then applied unchanged to the pooled test set; if precision 0.80 is never reached on validation data, test recall is 0). Matching: IoU >= 0.5, COCO score-greedy rule (`src/metrics/detection.py`), classes mapped to groups with group-wise de-duplication at IoU 0.7.
* Secondary: per-group R@P80; mAP50 over the 20 hazard classes (pooled test, Ultralytics-equivalent AP); per-aid mean R@P80 over the groups evaluated on that aid (same group thresholds); worst-aid mean recall; mean recall over the five groups without stairs (stairs has 67 pooled test instances on one aid, so its interval is wide).
* For personalised methods (Local-only, FedRep-style) each aid's own head produces its validation and test predictions, which are pooled in the same way.

## 5. Statistics

* Five seeds (0 to 4) per method. A seed fixes the teacher emulation (which vehicles are connected, which labels are kept), the SUMO contact trace, the client sampling of local batches and the initial random state; the partition and round 0 are the same for all seeds.
* Reported: mean and standard deviation over seeds of every metric.
* Primary comparisons (H1, H2, H3, H4): per-aid mean R@P80 averaged over the five seeds; paired two-sided Wilcoxon signed-rank tests over the 30 aids (H1, H4; H2 over the aids with at least one evaluated drop-off group); Holm correction across the 4 + 4 = 8 hypothesis tests of H1, H2 (four baselines), H4 and the H3 Wilcoxon (H3 is decided by its bootstrap bound; its Wilcoxon p-value is reported). Effect size: median per-aid difference. 95 % intervals: bootstrap over aids (10,000 resamples, seed 0), for the pooled primary metric (resampling aids, recomputing pooled recall at the fixed thresholds) and for per-aid differences.
* A hypothesis is supported if its Holm-adjusted p < 0.05 and the effect has the predicted sign (H3: the bound condition).

## 6. Federation protocol

| Item | Value |
|---|---|
| Aids, RSUs | 30 aids, 6 RSUs; association from SUMO contact traces of the mobility study (`results/phase2/contact_traces.json`, trace = seed) |
| Rounds | 25 cloud rounds, kappa = 2 edge rounds each (50 edge rounds); flat methods 50 rounds with the same local work per round |
| Local work | S = 16 SGD steps of batch 8 per edge round (momentum 0.9, lr 0.01, weight decay 5e-4, cosine-free constant lr) |
| Hierarchy rules | an aid trains on the edge model of the RSU of its first contact and uploads to the RSU of its last contact in the round; no contact: no training and no upload in that round; an RSU aggregates (label-mass weighted, Eq. edgeagg) once at least 3 reports have accumulated since its last aggregation, otherwise it defers; at the end of a cloud round the cloud averages the RSUs that aggregated in that round, weighted by aggregated label mass, and every RSU restarts from the global model |
| Norm bound | FedV2X-Aid bounds each update to Gamma = 90th percentile of the update norms of the first edge round |
| DP | none in E1 (z = 0; DP utility is E6) |
| Secure aggregation | emulated; numerically identical to plain weighted averaging |
| Evaluation | final global head (personalised methods: each aid's head) after the last round; validation curves every cloud round for time-to-target (E3) |

## 7. Teacher emulation (method fixed here; rates from E2b and E8)

Exactly as documented in `src/teachers/emulate.py`: V2X labels on proposals of connected CAM-class vehicles (penetration 0.5) with conditional recall and precision measured on real KITTI images in E2b (Phase 4, labeller C at 5 m GNSS error, per frame); range-sensor regions from fixed footprints with validity rate from a bench log of the range sensors (E8, Phase 5). Proposals: round-0 detections with score >= 0.05 (NMS IoU 0.7). Buffer: the training frames with at least one teacher signal. Label mass of an aid: sum of box-label weights plus 0.5 per range region.

## 8. Baseline protocol matrix (12 methods)

All methods: same frozen trunk and cached features, same partition and splits, same round-0 head as initial weights, same number of rounds and local work per round (unless stated), same optimiser, no tuning on test data, trainable parameters = the Detect head (DFL projection fixed), evaluation on the aids' test splits as in Section 4. "Teacher labels" = the emulated V2X and range labels of Section 7; "self labels" = the model's own detections with score >= 0.5 on the aid's training images (all 300), recomputed every round from the model the aid starts from.

| Method | Initial weights | Data | Label source | Clients / aggregation | Local work | Rounds | Tuning budget | Local objective | Evaluation |
|---|---|---|---|---|---|---|---|---|---|
| Pretrained | round 0 | none | none | none | none | 0 | none | none | global head |
| Local-only | round 0 | own buffer | teacher | each aid alone, no aggregation | S x B per round, every round | 50 | none | Eq. (local), source-scoped | own head |
| FedAvg | round 0 | buffers | teacher | 30 aids, flat, all upload every round (Uu) | S x B | 50 | none | Eq. (local) | global head |
| FedProx | round 0 | buffers | teacher | as FedAvg, proximal mu = 0.01 | S x B | 50 | none (mu from the paper's table) | Eq. (local) + prox | global head |
| SCAFFOLD | round 0 | buffers | teacher | as FedAvg, control variates (option II), server lr 1 | S x B | 50 | none | Eq. (local) with drift correction | global head |
| FedRep-style | round 0 | buffers | teacher | flat; the final 1x1 class convolutions (cv3[l][2]) stay on the aid, the rest is averaged | S x B: first 4 steps personal part, then 12 steps shared part | 50 | none | Eq. (local) | own head (shared + personal) |
| HierFAVG + self-training | round 0 | all 300 training images | self labels (score >= 0.5) | RSU hierarchy as FedV2X-Aid | S x B | 25 x 2 | none | standard detection loss (unlabelled = background) | global head |
| HierFAVG + mean teacher | round 0 | all 300 | labels of an EMA teacher (decay 0.9 per cloud round) at score >= 0.5 | RSU hierarchy | S x B | 25 x 2 | none | standard loss | global head |
| FedSTO-style | round 0 | all 300 | self labels at score >= 0.5 | flat FedAvg | S x B | 50 | none | standard loss + orthogonality penalty 1e-4 on head 1x1 convolutions (FedSTO's orthogonal enhancement; its selective backbone training is impossible with a frozen trunk) | global head |
| SemiFL-style | round 0 | all 300 + 2,000 labelled server images | self labels at score >= 0.5 on aids; merged labels on the server | flat FedAvg; every round the server first trains S x B steps on its labelled images (alternate training) | S x B | 50 | none | standard loss | global head |
| Centralised | round 0 | pooled buffers of all aids | teacher | one model | as many epochs over the pooled buffer as an aid makes over its own buffer, batch 32 | n/a | none | Eq. (local) | global head |
| **FedV2X-Aid** | round 0 | buffers | teacher | RSU hierarchy with SUMO association, label-mass weights, deferral, norm bound | S x B | 25 x 2 | none | Eq. (local) | global head |

## 9. What happens after the run

The results are compared with the expectations in `docs/preregistered_targets.csv` (experiment E1). If FedV2X-Aid fails H1 or H4, the cause is investigated (teacher label quality, loss terms, aggregation); a change is allowed only if it is justified independently of the test data, documented, and followed by a complete re-run of all methods and seeds; both runs are reported.

## 10. Amendment 1 (17 October 2026, 10:30, before any federated training run had finished)

At this time the only E1 result was the Pretrained row of seed 0, which involves no training. Two attempted runs were stopped before finishing, without results, because of the problems in items 1 and 2.

1. **Compute budget.** On the available machine (one 8 GB laptop GPU), one local SGD step of one aid costs 40 ms of GPU time (measured: forward 6.4 ms, teacher forward 7.3 ms, loss 10.1 ms, backward 16.3 ms per aid-step, ten aids stacked). In addition, the self-training baselines label all 9,000 training images in every edge round (18.5 s per round). The protocol above would need about 45 GPU-hours for E1 and E7. The protocol is therefore reduced, equally for every method:
   * local work is **S = 8** SGD steps of batch 8 per edge round, instead of 16. Cloud rounds, kappa, batch, learning rate and every other setting are unchanged. FedRep-style uses 2 personal steps, then 6 shared steps (the same 1:3 split). Centralised keeps "as many epochs as an aid", which now means half as many steps;
   * **three seeds (0, 1, 2)** per method instead of five. "Averaged over the five seeds" in Section 5 becomes "over the three seeds"; tests, corrections and intervals are unchanged.
2. **Evaluation scope, clarified.** "Groups attested by the aid's source" (Section 3) means: each test image is scored only on the hazard classes that the human labels of *its own* source attest. Aids that hold Roboflow images mix several projects, so this differs from using one source per aid. The federation statistics in the paper (for example, stairs evaluated on one aid with 67 test instances) were already counted this way. The first implementation used each aid's majority source; it was corrected before any training result.
3. **Implementation details that do not change the method.** Ten aids are trained together in one stacked head, instead of fifteen, so that GPU memory is not exceeded (each aid keeps its own parameters, optimiser state and normalised loss). The features of the next step are read on a host thread while the GPU works. The self-labels of the self-training baselines use a batched class-aware NMS (IoU 0.7, score >= 0.5). It gives the same boxes as Ultralytics' NMS on 1,146 of 1,152 tested images; the other 6 differ by one box, at the float32 rounding limit. Ultralytics' NMS is used for every evaluation. Validation curves for E3 are computed every 5 cloud rounds, not every cloud round, as fixed for Phase 7 in `docs/preregistration_E7.md`.

## 11. Amendment 2 (18 October 2026, reported after the runs)

Written after all 36 runs had finished, so it is a report of a change, not a pre-registration.

1. **Non-finite steps.** In some runs a local step produced a non-finite loss or gradient in FP16. The simulator now skips such a step, leaves that aid's parameters unchanged and counts the skip (`skipped_steps` in each result file). A run without non-finite steps is unaffected by the guard. Counts in the final results: FedV2X-Aid seed 1: 1; Local only: 219, 57, 68; SCAFFOLD: 557, 802, 712; every other run: 0.
2. **SCAFFOLD.** The scaling of the control-variate correction by the momentum term was corrected before the final SCAFFOLD runs.
3. **Invalid runs.** Runs produced before items 1 and 2 are kept, unchanged, in `results/phase6/E1_invalid/` and are not used.
4. **Effect on the conclusions.** H1 to H4 are decided on the final runs. The affected methods are a baseline that is reported as diverging (SCAFFOLD, two of three seeds) and the local-only baseline. FedV2X-Aid lost one step out of its run.
5. **Code versions.** A result file records the last commit at run time. local_s0 and scaffold_s2 record 2bc1d61, but they log skipped steps, so the guard was in the working tree; it was committed in 4adef3e. scaffold_s2 was re-run after its invalid version (kept in `E1_invalid/`). Whether that re-run already had the corrected momentum scaling cannot be read from the file. Either way, SCAFFOLD diverges in seed 0 (4adef3e, corrected) and in seed 2.
