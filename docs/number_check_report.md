# Manuscript number check

Manuscript: `FedV2X-Aid_journal.tex`

Each row recomputes printed values from `results/published/*.json` and compares them at the printed precision.

Summary: PASS 308

| Where | Item | Result | Detail |
|---|---|---|---|
| tab:det | Default | PASS | 5 values |
| tab:det | Recall x0.9 | PASS | 6 values |
| tab:det | Recall x0.8 | PASS | 6 values |
| tab:det | 0.5 false | PASS | 6 values |
| tab:det | 2.5 false | PASS | 6 values |
| tab:det | 4.0 false | PASS | 6 values |
| tab:det | Box noise 2% | PASS | 6 values |
| tab:det | Box noise 8% | PASS | 6 values |
| tab:det | Class confusion 5% | PASS | 6 values |
| tab:det | Class confusion 25% | PASS | 6 values |
| tab:det | Camera at 1.3 | PASS | 5 values |
| tab:temporal | i.i.d. 3 m | PASS | 9 values |
| tab:temporal | i.i.d. 5 m | PASS | 9 values |
| tab:temporal | i.i.d. 10 m | PASS | 9 values |
| tab:temporal | GM 3 m | PASS | 9 values |
| tab:temporal | GM 5 m | PASS | 9 values |
| tab:temporal | GM 10 m | PASS | 9 values |
| tab:temporal | GM + bias 3 m | PASS | 9 values |
| tab:temporal | GM + bias 5 m | PASS | 9 values |
| tab:temporal | GM + bias 10 m | PASS | 9 values |
| tab:temporal | GM 5 m, rho=0.1 | PASS | 10 values |
| tab:temporal | GM 5 m, rho=0.25 | PASS | 10 values |
| tab:temporal | GM 5 m, rho=1.0 | PASS | 10 values |
| tab:temporal | Position RMSE, GM 5 m | PASS | 4 values |
| tab:temporal | raw GNSS, same frames | PASS | 2 values |
| tab:misspec | As assumed | PASS | 9 values |
| tab:misspec | 3x faster | PASS | 9 values |
| tab:misspec | 4x slower | PASS | 9 values |
| tab:misspec | Under-reported, 8 | PASS | 7 values |
| tab:misspec | Over-reported, 3 | PASS | 7 values |
| tab:misspec | 8 m and 3x faster | PASS | 8 values |
| tab:robust | No latency, no loss | PASS | 4 values |
| tab:robust | Latency, dead reckoning | PASS | 4 values |
| tab:robust | Latency, raw reports | PASS | 4 values |
| tab:robust | 30% loss, dead reckoning | PASS | 5 values |
| tab:robust | 30% loss, raw reports | PASS | 5 values |
| tab:robust | Ghosts, 17% | PASS | 7 values |
| tab:robust | Ghosts, 38% | PASS | 7 values |
| tab:robust | Shifted, 10% | PASS | 7 values |
| tab:robust | Shifted, 30% | PASS | 7 values |
| tab:mobility | 30 / 6, 150 m | PASS | 8 values |
| tab:mobility | 30 / 6, fixed | PASS | 7 values |
| tab:mobility | 30 / 6, 100 m | PASS | 8 values |
| tab:mobility | 30 / 6, 250 m | PASS | 8 values |
| tab:mobility | 30 / 12, 150 m | PASS | 8 values |
| tab:mobility | 120 / 12, 150 m | PASS | 8 values |
| tab:mobility | 300 / 24, 150 m | PASS | 8 values |
| tab:modes | 30 / 6 random 1.0 | PASS | 9 values |
| tab:modes | 30 / 6 continuous 1.0 | PASS | 9 values |
| tab:modes | 30 / 6 habitual 1.0 | PASS | 9 values |
| tab:modes | 30 / 6 habitual 0.25 | PASS | 9 values |
| tab:modes | 120 / 12 habitual 1.0 | PASS | 9 values |
| tab:modes | 120 / 12 habitual 0.25 | PASS | 9 values |
| tab:modes | 300 / 24 habitual 1.0 | PASS | 9 values |
| tab:modes | 300 / 24 habitual 0.25 | PASS | 9 values |
| tab:gatedrift | vs. omega_rel | PASS | 6 values |
| tab:gatedrift | + fallback | PASS | 6 values |
| tab:gatedrift | + anchor | PASS | 6 values |
| tab:gatedrift | + fallback + anchor | PASS | 6 values |
| fig:grid | (a) C 2 deg | PASS | 5 points |
| fig:grid | (a) PM 2 deg | PASS | 5 points |
| fig:grid | (a) C 5 deg | PASS | 5 points |
| fig:grid | (a) PM 5 deg | PASS | 5 points |
| fig:grid | (a) C 10 deg | PASS | 5 points |
| fig:grid | (a) PM 10 deg | PASS | 5 points |
| fig:grid | (b) C 2 deg | PASS | 5 points |
| fig:grid | (b) PM 2 deg | PASS | 5 points |
| fig:grid | (b) C 5 deg | PASS | 5 points |
| fig:grid | (b) PM 5 deg | PASS | 5 points |
| fig:grid | (b) C 10 deg | PASS | 5 points |
| fig:grid | (b) PM 10 deg | PASS | 5 points |
| fig:grid | (c) C rho 0.1 | PASS | 5 points |
| fig:grid | (c) C rho 0.25 | PASS | 5 points |
| fig:grid | (c) C rho 0.5 | PASS | 5 points |
| fig:grid | (c) C rho 0.75 | PASS | 5 points |
| fig:grid | (c) C rho 1.0 | PASS | 5 points |
| fig:gate | +1 pt churn 2% | PASS | 4 points |
| fig:gate | +1 pt churn 5% | PASS | 4 points |
| fig:gate | +1 pt churn 5% indep | PASS | 4 points |
| fig:gate | +1 pt churn 10% | PASS | 4 points |
| fig:gate | no change churn 5% | PASS | 4 points |
| fig:gate | one group -2 pt churn 5% | PASS | 4 points |
| fig:theory | (a) grad norm vs eps | PASS | 5 points |
| fig:theory | (b) d=1005 | PASS | 3 points |
| fig:theory | (b) d=105 | PASS | 3 points |
| text | C4: update share | PASS |  |
| text | discussion: participation | PASS |  |
| text | RQ3: modes participation and deferral | PASS |  |
| text | RQ3: modes cross-RSU | PASS |  |
| text | budget: 80-class parameters | PASS |  |
| text | model budget | PASS |  |
| text | C2: precision and ratio | PASS |  |
| text | corollary: floor ratio | PASS |  |
| text | corollary: precisions | PASS |  |
| text | corollary: PM | PASS |  |
| text | corollary: head params | PASS |  |
| text | RQ1: P precision | PASS |  |
| text | RQ1: P at 5 m | PASS |  |
| text | RQ1: PM | PASS |  |
| text | RQ1: C | PASS |  |
| text | RQ1: unconnected share | PASS |  |
| text | RQ1: identity precision | PASS |  |
| text | RQ1: hard share | PASS |  |
| text | grid: more / twice | PASS |  |
| text | grid: recall exceptions | PASS |  |
| text | grid: precision settings | PASS |  |
| text | grid: precision gap | PASS |  |
| text | grid: compass | PASS |  |
| text | detector: min ratio | PASS |  |
| text | detector: false proposals | PASS |  |
| text | detector: box noise | PASS |  |
| text | detector: hard share | PASS |  |
| text | detector: camera height | PASS |  |
| text | temporal: iid 10 m | PASS |  |
| text | temporal: GM 5 m | PASS |  |
| text | temporal: pose frames | PASS |  |
| text | temporal: RMSE | PASS |  |
| text | temporal: C RMSE | PASS |  |
| text | temporal: bias and 10 m | PASS |  |
| text | temporal: SM | PASS |  |
| text | temporal: low penetration | PASS |  |
| text | misspec: fast | PASS |  |
| text | misspec: worst case | PASS |  |
| text | misspec: over-reported | PASS |  |
| text | latency: raw | PASS |  |
| text | latency: loss raw | PASS |  |
| text | latency: loss DR | PASS |  |
| text | RQ2: update size | PASS |  |
| text | RQ2: participation | PASS |  |
| text | RQ2: cross-RSU | PASS |  |
| text | RQ2: reporters | PASS |  |
| text | RQ2: WAN | PASS |  |
| text | RQ2: all-hop | PASS |  |
| text | RQ2: fixed association | PASS |  |
| text | RQ2: windows | PASS |  |
| text | RQ2: transfer | PASS |  |
| text | RQ2: range | PASS |  |
| text | RQ2: 120/12 | PASS |  |
| text | RQ2: 300/24 | PASS |  |
| text | RQ2: 120/24 | PASS |  |
| text | RQ2: 30/12 | PASS |  |
| text | RQ2: duty cycle | PASS |  |
| text | DP: frame-level eps | PASS |  |
| text | DP: PLD | PASS |  |
| text | DP: user level 30 | PASS |  |
| text | DP: user level 300 | PASS |  |
| text | DP: per direction | PASS |  |
| text | spoof: ghosts | PASS |  |
| text | spoof: shifted | PASS |  |
| text | spoof: wrong labels | PASS |  |
| text | spoof: CT | PASS |  |
| text | gate: regressing 2 pt | PASS |  |
| text | gate: regressing 1 pt | PASS |  |
| text | gate: release probabilities | PASS |  |
| text | gate: independent | PASS |  |
| text | gate: fallback 10% churn | PASS |  |
| text | gate: fallback multiplicity | PASS |  |
| text | gate: drift | PASS |  |
| text | gate: anchor | PASS |  |
| text | gate: releases | PASS |  |
| text | gate: safety-set optimism | PASS |  |
| text | theory: doubling | PASS |  |
| text | theory: kappa | PASS |  |
| text | theory: DP d=1005 | PASS |  |
| text | theory: DP d=105 | PASS |  |
| text | theory: DP ratio | PASS |  |
| text | theory: clipping | PASS |  |
| text | tab:datasets COCO | PASS |  |
| text | tab:datasets Open Images | PASS |  |
| text | tab:datasets Vistas | PASS |  |
| text | tab:datasets MTSD | PASS |  |
| text | tab:datasets Roboflow | PASS |  |
| text | methods: dataset size | PASS |  |
| text | device: RQ4 accuracy | PASS |  |
| text | device: person AP50 | PASS |  |
| text | device: car AP50 | PASS |  |
| text | device: hazard AP50 | PASS |  |
| text | device: binary size | PASS |  |
| text | device: benchmarks | PASS |  |
| text | device: service rate | PASS |  |
| text | device: safety loop and boot | PASS |  |
| text | device: vision outage | PASS |  |
| text | device: Tier 1 FOV | PASS |  |
| text | device: Tier 1 rate | PASS |  |
| text | tab:platform detector | PASS |  |
| text | tab:platform accuracy | PASS |  |
| text | tab:platform AP50 1 | PASS |  |
| text | tab:platform AP50 2 | PASS |  |
| text | tab:platform AP50 3 | PASS |  |
| text | tab:platform binary | PASS |  |
| text | tab:platform benchmark v3 | PASS |  |
| text | tab:platform benchmark v8s | PASS |  |
| text | tab:platform loop | PASS |  |
| text | tab:platform FOV | PASS |  |
| text | tab:platform first reading | PASS |  |
| text | tab:platform interval | PASS |  |
| text | tab:platform range noise | PASS |  |
| text | tab:platform boot | PASS |  |
| text | federation: aids | PASS |  |
| text | federation: splits | PASS |  |
| text | federation: dHash | PASS |  |
| text | federation: evaluated groups | PASS |  |
| text | federation: n_min | PASS |  |
| text | federation: round 0 | PASS |  |
| text | federation: cache size | PASS |  |
| text | federation: cache equality | PASS |  |
| text | federation: safety set | PASS |  |
| text | federation: safety floor | PASS |  |
| text | federation: rounds | PASS |  |
| text | tab:hyper local work | PASS |  |
| text | tab:hyper rounds | PASS |  |
| text | datasets: groups | PASS |  |
| text | datasets: manifest | PASS |  |
| text | reproducibility: metric agreement | PASS |  |
| text | abstract: KITTI ratio | PASS |  |
| text | tab:kitti caption sd | PASS |  |
| text | tab:kitti object true 5m | PASS |  |
| text | tab:kitti object det 0m | PASS |  |
| text | tab:kitti object det 5m | PASS |  |
| text | tab:kitti object det 10m | PASS |  |
| text | tab:kitti object rho .25 | PASS |  |
| text | tab:kitti seq GM 5 | PASS |  |
| text | tab:kitti seq GM 10 | PASS |  |
| text | tab:kitti seq iid | PASS |  |
| text | RQ1 KITTI: geometry | PASS |  |
| text | RQ1 KITTI: end to end | PASS |  |
| text | RQ1 KITTI: proposal share | PASS |  |
| text | RQ1 KITTI: failures | PASS |  |
| text | RQ1 KITTI: CT | PASS |  |
| text | RQ1 KITTI: position | PASS |  |
| text | RQ1 KITTI: iid | PASS |  |
| text | RQ1 KITTI: runtime | PASS |  |
| text | RQ1 98 deg | PASS |  |
| text | discussion: heights | PASS |  |
| text | E8 method: bench log | PASS |  |
| text | tab:rangelabels caption | PASS |  |
| text | tab:rangelabels caption frames | PASS |  |
| text | E8: events | PASS |  |
| text | E8: coverage | PASS |  |
| text | E8: promotion | PASS |  |
| text | E8: suppression | PASS |  |
| text | E8: distilled/protected | PASS |  |
| text | E8: pothole | PASS |  |
| text | tab:rangelabels curb | PASS |  |
| text | tab:rangelabels stairs | PASS |  |
| text | tab:rangelabels pothole | PASS |  |
| text | tab:rangelabels open hole | PASS |  |
| text | tab:rangelabels vehicle | PASS |  |
| text | tab:rangelabels person | PASS |  |
| text | tab:rangelabels curb | PASS |  |
| text | tab:rangelabels crosswalk | PASS |  |
| text | tab:rangelabels stairs | PASS |  |
| text | tab:rangelabels pothole | PASS |  |
| text | tab:rangelabels all | PASS |  |
| text | tab:e1 pretrained | PASS |  |
| text | tab:e1 local | PASS |  |
| text | tab:e1 fedavg | PASS |  |
| text | tab:e1 fedprox | PASS |  |
| text | tab:e1 scaffold | PASS |  |
| text | tab:e1 fedrep | PASS |  |
| text | tab:e1 hier_st | PASS |  |
| text | tab:e1 hier_mt | PASS |  |
| text | tab:e1 fedsto | PASS |  |
| text | tab:e1 semifl | PASS |  |
| text | tab:e1 central | PASS |  |
| text | tab:e1 fedv2x | PASS |  |
| text | abstract: E1 | PASS |  |
| text | abstract: E7 | PASS |  |
| text | C7: E1 | PASS |  |
| text | E1: means | PASS |  |
| text | E1: H1 | PASS |  |
| text | E1: H3 | PASS |  |
| text | E1: SemiFL | PASS |  |
| text | E1: H4 | PASS |  |
| text | E1: pooled steps | PASS |  |
| text | E1: H2 | PASS |  |
| text | discussion: SemiFL gap | PASS |  |
| text | E3: WAN | PASS |  |
| text | E3: target | PASS |  |
| text | tab:e7 A1 | PASS |  |
| text | tab:e7 A2 | PASS |  |
| text | tab:e7 A3 | PASS |  |
| text | tab:e7 A4 | PASS |  |
| text | tab:e7 A5 | PASS |  |
| text | tab:e7 A6 | PASS |  |
| text | tab:e7 A7 | PASS |  |
| text | tab:e7 A8 | PASS |  |
| text | tab:e7 I1 | PASS |  |
| text | tab:e7 I2 | PASS |  |
| text | E7: A2 | PASS |  |
| text | E7: participation | PASS |  |
| text | E8: scope worth | PASS |  |
| text | RQ4: laptop cost | PASS |  |
| text | tab:devices params | PASS |  |
| text | tab:devices gpu | PASS |  |
| text | tab:devices energy | PASS |  |
| text | tab:devices cpu | PASS |  |
| text | tab:devices cpu int8 | PASS |  |
| text | tab:devices size | PASS |  |
| text | tab:devices map | PASS |  |
| text | E6: dims | PASS |  |
| text | E6: trade-off | PASS |  |
| text | tab:privacy full 0.5 | PASS |  |
| text | tab:privacy full 1.0 | PASS |  |
| text | tab:privacy cls 1.0 | PASS |  |
| text | tab:privacy cls 2.0 | PASS |  |
| text | tab:privacy cls 4.0 | PASS |  |
| text | J4: results | PASS |  |
