# Manuscript number check

Manuscript: `FedV2X-Aid_journal.tex`

Each row recomputes printed values from `results/published/*.json` and compares them at the printed precision.

Summary: FAIL 5, PASS 172

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
| tab:mobility | 30 / 6, fixed | FAIL | (printed, computed): [('6.4', 6.346574)] |
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
| fig:mc | (a) P precision | PASS | 5 points |
| fig:mc | (a) PM precision | PASS | 5 points |
| fig:mc | (a) C precision | PASS | 5 points |
| fig:mc | (b) P recall_connected | PASS | 5 points |
| fig:mc | (b) PM recall_connected | PASS | 5 points |
| fig:mc | (b) C recall_connected | PASS | 5 points |
| fig:mc | (c) PM recall_all | PASS | 5 points |
| fig:mc | (c) C recall_all | PASS | 5 points |
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
| text | abstract: C/PM ratio and precision | PASS |  |
| text | abstract: settings | PASS |  |
| text | abstract: CT | PASS |  |
| text | abstract: WAN share | PASS |  |
| text | abstract: head-only factor | PASS |  |
| text | model budget | PASS |  |
| text | C2: precision and ratio | PASS |  |
| text | corollary: floor ratio | PASS |  |
| text | corollary: precisions | PASS |  |
| text | corollary: PM | FAIL | (printed, computed): [('1.7', 1.634639)] |
| text | corollary: head params | PASS |  |
| text | RQ1: P precision | PASS |  |
| text | RQ1: P at 5 m | PASS |  |
| text | RQ1: PM | FAIL | (printed, computed): [('18', 17.493266)] |
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
| text | RQ2: duty cycle | FAIL | (printed, computed): [('0.59', 0.598472)] |
| text | DP: eps z=0.6 | PASS |  |
| text | DP: eps z=1.0 | PASS |  |
| text | DP: eps z=1.5 | PASS |  |
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
| text | theory: DP ratio | FAIL | (printed, computed): [('23', 23.966814)] |
| text | theory: clipping | PASS |  |
