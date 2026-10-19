# Federation design (Phase 2, chunk 2.3; fixed 17 October 2026 before any federated run)

Code: `src/data/build_federation.py` (partition), `tools/leakage_check.py` (leakage audit), `src/data/federation_stats.py` (per-client statistics), `src/data/near_duplicates.py` (dHash). Output: `results/phase2/federation.json`, `docs/federation_clients.csv`, `results/phase2/federation_stats.json`, `results/phase2/leakage_check.json`.

## What the federation stands for

Thirty aids (AI mobility wearables) of different users. Each aid holds the street images it has seen; none of them is labelled by a person. Labels come only from the emulated teachers (V2X messages, range sensors) or from the model itself (self-training baselines). The server holds public data that no aid holds. Human labels of the aids' images are used **only to evaluate**, never to train or to tune a training run.

## Sources

| Role | Sources | Why |
|---|---|---|
| Clients (aids) | Mapillary Vistas (20 clients); Roboflow projects whose human labels cover a hazard group: rf_risk, rf_bump2, rf_obstacle, rf_openhole, rf_aid, rf_esera, rf_starvision (10 clients) | Street-level, pedestrian-like viewpoints with human labels of the hazard groups, so that every client can be evaluated |
| Server pool (round 0) | COCO, Open Images (oiv7, oiv7x), MTSD, the other 13 Roboflow projects, and every street-level image not given to a client or to the safety set | Public data a vendor could legally use |
| Safety set | Leftover street-level images (Vistas, the seven client Roboflow projects), Open Images for stairs | Public data, disjoint from clients and from round-0 training |

MTSD's human labels cover only traffic signs (none of the six hazard groups), so MTSD images could not be scored on a client and go to the server pool. Street-level human labels of stairs are scarce: 209 boxes, all in rf_starvision (`docs/attested_classes.yaml`); stairs therefore has a small client test set and a wide interval, and the safety set takes its stairs from Open Images.

## Units and leakage control

An *atom* is a connected component of images that share a source group key (one original photo: Roboflow exports contain several augmented copies `<photo>_jpg.rf.<hash>`) or are near duplicates (64-bit difference hash, Hamming distance at most 4, over all 263,857 images, including across sources). Atoms are never split: every image of an atom goes to the same cell (one client's train, validation or test split, the safety set, or the server pool). The leakage check re-derives every near-duplicate pair independently and fails if any pair, group or image crosses cells.

Mapillary does not release sequence identifiers with Vistas, so consecutive frames of one Mapillary sequence that are not near duplicates by dHash can fall into different cells. This is a known residual risk; it affects only Vistas.

## Client partition (non-IID)

1. Client pools contain only atoms with at least one human-labelled instance of the six hazard groups (vehicle, person, curb, crosswalk, stairs, pothole; `src/data/hazard_groups.yaml`). The primary group of an atom is its rarest hazard group, by instance frequency in the pool.
2. Each client draws a group mixture q_k ~ Dirichlet(0.5, ..., 0.5) over the groups present in its pool (Hsu et al., 2019) and fills a quota of 510 images by drawing atoms of each primary group in proportion to q_k (if a group runs out, its remaining quota is redistributed in proportion to q_k).
3. Source non-IID: Vistas clients see only Vistas; Roboflow clients see only the seven projects, and their groups are concentrated in single projects (potholes: rf_bump2, rf_risk, rf_obstacle, rf_openhole; crosswalks: rf_risk, rf_aid, rf_esera; stairs: rf_starvision).
4. Each client's atoms are split into 300 / 60 / 150 images (train / validation / test), stratified by primary group.
5. A group is *evaluated* on a client if the client's test split has at least 20 human instances of it. Per-aid statistics use only evaluated groups.

## Labels used where

| Use | Labels |
|---|---|
| Teacher emulation (which vehicles exist, where the drop-offs are) | merged labels (human where the source attests the class, otherwise pseudo-labels of public teacher detectors): the best available description of the scene |
| Client training | Only the teacher labels (or self-training pseudo-labels for those baselines); never human labels |
| Evaluation on clients | Human labels only, and only for groups attested by the client's source |
| Round-0 training (server) | merged labels of the server pool |
| Safety set | Human labels only |

## Seeds

Partition seed 20261017. Learning runs use seeds 0 to 4 (E1) or 0 to 2 (other experiments); the partition is the same for every run and method.
