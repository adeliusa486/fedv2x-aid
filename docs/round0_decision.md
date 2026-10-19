# Round-0 model and leakage decision (Phase 2, chunk 2.5; 17 October 2026)

## Options

| | (a) Official COCO-pretrained YOLO11-s trunk, head trained on the server pool | (b) Reference-detector trunk (the reference detector) |
|---|---|---|
| What the trunk has seen | COCO 2017 train (Ultralytics' official weights) | All 235,951 merged-set training images with labels, including the Vistas and Roboflow images that the clients hold |
| Leakage into client test data | None: no client source is in COCO | Severe: about two thirds of the client images (Vistas and Roboflow images of the merged training split) were training images of this trunk *and of its head*, with human labels |
| Room for federated gains | Realistic: the head starts from public data that no aid holds | Almost none, and gains would be confounded with memorisation |
| Accuracy of the starting model | Lower (frozen general-purpose trunk, head trained on public data) | Higher (fully fine-tuned for 152 classes) |

## Decision

**Option (a).** The trunk is Ultralytics' official COCO-pretrained YOLO11-s (`yolo11s.pt`), frozen; the 152-class Detect head of round 0 is trained only on the server pool of `results/phase2/federation.json`: COCO, Open Images and MTSD images, the 13 Roboflow projects that no client uses, and the street-level images that are neither in a client nor in the safety set. Training script: `src/fl/train_round0.py` (60,000 images, 10 epochs, frozen trunk; settings in its docstring). The script checks that the trained model's trunk is bit-identical to the official trunk, because the feature cache (`src/fl/build_cache.py`) is computed with the official trunk.

**Change of route (17 October 2026, before any federated run).** The Ultralytics training of `train_round0.py` ran at 50 to 70 images/s on this machine and its data loader failed twice (shared-event error, then exit code 4 on resume; logs in `results/phase2/train_round0*.log`). Round 0 is therefore trained by `src/fl/train_round0_cache.py` with the federation's own machinery: the same frozen official trunk, FP16 trunk features cached once (bit-identical head outputs, `results/phase2/cache_equivalence.json`), and the same 152-class head and loss, on 24,000 images drawn at random (seed 5) from the 58,000 selected server training images and validated on 1,000 of the 2,000 selected server validation images (seed 6), 10 epochs. The server pool, the leakage argument and every other decision above are unchanged; only the number of round-0 training images is smaller. Training log: `results/phase2/round0_training.json`.

## Reasons

1. Option (b) would let every method start from a model that has been trained, with human labels, on most of the images it is later tested on. Any comparison between methods would be meaningless.
2. Option (a) keeps every client image unseen by both trunk and round-0 head. The only shared property is the source domain: the server pool contains other images of Mapillary Vistas and of the client Roboflow projects. This mirrors deployment, where a vendor trains on public street datasets and aids then see new streets.
3. The reference detector stays the subject of the edge measurements (its compiled model and latency), not of the learning experiment.

## Consequences

* Round-0 accuracy on the clients is lower than the reference detector's validation accuracy; this is expected and is reported as the "Pretrained" row of E1.
* The trunk is frozen for every method, so method differences come from labels and aggregation only.
* Leakage between round-0 training data and client test data is excluded by construction (disjoint cells of the partition) and verified by `tools/leakage_check.py`.
