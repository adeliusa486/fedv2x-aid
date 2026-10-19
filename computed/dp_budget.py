"""Frame-level (example-level) DP budget for on-device DP-SGD on the detection head.

Configuration mirrors the paper: per-client hard-example buffer of n frames, Poisson
sampling with expected batch B, S local steps per edge round, kappa edge rounds per
cloud round, T cloud rounds. Worst case: the client participates in every edge round.
"""
import json
import sys

import dp_accounting

N_FRAMES = 2000
BATCH = 8
STEPS_PER_EDGE_ROUND = 50
EDGE_ROUNDS_PER_CLOUD_ROUND = 2
CLOUD_ROUNDS = 100
DELTA = 1e-5
NOISE_MULTIPLIERS = [0.6, 0.8, 1.0, 1.5]

q = BATCH / N_FRAMES
steps = STEPS_PER_EDGE_ROUND * EDGE_ROUNDS_PER_CLOUD_ROUND * CLOUD_ROUNDS
orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))

rows = []
for z in NOISE_MULTIPLIERS:
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(z)), steps
    )
    rdp = dp_accounting.rdp.RdpAccountant(orders)
    rdp.compose(event)
    eps_rdp = rdp.get_epsilon(DELTA)
    pld = dp_accounting.pld.PLDAccountant(value_discretization_interval=1e-3)
    pld.compose(event)
    eps_pld = pld.get_epsilon(DELTA)
    rows.append({"noise_multiplier": z, "eps_rdp": round(eps_rdp, 3), "eps_pld": round(eps_pld, 3)})

json.dump(
    {
        "sampling_rate": q,
        "steps": steps,
        "delta": DELTA,
        "n_frames": N_FRAMES,
        "batch": BATCH,
        "results": rows,
    },
    sys.stdout,
    indent=2,
)
print()
