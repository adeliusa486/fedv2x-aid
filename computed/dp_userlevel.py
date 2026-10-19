"""User-level DP for the federation: noise needed for client-level DP-FedAvg at the RSU tier,
and its size relative to the aggregated head update.

Each edge round, an RSU averages the clipped updates (norm S) of the aids it serves and adds
Gaussian noise with std z * S to the sum. Aids participate with probability q per edge round
(Poisson sampling as an upper bound on privacy amplification); kappa * T = 200 edge rounds.
For a target user-level (epsilon, delta), find the noise multiplier z and report the ratio
between the expected noise norm sqrt(d) * z * S / n and the norm S of a fully aligned average.

Usage: python dp_userlevel.py > dp_userlevel.json
"""
import json
import math
import sys

import dp_accounting

D = 878_216          # trainable head parameters (yolo11_budget.json, nc = 152)
ROUNDS = 200         # kappa * T edge rounds
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))


def eps_for(z, q, rounds, delta):
    ev = dp_accounting.SelfComposedDpEvent(dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(z)), rounds)
    acc = dp_accounting.rdp.RdpAccountant(ORDERS)
    acc.compose(ev)
    return acc.get_epsilon(delta)


def z_for(eps_target, q, rounds, delta):
    lo, hi = 0.3, 2000.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if eps_for(mid, q, rounds, delta) > eps_target:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    rows = []
    for n_aids, per_rsu in ((30, 5), (120, 10), (300, 12.5)):
        delta = 1.0 / (10 * n_aids)
        for q in (1.0, 0.5):
            for eps in (2.4, 8.0):
                z = z_for(eps, q, ROUNDS, delta)
                n = per_rsu * q  # expected aids per RSU aggregate
                ratio = math.sqrt(D) * z / n
                rows.append(dict(n_aids=n_aids, aids_per_rsu=per_rsu, q=q, delta=delta, eps=eps, z=round(z, 2), noise_to_signal_norm=round(ratio, 1)))
                print(rows[-1], file=sys.stderr, flush=True)
    json.dump(dict(d=D, rounds=ROUNDS, rows=rows), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
