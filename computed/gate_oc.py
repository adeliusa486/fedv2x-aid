"""Operating characteristic of the safety gate (Eq. gate).

The gate releases a candidate head only if, for every hazard group c and target h, the
paired-bootstrap lower 95 % confidence bound of its recall change on the safety set is at
least -eps. For paired detection outcomes the only quantities that matter per test are the
true recall change delta and the churn pi, the fraction of safety instances on which the
candidate and the released head disagree: an instance is gained with probability
(pi + delta) / 2 and lost with probability (pi - delta) / 2.

A bootstrap of the mean of d_i in {-1, 0, +1} is exactly a multinomial resample of the three
category counts. Targets are either identical (the compiled binaries agree on every instance,
6 tests) or independent (18 tests); real targets lie between these bounds.

The fallback of Eq. (fallback) interpolates towards the released head; to first order this
scales both delta and churn by alpha, which is evaluated as a separate study.

Usage: python gate_oc.py > gate_oc.json
"""
import json
import sys

import numpy as np

GROUPS = 6
B = 1000
REPS = 2000


def p_pass_one(rng, n, delta, churn, eps, reps):
    """Probability that one paired-bootstrap non-inferiority test passes."""
    pp, pm = (churn + delta) / 2, (churn - delta) / 2
    assert pm >= 0 and pp >= 0
    cnt = rng.multinomial(n, [pm, 1 - pp - pm, pp], size=reps)       # observed safety-set outcomes
    ok = np.empty(reps, dtype=bool)
    for i in range(reps):
        p = cnt[i] / n
        res = rng.multinomial(n, p, size=B)
        ok[i] = np.quantile((res[:, 2] - res[:, 0]) / n, 0.05) >= -eps
    return ok


def p_release(rng, n, deltas, churn, eps, independent_targets, reps=REPS):
    """Probability that all tests pass. deltas: per-group true recall change."""
    tests = []
    for c in range(GROUPS):
        k = 3 if independent_targets else 1
        for _ in range(k):
            tests.append(p_pass_one(rng, n, deltas[c], churn, eps, reps))
    return float(np.mean(np.all(np.vstack(tests), axis=0)))


def main():
    rng = np.random.default_rng(42)
    out = dict(bootstrap=B, reps=REPS, groups=GROUPS, uniform=[], regression=[], fallback=[])
    for eps in (0.01, 0.02):
        for churn in (0.02, 0.05, 0.10):
            for n in (500, 1000, 2000, 4000):
                for delta in (0.0, 0.005, 0.01, 0.02):
                    if delta > churn:
                        continue
                    row = dict(eps=eps, churn=churn, n=n, delta=delta)
                    for ind in (False, True):
                        row["p_release_" + ("indep" if ind else "shared")] = p_release(rng, n, np.full(GROUPS, delta), churn, eps, ind, reps=500)
                    out["uniform"].append(row)
                    print("uniform", row, file=sys.stderr, flush=True)
    # one group regresses, the other five improve by 1 point; churn 5 %
    for eps in (0.01, 0.02):
        for n in (500, 1000, 2000, 4000):
            for reg in (-0.01, -0.02, -0.03, -0.05):
                deltas = np.full(GROUPS, 0.01)
                deltas[-1] = reg
                churn = max(0.05, abs(reg))
                row = dict(eps=eps, churn=churn, n=n, delta_reg=reg)
                for ind in (False, True):
                    row["p_release_" + ("indep" if ind else "shared")] = p_release(rng, n, deltas, churn, eps, ind, reps=500)
                out["regression"].append(row)
                print("regression", row, file=sys.stderr, flush=True)
    # fallback: candidate with delta = +1 point and churn 10 %, interpolated by alpha
    for n in (1000, 2000):
        for alpha in (1.0, 0.5, 0.25):
            row = dict(eps=0.01, n=n, alpha=alpha, delta=0.01 * alpha, churn=0.10 * alpha)
            row["p_release_shared"] = p_release(rng, n, np.full(GROUPS, 0.01 * alpha), 0.10 * alpha, 0.01, False, reps=500)
            out["fallback"].append(row)
            print("fallback", row, file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
