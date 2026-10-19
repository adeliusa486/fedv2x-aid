"""Safety gate over many rounds: cumulative drift, fallback multiplicity and safety-set reuse.

gate_oc.py evaluates one candidate against one released head. This script simulates the whole
release process of Algorithm 3 at instance level:

  * each hazard group has a population of N_POP instances; the first N_SAFE form the server-side
    safety set (fixed and reused every round, as in the paper), the rest stand for the deployment
    distribution and give the TRUE recall;
  * the training head evolves every round regardless of the gate ("training always continues from
    omega^{t+1}"): instances are gained/lost so that the expected recall change of group c is
    Delta_c ~ N(mu_c, s^2) with churn 5 % between consecutive training heads;
  * the gate (Eq. gate) compares the candidate with the current release on the safety set by a
    paired bootstrap (1,000 resamples, lower 95 % bound >= -eps for all six groups);
  * optional fallback (Eq. fallback): omega_alpha adopts the candidate's outcome on each instance
    where candidate and release differ with probability alpha, for alpha = 1/2 then 1/4;
  * optional anchor test: the same bound must also hold against the round-0 release.

Study "drift": 100 rounds, scenarios plateau / one-group decay / learning; reports the true change of
the released head against round 0 (worst group), the probability that some group ends more than eps
below round 0, and the optimism of the safety-set estimate caused by reusing the set.
Study "fallback": one round, a candidate that improves five groups by 1 point and regresses one by
2-4 points; reports the probability that the released head regresses by more than eps.

Usage: python gate_drift.py > results/gate_drift.json
"""
import json
import sys

import numpy as np

GROUPS = 6
N_POP = 10000
N_SAFE = 2000
R0 = 0.60
CHURN = 0.05
EPS = 0.01
B = 1000
ROUNDS = 100


def lcb(rng, d):
    """Lower 95 % paired-bootstrap bound of mean(d), d in {-1, 0, +1}, via multinomial resampling."""
    n = d.size
    cnt = np.array([(d < 0).sum(), (d == 0).sum(), (d > 0).sum()])
    res = rng.multinomial(n, cnt / n, size=B)
    return np.quantile((res[:, 2] - res[:, 0]) / n, 0.05)


def passes(rng, cand, ref):
    return all(lcb(rng, cand[c, :N_SAFE].astype(np.int8) - ref[c, :N_SAFE].astype(np.int8)) >= -EPS for c in range(GROUPS))


def step(rng, head, deltas):
    """Flip instances so that each group's expected recall changes by deltas[c] with churn CHURN."""
    new = head.copy()
    for c in range(GROUPS):
        r = head[c].mean()
        pg = np.clip((CHURN + deltas[c]) / (2 * (1 - r)), 0, 1)
        pl = np.clip((CHURN - deltas[c]) / (2 * r), 0, 1)
        u = rng.random(N_POP)
        new[c] = np.where(head[c], u >= pl, u < pg)
    return new


def gate(rng, cand, rel, anchor, fallback, use_anchor):
    """Return the head that Algorithm 3 releases this round."""
    tries = [cand]
    if fallback:
        diff = cand != rel
        for a in (0.5, 0.25):
            take = diff & (rng.random(cand.shape) < a)
            tries.append(np.where(take, cand, rel))
    for h in tries:
        if passes(rng, h, rel) and (not use_anchor or passes(rng, h, anchor)):
            return h, True
    return rel, False


def true_rec(h):
    return h[:, N_SAFE:].mean(axis=1)


def safe_rec(h):
    return h[:, :N_SAFE].mean(axis=1)


VARIANTS = {"prev": (False, False), "prev+fallback": (True, False), "prev+anchor": (False, True), "prev+anchor+fallback": (True, True)}
SCENARIOS = {
    "plateau": dict(mu=np.zeros(GROUPS), s=0.003),
    "one_group_decay": dict(mu=np.array([0.0005] * 5 + [-0.001]), s=0.003),
    "learning": dict(mu=np.full(GROUPS, 0.001), s=0.003),
}


def drift_rep(args):
    sname, rep, seed = args
    sc = SCENARIOS[sname]
    rng = np.random.default_rng(seed + rep)
    head0 = rng.random((GROUPS, N_POP)) < R0
    # one training trajectory per repetition, shared by all variants (common random numbers)
    traj, h = [], head0
    for _ in range(ROUNDS):
        h = step(rng, h, rng.normal(sc["mu"], sc["s"]))
        traj.append(h)
    res = {}
    for v, (fb, anc) in VARIANTS.items():
        grng = np.random.default_rng(10_000_000 + seed + rep)
        rel, n_rel = head0, 0
        for cand in traj:
            rel, ok = gate(grng, cand, rel, head0, fb, anc)
            n_rel += ok
        d_true = true_rec(rel) - true_rec(head0)
        d_safe = safe_rec(rel) - safe_rec(head0)
        res[v] = (float(d_true.min()), bool(d_true.min() < -EPS), n_rel, float(np.mean(d_safe - d_true)))
    return sname, res


def drift(reps, seed, workers=16):
    from multiprocessing import Pool
    jobs = [(s, rep, seed) for s in SCENARIOS for rep in range(reps)]
    with Pool(workers) as pool:
        done = pool.map(drift_rep, jobs, chunksize=1)
    out = []
    for sname in SCENARIOS:
        rows = {v: dict(worst=[], below=[], releases=[], optimism=[]) for v in VARIANTS}
        for s, res in done:
            if s != sname:
                continue
            for v, (w, b, n, o) in res.items():
                rows[v]["worst"].append(w)
                rows[v]["below"].append(b)
                rows[v]["releases"].append(n)
                rows[v]["optimism"].append(o)
        for v, r in rows.items():
            row = dict(scenario=sname, variant=v, reps=reps,
                       worst_group_change_mean=float(np.mean(r["worst"])), worst_group_change_p05=float(np.quantile(r["worst"], 0.05)),
                       p_some_group_below_eps=float(np.mean(r["below"])), releases_mean=float(np.mean(r["releases"])),
                       safety_set_optimism_mean=float(np.mean(r["optimism"])))
            out.append(row)
            print("drift", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}, file=sys.stderr, flush=True)
    return out


def fallback(reps, seed):
    out = []
    for reg in (-0.02, -0.03, -0.04):
        deltas = np.array([0.01] * 5 + [reg])
        for v in ("prev", "prev+fallback"):
            fb = v.endswith("fallback")
            bad, rel_any = 0, 0
            for rep in range(reps):
                rng = np.random.default_rng(seed + rep)
                head0 = rng.random((GROUPS, N_POP)) < R0
                churn = np.full(GROUPS, CHURN)
                cand = head0.copy()
                for c in range(GROUPS):
                    pg = (churn[c] + deltas[c]) / (2 * (1 - R0))
                    pl = (churn[c] - deltas[c]) / (2 * R0)
                    u = rng.random(N_POP)
                    cand[c] = np.where(head0[c], u >= pl, u < pg)
                rel, ok = gate(rng, cand, head0, head0, fb, False)
                rel_any += ok
                bad += bool(ok and (true_rec(rel) - true_rec(head0)).min() < -EPS)
            row = dict(regression=reg, variant=v, reps=reps, p_release=rel_any / reps, p_release_beyond_tolerance=bad / reps)
            out.append(row)
            print("fallback", row, file=sys.stderr, flush=True)
    return out


def main():
    res = dict(groups=GROUPS, n_pop=N_POP, n_safe=N_SAFE, r0=R0, churn=CHURN, eps=EPS, bootstrap=B, rounds=ROUNDS,
               scenarios={k: dict(mu=v["mu"].tolist(), s=v["s"]) for k, v in SCENARIOS.items()})
    res["fallback"] = fallback(1000, 500)
    res["drift"] = drift(200, 900)
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
