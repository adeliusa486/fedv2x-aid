"""Mobility-driven RSU association, participation and uplink traffic (E3 system metrics, J2 scale).

SUMO (netgenerate + sumo, pedestrian striping model) simulates walkers on three grid towns
with signalised, crosswalk-equipped intersections. RSUs sit at signalised junctions chosen by
farthest-point sampling. From 1 Hz walker traces we cut 15-min edge-round segments and record,
per segment and RSU, the contact windows within range. A federation of N aids is then
assembled from these segments (80 % of an aid's edge rounds in its home town) and run for
kappa * T = 200 edge rounds:

  * an aid downloads the edge model in its first contact window that can carry the 1.76 MB
    head and uploads in its last such window of the round; with PC5 loss L and ARQ, a
    transfer of b bytes at R bit/s needs 8 b / (R (1 - L)) seconds of contact;
  * an aid without an upload window keeps its update and uploads at its next contact (stale);
  * an RSU aggregates when at least 3 aids report in the edge round, otherwise it defers;
  * each RSU forwards one edge model per cloud round (kappa = 2) if it aggregated at least once.

Flat FL baselines upload over Uu from anywhere, so every aid that finishes its local work
uploads every flat round (two flat rounds per cloud round).

Usage: python mobility_sumo.py --workdir <dir> > mobility.json
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

TOWNS = {"A": dict(grid=5, length=120.0), "B": dict(grid=5, length=150.0), "C": dict(grid=4, length=200.0)}
SIM_SECONDS = 6 * 3600
WARMUP = 300
SEG = 900               # 15-min edge round
HEAD_BYTES = 1_756_432  # FP16 head update (yolo11_budget.json)
EDGE_ROUNDS = 200
KAPPA = 2
MIN_AIDS = 3


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    return r


def build_town(workdir, name, grid, length, n_walkers, seed):
    import sumolib
    net = os.path.join(workdir, f"town{name}.net.xml")
    sh(["netgenerate", "--grid", f"--grid.number={grid}", f"--grid.length={length}", "--default.lanenumber=2",
        "--sidewalks.guess", "true", "--crossings.guess", "true", "--tls.guess", "true", f"--seed={seed}", "-o", net])
    n = sumolib.net.readNet(net)
    ped_edges = [e.getID() for e in n.getEdges() if not e.getID().startswith(":") and any(l.allows("pedestrian") for l in e.getLanes())]
    tls_nodes = [j for j in n.getNodes() if j.getType() == "traffic_light"]
    rng = random.Random(seed)
    rou = os.path.join(workdir, f"town{name}.rou.xml")
    with open(rou, "w") as f:
        f.write("<routes>\n")
        departs = sorted(rng.uniform(0, 60) for _ in range(n_walkers))  # SUMO needs sorted departures
        for p in range(n_walkers):
            cur = rng.choice(ped_edges)
            f.write(f'  <person id="w{p}" depart="{departs[p]:.1f}">\n')
            for k in range(80):
                nxt = rng.choice(ped_edges)
                while nxt == cur:
                    nxt = rng.choice(ped_edges)
                attr = f'from="{cur}" ' if k == 0 else ""
                f.write(f'    <personTrip {attr}to="{nxt}"/>\n')
                cur = nxt
            f.write("  </person>\n")
        f.write("</routes>\n")
    fcd = os.path.join(workdir, f"town{name}.fcd.xml")
    sh(["sumo", "-n", net, "-r", rou, "--end", str(SIM_SECONDS), "--step-length", "1", "--fcd-output", fcd,
        "--pedestrian.model", "striping", "--no-step-log", "true", "--seed", str(seed), "--ignore-route-errors", "true"])
    tls_xy = np.array([j.getCoord() for j in tls_nodes])
    return fcd, tls_xy


def farthest_points(xy, k, seed):
    rng = np.random.default_rng(seed)
    centre = xy.mean(axis=0)
    chosen = [int(np.argmin(np.linalg.norm(xy - centre, axis=1)))]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(xy[:, None, :] - xy[chosen][None, :, :], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return xy[chosen]


def read_traces(fcd):
    traces = {}
    for _, el in ET.iterparse(fcd, events=("end",)):
        if el.tag == "timestep":
            t = float(el.get("time"))
            for p in el.findall("person"):
                traces.setdefault(p.get("id"), []).append((t, float(p.get("x")), float(p.get("y"))))
            el.clear()
    return {k: np.array(v) for k, v in traces.items()}


def segments(traces, rsu_xy, rng_m):
    """Per 15-min segment: list over RSUs of contact windows [(start, end)] in seconds from segment start."""
    segs = []
    for pid, tr in traces.items():
        t0 = WARMUP
        while t0 + SEG <= tr[-1, 0]:
            m = (tr[:, 0] >= t0) & (tr[:, 0] < t0 + SEG)
            part = tr[m]
            if len(part) < 0.95 * SEG:
                t0 += SEG
                continue
            d = np.linalg.norm(part[:, None, 1:] - rsu_xy[None, :, :], axis=2)
            wins = []
            for e in range(len(rsu_xy)):
                inr = d[:, e] <= rng_m
                w, start = [], None
                for i, f in enumerate(inr):
                    if f and start is None:
                        start = part[i, 0] - t0
                    if not f and start is not None:
                        w.append((start, part[i, 0] - t0))
                        start = None
                if start is not None:
                    w.append((start, SEG))
                wins.append(w)
            segs.append(wins)
            t0 += SEG
    return segs


def federation(seg_pool, rsu_per_town, n_aids, loss, rate_bps, fixed, seed):
    """Run 200 edge rounds; seg_pool[town] = list of segments; RSU ids are (town, e)."""
    rng = np.random.default_rng(seed)
    towns = sorted(seg_pool)
    need = 8 * HEAD_BYTES / (rate_bps * (1 - loss))
    home = [towns[i % len(towns)] for i in range(n_aids)]
    home_rsu = [(home[i], int(rng.integers(rsu_per_town))) for i in range(n_aids)]
    pending = [False] * n_aids     # update trained but not yet uploaded
    reporters_hist, deferred, uploads, stale, cross, contact_any = [], 0, 0, 0, 0, 0
    win_lengths = []
    active_rsus_per_cloud = []
    last_down = [None] * n_aids
    agg_this_cloud = set()
    rsu_rounds = 0
    for r in range(EDGE_ROUNDS):
        rep = {}
        for k in range(n_aids):
            town = home[k] if rng.random() < 0.8 else rng.choice([t for t in towns if t != home[k]])
            seg = seg_pool[town][int(rng.integers(len(seg_pool[town])))]
            cands = []
            for e, wins in enumerate(seg):
                rid = (town, e)
                if fixed and rid != home_rsu[k]:
                    continue
                for (a, b) in wins:
                    win_lengths.append(b - a)
                    if b - a >= need:
                        cands.append((a, b, rid))
            if not cands:
                if last_down[k] is not None:
                    pending[k] = True
                continue
            contact_any += 1
            cands.sort()
            down = cands[0][2]
            up = cands[-1][2]
            # upload the update trained in the previous round(s), if any, then the fresh one
            if pending[k]:
                stale += 1
            uploads += 1
            if last_down[k] is not None and last_down[k] != up:
                cross += 1
            rep.setdefault(up, 0)
            rep[up] += 1
            pending[k] = False
            last_down[k] = down
        all_rsus = [(t, e) for t in towns for e in range(rsu_per_town)]
        for rid in all_rsus:
            n = rep.get(rid, 0)
            reporters_hist.append(n)
            rsu_rounds += 1
            if n >= MIN_AIDS:
                agg_this_cloud.add(rid)
            else:
                deferred += 1
        if (r + 1) % KAPPA == 0:
            active_rsus_per_cloud.append(len(agg_this_cloud))
            agg_this_cloud = set()
    reporters_hist = np.array(reporters_hist)
    cloud_rounds = EDGE_ROUNDS // KAPPA
    wan_hier = np.mean(active_rsus_per_cloud) * HEAD_BYTES
    wan_flat = KAPPA * n_aids * HEAD_BYTES
    pc5 = uploads / cloud_rounds * HEAD_BYTES
    wl = np.array(win_lengths) if win_lengths else np.zeros(1)
    return dict(n_aids=n_aids, rsus=len(towns) * rsu_per_town, loss=loss, fixed_association=fixed,
                upload_need_s=need, p_upload_per_aid_round=uploads / (n_aids * EDGE_ROUNDS),
                stale_share=stale / max(uploads, 1), cross_rsu_share=cross / max(uploads, 1),
                reporters_mean=float(reporters_hist.mean()), reporters_median=float(np.median(reporters_hist)),
                deferred_share=deferred / rsu_rounds, active_rsus_per_cloud=float(np.mean(active_rsus_per_cloud)),
                wan_mb_per_cloud_round_hier=wan_hier / 1e6, wan_mb_per_cloud_round_flat=wan_flat / 1e6,
                wan_ratio=wan_hier / wan_flat, all_hop_mb_per_cloud_round_hier=(wan_hier + pc5) / 1e6,
                window_s_median=float(np.median(wl)), window_s_p10=float(np.quantile(wl, 0.1)), window_count=int(len(wl)))


def federation_modes(walker_segs, rsu_per_town, n_aids, mode, duty, fixed, seed, routes=3):
    """Variant of federation() with explicit mobility modes (added to test the
    segment independence and duty cycle). walker_segs[town] = list over walkers of their segments in
    time order. Aids stay in their home town.

      mode "iid":        a random segment of the home town in every active edge round (aids jump);
      mode "continuous": aid k follows one walker's trace in time order, cycling over the 6 h trace;
      mode "habitual":   aid k repeats `routes` fixed segments of its home town (familiar routes);
      duty:              probability that the aid is out walking in an edge round; an aid that stays
                         in neither trains nor uploads, and keeps any pending update.
    """
    rng = np.random.default_rng(seed)
    towns = sorted(walker_segs)
    need = 8 * HEAD_BYTES / (rate_bps_default() * 1.0)
    home = [towns[i % len(towns)] for i in range(n_aids)]
    home_rsu = [(home[i], int(rng.integers(rsu_per_town))) for i in range(n_aids)]
    flat = {t: [s for w in walker_segs[t] for s in w] for t in towns}
    walker = [int(rng.integers(len(walker_segs[home[k]]))) for k in range(n_aids)]
    pos = [int(rng.integers(len(walker_segs[home[k]][walker[k]]))) for k in range(n_aids)]
    my_routes = [[flat[home[k]][int(rng.integers(len(flat[home[k]])))] for _ in range(routes)] for k in range(n_aids)]
    pending = [False] * n_aids
    last_down = [None] * n_aids
    reporters_hist, deferred, uploads, stale, cross, active = [], 0, 0, 0, 0, 0
    active_rsus_per_cloud, agg_this_cloud, rsu_rounds = [], set(), 0
    for r in range(EDGE_ROUNDS):
        rep = {}
        for k in range(n_aids):
            if rng.random() >= duty:
                continue
            active += 1
            town = home[k]
            if mode == "iid":
                seg = flat[town][int(rng.integers(len(flat[town])))]
            elif mode == "continuous":
                segs_w = walker_segs[town][walker[k]]
                seg = segs_w[pos[k] % len(segs_w)]
                pos[k] += 1
            else:
                seg = my_routes[k][int(rng.integers(routes))]
            cands = []
            for e, wins in enumerate(seg):
                rid = (town, e)
                if fixed and rid != home_rsu[k]:
                    continue
                for (a, b) in wins:
                    if b - a >= need:
                        cands.append((a, b, rid))
            if not cands:
                if last_down[k] is not None:
                    pending[k] = True
                continue
            cands.sort()
            down, up = cands[0][2], cands[-1][2]
            if pending[k]:
                stale += 1
            uploads += 1
            if last_down[k] is not None and last_down[k] != up:
                cross += 1
            rep[up] = rep.get(up, 0) + 1
            pending[k] = False
            last_down[k] = down
        for rid in [(t, e) for t in towns for e in range(rsu_per_town)]:
            n = rep.get(rid, 0)
            reporters_hist.append(n)
            rsu_rounds += 1
            if n >= MIN_AIDS:
                agg_this_cloud.add(rid)
            else:
                deferred += 1
        if (r + 1) % KAPPA == 0:
            active_rsus_per_cloud.append(len(agg_this_cloud))
            agg_this_cloud = set()
    wan_hier = np.mean(active_rsus_per_cloud) * HEAD_BYTES
    wan_flat = KAPPA * n_aids * duty * HEAD_BYTES   # flat FL: every active aid uploads over Uu
    return dict(n_aids=n_aids, rsus=len(towns) * rsu_per_town, mode=mode, duty=duty, fixed_association=fixed,
                active_share=active / (n_aids * EDGE_ROUNDS), p_upload_per_aid_round=uploads / (n_aids * EDGE_ROUNDS),
                p_upload_per_active_round=uploads / max(active, 1), stale_share=stale / max(uploads, 1),
                cross_rsu_share=cross / max(uploads, 1), reporters_mean=float(np.mean(reporters_hist)),
                deferred_share=deferred / rsu_rounds, active_rsus_per_cloud=float(np.mean(active_rsus_per_cloud)),
                wan_mb_per_cloud_round_hier=wan_hier / 1e6, wan_mb_per_cloud_round_flat=wan_flat / 1e6,
                wan_ratio=wan_hier / wan_flat)


def rate_bps_default():
    return 10e6


def main_modes(args):
    """Journal revision: mobility modes x duty cycle x association, 150 m RSU range, no PC5 loss."""
    traces, tls = {}, {}
    for i, (name, t) in enumerate(TOWNS.items()):
        fcd, tls_xy = build_town(args.workdir, name, t["grid"], t["length"], args.walkers, 11 + i)
        traces[name] = read_traces(fcd)
        tls[name] = tls_xy
    out = dict(towns=TOWNS, walkers_per_town=args.walkers, sim_hours=SIM_SECONDS / 3600, range_m=150.0, runs=[])
    for rsu_per_town, n_aids in ((2, 30), (4, 120), (8, 300)):
        ws = {}
        for name in TOWNS:
            rsu_xy = farthest_points(tls[name], rsu_per_town, 0)
            ws[name] = [s for s in (segments({pid: tr}, rsu_xy, 150.0) for pid, tr in sorted(traces[name].items())) if s]
        for mode in ("iid", "continuous", "habitual"):
            for duty in (1.0, 0.25):
                for fixed in (False, True):
                    rs = [federation_modes(ws, rsu_per_town, n_aids, mode, duty, fixed, s) for s in range(args.seeds)]
                    agg = {k: (float(np.mean([r[k] for r in rs])) if isinstance(rs[0][k], (int, float)) and not isinstance(rs[0][k], bool) else rs[0][k]) for k in rs[0]}
                    out["runs"].append(agg)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()}, file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--walkers", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--study", choices=["base", "modes"], default="base")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    if args.study == "modes":
        main_modes(args)
        return
    os.makedirs(args.workdir, exist_ok=True)
    traces, tls = {}, {}
    for i, (name, t) in enumerate(TOWNS.items()):
        fcd, tls_xy = build_town(args.workdir, name, t["grid"], t["length"], args.walkers, 11 + i)
        traces[name] = read_traces(fcd)
        tls[name] = tls_xy
        print(f"town {name}: {len(traces[name])} walkers, {len(tls_xy)} signalised junctions", file=sys.stderr, flush=True)
    out = dict(towns=TOWNS, walkers_per_town=args.walkers, sim_hours=SIM_SECONDS / 3600, runs=[])
    for rsu_per_town in (2, 4, 8):
        for rng_m in (100.0, 150.0, 250.0):
            if rsu_per_town != 2 and rng_m != 150.0:
                continue
            pool = {}
            for name in TOWNS:
                rsu_xy = farthest_points(tls[name], rsu_per_town, 0)
                pool[name] = segments(traces[name], rsu_xy, rng_m)
            n_aids_list = {2: (30,), 4: (30, 120), 8: (120, 300)}[rsu_per_town]
            for n_aids in n_aids_list:
                for loss in (0.0, 0.1, 0.3):
                    for fixed in (False, True):
                        if fixed and (loss != 0.0 or rng_m != 150.0):
                            continue
                        rs = [federation(pool, rsu_per_town, n_aids, loss, 10e6, fixed, s) for s in range(args.seeds)]
                        agg = {k: (float(np.mean([r[k] for r in rs])) if isinstance(rs[0][k], (int, float)) and not isinstance(rs[0][k], bool) else rs[0][k]) for k in rs[0]}
                        agg["range_m"] = rng_m
                        agg["segments_per_town"] = {k: len(v) for k, v in pool.items()}
                        out["runs"].append(agg)
                        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items() if k != "segments_per_town"}, file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
