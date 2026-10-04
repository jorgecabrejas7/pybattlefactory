"""Tables of the planner2 measurements (docs/RL_DECISIONS.md §27.x) from the probe's caches.

    python -m rl.planner2_report [--variants ';n=32,h=1;...'] [--out runs/probe/planner2/report.json]

C1: per variant, the reference planner's Regret_Q of its choices and its agreement with the old planner (same seeds).
C2: per variant, value of the chosen options (P(complete the round), wins left), the paired difference with the old
    planner's choices and with the v3 network's (state by state), validated regret.
A:  per battler (v3, v4), each variant vs the old planner (and vs the network tactician): paired difference of
    P(complete) per round (run i of round k is the same game start for both: same environment seed) with a 95 % normal
    interval, the mean over the rounds, and the difference of P(6 rounds) with a paired bootstrap interval (the runs of
    every round resampled jointly for the two tacticians).
"""
import argparse
import json
import os

import numpy as np

V3 = "runs/alphazero_v3/latest.pt"
V4 = "runs/alphazero_v4/stopped_it1111.pt"
VARIANTS = ["", "n=32", "n=32,h=1", "n=64,h=1", "n=16,h=1,d=2", "n=32,h=1,nol=mc", "p=fs,n=32,h=1"]
A_VARIANTS = ["", "n=32,h=1", "n=32,h=1,nol=mc", "n=16,h=1,d=2", "p=fs,n=32,h=1"]


def paired_crossplay(spec_a, spec_b, battler, runs=512, seed=777, B=4000):
    from . import agent_probe as AP
    sa = AP.summarize_crossplay(spec_a, battler, runs, seed)
    sb = AP.summarize_crossplay(spec_b, battler, runs, seed)
    rounds = sorted(set(sa["per_run"]) & set(sb["per_run"]))
    out = {"rounds": {}, "a": {}, "b": {}}
    rng = np.random.default_rng(0)
    pa_b, pb_b = np.ones(B), np.ones(B)
    pa, pb = 1.0, 1.0
    diffs = []
    for k in rounds:
        ids = sorted(set(sa["per_run"][k]) & set(sb["per_run"][k]))
        a = np.array([sa["per_run"][k][i] for i in ids], np.float64)
        b = np.array([sb["per_run"][k][i] for i in ids], np.float64)
        d = a - b
        out["rounds"][k] = (float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))), len(d))
        out["a"][k], out["b"][k] = float(a.mean()), float(b.mean())
        diffs.append(d)
        pa *= a.mean()
        pb *= b.mean()
        idx = rng.integers(0, len(d), (B, len(d)))
        pa_b *= a[idx].mean(1)
        pb_b *= b[idx].mean(1)
    if diffs:
        m = np.mean([d.mean() for d in diffs])
        se = np.sqrt(np.sum([d.var(ddof=1) / len(d) for d in diffs])) / len(diffs)
        out["mean"] = (float(m), float(1.96 * se))
        db = pa_b - pb_b
        out["p6"] = {"a": pa, "b": pb, "diff": (float(pa - pb), float(np.percentile(db, 2.5)),
                                                 float(np.percentile(db, 97.5)))}
    return out


def has_crossplay(spec, battler, runs=512, seed=777):
    from . import agent_probe as AP
    rows = AP.read_shards(AP.crossplay_path(spec, battler, runs, seed))
    return len({(r["round"], r["i"]) for r in rows}) >= 2 * runs        # (nol=mc: rounds 3 and 6 only)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default=";".join(VARIANTS))
    p.add_argument("--out", default="runs/probe/planner2/report.json")
    a = p.parse_args(argv)
    os.environ.setdefault("PYBATTLE_DIAGNOSTIC_PRIOR", "factory_sets")
    from . import agent_probe as AP
    variants = a.variants.split(";")
    rep = {"c1": {}, "c2": {}, "a": {}}
    old, net = f"planner:{V3}", f"net:{V3}"
    for v in variants:
        spec = f"planner2:{V3}:{v}"
        try:
            rep["c1"][v] = AP.summarize_c1(spec, old)
            # agreement with the old planner's choice (Regret_Q under the old planner is the same thing in wins)
        except Exception as e:                            # (not computed yet)
            rep["c1"][v] = {"error": str(e)}
    table = AP.outcome_table()
    for v in [None] + variants:
        spec = old if v is None else f"planner2:{V3}:{v}"
        try:
            rep["c2"]["old" if v is None else v] = {"vs_old": AP.summarize_c2(spec, table=table, ref_spec=old),
                                                    "vs_net": AP.summarize_c2(spec, table=table, ref_spec=net)}
        except Exception as e:
            rep["c2"]["old" if v is None else v] = {"error": str(e)}
    for name, ck in (("v3", V3), ("v4", V4)):
        bat = f"net:{ck}"
        for v in A_VARIANTS:
            spec = f"planner2:{ck}:{v}"
            if not has_crossplay(spec, bat):
                continue
            r = {}
            for ref_name, ref in (("old", f"planner:{ck}"), ("net", f"net:{ck}"), ("E8", f"planner2:{ck}:"),
                                  ("n32h1", f"planner2:{ck}:n=32,h=1")):
                if ref != spec and has_crossplay(ref, bat):
                    r[ref_name] = paired_crossplay(spec, ref, bat)
            rep["a"].setdefault(name, {})[v] = r
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, default=str)
    print_tables(rep)


def _f(x, pct=True, d=1):
    s = 100 if pct else 1
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return f"{x[0] * s:+.{d}f} ± {x[1] * s:.{d}f}"
    return f"{x * s:.{d}f}"


def print_tables(rep):
    print("\nC1 (Regret_Q under the old planner, wins; agreement with its argmax)")
    for v, r in rep["c1"].items():
        if "error" in r:
            print(f"  {v or 'E8':>18}: {r['error'][:60]}")
            continue
        print(f"  {v or 'E8':>18}: rental {r['rental']['regret'][0]:.3f} agree {r['rental']['agree'][0]:.2f} | "
              f"swap {r['swap']['regret'][0]:.3f} agree {r['swap']['agree'][0]:.2f}")
    print("\nC2 (P(complete) of the chosen option, points; paired diff vs old planner / vs v3 net; cv regret)")
    for v, r in rep["c2"].items():
        if "error" in r:
            print(f"  {v:>18}: {r['error'][:60]}")
            continue
        for kind in ("all", "rental", "swap"):
            o, n = r["vs_old"][kind], r["vs_net"][kind]
            print(f"  {v or 'E8':>18} {kind:>6}: value {o['complete_value'][0] * 100:.1f}  vs old "
                  f"{_f(o['complete_diff_ref'])}  vs net {_f(n['complete_diff_ref'])}  cv {_f(o['complete_cv'])}  "
                  f"wins vs old {_f(o['diff_ref'], False, 3)}")
    print("\nA (paired P(complete) differences per round, points; P(6) difference with bootstrap CI)")
    for bat, vs in rep["a"].items():
        for v, refs in vs.items():
            for ref, r in refs.items():
                if "mean" not in r:
                    continue
                rr = "  ".join(f"R{k} {_f(x)}" for k, x in r["rounds"].items())
                p6 = r["p6"]
                print(f"  [{bat}] {v or 'E8':>16} - {ref:>5}: {rr} | mean {_f(r['mean'])} | P(6) "
                      f"{p6['a'] * 100:.2f} vs {p6['b'] * 100:.2f}: {p6['diff'][0] * 100:+.2f} "
                      f"[{p6['diff'][1] * 100:+.2f}, {p6['diff'][2] * 100:+.2f}]")


if __name__ == "__main__":
    main()
