"""Cost and sampling noise of the planner variants (rl/planner2.py, docs §27.x) on the probe bench's C1 states.

    python -m rl.planner2_noise --ckpt runs/alphazero_v3/latest.pt --rentals 40 --swaps 60 --seeds 4

Per variant (planner2 option strings; "old" = rl.team_eval's planner, the Factory's list, diagnostic only):
    ms          time per decision (1 thread), rental / swap
    sd          the standard deviation over seeds of the centered Q_plan (Q - mean over the options), averaged over
                options and states: the noise that moves the argmax
    agree       P(two seeds pick the same option)
    regret_ref  Q_ref(best) - Q_ref(chosen), Q_ref = the same variant with 256 opponents (one seed): what the
                sampling noise costs, as judged by the converged planner
"""
import argparse
import json
import os
import time

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/alphazero_v3/latest.pt")
    p.add_argument("--rentals", type=int, default=40)
    p.add_argument("--swaps", type=int, default=60)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--variants", default="old;;n=16;n=32;n=64;n=32,h=1;n=64,h=1;d=2;n=16,h=1,d=2")
    p.add_argument("--ref-n", type=int, default=256)
    p.add_argument("--out", default="runs/probe/planner2/noise.json")
    a = p.parse_args(argv)
    os.environ.setdefault("PYBATTLE_DIAGNOSTIC_PRIOR", "factory_sets")    # "old" draws from the Factory's list
    import torch
    torch.set_num_threads(1)
    from . import agent_probe as AP, planner2 as P2, team_eval as T
    from .tactician_search import options_of
    bench = AP.Bench()
    agents = AP.Agents("cpu")
    hook = agents.hook(a.ckpt)
    ev = hook._load()
    model = P2.opponent_model_for(a.ckpt)
    samp = {"experience": P2.OpponentSampler("experience", model), "factory_sets": P2.OpponentSampler("factory_sets")}
    rng = np.random.default_rng(0)
    rent = [s for s in bench.tactician if s["kind"] == "rental"]
    swap = [s for s in bench.tactician if s["kind"] == "swap" and s["battle"] < 7]   # (k = 7: no planning)
    pick = [rent[i] for i in sorted(rng.choice(len(rent), a.rentals, replace=False))] + \
           [swap[i] for i in sorted(rng.choice(len(swap), a.swaps, replace=False))]
    variants = a.variants.split(";")
    res = {v: {"ms": {"rental": [], "swap": []}, "sd": {"rental": [], "swap": []}, "agree": {"rental": [], "swap": []},
               "regret_ref": {"rental": [], "swap": []}} for v in variants}
    t_all = time.time()
    for s, pre in AP.replay_states(bench, pick):
        kind, obs, ctx = AP.tactician_obs(pre)
        options = options_of(kind, obs)
        view = pre.view()
        hint = (int(view.hint_type), int(view.hint_style))
        vn = T.v_next_value(hook.table(), ctx)
        refs = {}
        for v in variants:
            cfg = P2.parse_options("p=fs" if v == "old" else v)
            qs = []
            for sd in range(a.seeds):
                c = P2.Planner2Config(**dict(cfg.__dict__, seed=AP.feat_seed(s["id"]) + 7919 * sd))
                t0 = time.perf_counter()
                q = P2.plan_values2(ev, view, kind, ctx, options, vn, hint, c, samp[c.prior])["q_plan"]
                res[v]["ms"][kind].append((time.perf_counter() - t0) * 1000)
                qs.append(np.asarray(q))
            Q = np.stack(qs)
            Qc = Q - Q.mean(1, keepdims=True)
            res[v]["sd"][kind].append(float(Qc.std(0, ddof=1).mean()))
            am = Q.argmax(1)
            res[v]["agree"][kind].append(float(np.mean([am[i] == am[j] for i in range(len(am))
                                                        for j in range(i + 1, len(am))])))
            rk = (cfg.prior, cfg.hint_next, cfg.depth)
            if rk not in refs:
                c = P2.Planner2Config(**dict(cfg.__dict__, n_opp=a.ref_n, depth=1, seed=AP.feat_seed(s["id"]) + 1))
                refs[rk] = np.asarray(P2.plan_values2(ev, view, kind, ctx, options, vn, hint, c,
                                                      samp[c.prior])["q_plan"])
            qr = refs[rk]
            res[v]["regret_ref"][kind].append(float(np.mean([qr.max() - qr[i] for i in am])))
        print(f"[noise] state {s['id']} {kind} done, {time.time() - t_all:.0f}s", flush=True)
    summ = {}
    for v in variants:
        summ[v] = {m: {k: (float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)
                       for k, x in d.items() if x} for m, d in res[v].items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"args": vars(a), "summary": summ, "raw": res}, open(a.out, "w"))
    for v in variants:
        r = summ[v]
        print(f"{v or 'default':>16} ms {r['ms']['rental'][0]:7.0f} / {r['ms']['swap'][0]:6.0f}  sd "
              f"{r['sd']['rental'][0]:.3f} / {r['sd']['swap'][0]:.3f}  agree {r['agree']['rental'][0]:.2f} / "
              f"{r['agree']['swap'][0]:.2f}  regret_ref {r['regret_ref']['rental'][0]:.3f} / "
              f"{r['regret_ref']['swap'][0]:.3f}")


if __name__ == "__main__":
    main()
