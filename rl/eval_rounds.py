"""Per-round evaluation: how well a policy plays each round, whatever the round it was trained from.

    python -m rl.eval_rounds --run runs/NAME [--steps 5e6,10e6,15e6] [--runs 192]

In-process mode (the battler sees the backend: decision-time search), one checkpoint:

    python -m rl.eval_rounds --ckpt runs/NAME/latest.pt --battler search --sims 256 --mode legal --runs 192 \
        [--dets 8] [--rounds 1-6] [--procs 8] [--out results.json]

Same protocol and seeds as eval_round (environment j of round k: seed (seed + 1000k) * 100_003 + j, streak 7(k-1),
FactoryEnv skipping decisions without a choice), tactician = the network (greedy), battler = the network (greedy,
--battler net) or rl.search.SearchBattler (--battler search). Environments are split over --procs forked workers,
each with its own network copy (--device, CPU by default), or, with --inference gpu, all sharing one GPU inference
server (rl/inference.py: the battler network, for --battler net and search; the tactician stays on the workers'
CPU copy). --mode perfect is the information ceiling (evaluation
only). Reports per round complete / battle rates and the battler's ms per decision.

For each round k = 1..6, runs start at the beginning of round k (streak 7(k-1), random feasible rental counter,
fixed seeds) and stop when round k is completed or lost. Per round it reports:
    complete  P(win all 7 battles of round k | start of round k)
    battle    per-battle win rate inside round k
Both are comparable between policies trained with different start distributions. Results are cached per
checkpoint in <run>/eval_rounds/.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from . import encode
from .envs import VecEnv, decode_action
from .policy import Policy


def eval_round(policy, device, k, n_runs, workers=4, envs_per_worker=8, seed=777):
    env = VecEnv(workers, envs_per_worker, seed=seed + 1000 * k, win_streak=7 * (k - 1))
    events = env.start()
    wins = np.zeros(env.n, int)
    done = []                                   # (wins in the round, completed)
    per_env = np.zeros(env.n, int)
    target = int(np.ceil(n_runs / env.n))
    while min(per_env) < target:
        acts = [None] * env.n
        by = {}
        for i, ev in enumerate(events):
            by.setdefault(ev["kind"], []).append(i)
        for kind, idx in by.items():
            a = policy.act(kind, encode.collate([events[i]["obs"] for i in idx], device), greedy=True)
            for j, i in enumerate(idx):
                acts[i] = decode_action(kind, a[j])
        for i, ev in enumerate(events):
            b = ev["stats"].get("battle")
            finished = False
            if b is not None and b["won"] is not None:
                if b["won"]:
                    wins[i] += 1
                    finished = wins[i] == 7
                else:
                    finished = True
            if ev["truncate"]:
                finished = True
            if finished:
                if per_env[i] < target:
                    done.append((int(wins[i]), bool(wins[i] == 7), bool(ev["truncate"])))
                per_env[i] += 1
                wins[i] = 0
                # after a loss the environment has already started its next run (this event is its rental, and
                # acts[i] answers it): a "reset" would draw a second run seed and shift every later run of this
                # environment, so the runs would depend on how the earlier ones ended
                if "run" not in ev["stats"]:
                    acts[i] = "reset"
        events = env.step(acts)
    env.close()
    w = np.array([d[0] for d in done]); c = np.array([d[1] for d in done]); t = np.array([d[2] for d in done])
    losses = (~c & ~t).sum()                    # a truncated run ends without a loss
    return {"complete": float(c.mean()), "battle": float(w.sum() / max(w.sum() + losses, 1)), "n": len(done),
            "truncated": int(t.sum())}


# ---- in-process mode ---------------------------------------------------------------------------------------------

def net_battler(policy, evaluator=None):
    """The network's greedy battler as a battler callable: (env, event) -> backend action. evaluator: a batch
    evaluator (rl.inference, e.g. a GPU server client) used instead of the policy's own network."""
    def act(env, ev, known=None):
        if evaluator is not None:
            from .inference import stack_obs
            pri, _ = evaluator(stack_obs([ev["obs"]]))
            a = int(np.argmax(pri[0]))
        else:
            a = policy.act("battle", encode.collate([ev["obs"]], policy.device), greedy=True)[0]
        return decode_action("battle", a)
    return act


def search_battler(policy, evaluator=None, **kw):
    import inspect
    from .search import SearchBattler
    native_ev = evaluator is not None and "evaluator" in inspect.signature(SearchBattler).parameters
    sb = SearchBattler(policy, **(dict(kw, evaluator=evaluator) if native_ev else kw))
    if evaluator is not None and not native_ev:   # leaves evaluated by the batch evaluator (GPU server client)
        from .inference import obs_evaluator
        sb.evaluate = obs_evaluator(evaluator)

    def act(env, ev, known=None):
        return sb.act_on(env.backend, known=known, decisions=env.decisions - 1)
    act.search = sb
    return act


def play_env(env_seed, k, n_runs, policy, battler, stats):
    """The first n_runs runs of one environment (eval_round's per-environment protocol) -> [(wins, completed)]."""
    from pybattle.emu.decode import SB2_RENTAL_MONS, decode_rental_mons
    from .determinize import ExclusionTracker
    from .envs import NO_ACTION, FactoryEnv
    env = FactoryEnv(env_seed, win_streak=7 * (k - 1))
    known = ExclusionTracker()
    ev = env._advance(NO_ACTION)
    out, wins = [], 0
    while len(out) < n_runs:
        b = ev["stats"].get("battle")
        finished = False
        if b is not None and b["won"] is not None:
            if b["won"]:
                wins += 1
                finished = wins == 7
            else:
                finished = True
        if ev["truncate"]:
            finished = True
        if finished:
            out.append((wins, wins == 7))
            wins = 0
            # a fresh environment per run: every run (env, j) gets its own seed and starts identically whatever
            # battler is evaluated (with one environment, a run's seed would depend on how the earlier ones ended)
            env = FactoryEnv(env_seed + 7919 * len(out), win_streak=7 * (k - 1))
            known = ExclusionTracker()
            ev = env._advance(NO_ACTION)
            continue
        kind = ev["kind"]
        be = env.backend
        if kind == "battle":
            t0 = time.perf_counter()
            action = battler(env, ev, known)
            stats["ms"] += (time.perf_counter() - t0) * 1000
            stats["decisions"] += 1
        else:
            a = policy.act(kind, encode.collate([ev["obs"]], policy.device), greedy=True)[0]
            action = decode_action(kind, a)
            view = be.view()
            if kind == "rental":
                known.on_rental(view)
                known.set_own_ids([view.frontier_ids[i] for i in action])
            else:
                own = [m.mon_id for m in decode_rental_mons(be.game.read_saveblock2(SB2_RENTAL_MONS, 72))[:3]]
                known.on_swap(view, own)
        ev = env.step(action)
        if kind == "swap":                      # the team that goes into battle, after the trade
            known.set_own_ids([m.mon_id for m in decode_rental_mons(be.game.read_saveblock2(SB2_RENTAL_MONS, 72))[:3]])
    return out


_SERVER = None                                  # the GPU inference server, set before forking the workers


def _worker(args):
    (ckpt, device, battler_kind, kw, k, env_seeds, n_runs, threads, slot) = args
    torch.set_num_threads(threads)
    policy = Policy(ckpt, torch.device(device))
    ev = _SERVER.client(slot).evaluate if _SERVER is not None else None
    battler = net_battler(policy, ev) if battler_kind == "net" else search_battler(policy, ev, **kw)
    stats = {"ms": 0.0, "decisions": 0}
    runs = []
    for s in env_seeds:
        runs += play_env(s, k, n_runs, policy, battler, stats)
    sb = getattr(battler, "search", None)
    stats["search_errors"] = sb.errors if sb is not None else 0
    return runs, stats


def eval_round_inprocess(ckpt, k, n_runs, battler="net", search_kw=None, workers=4, envs_per_worker=8, seed=777,
                         procs=1, device="cpu", threads=1, server=None):
    """eval_round with a battler that sees the backend. Returns complete / battle / n / ms_per_decision.
    server: an rl.inference.InferenceServer with >= procs client slots (the battler's network on the GPU)."""
    import multiprocessing as mp
    global _SERVER
    n_env = workers * envs_per_worker
    base = seed + 1000 * k                      # VecEnv(seed=seed + 1000k): env j gets base * 100_003 + j
    seeds = [base * 100_003 + j for j in range(n_env)]
    target = int(np.ceil(n_runs / n_env))
    kw = dict(search_kw or {})
    jobs = [(ckpt, device, battler, dict(kw, seed=kw.get("seed", 0) + 7919 * p + 104729 * k), k, seeds[p::procs],
             target, threads, p) for p in range(procs)]
    if server is not None and server.n_clients < procs:
        raise ValueError(f"the inference server has {server.n_clients} client slots, {procs} workers")
    _SERVER = server
    try:
        if procs == 1:
            results = [_worker(jobs[0])]
        else:
            with mp.get_context("fork").Pool(procs) as pool:
                results = pool.map(_worker, jobs)
    finally:
        _SERVER = None
    done = [r for res, _ in results for r in res]
    ms = sum(st["ms"] for _, st in results)
    dec = sum(st["decisions"] for _, st in results)
    w = np.array([d[0] for d in done]); c = np.array([d[1] for d in done])
    losses = (~c).sum()
    return {"complete": float(c.mean()), "battle": float(w.sum() / max(w.sum() + losses, 1)), "n": len(done),
            "decisions": int(dec), "ms_per_decision": ms / max(dec, 1),
            "search_errors": int(sum(st["search_errors"] for _, st in results))}


def _rounds(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main_inprocess(args):
    kw = {}
    if args.battler == "search":
        kw = {"n_sims": args.sims, "n_determinizations": args.dets, "c_puct": args.c_puct, "batch": args.batch,
              "mode": args.mode, "allow_perfect": args.mode == "perfect", "seed": args.seed}
        if args.impl:
            kw["impl"] = args.impl
    res = {"ckpt": args.ckpt, "battler": args.battler, "search": kw, "runs": args.runs, "seed": args.seed,
           "rounds": {}}
    server = None
    if args.inference == "gpu":
        from .inference import InferenceServer
        server = InferenceServer(args.ckpt, n_clients=args.procs, deadline_ms=args.deadline_ms,
                                 precision=args.precision)
        res["inference"] = {"kind": "gpu", "deadline_ms": args.deadline_ms, "precision": args.precision}
    try:
        for k in _rounds(args.rounds):
            t0 = time.time()
            r = eval_round_inprocess(args.ckpt, k, args.runs, args.battler, kw, seed=args.seed, procs=args.procs,
                                     device=args.device, threads=args.threads, server=server)
            r["seconds"] = time.time() - t0
            if server is not None:
                r["inference"] = server.stats()
            res["rounds"][k] = r
            print(f"round {k}: complete {r['complete']:.3f} battle {r['battle']:.3f} n {r['n']} "
                  f"{r['ms_per_decision']:.1f} ms/decision ({r['seconds']:.0f} s)", flush=True)
            if args.out:
                json.dump(res, open(args.out, "w"), indent=1)
    finally:
        if server is not None:
            server.close()
    if not args.out:
        print(json.dumps(res, indent=1))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--ckpt", default=None, help="in-process mode: one checkpoint")
    p.add_argument("--battler", choices=("net", "search"), default="net")
    p.add_argument("--sims", type=int, default=256)
    p.add_argument("--impl", default=None, choices=["cpp", "python"], help="search implementation (default: cpp)")
    p.add_argument("--dets", type=int, default=8, help="determinizations (trees) per decision")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--mode", choices=("legal", "perfect"), default="legal")
    p.add_argument("--rounds", default="1-6")
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--device", default="cpu", help="the workers' network copies (tactician; battler unless --inference gpu)")
    p.add_argument("--inference", choices=("cpu", "gpu"), default="cpu",
                   help="gpu: the battler network runs in one shared GPU inference server (rl/inference.py)")
    p.add_argument("--deadline-ms", type=float, default=0.75, help="--inference gpu: batching window")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32", help="--inference gpu")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--out", default=None)
    p.add_argument("--steps", default=None, help="comma list of battler steps; nearest checkpoint each")
    p.add_argument("--runs", type=int, default=192)
    args = p.parse_args()
    if args.ckpt:
        return main_inprocess(args)
    if not args.run:
        p.error("--run or --ckpt is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(os.path.join(args.run, "ckpt_*.pt")))
    steps_of = {c: int(os.path.basename(c)[5:-3]) for c in ckpts}
    if args.steps:
        wanted = [float(s) for s in args.steps.split(",")]
        ckpts = sorted({min(ckpts, key=lambda c: abs(steps_of[c] - w)) for w in wanted})
    out_dir = os.path.join(args.run, "eval_rounds")
    os.makedirs(out_dir, exist_ok=True)
    for c in ckpts:
        path = os.path.join(out_dir, os.path.basename(c)[:-3] + ".json")
        if os.path.exists(path):
            continue
        policy = Policy(c, device)
        res = {"steps": steps_of[c], "rounds": {k: eval_round(policy, device, k, args.runs) for k in range(1, 7)}}
        json.dump(res, open(path, "w"), indent=1)
        print(os.path.basename(c), {k: (round(v["complete"], 3), round(v["battle"], 3)) for k, v in res["rounds"].items()},
              flush=True)


if __name__ == "__main__":
    main()
