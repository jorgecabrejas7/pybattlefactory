"""Data and validation for the experience opponent model (rl/opponent_model.py, docs/RL_DECISIONS.md §22).

    # warm start: records of what the player saw while alphazero_v3 plays (both agents greedy, network only)
    python -m rl.opponent_model_data collect --ckpt runs/alphazero_v3/latest.pt --battles 30000 \
        --out runs/opp_model_v3/model.pt
    # held-out battles (other seeds) with the TRUE opponent, for validation only
    python -m rl.opponent_model_data heldout --ckpt runs/alphazero_v3/latest.pt --battles 3000 \
        --out runs/opp_model_v3/heldout.pkl
    # log-likelihood / top-k of the true opponent configuration: model vs the strict sampler
    python -m rl.opponent_model_data validate --model runs/opp_model_v3/model.pt --heldout runs/opp_model_v3/heldout.pkl

Protocol (collect and heldout): each worker plays runs that start at the beginning of round k (k cycling 1..6, a
random feasible rental counter, FactoryEnv), until they are lost, with the checkpoint's network greedy for both
agents (the tactician with its per-option features, rl.eval_rounds.features_hook), on its CPU.
    collect   an rl.opponent_model.ExperienceCollector watches the environment (player view only) -> model file.
    heldout   at every battle decision: the view of the opponent's 3 slots (what the player sees) and, for scoring
              only, the TRUE configuration of each slot (species, moves, item: read from gEnemyParty). This file is
              the only place that reads the true opponent; nothing here feeds a model.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import time

import numpy as np
import torch

from . import encode
from . import opponent_model as OM
from .envs import NO_ACTION, FactoryEnv, decode_action

SEED_COLLECT, SEED_HELDOUT = 4242, 9191
LOGP_FLOOR = math.log(1e-12)


def _true_configs(backend):
    """VALIDATION ONLY: the opponent's real (species, moves, item) per slot (gEnemyParty)."""
    from pybattle.emu.decode import SYMBOLS as S, decode_party
    party = decode_party(backend.game.read(S.addr("gEnemyParty"), 300))[:3]
    return [(int(m.species), tuple(int(x) for x in m.moves), int(m.held_item)) for m in party]


def _play(job):
    """One worker: play runs until n_battles battles ended. mode "collect" -> Counter of records; "heldout" ->
    list of decision samples."""
    ckpt, seeds, n_battles, mode, hook, threads = job
    torch.set_num_threads(threads)
    from .policy import Policy
    policy = Policy(ckpt, torch.device("cpu"))
    # 6 environments, one per start round (runs start at the beginning of round k and go on until lost); played in
    # turns, one run at a time; each has its own collector (the collector follows one environment's screens)
    envs, collectors, events, ended = [], [], [], []
    for k in range(1, 7):
        env = FactoryEnv(seeds[k - 1], win_streak=7 * (k - 1), obs_hook=hook)
        col = OM.ExperienceCollector()
        if mode == "collect":
            col.attach(env)
        envs.append(env)
        collectors.append(col)
        events.append(env._advance(NO_ACTION))
        ended.append(0)
    samples = []
    battles = 0
    e = 0
    while battles < n_battles:
        env, ev = envs[e], events[e]
        while battles < n_battles:
            kind = ev["kind"]
            if kind == "battle" and mode == "heldout":
                be = env.backend
                v = be.view()
                info = be.run_info()
                samples.append({"battle": (seeds[e], ended[e]), "challenge": info.challenge_num,
                                "noland": info.noland, "hint": (int(v.hint_type), int(v.hint_style)),
                                "enemy": v.enemy_party[:3], "truth": _true_configs(be), "turn": v.turn})
            a = policy.act(kind, encode.collate([ev["obs"]], policy.device), greedy=True)[0]
            ev = env.step(decode_action(kind, a))
            if ev["stats"].get("battle") is not None:
                battles += 1
                ended[e] += 1
            if ev["truncate"]:
                ev = env.step("reset")
                break
            if "run" in ev["stats"]:
                break
        events[e] = ev
        e = (e + 1) % len(envs)
    if mode == "collect":
        return OM.merge([c.drain() for c in collectors]), battles
    return samples, battles


def _run(ckpt, n_battles, procs, seed, mode, threads=1):
    from .eval_rounds import features_hook
    hook = features_hook(ckpt, argparse.Namespace(features="auto", team_eval=None))
    per = math.ceil(n_battles / procs)
    jobs = [(ckpt, [seed * 7919 + 104729 * p + j for j in range(64)], per, mode, hook, threads) for p in range(procs)]
    if procs == 1:
        return [_play(jobs[0])]
    with mp.get_context("fork").Pool(procs) as pool:
        return pool.map(_play, jobs)


def collect(args):
    t0 = time.time()
    res = _run(args.ckpt, args.battles, args.procs, args.seed if args.seed is not None else SEED_COLLECT, "collect")
    counts = OM.merge([r for r, _ in res])
    model = OM.ExperienceModel(json.loads(args.cfg), counts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save(args.out)
    info = dict(model.summary(), battles=sum(b for _, b in res), seconds=time.time() - t0, ckpt=args.ckpt)
    json.dump(info, open(os.path.splitext(args.out)[0] + "_info.json", "w"), indent=1)
    print(json.dumps(info), flush=True)


def heldout(args):
    t0 = time.time()
    res = _run(args.ckpt, args.battles, args.procs, args.seed if args.seed is not None else SEED_HELDOUT, "heldout")
    samples = [s for r, _ in res for s in r]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pickle.dump(samples, open(args.out, "wb"))
    print(f"{len(samples)} decisions of {sum(b for _, b in res)} battles ({time.time() - t0:.0f} s)", flush=True)


# ---- validation ------------------------------------------------------------------------------------------------------

class _View:
    def __init__(self, enemy):
        self.enemy_party = enemy


def factory_sets_slot_logp(view, slot, challenge, species, moves, item):
    """REFERENCE ONLY (reads the Factory's set list, like the factory_sets diagnostic ceiling): log P of the true
    configuration when the slot is drawn uniformly from the round's pool sets consistent with what the player saw
    (unseen: not a seen species, not a revealed item; seen: its species, revealed moves and item). The generation's
    exclusions (species on screen, Noland's set ids) are not applied."""
    from . import determinize as D
    enemy = view.enemy_party[:3]
    m = enemy[slot]
    ids = D.pool(challenge)
    used_items = {x.revealed_item for j, x in enumerate(enemy) if j != slot and x.seen and x.revealed_item}
    if m.seen:
        R = frozenset(x for x in m.revealed_moves if x)
        c = [k for k in ids if D.SET_SPECIES[k] == m.species and R <= D._SET_MOVESET[k]
             and (D.SET_ITEM[k] == m.revealed_item if m.revealed_item else D.SET_ITEM[k] not in used_items)]
    else:
        seen = {x.species for x in enemy if x.seen}
        c = [k for k in ids if D.SET_SPECIES[k] not in seen and (D.SET_ITEM[k] == 0 or D.SET_ITEM[k] not in used_items)]
    if not c:
        return -math.inf
    key = (int(species), OM._moves_key(moves), int(item))
    hit = sum(1 for k in c if (int(D.SET_SPECIES[k]), OM._moves_key(D.SET_MOVES_AS_BUILT[k]), int(D.SET_ITEM[k])) == key)
    return math.log(hit / len(c)) if hit else -math.inf


def score(model, samples, per_battle=None, seed=0, ks=(1, 5, 10), with_factory=True):
    """Per slot of each sampled decision: log P(true configuration) under the model and the strict sampler, the
    model's rank of it, and whether it is a configuration the model knows. Decisions: all, or per_battle random
    ones per battle. Slots: "unseen" (never sent out) and "seen" (sent out, alive). Log-probabilities are floored
    at log 1e-12 (an "impossible" configuration, counted apart). with_factory: also the factory_sets reference
    (factory_sets_slot_logp). -> metrics dict."""
    rng = random.Random(seed)
    by_battle = {}
    for s in samples:
        by_battle.setdefault(s["battle"], []).append(s)
    chosen = []
    for key in sorted(by_battle, key=str):
        ss = by_battle[key]
        chosen += ss if per_battle is None else rng.sample(ss, min(per_battle, len(ss)))
    acc = {g: {"n": 0, "lp_model": 0.0, "lp_strict": 0.0, "lp_factory": 0.0, "strict_impossible": 0,
               "model_impossible": 0, "factory_impossible": 0,
               "known": 0, **{f"top{k}": 0 for k in ks}} for g in ("unseen", "seen", "all")}
    per_round = {}
    for s in chosen:
        v = _View(s["enemy"])
        for i, m in enumerate(s["enemy"]):
            if m.seen and m.fainted:
                continue
            sp, moves, item = s["truth"][i]
            g = "seen" if m.seen else "unseen"
            lm = model.slot_logp(v, i, s["challenge"], sp, moves, item) if model is not None else -math.inf
            ls = OM.strict_slot_logp(v, i, sp, moves, item)
            lf = factory_sets_slot_logp(v, i, s["challenge"], sp, moves, item) if with_factory else -math.inf
            rank = model.slot_rank(v, i, s["challenge"], sp, moves, item, max_rank=max(ks)) \
                if model is not None else max(ks) + 1
            r = min(int(s["challenge"]) + 1, 7)
            for gg in (g, "all"):
                a = acc[gg]
                a["n"] += 1
                a["lp_model"] += max(lm, LOGP_FLOOR)
                a["lp_strict"] += max(ls, LOGP_FLOOR)
                a["lp_factory"] += max(lf, LOGP_FLOOR)
                a["factory_impossible"] += int(lf == -math.inf)
                a["model_impossible"] += int(lm == -math.inf)
                a["strict_impossible"] += int(ls == -math.inf)
                a["known"] += int(rank <= max(ks))
                for k in ks:
                    a[f"top{k}"] += int(rank <= k)
            pr = per_round.setdefault(r, {"n": 0, "lp_model": 0.0, "lp_strict": 0.0, "top1": 0})
            pr["n"] += 1
            pr["lp_model"] += max(lm, LOGP_FLOOR)
            pr["lp_strict"] += max(ls, LOGP_FLOOR)
            pr["top1"] += int(rank <= 1)
    out = {}
    for g, a in acc.items():
        n = max(a["n"], 1)
        out[g] = {"slots": a["n"], "mean_logp_model": a["lp_model"] / n, "mean_logp_strict": a["lp_strict"] / n,
                  "mean_logp_factory_sets": a["lp_factory"] / n if with_factory else None,
                  "model_impossible": a["model_impossible"] / n, "strict_impossible": a["strict_impossible"] / n,
                  "factory_impossible": a["factory_impossible"] / n if with_factory else None,
                  **{f"top{k}": a[f"top{k}"] / n for k in ks}}
    out["per_round"] = {r: {"slots": a["n"], "mean_logp_model": a["lp_model"] / max(a["n"], 1),
                            "mean_logp_strict": a["lp_strict"] / max(a["n"], 1), "top1": a["top1"] / max(a["n"], 1)}
                        for r, a in sorted(per_round.items())}
    out["decisions"] = len(chosen)
    return out


def validate(args):
    samples = pickle.load(open(args.heldout, "rb"))
    res = {}
    for spec in args.model:
        name, _, path = spec.rpartition("=")
        cfg = json.loads(args.cfg)
        model = OM.ExperienceModel.load(path, cfg)
        t0 = time.time()
        model.compile()
        r = score(model, samples, args.per_battle)
        r["summary"] = model.summary()
        r["compile_seconds"] = time.time() - t0
        res[name or path] = r
        a, u, s = r["all"], r["unseen"], r["seen"]
        print(f"{name or path}: slots {a['slots']}  mean log P  model {a['mean_logp_model']:.2f} / strict "
              f"{a['mean_logp_strict']:.2f}  (unseen {u['mean_logp_model']:.2f} / {u['mean_logp_strict']:.2f}; "
              f"seen {s['mean_logp_model']:.2f} / {s['mean_logp_strict']:.2f})  top1 {a['top1']:.3f} top5 "
              f"{a['top5']:.3f} top10 {a['top10']:.3f} (unseen {u['top1']:.3f}/{u['top10']:.3f}, seen "
              f"{s['top1']:.3f}/{s['top10']:.3f})  factory_sets ref {a['mean_logp_factory_sets']:.2f} (unseen "
              f"{u['mean_logp_factory_sets']:.2f}, seen {s['mean_logp_factory_sets']:.2f})", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "heldout"):
        q = sub.add_parser(name)
        q.add_argument("--ckpt", required=True)
        q.add_argument("--battles", type=int, default=30000)
        q.add_argument("--procs", type=int, default=18)
        q.add_argument("--seed", type=int, default=None)
        q.add_argument("--out", required=True)
        q.add_argument("--cfg", default="{}", help="JSON overrides of the model's cfg (collect)")
    q = sub.add_parser("validate")
    q.add_argument("--model", nargs="+", required=True, help="PATH or NAME=PATH (several: compared)")
    q.add_argument("--heldout", required=True)
    q.add_argument("--per-battle", type=int, default=None, help="random decisions per battle (default: all)")
    q.add_argument("--cfg", default="{}", help="JSON overrides of the model's cfg")
    q.add_argument("--out", default=None)
    a = p.parse_args()
    {"collect": collect, "heldout": heldout, "validate": validate}[a.cmd](a)


if __name__ == "__main__":
    main()
