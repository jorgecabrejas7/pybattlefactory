"""How strong are the tactician's simulated opponents? (docs/RL_DECISIONS.md §18)

At the start of round k (the rental screen, streak 7(k-1)), the network's own greedy rental is applied and the first
battle is played by the battler network (greedy, no search, as BattleSimulator plays it) against
    strict        an opponent drawn by the strict player-knowledge sampler (rl/determinize.py)
    factory_sets  an opponent drawn from the Factory's set list (training-time sampler)
    real          the real opponent the game generated (the RNG reseeded per battle)
and, for reference, the network's real per-round results (rl.eval_rounds.eval_round: battle win rate and round
completion, greedy, network only).

    python scripts/diag_sim_opponents.py --rounds 1,3,5 --starts 64 --per-start 4
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from pybattle.backend import Phase, SimBackend
from rl import alphazero as AZ
from rl import encode, search
from rl import tactician_search as TS
from rl.envs import decode_action


def rental_option(policy, b):
    ctx = search.run_ctx(b)
    obs = encode.rental(b.view(), ctx)
    a = policy.act("rental", encode.collate([obs], "cpu"), greedy=True)[0]
    return (int(a[0]), int(a[1])), obs


def real_start(b, option, rng):
    be = b.clone()
    act = decode_action("rental", option)
    assert be.game.factory_rent(*act)
    be.game.set_rng(rng.getrandbits(32))
    be.phase = Phase.BATTLE
    return be


def measure(name, policy, net, norm, version, rounds, starts, per_start, max_decisions, eval_runs):
    encode.set_version(version)
    ev = AZ.local_battler_evaluator(net, norm)
    sim = TS.BattleSimulator(ev, max_decisions=max_decisions)
    out = {}
    for k in rounds:
        res = {}
        for cond in ("strict", "factory_sets", "real"):
            rng = random.Random(1000 * k + 7)
            sims = []
            for s in range(starts):
                b = SimBackend(max_turns=10 ** 9)
                b.reset(seed=10_000 * k + s, win_streak=7 * (k - 1), rents_count=k - 1)
                opt, _ = rental_option(policy, b)
                for j in range(per_start):
                    be = real_start(b, opt, rng) if cond == "real" else \
                        TS.start_simulated_battle(b, "rental", opt, rng, cond)
                    sims.append(TS._Sim(be, 0, sim.cpp))
            sim.run(sims)
            won = np.array([s.won for s in sims], float)
            trunc = np.array([s.trunc for s in sims], float)
            res[cond] = {"win": float(won.mean()), "truncated": float(trunc.mean()), "n": len(sims)}
        if eval_runs:
            from rl.eval_rounds import eval_round
            r = eval_round(policy, "cpu", k, eval_runs, workers=8, envs_per_worker=4)
            res["eval_round"] = {"battle_win_rate": float(r["battle"]), "complete": float(r["complete"])}
        out[k] = res
        print(name, k, json.dumps(res), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ppo_joint_v3/latest.pt")
    p.add_argument("--rounds", default="1,3,5")
    p.add_argument("--starts", type=int, default=64)
    p.add_argument("--per-start", type=int, default=4)
    p.add_argument("--max-decisions", type=int, default=300)
    p.add_argument("--eval-runs", type=int, default=192)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    search.mark_training()                  # factory_sets is a training-time sampler: this diagnostic measures it
    torch.set_num_threads(4)
    rounds = [int(x) for x in args.rounds.split(",")]
    from rl.policy import Policy
    pol = Policy(args.ckpt, torch.device("cpu"))
    results = {"v3": measure("v3", pol, pol.net, (pol.value_norm or {}).get("battler"), pol.encode_version, rounds,
                             args.starts, args.per_start, args.max_decisions, args.eval_runs)}
    torch.manual_seed(0)
    encode.set_version(4)
    from rl.model import FactoryNet
    net = FactoryNet(share="embeddings").eval()
    results["untrained"] = measure("untrained", AZ.NetPolicy(net), net, None, 4, rounds, args.starts,
                                   args.per_start, args.max_decisions, args.eval_runs)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
