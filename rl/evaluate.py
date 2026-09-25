"""Evaluation on fixed run seeds, alongside training (docs/RL_DECISIONS.md §8).

    python -m rl.evaluate --run runs/NAME --watch        # evaluates each new checkpoint, logs eval/* to TensorBoard

Every evaluation plays the same runs (same starting seeds) with the greedy (argmax) and the sampled policy.
The baselines (random battler, max-damage battler, both with the random tactician) are played once on the
same seeds and logged at every step as reference lines.
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pybattle.backend import SimBackend
from . import encode
from .baselines import MaxDamageBattler, RandomBattler, RandomTactician, run_streak
from .envs import VecEnv, decode_action
from .policy import Policy


def run_seeds(env_seed, k):
    """The first k run seeds FactoryEnv(env_seed) will play (FactoryEnv draws one per run)."""
    rng = random.Random(env_seed)
    return [rng.getrandbits(32) for _ in range(k)]


def summarize(streaks, prefix):
    s = np.array(streaks, np.float64)
    boot = [np.random.default_rng(i).choice(s, len(s)).mean() for i in range(1000)]
    return {f"{prefix}/streak_mean": s.mean(), f"{prefix}/streak_median": np.median(s),
            f"{prefix}/streak_max": s.max(), f"{prefix}/streak_ci95_low": np.percentile(boot, 2.5),
            f"{prefix}/streak_ci95_high": np.percentile(boot, 97.5),
            f"{prefix}/challenge_completed_rate": float(np.mean(s >= 7)),
            f"{prefix}/battle_win_rate": s.sum() / (s.sum() + len(s)),   # each run ends with exactly one loss
            f"{prefix}/rounds_completed_mean": float(np.mean(s // 7)),
            **{f"{prefix}/reach_round_{k}": float(np.mean(s >= 7 * k)) for k in range(1, 7)}}   # 6 rounds = 42


def evaluate_policy(policy, device, make_env, runs_per_env, greedy, seed=0):
    torch.manual_seed(seed)                     # the sampled policy's draws are repeatable too
    env = make_env()
    per_env = [[] for _ in range(env.n)]
    events = env.start()
    while min(len(r) for r in per_env) < runs_per_env:
        actions = [None] * env.n
        by_kind = {}
        for i, ev in enumerate(events):
            by_kind.setdefault(ev["kind"], []).append(i)
        for kind, idx in by_kind.items():
            a = policy.act(kind, encode.collate([events[i]["obs"] for i in idx], device), greedy=greedy)
            for j, i in enumerate(idx):
                actions[i] = decode_action(kind, a[j])
        for i, ev in enumerate(events):
            if "run" in ev["stats"]:
                per_env[i].append(ev["stats"]["run"]["streak"])
            if ev["truncate"]:
                per_env[i].append(ev["stats"].get("wins", 0))
                actions[i] = "reset"
        events = env.step(actions)
    env.close()
    return [s for r in per_env for s in r[:runs_per_env]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--envs-per-worker", type=int, default=8)
    p.add_argument("--runs-per-env", type=int, default=9, help="runs per environment: 3x8x9 = 216 runs")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--backfill", type=float, default=0, help="first evaluate one checkpoint every this many steps "
                                                                  "where the log has no evaluation")
    p.add_argument("--rounds-runs", type=int, default=192,
                   help="also evaluate each round from its start (comparable across start distributions); 0: off")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb = SummaryWriter(args.run)
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    def make_env():      # a fresh pool per evaluation: every evaluation plays the same run seeds
        return VecEnv(args.workers, args.envs_per_worker, seed=args.seed, gamma=cfg["gamma_b"], beta=cfg["beta"],
                      max_decisions=cfg["max_decisions"])
    env_seeds = [args.seed * 100_003 + w * args.envs_per_worker + i
                 for w in range(args.workers) for i in range(args.envs_per_worker)]

    # baselines on the same run seeds
    baseline_file = os.path.join(args.run, "eval_baselines.json")
    if os.path.exists(baseline_file):
        baselines = json.load(open(baseline_file))
    else:
        baselines = {}
        seeds = [s for es in env_seeds for s in run_seeds(es, args.runs_per_env)]
        for name, B in (("random", RandomBattler), ("maxdamage", MaxDamageBattler)):
            streaks = [run_streak(SimBackend(), RandomTactician(k), B(k), seed=s,
                                  max_decisions=cfg["max_decisions"])[0] for k, s in enumerate(seeds)]
            baselines.update({k: float(v) for k, v in summarize(streaks, f"baseline_{name}").items()})
        json.dump(baselines, open(baseline_file, "w"), indent=2)
    print("baselines:", {k: round(v, 2) for k, v in baselines.items() if k.endswith("streak_mean")}, flush=True)

    done = set()
    backfill = []
    if args.backfill:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(args.run, size_guidance={"scalars": 0}); ea.Reload()
        have = [e.step for e in ea.Scalars("eval_greedy/streak_mean")] if "eval_greedy/streak_mean" in ea.Tags()["scalars"] else []
        by_step = {int(os.path.basename(c)[5:-3]): c for c in glob.glob(os.path.join(args.run, "ckpt_*.pt"))}
        for g in np.arange(args.backfill, max(by_step) + 1, args.backfill):
            s_ = min(by_step, key=lambda k: abs(k - g))
            if not any(abs(h - s_) < args.backfill / 2 for h in have) and by_step[s_] not in backfill:
                backfill.append(by_step[s_])
    while True:
        ckpts = sorted(glob.glob(os.path.join(args.run, "ckpt_*.pt")))
        todo = [c for c in ckpts if c not in done]
        if todo:
            if backfill:
                path = backfill.pop(0)                      # fill gaps on the grid first, oldest first
                done.add(path)
            else:
                path = todo[-1]                             # newest only: skip checkpoints we fell behind on
                done.update(todo)
            try:
                policy = Policy(path, device)
            except (RuntimeError, EOFError) as e:        # still being written by the trainer: retry next poll
                print(f"could not load {os.path.basename(path)} yet ({e}); retrying", flush=True)
                done.discard(path)
                time.sleep(10)
                continue
            step = policy.steps
            t = time.time()
            for greedy in (True, False):
                streaks = evaluate_policy(policy, device, make_env, args.runs_per_env, greedy)
                for k, v in summarize(streaks, "eval_greedy" if greedy else "eval_sampled").items():
                    tb.add_scalar(k, v, step)
            for k, v in baselines.items():
                tb.add_scalar(k.replace("baseline_", "eval_baseline_"), v, step)
            if args.rounds_runs:
                from .eval_rounds import eval_round
                for k in range(1, 7):
                    r = eval_round(policy, device, k, args.rounds_runs, workers=args.workers)
                    tb.add_scalar(f"eval_round_{k}/complete", r["complete"], step)
                    tb.add_scalar(f"eval_round_{k}/battle_win_rate", r["battle"], step)
            tb.flush()
            print(f"evaluated {os.path.basename(path)} in {time.time() - t:.0f}s", flush=True)
        elif not args.watch:
            break
        if not args.watch:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
