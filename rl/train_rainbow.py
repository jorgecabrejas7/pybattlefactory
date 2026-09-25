"""Joint Rainbow DQN training of the tactician and the battler (docs/RL_DECISIONS.md §12).

    python -m rl.train_rainbow --name rainbow_joint_v1
    python -m rl.evaluate --run runs/rainbow_joint_v1 --watch

Same environments, observations, rewards (+ potential shaping), truncation and network trunk as PPO.
"""

import argparse
import collections
import copy
import faulthandler
import json
import os
import signal
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import encode
from .envs import VecEnv, decode_action
from .rainbow import NStep, RainbowNet, Replay, act, learn, to_env_action

AGENT = {"battle": "battler", "rental": "tactician", "swap": "tactician"}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=time.strftime("rainbow_%Y%m%d_%H%M%S"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--envs-per-worker", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-battler-steps", type=float, default=2e8)
    p.add_argument("--atoms", type=int, default=51)
    p.add_argument("--vmin-b", type=float, default=-1.0)
    p.add_argument("--vmax-b", type=float, default=2.0)
    p.add_argument("--vmin-t", type=float, default=0.0)
    p.add_argument("--vmax-t", type=float, default=60.0)
    p.add_argument("--gamma-b", type=float, default=1.0, help="battler discount (as ppo_joint_v3)")
    p.add_argument("--gamma-t", type=float, default=1.0)
    p.add_argument("--n-b", type=int, default=5)
    p.add_argument("--n-t", type=int, default=3)
    p.add_argument("--replay-b", type=int, default=1_000_000)
    p.add_argument("--replay-t", type=int, default=100_000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta0", type=float, default=0.4)
    p.add_argument("--beta-steps", type=float, default=5e7, help="battler steps to anneal beta to 1")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--reuse", type=int, default=8, help="average times each transition is sampled "
                                                          "(Atari Rainbow: batch 32 every 4 steps = 8)")
    p.add_argument("--target-every", type=int, default=8000, help="gradient steps between target copies")
    p.add_argument("--warmup-b", type=int, default=20_000)
    p.add_argument("--warmup-t", type=int, default=2_000)
    p.add_argument("--beta", type=float, default=0.5, help="potential-based shaping weight")
    p.add_argument("--beta-anneal-steps", type=float, default=20e6,
                   help="battler steps over which the shaping weight goes linearly from --beta to 0 (0: constant)")
    p.add_argument("--start-p0", type=float, default=0.5, help="probability a run starts at round 1 (streak 0)")
    p.add_argument("--start-max-round", type=int, default=5, help="other runs start at streak 7k, k in 1..this")
    p.add_argument("--share", default="embeddings", choices=("all", "embeddings"),
                   help="what the tactician shares with the battler")
    p.add_argument("--max-decisions", type=int, default=300)
    p.add_argument("--d-emb", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--save-every", type=float, default=3e5, help="battler steps between checkpoints")
    p.add_argument("--encode-version", type=int, default=3)
    return p.parse_args()


def main():
    from .search import mark_training
    mark_training()     # no perfect-information search can exist in a training process (rl/search.py)
    faulthandler.register(signal.SIGUSR1)
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    cfg = dict(vars(args), algo="rainbow")
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    tb = SummaryWriter(run_dir)
    tb.add_text("config", "```\n" + json.dumps(cfg, indent=2) + "\n```")

    encode.set_version(args.encode_version)
    net = RainbowNet(args.atoms, args.d_emb, args.d, args.layers, args.heads, share=args.share).to(device)
    target = copy.deepcopy(net)
    for p_ in target.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1.5e-4)
    sup = {"battler": torch.linspace(args.vmin_b, args.vmax_b, args.atoms, device=device),
           "tactician": torch.linspace(args.vmin_t, args.vmax_t, args.atoms, device=device)}
    support = {"battle": sup["battler"], "rental": sup["tactician"], "swap": sup["tactician"]}

    env = VecEnv(args.workers, args.envs_per_worker, seed=args.seed, gamma=args.gamma_b, beta=args.beta,
                 max_decisions=args.max_decisions, start_p0=args.start_p0, start_max_round=args.start_max_round)
    beta_now = args.beta
    n = env.n
    events = env.reset()

    # replays are allocated from the first observations of each kind
    replay = {}
    examples = {}
    nstep = {"battler": [NStep(args.n_b, args.gamma_b) for _ in range(n)],
             "tactician": [NStep(args.n_t, args.gamma_t) for _ in range(n)]}
    new = {"battler": 0, "tactician": 0}

    def ensure_replay(agent, kind, obs):
        examples.setdefault(kind, obs)
        if agent in replay:
            return
        if agent == "battler":
            replay[agent] = Replay(args.replay_b, obs, args.alpha)
        elif "rental" in examples and "swap" in examples:
            ex = dict(examples["swap"])
            ex["lead_mask"], ex["pair_mask"] = examples["rental"]["lead_mask"], examples["rental"]["pair_mask"]
            replay[agent] = Replay(args.replay_t, ex, args.alpha)

    pending = collections.defaultdict(list)          # tactician transitions emitted before its replay exists

    def emitter(agent):
        def emit(kind, obs, action, R, disc, next_kind, next_obs):
            if agent not in replay:
                pending[agent].append((kind, obs, action, R, disc, next_kind, next_obs))
                return
            for item in pending.pop(agent, []):
                replay[agent].add(*item)
            replay[agent].add(kind, obs, action, R, disc, next_kind, next_obs)
            new[agent] += 1
        return emit

    emit = {a: emitter(a) for a in ("battler", "tactician")}

    runs = collections.deque(maxlen=500)
    battles = collections.deque(maxlen=2000)
    act_counts, swap_counts = collections.Counter(), collections.Counter()
    stats = {"battler": collections.defaultdict(list), "tactician": collections.defaultdict(list)}
    truncs = 0
    b_steps = t_steps = 0
    grad_steps = {"battler": 0, "tactician": 0}
    credit = {"battler": 0.0, "tactician": 0.0}
    period = args.batch / args.reuse             # new transitions per gradient step
    t0 = last_log = time.time()
    steps_at_log, next_save = 0, args.save_every
    learn_time = 0.0

    while b_steps < args.total_battler_steps:
        net.reset_noise()
        actions = [None] * n
        by_kind = collections.defaultdict(list)
        for i, ev in enumerate(events):
            by_kind[ev["kind"]].append(i)
        chosen = {}
        for kind, idx in by_kind.items():
            a, _ = act(net, kind, encode.collate([events[i]["obs"] for i in idx], device), support[kind])
            for j, i in enumerate(idx):
                chosen[i] = int(a[j])

        for i, ev in enumerate(events):
            kind = ev["kind"]
            if ev["close_b"] is not None:
                r, done, trunc = ev["close_b"]
                nxt = ev["obs"] if (kind == "battle" and not done) else None
                nstep["battler"][i].close(r, done, trunc, "battle", nxt, emit["battler"])
            if ev["close_t"] is not None:
                r, done, trunc = ev["close_t"]
                if trunc:
                    nstep["tactician"][i].clear()           # no tactician observation to bootstrap from
                else:
                    nxt = ev["obs"] if (kind != "battle" and not done) else None
                    nstep["tactician"][i].close(r, done, False, kind if nxt is not None else "swap", nxt,
                                                emit["tactician"])
            st = ev["stats"]
            if "battle" in st:
                battles.append(st["battle"])
            if "run" in st:
                runs.append(st["run"]["streak"])
            truncs += st.get("truncated", 0)
            if ev["truncate"]:
                actions[i] = "reset"
                continue
            agent = AGENT[kind]
            ensure_replay(agent, kind, ev["obs"])
            a = chosen[i]
            nstep[agent][i].push(kind, ev["obs"], a)
            if agent == "battler":
                b_steps += 1
                act_counts["move" if a < 4 else "switch"] += 1
            else:
                t_steps += 1
                if kind == "swap":
                    swap_counts["keep" if a == 0 else "swap"] += 1
            actions[i] = decode_action(kind, to_env_action(kind, a))
        events = env.step(actions)
        if args.beta_anneal_steps > 0:                  # the shaping weight is withdrawn as in ppo_joint_v3
            b_target = args.beta * max(0.0, 1.0 - b_steps / args.beta_anneal_steps)
            if abs(b_target - beta_now) > 0.005 or (b_target == 0.0 and beta_now != 0.0):
                env.set("beta", b_target)
                beta_now = b_target

        # ---- learning: keep (sampled transitions) / (new transitions) = reuse --------------------------------
        tl = time.time()
        beta = min(1.0, args.beta0 + (1 - args.beta0) * b_steps / args.beta_steps)
        for agent, warm in (("battler", args.warmup_b), ("tactician", args.warmup_t)):
            credit[agent] += new[agent]
            new[agent] = 0
            if agent not in replay or replay[agent].n < warm:
                credit[agent] = 0.0
                continue
            while credit[agent] >= period:
                credit[agent] -= period
                s = learn(net, target, opt, replay[agent], sup[agent], args.batch, beta, device)
                for k, v in s.items():
                    stats[agent][k].append(v)
                grad_steps[agent] += 1
                if sum(grad_steps.values()) % args.target_every == 0:
                    target.load_state_dict(net.state_dict())
        learn_time += time.time() - tl

        if b_steps >= next_save:
            next_save += args.save_every
            ck = {"net": net.state_dict(), "opt": opt.state_dict(), "args": cfg, "battler_steps": b_steps,
                  "tactician_steps": t_steps}
            torch.save(ck, os.path.join(run_dir, "latest.pt"))
            torch.save(ck, os.path.join(run_dir, f"ckpt_{b_steps:011d}.pt"))

        if time.time() - last_log > 30:
            dt = time.time() - last_log
            tb.add_scalar("perf/battler_steps_per_s", (b_steps - steps_at_log) / dt, b_steps)
            tb.add_scalar("perf/learn_time_frac", learn_time / dt, b_steps)
            tb.add_scalar("perf/hours", (time.time() - t0) / 3600, b_steps)
            if torch.cuda.is_available():
                tb.add_scalar("perf/gpu_mem_peak_mb", torch.cuda.max_memory_allocated() / 2 ** 20, b_steps)
            learn_time = 0.0
            for agent in ("battler", "tactician"):
                for k, v in stats[agent].items():
                    tb.add_scalar(f"{agent}/{k}", float(np.mean(v)), b_steps)
                stats[agent].clear()
                tb.add_scalar(f"{agent}/grad_steps", grad_steps[agent], b_steps)
                if agent in replay:
                    tb.add_scalar(f"{agent}/replay_size", replay[agent].n, b_steps)
            tb.add_scalar("rainbow/beta", beta, b_steps)
            sig = [m.w_sigma.abs().mean().item() for m in net.noisy_layers()]
            tb.add_scalar("rainbow/noisy_sigma_mean", float(np.mean(sig)), b_steps)
            if runs:
                r = np.array(runs)
                tb.add_scalar("train/streak_mean", float(r.mean()), b_steps)
                tb.add_scalar("train/challenge_completed_rate", float(np.mean(r >= 7)), b_steps)
                tb.add_scalar("train/rounds_completed_mean", float(np.mean(r // 7)), b_steps)
            done_b = [b for b in battles if b["won"] is not None]
            if done_b:
                tb.add_scalar("train/battle_win_rate", float(np.mean([b["won"] for b in done_b])), b_steps)
                tb.add_scalar("train/battle_decisions", float(np.mean([b["decisions"] for b in battles])), b_steps)
            tb.add_scalar("train/truncated_battles_total", truncs, b_steps)
            tot = sum(act_counts.values()) or 1
            tb.add_scalar("actions/battler_switch_frac", act_counts["switch"] / tot, b_steps)
            tot = sum(swap_counts.values()) or 1
            tb.add_scalar("actions/tactician_swap_frac", swap_counts["swap"] / tot, b_steps)
            act_counts.clear(); swap_counts.clear()
            win = np.mean([b["won"] for b in done_b]) if done_b else 0
            print(f"[{(time.time() - t0) / 60:7.1f} min] battler steps {b_steps:,}  grad b/t {grad_steps['battler']}/"
                  f"{grad_steps['tactician']}  streak {np.mean(runs) if runs else 0:.2f}  win {win:.3f}  "
                  f"{(b_steps - steps_at_log) / dt:,.0f} steps/s", flush=True)
            last_log, steps_at_log = time.time(), b_steps
            tb.flush()

    torch.save({"net": net.state_dict(), "args": cfg, "battler_steps": b_steps}, os.path.join(run_dir, "latest.pt"))
    env.close()


if __name__ == "__main__":
    main()
