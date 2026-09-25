"""Joint PPO training of the tactician and the battler (docs/RL_DECISIONS.md).

    python -m rl.train --name run1
    tensorboard --logdir runs

Every environment is a Factory run. Each round, every environment is at one decision; decisions are batched by
kind (battle / rental / swap) through the shared network. Each agent keeps its own buffer and is updated when
it holds enough complete transitions.
"""

import argparse
import collections
import json
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import encode
from .envs import VecEnv, decode_action
from .model import FactoryNet
from .ppo import AgentBuffer, Transition, ValueNorm, act, ppo_update

AGENT = {"battle": "battler", "rental": "tactician", "swap": "tactician"}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=time.strftime("ppo_%Y%m%d_%H%M%S"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--envs-per-worker", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-battler-steps", type=float, default=2e8)
    p.add_argument("--battler-batch", type=int, default=8192)
    p.add_argument("--tactician-batch", type=int, default=1024)
    p.add_argument("--battler-minibatches", type=int, default=8)
    p.add_argument("--tactician-minibatches", type=int, default=4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma-b", type=float, default=1.0, help="battler discount (v3: 1; v1/v2: 0.99)")
    p.add_argument("--gamma-t", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5, help="potential-based shaping weight")
    p.add_argument("--max-decisions", type=int, default=300, help="battler decisions per battle before truncation")
    p.add_argument("--d-emb", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--save-every", type=int, default=20, help="battler updates between checkpoints")
    p.add_argument("--encode-version", type=int, default=3, help="2 = ppo_joint_v1/v2 layout; 3 = + damage/speed "
                                                                   "estimates and round one-hot")
    p.add_argument("--share", default="embeddings", choices=("all", "embeddings"),
                   help="what the tactician shares with the battler")
    p.add_argument("--value-norm", type=int, default=1, help="normalize each agent's value targets")
    p.add_argument("--beta-anneal-steps", type=float, default=20e6,
                   help="battler steps over which the shaping weight goes linearly from --beta to 0 (0: constant)")
    p.add_argument("--init-from", default=None, help="checkpoint to start from (weights, and optimizer if saved)")
    p.add_argument("--start-p0", type=float, default=0.5, help="probability a run starts at round 1 (streak 0)")
    p.add_argument("--start-max-round", type=int, default=5, help="other runs start at streak 7k, k in 1..this")
    p.add_argument("--target-kl-t", type=float, default=None,
                   help="tactician: stop an update's epochs once a minibatch's approx KL > 1.5 x this")
    p.add_argument("--refresh-logp-t", type=int, default=0,
                   help="tactician: recompute the old log-probs with the network at the start of each update "
                        "(its batches mix transitions from before its previous update: see ppo_update)")
    return p.parse_args()


def main():
    from .search import mark_training
    mark_training()     # no perfect-information search can exist in a training process (rl/search.py)
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    tb = SummaryWriter(run_dir)

    encode.set_version(args.encode_version)       # before building the network and forking the workers
    net = FactoryNet(args.d_emb, args.d, args.layers, args.heads, share=args.share).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    b_steps0 = 0
    norms = {a: (ValueNorm() if args.value_norm else None) for a in ("battler", "tactician")}
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        net.load_state_dict(ck["net"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        for agent, state in (ck.get("value_norm") or {}).items():
            if state and norms.get(agent) is not None:
                norms[agent].load(state)        # the critic predicts values normalized with these statistics
        b_steps0 = ck.get("battler_steps", 0)
        print(f"initialized from {args.init_from} ({b_steps0:,} battler steps)", flush=True)
    tb.add_text("config", "```\n" + json.dumps(vars(args), indent=2) + "\n```")
    tb.add_scalar("model/parameters", sum(p.numel() for p in net.parameters()), 0)

    env = VecEnv(args.workers, args.envs_per_worker, seed=args.seed, gamma=args.gamma_b, beta=args.beta,
                 max_decisions=args.max_decisions, start_p0=args.start_p0, start_max_round=args.start_max_round)
    n = env.n
    bufs = {"battler": AgentBuffer(args.gamma_b, args.lam), "tactician": AgentBuffer(args.gamma_t, args.lam)}
    for b in bufs.values():
        b.ensure(n)
    beta_now = args.beta
    last_t_value = np.zeros(n, np.float32)       # the tactician's latest value per env (bootstrap on truncation)

    # running statistics for logging
    runs = collections.deque(maxlen=500)
    runs_by_start = collections.defaultdict(lambda: collections.deque(maxlen=300))
    battles = collections.deque(maxlen=2000)
    act_counts = collections.Counter()
    swap_counts = collections.Counter()
    truncs = 0
    b_steps, t_steps = b_steps0, 0
    b_updates = t_updates = 0
    t0 = time.time()
    last_log, steps_at_log = time.time(), 0
    update_time = 0.0

    def save(extra=None):
        """latest.pt (and `extra`), atomically: readers never see half a file. Everything needed to resume or to
        read values: the network, the optimizer, the value normalization."""
        ck = {"net": net.state_dict(), "opt": opt.state_dict(), "args": vars(args), "battler_steps": b_steps,
              "tactician_steps": t_steps, "value_norm": {k: (v.state() if v else None) for k, v in norms.items()}}
        for name in ("latest.pt",) + ((extra,) if extra else ()):
            torch.save(ck, os.path.join(run_dir, name + ".tmp"))
            os.replace(os.path.join(run_dir, name + ".tmp"), os.path.join(run_dir, name))

    events = env.reset()
    while b_steps < args.total_battler_steps:
        # ---- choose an action for every environment, batched by decision kind -----------------------------
        actions = [None] * n
        by_kind = collections.defaultdict(list)
        for i, ev in enumerate(events):
            by_kind[ev["kind"]].append(i)
        out = {}
        for kind, idx in by_kind.items():
            x = encode.collate([events[i]["obs"] for i in idx], device)
            a, lp, v = act(net, kind, x)
            nm = norms[AGENT[kind]]
            if nm is not None:
                v = nm.denorm(v)                              # the critic predicts normalized values
            for j, i in enumerate(idx):
                out[i] = (a[j], float(lp[j]), float(v[j]))

        for i, ev in enumerate(events):
            kind = ev["kind"]
            a, lp, v = out[i]
            # close previous transitions
            if ev["close_b"] is not None:
                r, done, trunc = ev["close_b"]
                t = bufs["battler"].per_env[i][-1]
                t.reward, t.done, t.trunc, t.complete = r, done, trunc, True
                t.next_value = v if kind == "battle" else 0.0
            if ev["close_t"] is not None:
                r, done, trunc = ev["close_t"]
                t = bufs["tactician"].per_env[i][-1]
                t.reward, t.done, t.trunc, t.complete = r, done, trunc, True
                if kind != "battle":
                    t.next_value = v
                else:
                    # truncated in a battle: no tactician observation here. Its value at the decision estimated
                    # all the wins from there, `r` of which are already in the reward: bootstrap with the rest
                    t.next_value = max(float(last_t_value[i]) - r, 0.0)
            st = ev["stats"]
            if "battle" in st:
                battles.append(st["battle"])
            if "run" in st:
                runs.append(st["run"]["streak"])
                runs_by_start[st["run"].get("start", 0)].append(st["run"]["streak"])
            truncs += st.get("truncated", 0)
            if ev["truncate"]:
                actions[i] = "reset"
                continue
            agent = AGENT[kind]
            bufs[agent].per_env[i].append(Transition(kind, ev["obs"], a, lp, v))
            if agent == "tactician":
                last_t_value[i] = v
                t_steps += 1
                if kind == "swap":
                    swap_counts["keep" if int(a) == 0 else "swap"] += 1
            else:
                b_steps += 1
                act_counts["move" if int(a) < 4 else "switch"] += 1
            actions[i] = decode_action(kind, a)
        events = env.step(actions)
        if args.beta_anneal_steps > 0:
            b_target = args.beta * max(0.0, 1.0 - b_steps / args.beta_anneal_steps)
            if abs(b_target - beta_now) > 0.005 or (b_target == 0.0 and beta_now != 0.0):
                env.set("beta", b_target)
                beta_now = b_target

        # ---- updates ---------------------------------------------------------------------------------------
        tu = time.time()
        if bufs["battler"].n_complete() >= args.battler_batch:
            tr, adv, ret = bufs["battler"].take()
            s = ppo_update(net, opt, tr, adv, ret, device, args.epochs, args.battler_minibatches, args.clip,
                           args.vf_coef, args.ent_coef, norm=norms["battler"])
            b_updates += 1
            for k, v in s.items():
                tb.add_scalar(f"battler/{k}", v, b_steps)
            tb.add_scalar("battler/batch_size", len(tr), b_steps)
            if b_updates % args.save_every == 0:
                save(f"ckpt_{b_steps:011d}.pt")
        if bufs["tactician"].n_complete() >= args.tactician_batch:
            tr, adv, ret = bufs["tactician"].take()
            s = ppo_update(net, opt, tr, adv, ret, device, args.epochs, args.tactician_minibatches, args.clip,
                           args.vf_coef, args.ent_coef, norm=norms["tactician"], target_kl=args.target_kl_t,
                           refresh_logp=bool(args.refresh_logp_t))
            t_updates += 1
            for k, v in s.items():
                tb.add_scalar(f"tactician/{k}", v, b_steps)
            tb.add_scalar("tactician/batch_size", len(tr), b_steps)
            tb.add_scalar("tactician/steps", t_steps, b_steps)
        update_time += time.time() - tu

        # ---- logging ---------------------------------------------------------------------------------------
        if time.time() - last_log > 30:
            dt = time.time() - last_log
            tb.add_scalar("perf/battler_steps_per_s", (b_steps - steps_at_log) / dt, b_steps)
            tb.add_scalar("perf/update_time_frac", update_time / dt, b_steps)
            update_time = 0.0
            tb.add_scalar("perf/hours", (time.time() - t0) / 3600, b_steps)
            if torch.cuda.is_available():
                tb.add_scalar("perf/gpu_mem_peak_mb", torch.cuda.max_memory_allocated() / 2 ** 20, b_steps)
            if runs:
                tb.add_scalar("train/streak_mean", float(np.mean(runs)), b_steps)
                tb.add_scalar("train/streak_max_recent", float(np.max(runs)), b_steps)
                tb.add_scalar("train/challenge_completed_rate", float(np.mean([r >= 7 for r in runs])), b_steps)
                tb.add_histogram("train/streaks", np.array(runs), b_steps)
            for start, rr in runs_by_start.items():
                # wins in the run and the round reached, by starting round
                tb.add_scalar(f"train_by_start/wins_from_round_{start // 7 + 1}", float(np.mean(rr)), b_steps)
                tb.add_scalar(f"train_by_start/rounds_reached_from_round_{start // 7 + 1}",
                              float(np.mean([(start + w) // 7 for w in rr])), b_steps)
            done_b = [b for b in battles if b["won"] is not None]
            if done_b:
                tb.add_scalar("train/battle_win_rate", float(np.mean([b["won"] for b in done_b])), b_steps)
                tb.add_scalar("train/battle_decisions", float(np.mean([b["decisions"] for b in battles])), b_steps)
                tb.add_scalar("train/battle_shaped_return", float(np.mean([b["shaped_return"] for b in battles])),
                              b_steps)
            tb.add_scalar("train/truncated_battles_total", truncs, b_steps)
            tot = sum(act_counts.values()) or 1
            tb.add_scalar("actions/battler_switch_frac", act_counts["switch"] / tot, b_steps)
            tot = sum(swap_counts.values()) or 1
            tb.add_scalar("actions/tactician_swap_frac", swap_counts["swap"] / tot, b_steps)
            act_counts.clear(); swap_counts.clear()
            tb.add_scalar("perf/battler_updates", b_updates, b_steps)
            tb.add_scalar("train/shaping_beta", beta_now, b_steps)
            for k, nm in norms.items():
                if nm is not None:
                    tb.add_scalar(f"{k if k == 'tactician' else 'battler'}/value_norm_std", nm.std, b_steps)
            tb.add_scalar("perf/tactician_updates", t_updates, b_steps)
            print(f"[{(time.time() - t0) / 60:7.1f} min] battler steps {b_steps:,}  updates b/t {b_updates}/{t_updates}"
                  f"  streak {np.mean(runs) if runs else 0:.2f}  win {np.mean([b['won'] for b in done_b]) if done_b else 0:.3f}"
                  f"  {(b_steps - steps_at_log) / dt:,.0f} steps/s", flush=True)
            last_log, steps_at_log = time.time(), b_steps
            tb.flush()

    save()
    env.close()


if __name__ == "__main__":
    main()
