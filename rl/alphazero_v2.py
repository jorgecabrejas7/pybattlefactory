"""alphazero_v2 (docs/RL_DECISIONS.md §19): Gumbel AlphaZero battler + hybrid tactician, from scratch.

    python -m rl.alphazero_v2 --name alphazero_v2 --team-eval runs/team_eval/latest.pt
    tensorboard --logdir runs

A separate entry point that reuses rl/alphazero.py (curriculum, worker pool, replay buffer, battler training step,
evaluation), so alphazero_v1 (python -m rl.alphazero) stays exactly as it was.

Each iteration:
  1. Self-play with a frozen snapshot: --workers processes play Factory runs (curriculum over the starting round as
     in v1); the battler network in the GPU inference server, the tactician on each worker's CPU copy.
       battler     Gumbel AlphaZero over the legal-mode ensemble of §15 (rl/gumbel.py: K = --dets determinizations
                   with the turn redrawn, the opponent drawn with --opponent-prior, factory_sets in training):
                   Gumbel top-m (m = min(legal, --max-considered)) + sequential halving with --sims simulations in
                   total; below the root the deterministic Gumbel rule (--non-root gumbel) or PUCT. Self-play plays
                   the halving winner (no temperature, no Dirichlet noise). Targets: policy = softmax(logits +
                   sigma(completed Q)) (c_visit --c-visit, c_scale --c-scale, Q min-max rescaled: --gumbel-rescale);
                   value = (1 - m) z + m q_root, z the battle's outcome, q_root the visit-weighted root Q,
                   m = --value-mix-b.
       tactician   no search: the action is sampled from the network (on-policy), which also sees the team
                   evaluator's per-option features [E_round, W(normal), W(last|Noland), Delta] (rl/tactician_features.py
                   with --team-eval; zeros without it). Each run is a trajectory of tactician decisions (rental,
                   swap, ..., end of the run) with rewards r = battles won since the previous decision plus the
                   potential-based shaping Phi(s') - Phi(s) (gamma 1, Phi(end) = 0; --t-shaping), Phi = E_round +
                   P(complete) v_next, v_next = table[round + 1] of mean real wins from the start of a round, a
                   running mean (--v-next-rate per iteration) of self-play outcomes.
  2. Training:
       battler     cross-entropy to the Gumbel target + value MSE on a replay buffer of --window iterations
                   (rl.alphazero.train_iteration).
       tactician   PPO on this iteration's samples only (on-policy; the decision still open at the end of the
                   iteration is carried to the next one as a bootstrap and never trained on): --t-ppo-epochs epochs,
                   clipped ratio (--t-clip), GAE (lambda --t-gae-lambda, gamma 1) on the shaped rewards, value MSE on
                   ValueNorm-normalized returns, entropy bonus (--t-ent), and --t-imitation x CE(pi_eval, pi_net) with
                   pi_eval = softmax(Q_eval / --t-eval-temp), Q_eval(option) = Phi of the state the option leads to.
  3. Evaluation: per round, network only, every --eval-every iterations (eval_round_k/*); network + legal PUCT search
     of --eval-search-sims (strict sampler: inference never uses factory_sets) every --eval-search-every iterations
     (eval_search_round_k/*). Checkpoints as v1 (args["algo"] = "alphazero", args["version"] = 2, args["opt_feat"] = 4,
     and the v_next table).

Perfect-information search is impossible here: mark_training() runs first, GumbelBattler has no mode argument, and
the C++ Searcher refuses roots that are not full determinizations.
"""

import argparse
import collections
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import alphazero as AZ
from . import encode
from .envs import decode_action
from .gumbel import GumbelBattler
from .model import FactoryNet
from .ppo import ValueNorm
from .tactician_features import N_FEAT, V_NEXT_ROUNDS, TeamFeatures, softmax, to_action_space

VERSION = 2
T_KINDS = ("rental", "swap")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", default=time.strftime("alphazero_v2_%Y%m%d_%H%M%S"))
    p.add_argument("--run-dir", default=None, help="default runs/<name>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=200, help="planned iterations (cosine learning rate)")
    p.add_argument("--decisions-per-iter", type=int, default=40_000, help="self-play battler decisions per iteration")
    p.add_argument("--workers", type=int, default=18)
    p.add_argument("--inference", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--device", default=None, help="training device (default cuda if available)")
    # battler: Gumbel search
    p.add_argument("--sims", type=int, default=256, help="simulations per decision, all trees together")
    p.add_argument("--dets", type=int, default=8, help="determinizations (trees) per decision")
    p.add_argument("--max-considered", type=int, default=16, help="Gumbel top-m: m = min(legal, this)")
    p.add_argument("--c-visit", type=float, default=50.0)
    p.add_argument("--c-scale", type=float, default=0.1)
    p.add_argument("--gumbel-rescale", type=int, default=1, help="min-max rescale the completed Q (mctx default)")
    p.add_argument("--non-root", choices=("gumbel", "puct"), default="gumbel",
                   help="selection below the root: the paper's deterministic rule or PUCT (--c-puct)")
    p.add_argument("--c-puct", type=float, default=1.5, help="--non-root puct, and the evaluation search")
    p.add_argument("--search-batch", type=int, default=32)
    p.add_argument("--max-decisions", type=int, default=300, help="battler decisions per battle before truncation")
    p.add_argument("--opponent-prior", choices=("factory_sets", "strict"), default="factory_sets",
                   help="the battler search's opponent sampler (factory_sets: training only)")
    p.add_argument("--value-mix-b", type=float, default=0.5, help="battler value target: (1-m) z + m q_root")
    # tactician: hybrid
    p.add_argument("--team-eval", default=None,
                   help="team evaluator checkpoint (rl/team_eval.py). None: no features (zeros), no shaping, no "
                        "imitation")
    p.add_argument("--team-eval-module", default=None,
                   help="module with the team evaluator API (default rl.team_eval, else rl.team_eval_stub)")
    p.add_argument("--t-shaping", type=int, default=1, help="potential-based shaping Phi(s') - Phi(s)")
    p.add_argument("--t-imitation", type=float, default=0.0, help="coefficient of CE(pi_eval, pi_net)")
    p.add_argument("--t-eval-temp", type=float, default=0.5, help="pi_eval = softmax(Q_eval / T), T in wins")
    p.add_argument("--t-ppo-epochs", type=int, default=4)
    p.add_argument("--t-clip", type=float, default=0.2)
    p.add_argument("--t-ent", type=float, default=0.01)
    p.add_argument("--t-gae-lambda", type=float, default=0.95)
    p.add_argument("--t-minibatch", type=int, default=256, help="tactician samples per PPO step")
    p.add_argument("--t-vf-coef", type=float, default=1.0)
    p.add_argument("--v-next-rate", type=float, default=0.3,
                   help="v_next table: per-iteration running mean rate (first observation: taken as is)")
    # battler training (rl.alphazero.train_iteration)
    p.add_argument("--window", type=int, default=4, help="battler replay buffer: the last N iterations")
    p.add_argument("--reuse", type=float, default=4.0, help="times each battler sample is used on average")
    p.add_argument("--batch-b", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--value-norm-rate", type=float, default=0.3)
    p.add_argument("--holdout", type=float, default=0.05, help="fraction of battles / runs kept out of training")
    # network
    p.add_argument("--d-emb", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--encode-version", type=int, default=4)
    # curriculum
    p.add_argument("--start-p0", type=float, default=0.3)
    # evaluation
    p.add_argument("--eval-every", type=int, default=1, help="iterations between network-only evaluations")
    p.add_argument("--eval-runs", type=int, default=192)
    p.add_argument("--eval-rounds", default="1-6")
    p.add_argument("--eval-workers", type=int, default=16)
    p.add_argument("--eval-envs-per-worker", type=int, default=4)
    p.add_argument("--eval-search-every", type=int, default=5, help="iterations between search evaluations (0: never)")
    p.add_argument("--eval-search-sims", type=int, default=512)
    p.add_argument("--eval-search-runs", type=int, default=64)
    p.add_argument("--eval-search-procs", type=int, default=12)
    p.add_argument("--init-from", default=None, help="checkpoint to resume from")
    a = p.parse_args(argv)
    # (rl.alphazero.train_iteration / heldout_stats read these; the tactician is not trained there in v2)
    a.value_mix_t, a.min_batch_t = 0.0, 1
    return a


# ---- self-play ----------------------------------------------------------------------------------------------------

def gae(rewards, values, dones, boot, lam, gamma=1.0):
    """GAE over one run's segment: values[t] of each decision, boot the value after the last one (0 if done)."""
    n = len(rewards)
    adv = np.zeros(n, np.float64)
    last = 0.0
    for t in reversed(range(n)):
        nxt = 0.0 if dones[t] else (values[t + 1] if t + 1 < n else boot)
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + (0.0 if dones[t] else gamma * lam * last)
        adv[t] = last
    return adv


def rental_joint_logp(ll, pl):
    """lead logits [6], pair logits [6, 15] -> log pi(lead, pair) [6, 15] (numpy)."""
    def ls(z):
        z = np.asarray(z, np.float64)
        m = z.max(-1, keepdims=True)
        return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))
    return ls(ll)[:, None] + ls(pl)


class SelfPlayerV2(AZ.SelfPlayer):
    """One self-play worker (AZ.WorkerPool with player=SelfPlayerV2)."""

    def __init__(self, cfg, seed, evaluator=None):
        self.cfg = cfg
        encode.set_version(cfg["encode_version"])
        self.net = FactoryNet(cfg["d_emb"], cfg["d"], cfg["layers"], cfg["heads"], share="embeddings",
                              opt_feat=N_FEAT).eval()
        self.norms = {"battler": None, "tactician": None}
        self._local = evaluator is None
        self._local_eval = AZ.local_battler_evaluator(self.net, None)
        self.evaluator = evaluator if evaluator is not None else (lambda b: self._local_eval(b))
        self.battler = GumbelBattler(self.evaluator, cfg["sims"], cfg["dets"], cfg["search_batch"], seed=seed,
                                     max_considered=cfg["max_considered"], c_visit=cfg["c_visit"],
                                     c_scale=cfg["c_scale"], rescale=bool(cfg["gumbel_rescale"]),
                                     non_root=cfg["non_root"], c_puct=cfg["c_puct"],
                                     opponent_prior=cfg["opponent_prior"])
        self.features = TeamFeatures(cfg["team_eval"], cfg["team_eval_module"], shaping=bool(cfg["t_shaping"]))
        self.rng = random.Random(seed ^ 0xA2)
        self.np_rng = np.random.default_rng(seed + 2)
        self.env = AZ.CurriculumEnv(seed, max_decisions=cfg["max_decisions"])
        self.ev = None
        self.out_b, self.out_t, self.v_samples = [], [], []
        self.traj = []
        self.stats = self._new_stats()

    def set_v_next(self, table):
        self.features.set_v_next(table)

    @torch.no_grad()
    def t_act(self, kind, obs):
        """Sample from pi_net. -> (action index: (lead, pair) or int, log pi, value in return units)."""
        x = encode.collate([obs], "cpu")
        if kind == "rental":
            ll, pl, v = self.net.rental_joint(x)
            lp = rental_joint_logp(ll[0].numpy(), pl[0].numpy())
            legal = obs["pair_mask"] & obs["lead_mask"][:, None]
            p = np.where(legal, np.exp(lp), 0.0).ravel()
            i = int(self.np_rng.choice(p.size, p=p / p.sum()))
            a = (i // 15, i % 15)
            logp = float(lp.ravel()[i])
        else:
            lg, v = self.net.swap(x)
            z = lg[0].numpy().astype(np.float64)
            z = np.where(obs["mask"], z, -np.inf)
            lp = z - z.max() - np.log(np.exp(z - z.max()).sum())
            p = np.exp(lp)
            a = int(self.np_rng.choice(10, p=p / p.sum()))
            logp = float(lp[a])
        return a, logp, float(self._denorm_t(v.numpy().astype(np.float64))[0])

    def _new_run(self):
        super()._new_run()
        self.traj = []
        self.round_starts = []

    def _finish_battle(self, z):
        for obs, pi, q in self.pending_b:
            self.out_b.append((obs, pi, float(z), q, self.battle_held))
        self._new_battle()

    def play(self, n_decisions):
        t0 = time.perf_counter()
        if self.ev is None:
            self.ev = self.env.reset()
            self._new_run()
            self._new_battle()
        made = 0
        while made < n_decisions:
            ev, env = self.ev, self.env
            kind = ev["kind"]
            if kind == "battle":
                r = self.battler.search_root(env.backend, env.decisions - 1, ev["obs"], noise=True)
                a = r["action"]
                self.pending_b.append((ev["obs"], r["target"].astype(np.float32), r["root_q"]))
                self._battle_stats(r, a)
                action = decode_action("battle", a)
                made += 1
            else:
                action = self._tactician(kind, ev["obs"])
            self.ev = env.step(action)
            self._after(self.ev)
        # the run in progress: its closed decisions are trained now; the open one only bootstraps them and is
        # never trained (its log-prob belongs to this iteration's snapshot)
        if self.traj:
            open_t = self.traj[-1]
            self._emit(self.traj[:-1], boot=open_t["value"])
            open_t["stale"] = True
            self.traj = [open_t]
        st = self.stats
        st["seconds"] = time.perf_counter() - t0
        st["battler_decisions"] = made
        out = {"battle": _stack_b(self.out_b), "rental": _stack_t(self.out_t, "rental"),
               "swap": _stack_t(self.out_t, "swap"), "v_next": list(self.v_samples)}
        self.out_b, self.out_t, self.v_samples = [], [], []
        self.stats = self._new_stats()
        return out, dict(st)

    def _tactician(self, kind, obs):
        env = self.env
        ctx = env._ctx()
        t0 = time.perf_counter()
        fe = self.features.compute(env.backend, kind, obs, ctx)
        obs = dict(obs, opt_feat=fe["opt_feat"])
        a, logp, value = self.t_act(kind, obs)
        opts = fe["options"]
        if self.features.enabled and len(opts):
            pi_eval = to_action_space(kind, opts, softmax(fe["q_eval"], self.cfg["t_eval_temp"]))
            best = opts[int(np.argmax(fe["q_eval"]))]
            self.stats["t_net_is_eval_argmax"] += float(tuple(np.atleast_1d(best)) == tuple(np.atleast_1d(a)))
        else:
            pi_eval = np.zeros((6, 15) if kind == "rental" else (10,), np.float32)
        phi = fe["phi"]
        if self.traj:                                   # close the previous decision of this run
            prev = self.traj[-1]
            prev["reward"] = (self.run_wins - prev["wins_before"]) + phi - prev["phi"]
            prev["done"] = False
        self.traj.append({"kind": kind, "obs": obs, "action": a, "logp": logp, "value": value, "phi": phi,
                          "pi_eval": pi_eval, "wins_before": self.run_wins, "stale": False, "held": self.run_held})
        if kind == "rental":
            self.round_starts.append((int(ctx["challenge"]) + 1, self.run_wins))
        st = self.stats
        st[f"t_{kind}_decisions"] += 1
        st["t_decisions"] += 1
        st["t_ms"] += (time.perf_counter() - t0) * 1000
        st["t_value_net_sum"] += value
        st["t_phi_sum"] += phi
        if kind == "swap":
            st["t_swaps"] += float(a != 0)
        return decode_action(kind, a)

    def _emit(self, closed, boot):
        """GAE over a run's closed decisions (boot: the value after the last one unless it is done)."""
        if not closed:
            return
        adv = gae([t["reward"] for t in closed], [t["value"] for t in closed], [t["done"] for t in closed], boot,
                  self.cfg["t_gae_lambda"])
        for t, a in zip(closed, adv):
            if t["stale"]:
                continue
            self.out_t.append(dict(t, adv=float(a), ret=float(a + t["value"])))

    def _end_run(self, extra=0.0, truncated=False):
        if self.traj:
            last = self.traj[-1]
            last["reward"] = (self.run_wins - last["wins_before"]) - last["phi"] + extra
            last["done"] = True
            self._emit(self.traj, 0.0)
        self.traj = []
        if not truncated:
            for rnd, w0 in self.round_starts:
                self.v_samples.append((rnd, self.run_wins - w0))

    def _after(self, ev):
        st = ev["stats"]
        b = st.get("battle")
        if b is not None and b["won"] is not None:
            self._finish_battle(float(b["won"]))
            self.run_wins += int(b["won"])
            self.stats["battles"] += 1
            self.stats["battles_won"] += float(b["won"])
        if ev["truncate"]:
            # the battle hit max_decisions (not a loss): the battler's z is its value at the cut; the run ends
            # here: the tactician's last decision bootstraps with its value (return units: shaped value + Phi)
            _, v = self.evaluator({k: np.stack([ev["obs"][k]]) for k in ev["obs"]})
            self._finish_battle(float(v[0]))
            if self.traj:
                last = self.traj[-1]
                raw = last["value"] + last["phi"] - (self.run_wins - last["wins_before"])
                self._end_run(extra=max(raw, 0.0), truncated=True)
            else:
                self._end_run(truncated=True)
            self.stats["truncated"] += 1
            self.stats["runs"].append((self.env.start_streak, self.run_wins))
            self.ev = self.env.step("reset")
            self._new_run()
            return
        if "run" in st:
            self._end_run()
            self.stats["runs"].append((st["run"]["start"], st["run"]["streak"]))
            self._new_run()

    def _battle_stats(self, r, a):
        super()._battle_stats(r, a)
        st = self.stats
        if r["searched"]:
            t, p, legal = r["target"], r["prior"], r["legal"]
            st["b_target_entropy"] += float(-(t[t > 0] * np.log(t[t > 0])).sum())
            pn = np.where(legal, p, 0.0)
            pn = pn / max(pn.sum(), 1e-12)
            m = t > 0
            st["b_kl_target_prior"] += float((t[m] * (np.log(t[m]) - np.log(np.maximum(pn[m], 1e-12)))).sum())
            st["b_winner_ne_prior"] += float(a != int(np.argmax(np.where(legal, p, -1.0))))
            st["b_winner_ne_target_argmax"] += float(a != int(np.argmax(t)))
            st["b_considered"] += float(np.sum(r.get("considered", legal)))


def _stack_b(samples):
    return AZ._stack(samples, "battle") if samples else None


def _stack_t(samples, kind):
    s = [t for t in samples if t["kind"] == kind]
    if not s:
        return None
    obs = {k: np.stack([t["obs"][k] for t in s]) for k in s[0]["obs"]}
    act = np.array([t["action"] for t in s], np.int64)
    return {"obs": obs, "action": act, "logp": np.array([t["logp"] for t in s], np.float32),
            "value": np.array([t["value"] for t in s], np.float32),
            "adv": np.array([t["adv"] for t in s], np.float32), "ret": np.array([t["ret"] for t in s], np.float32),
            "reward": np.array([t["reward"] for t in s], np.float32), "phi": np.array([t["phi"] for t in s],
                                                                                        np.float32),
            "pi_eval": np.stack([t["pi_eval"] for t in s]).astype(np.float32),
            "held": np.array([t["held"] for t in s], bool)}


def merge_t(results):
    out = {}
    for kind in T_KINDS:
        parts = [r[kind] for r in results if r.get(kind) is not None]
        if not parts:
            out[kind] = None
            continue
        out[kind] = {"obs": {k: np.concatenate([p["obs"][k] for p in parts]) for k in parts[0]["obs"]},
                     **{f: np.concatenate([p[f] for p in parts]) for f in parts[0] if f != "obs"}}
    return out


# ---- the v_next table -------------------------------------------------------------------------------------------

class VNext:
    """table[r]: mean real wins from the start of round r (to the end of the run), a running mean over iterations."""

    def __init__(self, rate=0.3, n=V_NEXT_ROUNDS):
        self.rate = rate
        self.table = np.zeros(n)
        self.seen = np.zeros(n, bool)

    def update(self, samples):
        by = collections.defaultdict(list)
        for r, w in samples:
            by[min(int(r), len(self.table) - 1)].append(float(w))
        for r, ws in by.items():
            m = float(np.mean(ws))
            self.table[r] = m if not self.seen[r] else self.table[r] + self.rate * (m - self.table[r])
            self.seen[r] = True
        return {r: len(ws) for r, ws in by.items()}

    def state(self):
        return {"table": self.table.tolist(), "seen": self.seen.tolist(), "rate": self.rate}

    def load(self, st):
        self.table = np.asarray(st["table"], float)
        self.seen = np.asarray(st["seen"], bool)


# ---- the tactician's PPO ----------------------------------------------------------------------------------------

def tactician_terms(net, kind, x, action):
    """-> (log pi(action) [B], entropy [B], log pi over the flat action space [B, n] (-1e9 illegal), value [B])."""
    if kind == "rental":
        ll, pl, v = net.rental_joint(x)
        logq = (F.log_softmax(ll, -1)[:, :, None] + F.log_softmax(pl, -1)).flatten(1)
        legal = (x["pair_mask"] & x["lead_mask"][:, :, None]).flatten(1)
        idx = action[:, 0] * 15 + action[:, 1]
    else:
        logits, v = net.swap(x)
        logq = F.log_softmax(logits, -1)
        legal = x["mask"]
        idx = action
    logq = logq.masked_fill(~legal, -1e9)
    logp = logq.gather(1, idx[:, None])[:, 0]
    q = logq.exp()
    ent = -(q * logq.clamp_min(-1e4)).sum(-1)
    return logp, ent, logq, v


def tactician_ppo(net, opt, data, norm, args, device, rng):
    """PPO epochs on this iteration's tactician samples (rentals and swaps mixed in each minibatch)."""
    parts = [(k, np.flatnonzero(~data[k]["held"])) for k in T_KINDS if data.get(k) is not None]
    parts = [(k, i) for k, i in parts if len(i)]
    if not parts:
        return {}
    all_ret = np.concatenate([data[k]["ret"][i] for k, i in parts])
    all_adv = np.concatenate([data[k]["adv"][i] for k, i in parts])
    norm.update(all_ret)
    items = [(k, j) for k, i in parts for j in i]
    n = len(items)
    kinds = np.array([k for k, _ in items])
    rows = np.array([j for _, j in items])
    stats = collections.defaultdict(list)
    net.train()
    n_mb = max(1, math.ceil(n / args.t_minibatch))
    for _ in range(args.t_ppo_epochs):
        perm = rng.permutation(n)
        for mb in np.array_split(perm, n_mb):
            logp_l, ent_l, v_l, ce_l, old_l, adv_l, ret_l = [], [], [], [], [], [], []
            for kind in T_KINDS:
                sel = rows[mb[kinds[mb] == kind]]
                if not len(sel):
                    continue
                d = data[kind]
                x = AZ.to_device(d["obs"], sel, device)
                act = torch.from_numpy(d["action"][sel]).to(device)
                lp, ent, logq, v = tactician_terms(net, kind, x, act)
                pe = torch.from_numpy(d["pi_eval"][sel].reshape(len(sel), -1)).to(device)
                ce_l.append(-(pe * logq).sum(-1))
                logp_l.append(lp); ent_l.append(ent); v_l.append(v)
                old_l.append(torch.from_numpy(d["logp"][sel]).to(device))
                adv_l.append(torch.from_numpy(d["adv"][sel]).to(device))
                ret_l.append(torch.from_numpy(((d["ret"][sel] - norm.mean) / norm.std).astype(np.float32)).to(device))
            logp, ent, v, ce = torch.cat(logp_l), torch.cat(ent_l), torch.cat(v_l), torch.cat(ce_l)
            old, adv, ret = torch.cat(old_l), torch.cat(adv_l), torch.cat(ret_l)
            a = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv * 0
            ratio = torch.exp(logp - old)
            pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.t_clip, 1 + args.t_clip) * a).mean()
            vl = 0.5 * ((v - ret) ** 2).mean()
            loss = pl + args.t_vf_coef * vl - args.t_ent * ent.mean() + args.t_imitation * ce.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()
            with torch.no_grad():
                lr = logp - old
                stats["approx_kl"].append(((torch.exp(lr) - 1) - lr).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > args.t_clip).float().mean().item())
            stats["policy_loss"].append(pl.item())
            stats["value_loss"].append(vl.item())
            stats["entropy"].append(ent.mean().item())
            stats["imitation_ce"].append(ce.mean().item())
            stats["grad_norm"].append(float(gn))
    net.eval()
    out = {f"tactician/{k}": float(np.mean(v)) for k, v in stats.items()}
    old_v = np.concatenate([data[k]["value"][i] for k, i in parts])
    out["tactician/explained_variance_train"] = AZ.explained_variance(old_v, all_ret)
    out["tactician/advantage_std"] = float(all_adv.std())
    out["tactician/advantage_mean"] = float(all_adv.mean())
    out["tactician/return_mean"] = float(all_ret.mean())
    out["tactician/samples"] = n
    out["tactician/ppo_steps"] = n_mb * args.t_ppo_epochs
    for k, i in parts:
        out[f"tactician_{k}/samples"] = len(i)
    return out


@torch.no_grad()
def tactician_heldout(net, data, norm, device):
    """Critic generalization on held-out runs: explained variance of the value (after the update) vs the return."""
    out = {}
    preds, rets = [], []
    for kind in T_KINDS:
        d = data.get(kind)
        if d is None:
            continue
        idx = np.flatnonzero(d["held"])
        if not len(idx):
            continue
        for chunk in np.array_split(idx, max(1, len(idx) // 2048)):
            x = AZ.to_device(d["obs"], chunk, device)
            _, _, _, v = tactician_terms(net, kind, x, torch.from_numpy(d["action"][chunk]).to(device))
            preds.append(norm.denorm(v.float().cpu().numpy()))
            rets.append(d["ret"][chunk])
    if preds and sum(len(p) for p in preds) >= 8:
        out["tactician/explained_variance_heldout"] = AZ.explained_variance(np.concatenate(preds),
                                                                            np.concatenate(rets))
        out["tactician/heldout_samples"] = int(sum(len(p) for p in preds))
    return out


# ---- main -------------------------------------------------------------------------------------------------------

def worker_cfg(args):
    keys = ("encode_version", "d_emb", "d", "layers", "heads", "sims", "dets", "c_puct", "search_batch",
            "max_considered", "c_visit", "c_scale", "gumbel_rescale", "non_root", "max_decisions", "holdout",
            "opponent_prior", "team_eval", "team_eval_module", "t_shaping", "t_eval_temp", "t_gae_lambda")
    return {k: getattr(args, k) for k in keys}


def eval_hook(args, v_next):
    """The evaluation environments' obs_hook: the same per-option features as in self-play (no shaping)."""
    return TeamFeatures(args.team_eval, args.team_eval_module, v_next=v_next, shaping=False)


def main(argv=None):
    from .search import mark_training
    mark_training()     # no perfect-information search can exist in a training process (rl/search.py)
    args = parse(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = args.run_dir or os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    config = dict(vars(args), algo=AZ.ALGO, version=VERSION, opt_feat=N_FEAT, share="embeddings", gamma_b=1.0,
                  beta=0.0)
    json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(run_dir)
    tb.add_text("config", "```\n" + json.dumps(config, indent=2) + "\n```")

    encode.set_version(args.encode_version)
    net = FactoryNet(args.d_emb, args.d, args.layers, args.heads, share="embeddings", opt_feat=N_FEAT).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-5)
    norms = {a: ValueNorm(args.value_norm_rate) for a in ("battler", "tactician")}
    vnext = VNext(args.v_next_rate)
    it0, b_steps, t_steps = 0, 0, 0
    round_eval = {}
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        net.load_state_dict(ck["net"])
        if "opt" in ck and ck.get("args", {}).get("version") == VERSION:
            opt.load_state_dict(ck["opt"])
        for a, st in (ck.get("value_norm") or {}).items():
            if st and a in norms:
                norms[a].load(st)
        if ck.get("v_next"):
            vnext.load(ck["v_next"])
        it0 = ck.get("iteration", -1) + 1
        b_steps, t_steps = ck.get("battler_steps", 0), ck.get("tactician_steps", 0)
        round_eval = {int(k): v for k, v in (ck.get("round_eval") or {}).items()}
        print(f"resumed from {args.init_from}: iteration {it0}, {b_steps:,} battler steps", flush=True)
    tb.add_scalar("model/parameters", sum(p.numel() for p in net.parameters()), b_steps)
    net.eval()
    snap_path = os.path.join(run_dir, "snapshot.pt")

    def ckpt_dict(it):
        return {"net": net.state_dict(), "opt": opt.state_dict(), "args": config, "battler_steps": b_steps,
                "tactician_steps": t_steps, "iteration": it, "round_eval": round_eval, "v_next": vnext.state(),
                "value_norm": {k: (v.state() if v.seen else None) for k, v in norms.items()}}

    def write_snapshot():
        AZ.save_atomic({"net": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "args": config,
                        "value_norm": {k: (v.state() if v.seen else None) for k, v in norms.items()},
                        "v_next": vnext.state()}, snap_path)

    write_snapshot()
    server = None
    eval_slots = args.eval_search_procs if args.eval_search_every > 0 and args.eval_search_sims > 0 else 0
    if args.inference == "gpu":
        from .inference import InferenceServer
        server = InferenceServer(snap_path, n_clients=args.workers + eval_slots)
    pool = AZ.WorkerPool(args.workers, worker_cfg(args), server, args.seed, player=SelfPlayerV2)
    replay = AZ.Replay(args.window)
    rng = np.random.default_rng(args.seed + 17)
    t_start = time.time()
    try:
        for it in range(it0, args.iterations):
            t_it = time.time()
            lr = AZ.cosine_lr(args, it)
            for g in opt.param_groups:
                g["lr"] = lr
            weights = AZ.start_weights_from_eval(round_eval, args.start_p0)
            pool.broadcast("load", snap_path)
            pool.broadcast("weights", weights.tolist())
            pool.broadcast("call", ("set_v_next", (vnext.table.tolist(),)))

            # ---- 1. self-play ---------------------------------------------------------------------------------
            per_worker = math.ceil(args.decisions_per_iter / args.workers)
            srv0 = server.stats() if server is not None else None
            t0 = time.time()
            results = pool.broadcast("play", per_worker)
            t_play = time.time() - t0
            stats = [s for _, s in results]
            made = sum(s["battler_decisions"] for s in stats)
            if server is not None:
                srv1 = server.stats()
                nb = max(srv1["batches"] - srv0["batches"], 1)
                tb.add_scalar("perf/server_mean_batch", (srv1["samples"] - srv0["samples"]) / nb, b_steps + made)
                tb.add_scalar("perf/server_forward_ms", (srv1["forward_ms"] * srv1["batches"]
                                                         - srv0["forward_ms"] * srv0["batches"]) / nb, b_steps + made)
            bdata = AZ.merge([{"battle": r["battle"]} for r, _ in results])
            tdata = merge_t([r for r, _ in results])
            v_samples = [x for r, _ in results for x in r["v_next"]]
            new_counts = {"battle": int((~bdata["battle"]["held"]).sum()) if bdata["battle"] is not None else 0}
            b_steps += made
            t_steps += int(sum(s.get("t_decisions", 0) for s in stats))
            replay.add(bdata)
            vcount = vnext.update(v_samples)
            log_selfplay_v2(tb, stats, bdata, tdata, made, t_play, weights, b_steps, vnext, vcount)

            # ---- 2. training ----------------------------------------------------------------------------------
            t0 = time.time()
            tr = AZ.train_iteration(net, opt, replay, new_counts, norms, args, device, rng)
            tr.update(AZ.heldout_stats(net, replay, norms, args, device))
            tr.update(tactician_ppo(net, opt, tdata, norms["tactician"], args, device, rng))
            tr.update(tactician_heldout(net, tdata, norms["tactician"], device))
            t_train = time.time() - t0
            for k, v in tr.items():
                tb.add_scalar(k.replace("battle/", "battler/"), v, b_steps)
            tb.add_scalar("train/lr", lr, b_steps)
            tb.add_scalar("train/iteration", it, b_steps)
            for a, nm in norms.items():
                tb.add_scalar(f"{a}/value_norm_mean", nm.mean, b_steps)
                tb.add_scalar(f"{a}/value_norm_std", nm.std, b_steps)

            # ---- 3. checkpoint, new snapshot, evaluation -----------------------------------------------------
            write_snapshot()
            if server is not None:
                server.reload(snap_path)
            AZ.save_atomic(ckpt_dict(it), os.path.join(run_dir, f"ckpt_{b_steps:011d}.pt"))
            t0 = time.time()
            if args.eval_every and (it + 1) % args.eval_every == 0 and args.eval_runs > 0:
                with_search = bool(args.eval_search_every and (it + 1) % args.eval_search_every == 0
                                   and args.eval_search_sims > 0 and server is not None)
                round_eval = evaluate_rounds_v2(net, device, args, tb, b_steps, server, snap_path,
                                                vnext.table.tolist(), with_search)
            t_eval = time.time() - t0
            AZ.save_atomic(ckpt_dict(it), os.path.join(run_dir, "latest.pt"))
            tb.add_scalar("perf/selfplay_seconds", t_play, b_steps)
            tb.add_scalar("perf/train_seconds", t_train, b_steps)
            tb.add_scalar("perf/eval_seconds", t_eval, b_steps)
            tb.add_scalar("perf/iteration_seconds", time.time() - t_it, b_steps)
            tb.add_scalar("perf/hours", (time.time() - t_start) / 3600, b_steps)
            if torch.cuda.is_available() and device.type == "cuda":
                tb.add_scalar("perf/gpu_mem_peak_mb", torch.cuda.max_memory_allocated() / 2 ** 20, b_steps)
            tb.flush()
            won = sum(s.get("battles_won", 0) for s in stats) / max(sum(s.get("battles", 0) for s in stats), 1)
            print(f"[it {it:4d} | {(time.time() - t_start) / 60:7.1f} min] {b_steps:,} battler steps  "
                  f"{made / max(t_play, 1e-9):6.0f} dec/s  win {won:.3f}  "
                  f"CE b {tr.get('battle/policy_loss', float('nan')):.3f}  "
                  f"EV b {tr.get('battle/explained_variance_train', float('nan')):.2f}/"
                  f"{tr.get('battle/explained_variance_heldout', float('nan')):.2f}  "
                  f"t kl {tr.get('tactician/approx_kl', float('nan')):.4f} "
                  f"EV t {tr.get('tactician/explained_variance_train', float('nan')):.2f}  "
                  f"(play {t_play:.0f}s train {t_train:.0f}s eval {t_eval:.0f}s)"
                  + ("  rounds " + " ".join(f"{round_eval[k]:.2f}" for k in sorted(round_eval)) if round_eval else ""),
                  flush=True)
    finally:
        pool.close()
        if server is not None:
            server.close()
        tb.close()
    return run_dir


def log_selfplay_v2(tb, stats, bdata, tdata, made, t_play, weights, step, vnext, vcount):
    """v1's self-play scalars that still apply, plus the Gumbel search's and the tactician's."""
    AZ.log_selfplay(tb, stats, {"battle": bdata["battle"], **tdata}, made, t_play, weights, step, tsearch=False)
    tot = collections.defaultdict(float)
    for s in stats:
        for k, v in s.items():
            if isinstance(v, (int, float)):
                tot[k] += v
    ns = max(tot["b_searched"], 1)
    tb.add_scalar("gumbel/target_entropy", tot["b_target_entropy"] / ns, step)
    tb.add_scalar("gumbel/kl_target_prior", tot["b_kl_target_prior"] / ns, step)
    tb.add_scalar("gumbel/winner_ne_prior_argmax", tot["b_winner_ne_prior"] / ns, step)
    tb.add_scalar("gumbel/winner_ne_target_argmax", tot["b_winner_ne_target_argmax"] / ns, step)
    tb.add_scalar("gumbel/considered", tot["b_considered"] / ns, step)
    nt = max(tot["t_decisions"], 1)
    tb.add_scalar("tactician/ms_per_decision", tot["t_ms"] / nt, step)
    tb.add_scalar("tactician/value_net", tot["t_value_net_sum"] / nt, step)
    tb.add_scalar("tactician/phi_mean", tot["t_phi_sum"] / nt, step)
    tb.add_scalar("tactician/net_is_eval_argmax", tot["t_net_is_eval_argmax"] / nt, step)
    rs = [d["reward"] for d in tdata.values() if d is not None]
    if rs:
        r = np.concatenate(rs)
        tb.add_scalar("tactician/shaped_reward_mean", float(r.mean()), step)
        tb.add_scalar("tactician/shaped_reward_std", float(r.std()), step)
    for r in range(1, len(vnext.table)):
        if vnext.seen[r]:
            tb.add_scalar(f"v_next/round_{r}", float(vnext.table[r]), step)
            tb.add_scalar(f"v_next/samples_round_{r}", vcount.get(r, 0), step)


def evaluate_rounds_v2(net, device, args, tb, step, server, snap_path, v_next, with_search):
    """Per-round evaluation, network only (greedy), with the tactician's features; optionally network + legal PUCT
    search (--eval-search-sims, the strict sampler: never factory_sets outside training's self-play)."""
    from .eval_rounds import eval_round, eval_round_inprocess
    hook = eval_hook(args, v_next)
    res = {}
    policy = AZ.NetPolicy(net)
    for k in AZ._rounds(args.eval_rounds):
        r = eval_round(policy, device, k, args.eval_runs, workers=args.eval_workers,
                       envs_per_worker=args.eval_envs_per_worker, obs_hook=hook)
        tb.add_scalar(f"eval_round_{k}/complete", r["complete"], step)
        tb.add_scalar(f"eval_round_{k}/battle_win_rate", r["battle"], step)
        res[k] = r["complete"]
    if with_search:
        kw = {"n_sims": args.eval_search_sims, "n_determinizations": args.dets, "c_puct": args.c_puct,
              "batch": args.search_batch, "mode": "legal", "seed": 0,
              "opponent_prior": "strict"}   # evaluation: the strict sampler, never the training one
        prod = 1.0
        for k in AZ._rounds(args.eval_rounds):
            r = eval_round_inprocess(snap_path, k, args.eval_search_runs, "search", kw,
                                     procs=args.eval_search_procs, server=server, slot0=args.workers, obs_hook=hook)
            tb.add_scalar(f"eval_search_round_{k}/complete", r["complete"], step)
            tb.add_scalar(f"eval_search_round_{k}/battle_win_rate", r["battle"], step)
            tb.add_scalar(f"eval_search_round_{k}/ms_per_decision", r["ms_per_decision"], step)
            prod *= r["complete"]
        tb.add_scalar("eval_search/product_complete", prod, step)
    if res:
        tb.add_scalar("eval/product_complete", float(np.prod(list(res.values()))), step)
    return res


if __name__ == "__main__":
    main()
