"""Expert iteration from scratch for both agents (alphazero_v1, docs/RL_DECISIONS.md §18).

    python -m rl.alphazero --name alphazero_v1
    tensorboard --logdir runs

Each iteration:
  1. Self-play with a frozen snapshot of the network: --workers processes play Factory runs; the battler network
     runs in one GPU inference server shared by all of them (rl/inference.py: search leaves and the simulated
     battles' plain evaluations), the tactician network on each worker's CPU copy.
       battler     the C++ ensemble MCTS in legal mode only (rl/search.py; K determinizations, every root's turn
                   redrawn; the opponent drawn with --opponent-prior: factory_sets = the Factory's set list, the
                   training-only sampler, IVs / EVs / nature random; strict = player knowledge, the evaluation /
                   inference sampler), with Dirichlet noise on the root priors; the action is
                   sampled from the root visits with temperature --temperature. Targets: policy = root visit
                   distribution; value = (1 - m) z + m q, z = the battle's outcome (1 / 0), q = the search's root value
                   (visit-weighted mean Q), m = --value-mix-b.
       tactician   search by simulation (rl/tactician_search.py): every rental / swap option is valued by battles
                   against opponents drawn with --opponent-prior, played by the frozen battler network (greedy, no
                   search), with Gumbel top-m + sequential halving. Targets: policy = the search policy (visit
                   distribution, or softmax of the option values); value = (1 - m) z + m q, z = battles won from the
                   decision until the run ends (gamma 1), q = the chosen option's value, m = --value-mix-t.
     A battle cut at --max-decisions (truncation, not a loss) ends the run as in FactoryEnv: the battler's z is its
     network value at the cut, the tactician's z is bootstrapped with its last value (as rl/train.py).
     Runs start at round 1 with probability --start-p0, else at round k = 2..6 with weights proportional to
     1 - P(complete round k) from the latest per-round evaluation (uniform before the first); SimBackend gives
     the symbols a player with that streak holds (silver from 21, gold from 42).
     Runs continue across iterations (a sample is finished when its battle / run ends).
  2. Training on a replay buffer of the last --window iterations: each new sample is used --reuse times on
     average; cross-entropy to the search policy + value MSE on ValueNorm-normalized targets + weight decay (AdamW),
     cosine learning rate over --iterations. The rental policy is trained with the joint target: lead marginal +
     pair conditional (-sum pi(l, p) [log q(l) + log q(p | l)]). --holdout of the battles (battler) and of the runs
     (tactician) are never trained on: they measure the critics' generalization (explained variance).
  3. Evaluation (per round, network only, greedy: eval_round_k/*; optionally with search: eval_search_round_k/*),
     checkpoint (latest.pt and ckpt_<battler steps>.pt, atomic, with the optimizer and the value normalization,
     args["algo"] = "alphazero"): rl.policy / rl.evaluate / rl.eval_rounds / rl.full_report read them like PPO's.

Perfect-information search is impossible here: mark_training() runs first, the battler search is built in legal
mode only (no mode argument), and the C++ Searcher refuses roots that are not full determinizations.
"""

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import random
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from pybattle.backend import Phase
from . import encode
from .envs import FactoryEnv, decode_action
from .model import FactoryNet
from .ppo import ValueNorm, act as net_act
from .search import SearchBattler, run_ctx
from .tactician_search import TacticianSearch, option_logp, options_of, policy_to_action_space

ALGO = "alphazero"
KINDS = ("battle", "rental", "swap")
PI_SHAPE = {"battle": (7,), "rental": (6, 15), "swap": (10,)}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", default=time.strftime("alphazero_%Y%m%d_%H%M%S"))
    p.add_argument("--run-dir", default=None, help="default runs/<name>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=200, help="planned iterations (cosine learning rate)")
    p.add_argument("--decisions-per-iter", type=int, default=40_000, help="self-play battler decisions per iteration")
    p.add_argument("--workers", type=int, default=18)
    p.add_argument("--inference", choices=("gpu", "cpu"), default="gpu",
                   help="gpu: the battler network in one GPU inference server; cpu: a copy in every worker (tests)")
    p.add_argument("--device", default=None, help="training device (default cuda if available)")
    # battler search
    p.add_argument("--sims", type=int, default=128)
    p.add_argument("--dets", type=int, default=8, help="determinizations (trees) per decision")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--search-batch", type=int, default=32)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-frac", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=1.0, help="self-play action ~ visits^(1/T); 0: argmax")
    p.add_argument("--max-decisions", type=int, default=300, help="battler decisions per battle before truncation")
    p.add_argument("--opponent-prior", choices=("factory_sets", "strict"), default="factory_sets",
                   help="opponent sampler of both searches: the Factory's set list (training only; IVs / EVs / "
                        "nature random) or strict player knowledge (the evaluation / inference sampler)")
    # tactician search
    p.add_argument("--t-budget", type=int, default=256, help="simulated battles per tactician decision")
    p.add_argument("--t-considered", type=int, default=16, help="options kept by Gumbel top-m")
    p.add_argument("--t-bootstrap", type=int, default=1,
                   help="value of a won simulated battle: 1 + V_t(next decision) (1) or 1 (0)")
    p.add_argument("--t-target", choices=("visits", "softmax"), default="visits")
    p.add_argument("--t-target-temp", type=float, default=0.5, help="--t-target softmax: temperature (battles)")
    p.add_argument("--t-local-batch", type=int, default=8,
                   help="simulated-battle steps with at most this many battles use the worker's CPU copy of the "
                        "battler network instead of the GPU server (a round trip costs more than a small forward)")
    p.add_argument("--t-max-decisions", type=int, default=100,
                   help="simulated battles are cut after this many decisions (value: the battler network's)")
    # targets
    p.add_argument("--value-mix-b", type=float, default=0.5, help="battler value target: (1-m) z + m q_search")
    p.add_argument("--value-mix-t", type=float, default=0.5, help="tactician value target: (1-m) z + m q_search")
    # training
    p.add_argument("--window", type=int, default=4, help="replay buffer: the last N iterations")
    p.add_argument("--reuse", type=float, default=4.0, help="times each sample is used on average")
    p.add_argument("--batch-b", type=int, default=512, help="battler samples per gradient step")
    p.add_argument("--min-batch-t", type=int, default=16,
                   help="tactician samples (rentals + swaps) per gradient step, at least")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min-frac", type=float, default=0.05, help="cosine schedule floor, as a fraction of --lr")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--value-norm-rate", type=float, default=0.3, help="ValueNorm update rate, once per iteration")
    p.add_argument("--holdout", type=float, default=0.05, help="fraction of battles / runs kept out of training")
    # network
    p.add_argument("--d-emb", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--encode-version", type=int, default=4,
                   help="observation layout (rl/encode.py): 4 = no opponent IVs in the estimates + defeated-foe "
                        "records at swaps (docs §17); 3 = ppo_joint_v3's")
    # curriculum
    p.add_argument("--start-p0", type=float, default=0.3, help="probability a run starts at round 1")
    # evaluation
    p.add_argument("--eval-every", type=int, default=1, help="iterations between per-round evaluations (0: never)")
    p.add_argument("--eval-runs", type=int, default=192, help="runs per round, network only (greedy)")
    p.add_argument("--eval-rounds", default="1-6")
    p.add_argument("--eval-workers", type=int, default=16)
    p.add_argument("--eval-envs-per-worker", type=int, default=4)
    p.add_argument("--eval-search-sims", type=int, default=0, help="also evaluate with search at this many sims")
    p.add_argument("--eval-search-runs", type=int, default=64)
    p.add_argument("--eval-search-procs", type=int, default=12)
    p.add_argument("--init-from", default=None, help="checkpoint to resume from (network, optimizer, norms, state)")
    return p.parse_args(argv)


# ---- the battler's search -----------------------------------------------------------------------------------------

class AZBattler(SearchBattler):
    """The legal-mode ensemble MCTS (always: there is no mode argument) with Dirichlet noise on the root priors.
    search_root returns the whole root statistics; the caller picks the action."""

    def __init__(self, evaluator, n_sims=128, n_determinizations=8, c_puct=1.5, batch=32, seed=0,
                 dirichlet_alpha=0.3, dirichlet_frac=0.25, observer="auto", opponent_prior="strict"):
        if observer == "auto" and encode.VERSION not in (3, 4):
            observer = "python"                       # the C++ observer encodes versions 3 and 4
        super().__init__(None, n_sims=n_sims, n_determinizations=n_determinizations, c_puct=c_puct, batch=batch,
                         mode="legal", seed=seed, impl="cpp", observer=observer, evaluator=evaluator,
                         opponent_prior=opponent_prior)
        self.alpha, self.frac = dirichlet_alpha, dirichlet_frac
        self.np_rng = np.random.default_rng(seed)

    def search_root(self, backend, decisions=0, root_x=None, noise=True):
        t0 = time.perf_counter()
        self._net_calls, self._eval_ms = 0, 0.0
        if backend.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
            raise RuntimeError(f"not at a battle decision: {backend.phase}")
        forced = backend.phase == Phase.FORCED_SWITCH
        view = backend.view()
        ctx = run_ctx(backend)
        if root_x is None:
            root_x = encode.battle(view, ctx)
        legal = root_x["mask"].astype(bool)
        pri, val = self.evaluate([root_x])
        prior, value = pri[0].astype(np.float64), float(val[0])
        out = {"prior": prior, "value": value, "legal": legal}
        if legal.sum() <= 1:
            a = int(np.flatnonzero(legal)[0]) if legal.any() else 0
            out.update(visits=np.eye(7)[a], q=np.zeros(7), root_q=float("nan"), searched=False, leaves=0,
                       max_depth=0, mean_depth=0.0, ms=(time.perf_counter() - t0) * 1000)
            return out
        p = prior.copy()
        if noise and self.frac > 0:
            eta = self.np_rng.dirichlet([self.alpha] * int(legal.sum()))
            p[legal] = (1 - self.frac) * p[legal] + self.frac * eta
        roots = self._roots(backend, view, None, forced, native_obs=self.observer == "cpp")
        r = self.searcher.search([(g, obs, decisions) for g, obs in roots], ctx, p.tolist(), legal.tolist(), value,
                                 self.evaluate_batch)
        self.errors += r["errors"]
        visits, q = np.asarray(r["visits"], float), np.asarray(r["q"], float)
        visits = np.where(legal, visits, 0.0)
        out.update(visits=visits, q=q, root_q=float((visits * q).sum() / max(visits.sum(), 1e-9)), searched=True,
                   leaves=r["leaves"], max_depth=r.get("max_depth", 0), mean_depth=r.get("mean_depth", 0.0),
                   noisy_prior=p, ms=(time.perf_counter() - t0) * 1000)
        return out


def sample_from_visits(visits, legal, temperature, rng):
    v = np.where(legal, visits, 0.0)
    if temperature <= 0 or v.sum() <= 0:
        return int(np.argmax(np.where(legal, v, -1.0)))
    w = v ** (1.0 / temperature)
    return int(rng.choice(len(w), p=w / w.sum()))


# ---- environments with a curriculum over the starting round -----------------------------------------------------

def start_weights_from_eval(complete, p0=0.3):
    """Starting-round distribution: p0 for round 1, the rest proportional to 1 - P(complete round k) for k = 2..6
    (uniform while there is no evaluation, or if every round is always completed)."""
    w = np.zeros(6)
    w[0] = p0
    hard = np.array([1.0 - float(complete[k]) if complete and k in complete else 1.0 for k in range(2, 7)])
    hard = np.clip(hard, 0.0, 1.0)
    w[1:] = (1 - p0) * (hard / hard.sum() if hard.sum() > 0 else np.full(5, 0.2))
    return w


class CurriculumEnv(FactoryEnv):
    """FactoryEnv whose runs start at the beginning of round k (streak 7(k-1)) with probability weights[k-1],
    with a random feasible rental counter in [k-1, 7(k-1)] (FactoryEnv's rule). No shaping (beta 0)."""

    def __init__(self, seed, weights=None, max_decisions=300):
        self.weights = np.asarray(weights if weights is not None else start_weights_from_eval(None), float)
        super().__init__(seed, gamma=1.0, beta=0.0, max_decisions=max_decisions)

    def _new_run(self):
        seed = self.rng.getrandbits(32)
        w = getattr(self, "weights", None)
        if w is None:
            w = start_weights_from_eval(None)
        k = int(np.searchsorted(np.cumsum(w / w.sum()), self.start_rng.random(), side="right"))
        k = min(k, 5)                                             # round k + 1
        self.start_streak, rents = 7 * k, (self.start_rng.randint(k, 7 * k) if k else 0)
        self.backend.reset(seed=seed, win_streak=self.start_streak, rents_count=rents)
        self.b_pending, self.b_phi = False, 0.0
        self.t_pending, self.t_acc = False, 0
        self.decisions = 0
        self.battle_rewards = 0.0


# ---- self-play ----------------------------------------------------------------------------------------------------------

def local_battler_evaluator(net, norm):
    """The battler network on this process's CPU as a batch evaluator (rl/inference.py interface)."""
    from .inference import battler_outputs

    def evaluate(batch):
        m, s = (norm["mean"], max(norm["var"], 1e-4) ** 0.5) if norm else (0.0, 1.0)
        with torch.inference_mode():
            x = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in batch.items()}
            pri, val = battler_outputs(net, x, m, s)
        return pri.numpy(), val.numpy()
    return evaluate


class SelfPlayer:
    """One self-play worker: a Factory run at a time, both agents searching. Usable in-process (tests) or behind
    a pipe (WorkerPool). evaluator: the battler network as a batch evaluator (a GPU server client), or None for a
    CPU copy of the loaded snapshot."""

    def __init__(self, cfg, seed, evaluator=None):
        self.cfg = cfg
        encode.set_version(cfg["encode_version"])
        self.net = FactoryNet(cfg["d_emb"], cfg["d"], cfg["layers"], cfg["heads"], share="embeddings").eval()
        self.norms = {"battler": None, "tactician": None}
        self._local = evaluator is None
        self.evaluator = evaluator if evaluator is not None else (lambda b: self._local_eval(b))
        self._local_eval = local_battler_evaluator(self.net, None)
        self.battler = AZBattler(self.evaluator, cfg["sims"], cfg["dets"], cfg["c_puct"], cfg["search_batch"],
                                 seed=seed, dirichlet_alpha=cfg["dirichlet_alpha"],
                                 dirichlet_frac=cfg["dirichlet_frac"], opponent_prior=cfg["opponent_prior"])
        self.tsearch = TacticianSearch(self._sim_eval, self.t_values, budget=cfg["t_budget"],
                                       max_considered=cfg["t_considered"], max_decisions=cfg["t_max_decisions"],
                                       bootstrap=bool(cfg["t_bootstrap"]), target=cfg["t_target"],
                                       temperature=cfg["t_target_temp"], seed=seed + 1,
                                       opponent_prior=cfg["opponent_prior"])
        self.rng = random.Random(seed ^ 0xA2)
        self.np_rng = np.random.default_rng(seed + 2)
        self.env = CurriculumEnv(seed, max_decisions=cfg["max_decisions"])
        self.ev = None
        self.out = {k: [] for k in KINDS}                 # finished samples
        self.stats = self._new_stats()

    def _sim_eval(self, batch):
        """The simulated battles' battler evaluations: small batches on this CPU, the rest on the server."""
        if not self._local and len(batch["active"]) <= self.cfg["t_local_batch"]:
            return self._local_eval(batch)
        return self.evaluator(batch)

    # --- the frozen snapshot ---------------------------------------------------------------------------------------

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        self.net.load_state_dict(ck["net"])
        self.norms = dict(ck.get("value_norm") or {})
        self._local_eval = local_battler_evaluator(self.net, self.norms.get("battler"))

    def set_weights(self, w):
        self.env.weights = np.asarray(w, float)

    def _denorm_t(self, v):
        n = self.norms.get("tactician")
        return v * max(n["var"], 1e-4) ** 0.5 + n["mean"] if n else v

    @torch.no_grad()
    def t_net(self, kind, obs_list):
        """-> (logits per sample, values in return units)."""
        x = encode.collate(obs_list, "cpu")
        if kind == "rental":
            ll, pl, v = self.net.rental_joint(x)
            logits = [(a, b) for a, b in zip(ll.numpy(), pl.numpy())]
        else:
            lg, v = self.net.swap(x)
            logits = list(lg.numpy())
        return logits, self._denorm_t(v.numpy().astype(np.float64))

    def t_values(self, kinds, obs_list):
        vals = np.zeros(len(obs_list))
        for kind in set(kinds):
            idx = [i for i, k in enumerate(kinds) if k == kind]
            vals[idx] = self.t_net(kind, [obs_list[i] for i in idx])[1]
        return vals

    # --- playing -----------------------------------------------------------------------------------------------------

    def _new_stats(self):
        return collections.defaultdict(float, runs=[], starts=[])

    def _new_run(self):
        self.run_wins = 0
        self.run_held = self.rng.random() < self.cfg["holdout"]
        self.pending_t = []
        self.last_t_value, self.wins_at_last_t = 0.0, 0
        self.stats["starts"].append(self.env.start_streak // 7 + 1)

    def _new_battle(self):
        self.pending_b = []
        self.battle_held = self.rng.random() < self.cfg["holdout"]

    def play(self, n_decisions):
        """Play until n_decisions battler decisions were made (runs in progress are kept for the next call).
        -> (finished samples by kind, statistics)."""
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
                a = sample_from_visits(r["visits"], r["legal"], self.cfg["temperature"], self.np_rng)
                pi = r["visits"] / r["visits"].sum()
                self.pending_b.append((ev["obs"], pi.astype(np.float32), r["root_q"]))
                self._battle_stats(r, a)
                action = decode_action("battle", a)
                made += 1
            else:
                logits, v = self.t_net(kind, [ev["obs"]])
                opts = options_of(kind, ev["obs"])
                res = self.tsearch.search(env.backend, kind, ev["obs"], option_logp(kind, logits[0], opts),
                                          noise=True)
                pi = policy_to_action_space(kind, res["options"], res["policy"])
                self.pending_t.append((kind, ev["obs"], pi, res["value"], self.run_wins))
                self.last_t_value, self.wins_at_last_t = float(v[0]), self.run_wins
                st, s = self.stats, self.tsearch.last_stats
                st[f"t_{kind}_decisions"] += 1
                for key in ("sims", "ms", "sim_steps", "net_calls", "options", "chosen_is_prior_argmax"):
                    st[f"t_{key}"] += s[key]
                st["t_decisions"] += 1
                if not math.isnan(s["value"]):
                    st["t_value_sum"] += s["value"]
                    st["t_value_n"] += 1
                st["t_value_net_sum"] += float(v[0])
                if kind == "swap":
                    st["t_swaps"] += float(res["action"] != 0)
                action = decode_action(kind, res["action"])
            self.ev = env.step(action)
            self._after(self.ev)
        st = self.stats
        st["seconds"] = time.perf_counter() - t0
        st["battler_decisions"] = made
        out = {k: _stack(v, k) for k, v in self.out.items()}
        self.out = {k: [] for k in KINDS}
        self.stats = self._new_stats()
        return out, dict(st)

    def _battle_stats(self, r, a):
        st = self.stats
        st["b_decisions"] += 1
        st["b_ms"] += r["ms"]
        st["b_switch"] += float(a >= 4)
        if r["searched"]:
            st["b_searched"] += 1
            st["b_leaves"] += r["leaves"]
            st["b_mean_depth"] += r["mean_depth"]
            st["b_max_depth"] = max(st["b_max_depth"], r["max_depth"])
            st["b_root_q"] += r["root_q"]
            st["b_root_value_net"] += r["value"]
            v = r["visits"] / r["visits"].sum()
            st["b_visit_entropy"] += float(-(v[v > 0] * np.log(v[v > 0])).sum())
            st["b_changed"] += float(int(np.argmax(r["visits"])) != int(np.argmax(np.where(r["legal"], r["prior"],
                                                                                              -1))))

    def _finish_battle(self, z):
        for obs, pi, q in self.pending_b:
            self.out["battle"].append((obs, pi, float(z), q, self.battle_held))
        self._new_battle()

    def _finish_run(self, extra=0.0):
        for kind, obs, pi, q, wins_before in self.pending_t:
            self.out[kind].append((obs, pi, float(self.run_wins - wins_before + extra), q, self.run_held))
        self.pending_t = []

    def _after(self, ev):
        st = ev["stats"]
        b = st.get("battle")
        if b is not None and b["won"] is not None:
            self._finish_battle(float(b["won"]))
            self.run_wins += int(b["won"])
            self.stats["battles"] += 1
            self.stats["battles_won"] += float(b["won"])
        if ev["truncate"]:
            # the battle hit max_decisions: not a loss. Battler: its value at the cut; the run ends here (FactoryEnv
            # cannot continue it): the tactician bootstraps with its last value minus the wins since then
            _, v = self.evaluator({k: np.stack([ev["obs"][k]]) for k in ev["obs"]})
            self._finish_battle(float(v[0]))
            boot = max(self.last_t_value - (self.run_wins - self.wins_at_last_t), 0.0)
            self._finish_run(boot)
            self.stats["truncated"] += 1
            self.stats["runs"].append((self.env.start_streak, self.run_wins))
            self.ev = self.env.step("reset")
            self._new_run()
            return
        if "run" in st:
            self._finish_run()
            self.stats["runs"].append((st["run"]["start"], st["run"]["streak"]))
            self._new_run()


def _stack(samples, kind):
    if not samples:
        return None
    obs = {k: np.stack([s[0][k] for s in samples]) for k in samples[0][0]}
    return {"obs": obs, "pi": np.stack([s[1] for s in samples]).astype(np.float32),
            "z": np.array([s[2] for s in samples], np.float32), "q": np.array([s[3] for s in samples], np.float32),
            "held": np.array([s[4] for s in samples], bool)}


# ---- worker processes -------------------------------------------------------------------------------------------------

def _worker_proc(conn, wid, cfg, server, seed):
    try:
        torch.set_num_threads(1)
        ev = server.client(wid).evaluate if server is not None else None
        sp = SelfPlayer(cfg, seed, ev)
        while True:
            cmd, arg = conn.recv()
            try:
                if cmd == "load":
                    sp.load(arg)
                    conn.send(("ok", None))
                elif cmd == "weights":
                    sp.set_weights(arg)
                    conn.send(("ok", None))
                elif cmd == "play":
                    conn.send(("ok", sp.play(arg)))
                elif cmd == "close":
                    break
                else:
                    raise ValueError(cmd)
            except Exception:
                conn.send(("error", traceback.format_exc()))
    except (EOFError, KeyboardInterrupt):
        pass


class WorkerPool:
    """n forked SelfPlayer processes (fork after the inference server is created: they inherit its client slots)."""

    def __init__(self, n, cfg, server=None, seed=0):
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        for w in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker_proc, args=(b, w, cfg, server, seed * 1_000_003 + 7919 * w), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def call(self, cmd, args):
        for c, a in zip(self.conns, args):
            c.send((cmd, a))
        out = []
        for c in self.conns:
            status, res = c.recv()
            if status != "ok":
                raise RuntimeError(f"self-play worker failed:\n{res}")
            out.append(res)
        return out

    def broadcast(self, cmd, arg):
        return self.call(cmd, [arg] * len(self.conns))

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


# ---- replay buffer ------------------------------------------------------------------------------------------------------

class Replay:
    """The samples of the last `window` iterations, by kind."""

    def __init__(self, window):
        self.iters = collections.deque(maxlen=window)
        self.cat = {}

    def add(self, data):
        self.iters.append(data)
        self.cat = {}
        for kind in KINDS:
            parts = [d[kind] for d in self.iters if d.get(kind) is not None]
            if not parts:
                self.cat[kind] = None
                continue
            self.cat[kind] = {"obs": {k: np.concatenate([p["obs"][k] for p in parts]) for k in parts[0]["obs"]},
                              **{f: np.concatenate([p[f] for p in parts]) for f in ("pi", "z", "q", "held")}}

    def get(self, kind, held):
        d = self.cat.get(kind)
        if d is None:
            return None
        idx = np.flatnonzero(d["held"] == held)
        return (d, idx) if len(idx) else None


def merge(results):
    """Concatenate the workers' sample dicts by kind."""
    out = {}
    for kind in KINDS:
        parts = [r[kind] for r in results if r.get(kind) is not None]
        out[kind] = None if not parts else {
            "obs": {k: np.concatenate([p["obs"][k] for p in parts]) for k in parts[0]["obs"]},
            **{f: np.concatenate([p[f] for p in parts]) for f in ("pi", "z", "q", "held")}}
    return out


def value_target(z, q, mix):
    """(1 - mix) z + mix q, and z alone where there was no search value (q NaN)."""
    return np.where(np.isnan(q), z, (1.0 - mix) * z + mix * np.nan_to_num(q)).astype(np.float32)


# ---- losses ---------------------------------------------------------------------------------------------------------------

def policy_terms(kind, net, x, pi):
    """-> (cross-entropy to pi [B], KL(pi || net) [B], entropy of the net's policy [B], value [B])."""
    if kind == "rental":
        ll, pl, v = net.rental_joint(x)
        logq = F.log_softmax(ll, -1)[:, :, None] + F.log_softmax(pl, -1)          # [B, 6, 15] joint
        logq = logq.flatten(1)
        pi = pi.flatten(1)
        legal = (x["pair_mask"] & x["lead_mask"][:, :, None]).flatten(1)
    else:
        logits, v = net.battler(x) if kind == "battle" else net.swap(x)
        logq = F.log_softmax(logits, -1)
        legal = x["mask"]
    logq = logq.masked_fill(~legal, -1e9)
    ce = -(pi * logq).sum(-1)
    h_pi = -(pi * torch.log(pi.clamp_min(1e-12))).sum(-1)
    q = logq.exp()
    ent = -(q * logq.clamp_min(-1e4)).sum(-1)
    return ce, ce - h_pi, ent, v


def explained_variance(pred, target):
    var = np.var(target)
    return float(1 - np.var(target - pred) / var) if var > 0 else 0.0


def to_device(obs, idx, device):
    return {k: torch.from_numpy(np.ascontiguousarray(v[idx])).to(device, non_blocking=True) for k, v in obs.items()}


def train_iteration(net, opt, replay, new_counts, norms, args, device, rng):
    """Gradient steps on the replay buffer: sum over the kinds present of CE + vf_coef * value MSE."""
    agent = {"battle": "battler", "rental": "tactician", "swap": "tactician"}
    mix = {"battler": args.value_mix_b, "tactician": args.value_mix_t}
    data, targets = {}, {}
    for kind in KINDS:
        got = replay.get(kind, held=False)
        if got is not None:
            d, idx = got
            data[kind] = (d, idx)
            targets[kind] = value_target(d["z"], d["q"], mix[agent[kind]])
    # value normalization: once per iteration, on every training target in the buffer
    for a in ("battler", "tactician"):
        ts = [targets[k][data[k][1]] for k in data if agent[k] == a]
        if ts and norms[a] is not None:
            norms[a].update(np.concatenate(ts))
    if "battle" not in data:
        return {}
    n_new_b = max(new_counts.get("battle", 0), 1)
    steps = max(1, math.ceil(args.reuse * n_new_b / args.batch_b))
    per_step = {"battle": args.batch_b}
    t_kinds = [k for k in ("rental", "swap") if k in data]
    if t_kinds:
        # the tactician's samples are ~26x fewer: rentals and swaps share one minibatch, split by their new counts
        new_t = {k: max(new_counts.get(k, 0), 1) for k in t_kinds}
        n_t = max(args.min_batch_t, math.ceil(args.reuse * sum(new_t.values()) / steps))
        for k in t_kinds:
            per_step[k] = max(1, round(n_t * new_t[k] / sum(new_t.values())))
    logs = collections.defaultdict(list)
    preds = collections.defaultdict(list)
    net.train()
    for _ in range(steps):
        loss = 0.0
        for kind, n in per_step.items():
            d, idx = data[kind]
            pick = idx[rng.integers(0, len(idx), min(n, len(idx)) if kind != "battle" else n)]
            x = to_device(d["obs"], pick, device)
            pi = torch.from_numpy(d["pi"][pick]).to(device)
            nm = norms[agent[kind]]
            t = targets[kind][pick]
            tn = torch.from_numpy((t - nm.mean) / nm.std if nm is not None else t).to(device)
            ce, kl, ent, v = policy_terms(kind, net, x, pi)
            vl = 0.5 * ((v - tn) ** 2).mean()
            loss = loss + ce.mean() + args.vf_coef * vl
            logs[f"{kind}/policy_loss"].append(ce.mean().item())
            logs[f"{kind}/kl_to_search"].append(kl.mean().item())
            logs[f"{kind}/entropy"].append(ent.mean().item())
            logs[f"{kind}/value_loss"].append(vl.item())
            vd = v.detach().float().cpu().numpy()
            preds[kind].append((nm.denorm(vd) if nm is not None else vd, t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
        opt.step()
        logs["train/grad_norm"].append(float(gn))
        logs["train/loss"].append(float(loss.item()))
    net.eval()
    out = {k: float(np.mean(v)) for k, v in logs.items()}
    for kind, pv in preds.items():
        p = np.concatenate([a for a, _ in pv]); t = np.concatenate([b for _, b in pv])
        out[f"{kind}/explained_variance_train"] = explained_variance(p, t)
    out["train/steps"] = steps
    for kind, n in per_step.items():
        out[f"{kind}/batch_size"] = n
        out[f"{kind}/buffer_train"] = len(data[kind][1])
    return out


@torch.no_grad()
def heldout_stats(net, replay, norms, args, device):
    """Critic generalization on held-out battles / runs: explained variance against the training target and
    against the outcome z alone."""
    agent = {"battle": "battler", "rental": "tactician", "swap": "tactician"}
    mix = {"battler": args.value_mix_b, "tactician": args.value_mix_t}
    out = {}
    for kind in KINDS:
        got = replay.get(kind, held=True)
        if got is None or len(got[1]) < 8:
            continue
        d, idx = got
        vals, kls = [], []
        for chunk in np.array_split(idx, max(1, len(idx) // 2048)):
            x = to_device(d["obs"], chunk, device)
            _, kl, _, v = policy_terms(kind, net, x, torch.from_numpy(d["pi"][chunk]).to(device))
            vals.append(v.float().cpu().numpy())
            kls.append(kl.cpu().numpy())
        nm = norms[agent[kind]]
        pred = np.concatenate(vals)
        pred = nm.denorm(pred) if nm is not None else pred
        t = value_target(d["z"], d["q"], mix[agent[kind]])[idx]
        out[f"{kind}/explained_variance_heldout"] = explained_variance(pred, t)
        out[f"{kind}/explained_variance_heldout_z"] = explained_variance(pred, d["z"][idx])
        out[f"{kind}/kl_to_search_heldout"] = float(np.concatenate(kls).mean())
        out[f"{kind}/buffer_heldout"] = len(idx)
    return out


# ---- evaluation -----------------------------------------------------------------------------------------------------------

class NetPolicy:
    """The network being trained, as rl.eval_rounds.eval_round's policy (greedy actions)."""

    def __init__(self, net):
        self.net = net

    def act(self, kind, x, greedy=True):
        return net_act(self.net, kind, x, greedy=greedy)[0]


def _rounds(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


# ---- main -----------------------------------------------------------------------------------------------------------------

def cosine_lr(args, it):
    frac = min(it / max(args.iterations, 1), 1.0)
    lo = args.lr * args.lr_min_frac
    return lo + 0.5 * (args.lr - lo) * (1 + math.cos(math.pi * frac))


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def worker_cfg(args):
    keys = ("encode_version", "d_emb", "d", "layers", "heads", "sims", "dets", "c_puct", "search_batch",
            "dirichlet_alpha", "dirichlet_frac", "temperature", "max_decisions", "t_budget", "t_considered",
            "t_bootstrap", "t_target", "t_target_temp", "t_max_decisions", "t_local_batch", "holdout",
            "opponent_prior")
    return {k: getattr(args, k) for k in keys}


def main(argv=None):
    from .search import mark_training
    mark_training()     # no perfect-information search can exist in a training process (rl/search.py)
    args = parse(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = args.run_dir or os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    # (gamma_b / beta / share / algo: what rl.evaluate and rl.policy read from a run's config and checkpoints)
    config = dict(vars(args), algo=ALGO, share="embeddings", gamma_b=1.0, beta=0.0)
    json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(run_dir)
    tb.add_text("config", "```\n" + json.dumps(config, indent=2) + "\n```")

    encode.set_version(args.encode_version)           # before building the network and forking the workers
    net = FactoryNet(args.d_emb, args.d, args.layers, args.heads, share="embeddings").to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-5)
    norms = {a: ValueNorm(args.value_norm_rate) for a in ("battler", "tactician")}
    it0, b_steps, t_steps = 0, 0, 0
    round_eval = {}
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        net.load_state_dict(ck["net"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        for a, st in (ck.get("value_norm") or {}).items():
            if st and a in norms:
                norms[a].load(st)
        it0 = ck.get("iteration", -1) + 1
        b_steps, t_steps = ck.get("battler_steps", 0), ck.get("tactician_steps", 0)
        round_eval = {int(k): v for k, v in (ck.get("round_eval") or {}).items()}
        print(f"resumed from {args.init_from}: iteration {it0}, {b_steps:,} battler steps (empty replay buffer)",
              flush=True)
    tb.add_scalar("model/parameters", sum(p.numel() for p in net.parameters()), b_steps)
    net.eval()

    snap_path = os.path.join(run_dir, "snapshot.pt")

    def ckpt_dict(it):
        return {"net": net.state_dict(), "opt": opt.state_dict(), "args": config, "battler_steps": b_steps,
                "tactician_steps": t_steps, "iteration": it, "round_eval": round_eval,
                "value_norm": {k: (v.state() if v.seen else None) for k, v in norms.items()}}

    def write_snapshot():
        save_atomic({"net": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "args": config,
                     "value_norm": {k: (v.state() if v.seen else None) for k, v in norms.items()}}, snap_path)

    write_snapshot()
    server = None
    eval_slots = args.eval_search_procs if args.eval_search_sims > 0 else 0
    if args.inference == "gpu":
        from .inference import InferenceServer
        server = InferenceServer(snap_path, n_clients=args.workers + eval_slots)
    pool = WorkerPool(args.workers, worker_cfg(args), server, args.seed)
    replay = Replay(args.window)
    rng = np.random.default_rng(args.seed + 17)
    t_start = time.time()
    try:
        for it in range(it0, args.iterations):
            t_it = time.time()
            lr = cosine_lr(args, it)
            for g in opt.param_groups:
                g["lr"] = lr
            weights = start_weights_from_eval(round_eval, args.start_p0)
            pool.broadcast("load", snap_path)
            pool.broadcast("weights", weights.tolist())

            # ---- 1. self-play -------------------------------------------------------------------------------------
            per_worker = math.ceil(args.decisions_per_iter / args.workers)
            srv0 = server.stats() if server is not None else None
            t0 = time.time()
            results = pool.broadcast("play", per_worker)
            t_play = time.time() - t0
            if server is not None:
                srv1 = server.stats()
                nb = max(srv1["batches"] - srv0["batches"], 1)
                tb.add_scalar("perf/server_mean_batch", (srv1["samples"] - srv0["samples"]) / nb, b_steps + sum(
                    s["battler_decisions"] for _, s in results))
                tb.add_scalar("perf/server_forward_ms", (srv1["forward_ms"] * srv1["batches"]
                                                         - srv0["forward_ms"] * srv0["batches"]) / nb,
                              b_steps + sum(s["battler_decisions"] for _, s in results))
            data = merge([r for r, _ in results])
            stats = [s for _, s in results]
            new_counts = {k: int((~data[k]["held"]).sum()) if data[k] is not None else 0 for k in KINDS}
            made = sum(s["battler_decisions"] for s in stats)
            b_steps += made
            t_steps += int(sum(s.get("t_decisions", 0) for s in stats))
            replay.add(data)
            log_selfplay(tb, stats, data, made, t_play, weights, b_steps)

            # ---- 2. training ---------------------------------------------------------------------------------------
            t0 = time.time()
            tr = train_iteration(net, opt, replay, new_counts, norms, args, device, rng)
            tr.update(heldout_stats(net, replay, norms, args, device))
            t_train = time.time() - t0
            for k, v in tr.items():
                tb.add_scalar(k.replace("battle/", "battler/").replace("rental/", "tactician_rental/")
                              .replace("swap/", "tactician_swap/"), v, b_steps)
            tb.add_scalar("train/lr", lr, b_steps)
            tb.add_scalar("train/iteration", it, b_steps)
            for a, nm in norms.items():
                tb.add_scalar(f"{a}/value_norm_mean", nm.mean, b_steps)
                tb.add_scalar(f"{a}/value_norm_std", nm.std, b_steps)

            # ---- 3. checkpoint, new snapshot, evaluation ---------------------------------------------------------
            write_snapshot()
            if server is not None:
                server.reload(snap_path)
            ck = ckpt_dict(it)
            save_atomic(ck, os.path.join(run_dir, f"ckpt_{b_steps:011d}.pt"))
            t0 = time.time()
            if args.eval_every and (it + 1) % args.eval_every == 0 and args.eval_runs > 0:
                round_eval = evaluate_rounds(net, device, args, tb, b_steps, server, snap_path)
            t_eval = time.time() - t0
            save_atomic(ckpt_dict(it), os.path.join(run_dir, "latest.pt"))
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
                  f"(play {t_play:.0f}s train {t_train:.0f}s eval {t_eval:.0f}s)"
                  + ("  rounds " + " ".join(f"{round_eval[k]:.2f}" for k in sorted(round_eval)) if round_eval else ""),
                  flush=True)
    finally:
        pool.close()
        if server is not None:
            server.close()
        tb.close()
    return run_dir


def log_selfplay(tb, stats, data, made, t_play, weights, step):
    tot = collections.defaultdict(float)
    runs, starts = [], []
    for s in stats:
        for k, v in s.items():
            if k == "runs":
                runs += v
            elif k == "starts":
                starts += v
            elif k == "b_max_depth":
                tot[k] = max(tot[k], v)
            else:
                tot[k] += v
    tb.add_scalar("perf/battler_decisions_per_s", made / max(t_play, 1e-9), step)
    tb.add_scalar("perf/tactician_decisions_per_s", tot["t_decisions"] / max(t_play, 1e-9), step)
    n, ns = max(tot["b_decisions"], 1), max(tot["b_searched"], 1)
    tb.add_scalar("search/ms_per_decision", tot["b_ms"] / n, step)
    tb.add_scalar("search/searched_frac", tot["b_searched"] / n, step)
    tb.add_scalar("search/leaves_per_decision", tot["b_leaves"] / ns, step)
    tb.add_scalar("search/mean_depth", tot["b_mean_depth"] / ns, step)
    tb.add_scalar("search/max_depth", tot["b_max_depth"], step)
    tb.add_scalar("search/root_q", tot["b_root_q"] / ns, step)
    tb.add_scalar("search/root_value_net", tot["b_root_value_net"] / ns, step)
    tb.add_scalar("search/visit_entropy", tot["b_visit_entropy"] / ns, step)
    tb.add_scalar("search/argmax_changed_frac", tot["b_changed"] / ns, step)
    tb.add_scalar("actions/battler_switch_frac", tot["b_switch"] / n, step)
    nt = max(tot["t_decisions"], 1)
    tb.add_scalar("tsearch/ms_per_decision", tot["t_ms"] / nt, step)
    tb.add_scalar("tsearch/sims_per_decision", tot["t_sims"] / nt, step)
    tb.add_scalar("tsearch/sim_steps_per_decision", tot["t_sim_steps"] / nt, step)
    tb.add_scalar("tsearch/net_calls_per_decision", tot["t_net_calls"] / nt, step)
    tb.add_scalar("tsearch/options", tot["t_options"] / nt, step)
    tb.add_scalar("tsearch/chosen_is_prior_argmax", tot["t_chosen_is_prior_argmax"] / nt, step)
    tb.add_scalar("tsearch/value", tot["t_value_sum"] / max(tot["t_value_n"], 1), step)
    tb.add_scalar("tsearch/value_net", tot["t_value_net_sum"] / nt, step)
    tb.add_scalar("actions/tactician_swap_frac", tot["t_swaps"] / max(tot["t_swap_decisions"], 1), step)
    if tot["battles"]:
        tb.add_scalar("train/battle_win_rate", tot["battles_won"] / tot["battles"], step)
    tb.add_scalar("train/truncated_battles", tot["truncated"], step)
    tb.add_scalar("selfplay/runs_finished", len(runs), step)
    if runs:
        wins = np.array([w for _, w in runs], float)
        tb.add_scalar("train/streak_mean", float(wins.mean()), step)
        by = collections.defaultdict(list)
        for start, w in runs:
            by[start // 7 + 1].append(w)
        for k, ws in by.items():
            tb.add_scalar(f"train_by_start/wins_from_round_{k}", float(np.mean(ws)), step)
    for k in range(6):
        tb.add_scalar(f"curriculum/weight_round_{k + 1}", float(weights[k]), step)
    if starts:
        c = collections.Counter(starts)
        for k in range(1, 7):
            tb.add_scalar(f"curriculum/started_round_{k}", c[k] / len(starts), step)
    for kind in KINDS:
        d = data.get(kind)
        tb.add_scalar(f"selfplay/new_samples_{kind}", 0 if d is None else len(d["z"]), step)


def evaluate_rounds(net, device, args, tb, step, server, snap_path):
    """Per-round evaluation, network only (greedy), same seeds every time; optionally with search."""
    from .eval_rounds import eval_round
    res = {}
    policy = NetPolicy(net)
    for k in _rounds(args.eval_rounds):
        r = eval_round(policy, device, k, args.eval_runs, workers=args.eval_workers,
                       envs_per_worker=args.eval_envs_per_worker)
        tb.add_scalar(f"eval_round_{k}/complete", r["complete"], step)
        tb.add_scalar(f"eval_round_{k}/battle_win_rate", r["battle"], step)
        res[k] = r["complete"]
    if args.eval_search_sims > 0 and server is not None:
        from .eval_rounds import eval_round_inprocess
        kw = {"n_sims": args.eval_search_sims, "n_determinizations": args.dets, "c_puct": args.c_puct,
              "batch": args.search_batch, "mode": "legal", "seed": 0,
              "opponent_prior": "strict"}   # evaluation: the strict sampler, never the training one
        for k in _rounds(args.eval_rounds):
            r = eval_round_inprocess(snap_path, k, args.eval_search_runs, "search", kw,
                                     procs=args.eval_search_procs, server=server, slot0=args.workers)
            tb.add_scalar(f"eval_search_round_{k}/complete", r["complete"], step)
            tb.add_scalar(f"eval_search_round_{k}/battle_win_rate", r["battle"], step)
            tb.add_scalar(f"eval_search_round_{k}/ms_per_decision", r["ms_per_decision"], step)
    return res


if __name__ == "__main__":
    main()
