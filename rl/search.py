"""Decision-time search for the battler: ensemble determinized MCTS with the PPO network (docs/RL_DECISIONS.md).

    battler = SearchBattler(Policy(ckpt, device), n_sims=256, n_determinizations=8)
    action = battler.act_on(backend)          # backend: SimBackend at a BATTLE / FORCED_SWITCH decision
    battler.last_stats                        # visits, Q, priors, ms of the decision

K determinizations, each a deterministic tree over our decisions in this battle:
    legal    the opponent's hidden information (unseen Pokemon, moves, items, abilities, IVs, EVs, natures, exact
             HP inside the HP bar, hidden counters) is sampled from what a player knows, never from the Factory's
             set list or the true state (rl/determinize.py -> Gen3Game.determinize), redrawn if it would change
             what the player can do now (e.g. a trapping ability drawn while the view allows switching), and the
             observer is rebased on it (BattleObserver.rebase). opponent_prior="factory_sets" (training only:
             refused unless in_training(); rl/alphazero.py's default) draws the opponent's species, moves and
             items from the Factory's set list instead (rl/determinize.py sample_factory_sets; IVs, EVs and
             nature still random), with what the player knows of the team's generation
             (backend.opponent_knowledge);
    perfect  the true state. A ceiling for evaluation only: it needs allow_perfect=True and is refused inside
             training (mark_training(), called by rl/train.py, rl/train_rainbow.py and rl/alphazero.py), by the
             SearchBattler (its
             mode cannot be changed after construction) and by the C++ Searcher (in training, every root must
             be a full determinization: Gen3Game.determinized).
Both modes then redraw the random state of the turn (Gen3Game.redraw_turn): a fresh RNG seed, this turn's Quick Claw
roll, and the opponent's choice for this turn, which its AI makes while the player decides and which the root must
not know: the opponent's AI chooses again in each root.
Each node holds a cloned Gen3Game at a player decision and a BattleObserver advanced along its path (fast_copy +
observe), so a leaf's observation is exactly what a player would see there. Leaves are evaluated in batches by
FactoryNet.battler: masked softmax priors, and the critic's value denormalized with the checkpoint's value_norm
(battler return: 1 win / 0 loss, gamma 1, so it is a win probability), clipped to [0, 1]. Battle over: 1 win, 0
otherwise; a node at the truncation limit (300 decisions in the battle, as FactoryEnv) keeps the network value.
Like FactoryEnv, decisions without a choice (no usable move, no switch) are played automatically (move 0).
The decision is the argmax of the root visits summed over the K trees.

The C++ side (Gen3Game.sim_step / set_rng / determinize, MctsTree) is used when the extension has it; otherwise
sim_step and set_rng fall back to run() / a write of gRngValue, and the tree to PyMctsTree (same interface).
Legal mode needs Gen3Game.determinize.

impl="cpp" (the default when the extension has Searcher, src/gen3/search.cpp) runs the same loop in C++: the roots
are prepared here (determinization, redraws, rebase, set_rng: same RNG draws), then every node's clone / sim_step /
observe / encode and the tree bookkeeping happen in C++, and the network is called once per batch with the stacked
numpy arrays (evaluate_batch). Node observers are the C++ ObsMemory when the extension has it (observer="cpp"), or
the Python BattleObserver + rl.encode.battle called from C++ (observer="python"). Same selection and backup order
as impl="python", so both give the same visits for the same seeds (tests/python/test_search_cpp.py).
"""

import math
import os
import random
import sys
import time

import numpy as np
import torch

from pybattle.backend import Phase
from pybattle.emu.decode import SB2_RENTAL_MONS, SB2_TOWER_WIN_STREAKS, SYMBOLS as S, decode_rental_mons
from pybattle.pybattle_native import Gen3Game
from . import determinize as DET
from . import encode

try:
    from pybattle.pybattle_native import MctsTree as NativeMctsTree
except ImportError:                     # pragma: no cover - depends on the build
    NativeMctsTree = None
try:
    from pybattle.pybattle_native import Searcher as NativeSearcher
except ImportError:                     # pragma: no cover - depends on the build
    NativeSearcher = None
try:
    from pybattle.pybattle_native import ObsMemory as NativeObsMemory
except ImportError:                     # pragma: no cover - depends on the build
    NativeObsMemory = None

ACTION, SWITCH, BATTLE_OVER, TIMEOUT = 1, 2, 3, 4
OUTCOME_WON = 1
N_ACTIONS = 7
MAX_DECISIONS = 300                     # FactoryEnv.max_decisions
HAS_SIM_STEP = hasattr(Gen3Game, "sim_step")
HAS_SET_RNG = hasattr(Gen3Game, "set_rng")
HAS_DETERMINIZE = hasattr(Gen3Game, "determinize")
HAS_REDRAW_TURN = hasattr(Gen3Game, "redraw_turn")
_OUTCOME = S.addr("gBattleOutcome")
_RNG = S.addr("gRngValue")

# ---- the perfect-information guard ----------------------------------------------------------------------------
TRAINING_ENV = "PYB_TRAINING"
_TRAINING_MAINS = ("rl.train", "rl.train_rainbow", "rl.alphazero")


def mark_training():
    """Called first thing by the training entry points: no perfect-information search can be built afterwards in
    this process or its children (the environment variable is inherited)."""
    os.environ[TRAINING_ENV] = "1"


def in_training() -> bool:
    if os.environ.get(TRAINING_ENV):
        return True
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return spec is not None and spec.name in _TRAINING_MAINS


# ---- simulator steps ------------------------------------------------------------------------------------------

def sim_step(game, kind: int, index: int):
    """Apply a move slot (kind 0) or a switch to a party index (kind 1) and run to the next player decision or the
    end of the battle, without advancing the Factory. -> (decision, gBattleOutcome)."""
    if HAS_SIM_STEP:
        return game.sim_step(kind, index)
    if kind == 0:
        game.choose_move(index)
    else:
        game.choose_switch(index)
    d = int(game.run())
    return d, (game.read(_OUTCOME, 1)[0] if d == BATTLE_OVER else 0)


def set_rng(game, seed: int):
    if HAS_SET_RNG:
        game.set_rng(seed & 0xFFFFFFFF)
    else:
        game.write(_RNG, (seed & 0xFFFFFFFF).to_bytes(4, "little"))


def redraw_turn(game, seed: int):
    """A root's fresh random turn: RNG seed, this turn's Quick Claw roll and the opponent's choice for this turn
    (made by its AI while the player decides) redrawn. Every search root goes through it."""
    if not HAS_REDRAW_TURN:
        raise RuntimeError("search needs Gen3Game.redraw_turn (rebuild the extension)")
    game.redraw_turn(seed & 0xFFFFFFFF)


# ---- a pure-Python PUCT tree with the native MctsTree's interface ------------------------------------------------

class PyMctsTree:
    """PUCT over 7 actions, values in [0, 1] (maximized), lazy children, virtual loss for batched selection.
    Same interface as the C++ MctsTree (fallback and tests)."""

    def __init__(self, n_actions=N_ACTIONS, c_puct=1.5, virtual_loss=1.0):
        self.A, self.c, self.vl = n_actions, c_puct, virtual_loss
        self.reset()

    def reset(self):
        self.parent, self.action = [-1], [-1]
        self.prior, self.legal = [None], [None]
        self.child = [[-1] * self.A]
        self.n = [[0.0] * self.A]           # edge visits (including virtual loss in flight)
        self.w = [[0.0] * self.A]           # edge value sums
        self.terminal = [None]
        self.node_n, self.node_w = [0.0], [0.0]

    def _new(self, parent, action):
        i = len(self.parent)
        self.parent.append(parent); self.action.append(action)
        self.prior.append(None); self.legal.append(None)
        self.child.append([-1] * self.A); self.n.append([0.0] * self.A); self.w.append([0.0] * self.A)
        self.terminal.append(None); self.node_n.append(0.0); self.node_w.append(0.0)
        self.child[parent][action] = i
        return i

    def node_count(self):
        return len(self.parent)

    def is_expanded(self, node):
        return self.prior[node] is not None

    def expand(self, node, priors, legal):
        legal = [bool(x) for x in legal]
        p = [float(priors[a]) if legal[a] else 0.0 for a in range(self.A)]
        s = sum(p)
        self.prior[node] = [x / s for x in p] if s > 0 else [1.0 / max(sum(legal), 1) if l else 0.0 for l in legal]
        self.legal[node] = legal

    def set_terminal(self, node, value):
        self.terminal[node] = float(value)

    def _pick(self, node):
        n, w, p = self.n[node], self.w[node], self.prior[node]
        tot = sum(n)
        fpu = self.node_w[node] / self.node_n[node] if self.node_n[node] else 0.5
        sq = math.sqrt(tot + 1.0)
        best, best_a = -1e18, -1
        for a in range(self.A):
            if not self.legal[node][a]:
                continue
            q = w[a] / n[a] if n[a] else fpu
            u = q + self.c * p[a] * sq / (1.0 + n[a])
            if u > best:
                best, best_a = u, a
        return best_a

    def select(self, k):
        out = []
        for _ in range(k):
            node = 0
            if not self.is_expanded(0):
                out.append((0, -1, -1))
                continue
            while True:
                if self.terminal[node] is not None or not self.is_expanded(node):
                    out.append((node, self.parent[node], self.action[node]))
                    break
                a = self._pick(node)
                self.n[node][a] += self.vl
                c = self.child[node][a]
                if c < 0:
                    c = self._new(node, a)
                    out.append((c, node, a))
                    break
                node = c
        return out

    def backup(self, node, value):
        value = float(value)
        self.node_n[node] += 1
        self.node_w[node] += value
        while self.parent[node] >= 0:
            p, a = self.parent[node], self.action[node]
            self.n[p][a] += 1.0 - self.vl
            self.w[p][a] += value
            node = p
            self.node_n[node] += 1
            self.node_w[node] += value

    def root_visits(self):
        return list(self.n[0])

    def root_q(self):
        return [self.w[0][a] / self.n[0][a] if self.n[0][a] else 0.0 for a in range(self.A)]


def make_tree(c_puct=1.5, virtual_loss=1.0, native=None):
    use_native = NativeMctsTree is not None if native is None else native
    if use_native:
        return NativeMctsTree(N_ACTIONS, c_puct, virtual_loss)
    return PyMctsTree(N_ACTIONS, c_puct, virtual_loss)


# ---- the search battler -------------------------------------------------------------------------------------------

def run_ctx(backend):
    """The streak context the network sees (FactoryEnv._ctx)."""
    info = backend.run_info()
    return {"streak": info.win_streak, "battle": info.battle_in_challenge, "challenge": info.challenge_num,
            "rents": info.rents}


def own_set_ids(game):
    """Set ids of our 3 Pokemon (frontier.rentalMons[0..2]): the player knows its own rentals' sets."""
    return [m.mon_id for m in decode_rental_mons(game.read_saveblock2(SB2_RENTAL_MONS, 72))[:3]]


def to_backend_action(a: int):
    return ("move", a) if a < 4 else ("switch", a - 4)


class SearchBattler:
    def __init__(self, policy, n_sims=256, n_determinizations=8, c_puct=1.5, batch=32, mode="legal",
                 allow_perfect=False, seed=0, max_decisions=MAX_DECISIONS, virtual_loss=1.0, native_tree=None,
                 impl=None, observer="auto", evaluator=None, opponent_prior="strict"):
        """opponent_prior: the legal mode's opponent sampler, "strict" (player knowledge, the default and the only
        one outside training) or "factory_sets" (the Factory's set list: training-time search only, refused
        otherwise; rl/determinize.py). impl: "cpp" | "python" | None (cpp when built and the native tree is not refused). observer (cpp impl):
        "cpp" (ObsMemory) | "python" (BattleObserver called from C++) | "auto". evaluator: optional
        f(batch dict of numpy arrays) -> (priors [B, 7], values [B] in [0, 1]) replacing the policy's network
        (tests, an inference server)."""
        if mode not in ("legal", "perfect"):
            raise ValueError(f"mode must be 'legal' or 'perfect', not {mode!r}")
        if mode == "perfect":
            if not allow_perfect:
                raise PermissionError("perfect-information search is an evaluation ceiling: pass allow_perfect=True")
            if in_training():
                raise PermissionError("perfect-information search is never allowed in training")
        DET.check_prior(opponent_prior)                 # factory_sets: training only (PermissionError otherwise)
        self._prior = opponent_prior
        if mode == "legal" and not HAS_DETERMINIZE:
            raise RuntimeError("legal-mode search needs Gen3Game.determinize (rebuild the extension)")
        if impl is None:
            impl = "cpp" if NativeSearcher is not None and native_tree is not False else "python"
        if impl not in ("cpp", "python"):
            raise ValueError(f"impl must be 'cpp' or 'python', not {impl!r}")
        if impl == "cpp":
            if NativeSearcher is None:
                raise RuntimeError("impl='cpp' needs pybattle_native.Searcher (rebuild the extension)")
            if native_tree is False:
                raise ValueError("impl='cpp' always uses the native MctsTree")
        if observer == "auto":
            observer = "cpp" if NativeObsMemory is not None else "python"
        if observer not in ("cpp", "python"):
            raise ValueError(f"observer must be 'cpp', 'python' or 'auto', not {observer!r}")
        if impl == "cpp" and observer == "cpp" and NativeObsMemory is None:
            raise RuntimeError("observer='cpp' needs pybattle_native.ObsMemory (rebuild the extension)")
        self.impl, self.observer = impl, observer if impl == "cpp" else "python"
        self._mode = mode
        if observer == "cpp" and impl == "cpp" and policy is not None and getattr(policy, "encode_version", 3) not in (3, 4):
            raise ValueError("the C++ observer encodes versions 3 and 4 only: use observer='python' for this checkpoint")
        self._evaluator = evaluator
        self.policy = policy
        self.net = getattr(policy, "net", None)
        self.device = getattr(policy, "device", "cpu")
        norm = (getattr(policy, "value_norm", None) or {}).get("battler")
        self.v_mean, self.v_std = (norm["mean"], max(norm["var"], 1e-4) ** 0.5) if norm else (0.0, 1.0)
        self.n_sims, self.K, self.c_puct, self.batch = n_sims, n_determinizations, c_puct, batch
        self.max_decisions = max_decisions
        self.rng = random.Random(seed)
        self.trees = [make_tree(c_puct, virtual_loss, native_tree) for _ in range(self.K)]
        self.searcher = NativeSearcher(n_sims, batch, c_puct, virtual_loss, max_decisions) if impl == "cpp" else None
        self.last_stats = None
        self.n_decisions, self.total_ms, self.total_leaves, self.errors, self.redrawn = 0, 0.0, 0, 0, 0
        self.redraws = 8
        self.redraw_failed = 0          # roots whose last redraw still changed what the player can do
        self._net_calls, self._eval_ms = 0, 0.0

    @property
    def opponent_prior(self):
        """The legal mode's opponent sampler, fixed at construction."""
        return self._prior

    @property
    def mode(self):
        """'legal' or 'perfect', fixed at construction (the perfect-information guard checks it there)."""
        return self._mode

    # --- network ---------------------------------------------------------------------------------------------

    def evaluate(self, obs_list):
        """Encoded battle observations -> (priors [n, 7], values [n] in [0, 1])."""
        return self.evaluate_batch({k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]})

    @torch.no_grad()
    def evaluate_batch(self, batch):
        """A stacked batch (dict of numpy arrays, as encode.collate stacks them) -> (priors float32 [n, 7],
        values float32 [n] in [0, 1]). The C++ search calls this once per batch."""
        t = time.perf_counter()
        if self._evaluator is not None:
            pri, val = self._evaluator(batch)
        else:
            x = {k: torch.from_numpy(v).to(self.device, non_blocking=True) for k, v in batch.items()}
            logits, v = self.net.battler(x)
            pri = torch.softmax(logits.float(), -1).cpu().numpy()
            val = (v.float() * self.v_std + self.v_mean).clamp(0.0, 1.0).cpu().numpy()
        self._net_calls += 1
        self._eval_ms += (time.perf_counter() - t) * 1000
        return np.asarray(pri, np.float32), np.asarray(val, np.float32)

    # --- tree nodes ------------------------------------------------------------------------------------------

    def _advance(self, game, obs, action, ndec):
        """From a node's state (game already cloned, observer already copied): play `action` and any decisions
        without a choice. -> ("over", value) or ("node", view, ndec)."""
        kind, idx = (0, action) if action < 4 else (1, action - 4)
        while True:
            try:
                d, out = sim_step(game, kind, idx)
            except (RuntimeError, IndexError, ValueError):
                self.errors += 1                             # the game refused the action: count it as a loss
                return ("over", 0.0)
            ndec += 1
            if d == BATTLE_OVER:
                return ("over", 1.0 if out == OUTCOME_WON else 0.0)
            if d not in (ACTION, SWITCH):
                return ("over", 0.0)                         # TIMEOUT: the battle stalled
            view = obs.observe(game, d == SWITCH, game.unusable_moves(0), game.can_switch(0))
            if d == ACTION and not any(view.usable_moves) and not view.switch_targets \
                    and ndec < self.max_decisions:
                kind, idx = 0, 0                             # nothing to choose: FactoryEnv plays move 0
                continue
            return ("node", view, ndec)

    def _roots(self, backend, view, known, forced, native_obs=False):
        """K root states: (game, observer). native_obs: the observers are C++ ObsMemory copies."""
        roots = []
        if native_obs:
            base = NativeObsMemory.from_python(backend._observer)
            copy_obs = base.copy
        else:
            copy_obs = backend._observer.fast_copy
        if self._mode != "legal" and in_training():
            raise PermissionError("perfect-information search is never allowed in training")
        if self._prior != "strict":
            DET.check_prior(self._prior)
        hidden = DET.hidden_counters(view) if self._mode == "legal" else None
        knowledge = getattr(backend, "opponent_knowledge", None) if self._prior == "factory_sets" else None
        if self._prior == "factory_sets" and knowledge is None:
            raise ValueError("factory_sets needs backend.opponent_knowledge (a SimBackend run past a rental)")
        truth = (backend.game.unusable_moves(0), backend.game.can_switch(0))
        for _ in range(self.K):
            obs = copy_obs()
            if self._mode == "legal":
                # what the player can do now is visible (the game refuses a trapped switch / a disabled move):
                # redraw determinizations that would change it (e.g. a Shadow Tag / Arena Trap ability drawn)
                for attempt in range(self.redraws):
                    g = backend.game.clone()
                    specs = DET.determinization(view, knowledge, self.rng, backend.open_level, self._prior)
                    g.determinize(specs, hidden, self.rng.getrandbits(32))
                    if forced or (g.unusable_moves(0), g.can_switch(0)) == truth:
                        break
                    self.redrawn += 1
                else:
                    self.redraw_failed += 1
                obs.rebase(g, forced)
            else:
                g = backend.game.clone()
            redraw_turn(g, self.rng.getrandbits(32))
            roots.append((g, obs))
        return roots

    # --- the decision ----------------------------------------------------------------------------------------

    def act_on(self, backend, known=None, decisions=0):
        """Search from `backend`'s current decision and return a backend action.
        known: ignored (kept for callers that pass an ExclusionTracker: the strict legal sampler uses no generation
        rules of the Factory);
        decisions: decisions already taken in this battle (truncation at max_decisions)."""
        t0 = time.perf_counter()
        self._net_calls, self._eval_ms = 0, 0.0
        if backend.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
            raise RuntimeError(f"not at a battle decision: {backend.phase}")
        forced = backend.phase == Phase.FORCED_SWITCH
        view = backend.view()
        ctx = run_ctx(backend)
        root_x = encode.battle(view, ctx)
        legal = root_x["mask"].astype(bool)
        pri, val = self.evaluate([root_x])
        root_p, root_v = pri[0], float(val[0])
        if legal.sum() <= 1:
            a = int(np.flatnonzero(legal)[0]) if legal.any() else 0
            self._record(t0, a, np.eye(N_ACTIONS)[a], np.zeros(N_ACTIONS), root_p, root_v, 0, 0)
            return to_backend_action(a)

        roots = self._roots(backend, view, known, forced, native_obs=self.impl == "cpp" and self.observer == "cpp")
        if self.impl == "cpp":
            r = self.searcher.search([(g, obs, decisions) for g, obs in roots], ctx, root_p.tolist(),
                                     legal.tolist(), root_v, self.evaluate_batch)
            self.errors += r["errors"]
            visits, q = np.asarray(r["visits"], float), np.asarray(r["q"], float)
            score = np.where(legal, visits + 1e-6 * root_p, -1.0)
            a = int(np.argmax(score))
            self._record(t0, a, visits, q, root_p, root_v, r["leaves"], r["nodes"], ms_cpp=r["ms_cpp"])
            return to_backend_action(a)
        states, terms = [], []
        for k, (g, obs) in enumerate(roots):
            t = self.trees[k]
            t.reset()
            sel = t.select(1)
            t.expand(0, root_p.tolist(), legal.tolist())
            t.backup(0, root_v)
            states.append({0: (g, obs, decisions)})
            terms.append({})
        budget = max(1, self.n_sims // self.K)
        done = [1] * self.K
        per = max(1, self.batch // self.K)
        leaves = 0
        while True:
            pending = []
            for k, t in enumerate(self.trees[:self.K]):
                n = min(per, budget - done[k])
                if n > 0:
                    sel = t.select(n)
                    done[k] = done[k] + len(sel) if sel else budget      # (nothing left to select: stop)
                    pending += [(k, leaf, parent, act) for leaf, parent, act in sel]
            if not pending:
                break
            to_eval, dups = [], []
            for k, leaf, parent, act in pending:
                t = self.trees[k]
                if leaf in terms[k]:
                    t.backup(leaf, terms[k][leaf])
                    continue
                if leaf in states[k]:
                    dups.append((k, leaf))                   # selected twice in this batch, evaluated once
                    continue
                if parent < 0:                               # (root: already expanded)
                    t.backup(leaf, root_v)
                    continue
                g0, o0, d0 = states[k][parent]
                g, obs = g0.clone(), o0.fast_copy()
                res = self._advance(g, obs, act, d0)
                if res[0] == "over":
                    t.set_terminal(leaf, res[1])
                    terms[k][leaf] = res[1]
                    t.backup(leaf, res[1])
                    continue
                _, v, nd = res
                states[k][leaf] = (g, obs, nd)
                to_eval.append((k, leaf, encode.battle(v, ctx), nd >= self.max_decisions))
            if to_eval:
                pri, val = self.evaluate([e[2] for e in to_eval])
                leaves += len(to_eval)
                value_of = {}
                for (k, leaf, x, trunc), p, v in zip(to_eval, pri, val):
                    t = self.trees[k]
                    v = float(v)
                    if trunc:
                        t.set_terminal(leaf, v)
                        terms[k][leaf] = v
                    else:
                        t.expand(leaf, p.tolist(), x["mask"].astype(bool).tolist())
                    t.backup(leaf, v)
                    value_of[(k, leaf)] = v
                for k, leaf in dups:
                    self.trees[k].backup(leaf, value_of.get((k, leaf), terms[k].get(leaf, root_v)))
            elif dups:
                for k, leaf in dups:
                    self.trees[k].backup(leaf, terms[k].get(leaf, root_v))

        visits = np.zeros(N_ACTIONS)
        qsum = np.zeros(N_ACTIONS)
        nodes = 0
        for k in range(self.K):
            t = self.trees[k]
            vis = np.asarray(t.root_visits(), float)
            visits += vis
            qsum += np.asarray(t.root_q(), float) * vis
            nodes += t.node_count()
        q = np.where(visits > 0, qsum / np.maximum(visits, 1e-9), 0.0)
        score = np.where(legal, visits + 1e-6 * root_p, -1.0)
        a = int(np.argmax(score))
        self._record(t0, a, visits, q, root_p, root_v, leaves, nodes)
        return to_backend_action(a)

    def _record(self, t0, a, visits, q, prior, value, leaves, nodes, ms_cpp=None):
        ms = (time.perf_counter() - t0) * 1000
        self.last_stats = {"action": a, "visits": [float(x) for x in visits], "q": [float(x) for x in q],
                           "prior": [float(x) for x in prior], "value": value, "leaves": leaves, "nodes": nodes,
                           "ms": ms, "searched": leaves > 0 or nodes > 0, "errors": self.errors,
                           "redraw_failed": self.redraw_failed,
                           "impl": self.impl, "net_calls": self._net_calls, "ms_eval": self._eval_ms,
                           "ms_cpp": ms_cpp}
        self.n_decisions += 1
        self.total_ms += ms
        self.total_leaves += leaves

    def __call__(self, backend, known=None, decisions=0):
        return self.act_on(backend, known, decisions)
