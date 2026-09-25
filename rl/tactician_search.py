"""Decision-time search for the tactician (rentals and swaps) by simulation (alphazero_v1, docs/RL_DECISIONS.md §18).

    ts = TacticianSearch(battler_evaluator, tactician_values, budget=256)
    res = ts.search(backend, "rental" | "swap", obs, logp)     # at a RENTAL / SWAP decision of a SimBackend
    res["options"], res["visits"], res["q"], res["chosen"], res["value"], res["policy"]

Every option (a legal (lead, pair) rental, or keep / one of the legal trades) is valued by simulated battles:
    1. clone the game and apply the option (Gen3Game.factory_rent / factory_swap start the next battle: the game
       fills gEnemyParty with the real next opponent, generated before the decision);
    2. before the battle's first frame, replace the whole opposing team with one sampled from player knowledge only
       (rl/determinize.py sample_determinization with the 3 slots unseen: species uniform over the Frontier
       species, learnable moves, battle items, random IVs / EVs / nature / ability, full HP), or, with
       opponent_prior="factory_sets" (training only; rl/alphazero.py's default), from the Factory's set list the way
       the game draws the team (rl/determinize.py sample_factory_sets: the round's pool, the species on screen when
       it was generated excluded, SimBackend.opponent_knowledge_for; species / moves / item from the set, IVs / EVs /
       nature still random). The real team is
       overwritten first by a copy of our own party (the template Gen3Game.determinize rebuilds from: level, OT id,
       and "no item" kept only if our Pokemon has none), and the RNG is reseeded from the search's own stream, so
       nothing of the real next opponent (RAM, hint, RNG position) reaches the simulation;
    3. play the battle with the frozen battler network, greedy (argmax of its masked policy), no search, many
       simulated battles in lockstep so that each step is one batched network call (the GPU inference server);
       decisions without a choice are played as FactoryEnv plays them (move 0), and a battle is cut at
       max_decisions (value: the battler network's estimate of winning);
    4. value of a simulation, in the tactician's return units (battles won from the decision until the run ends,
       gamma 1): 0 if lost; if won, 1 + V_t(next tactician decision) with the tactician's value network
       (bootstrap=True; the next swap screen shows the defeated team as the player saw it in the simulated battle:
       reveals and v4 defeated-foe records, from the simulation's observer, finished when the battle is decided as
       SimBackend does) or just 1 (bootstrap=False).

Budget allocation: Gumbel top-m + sequential halving (Danihelka et al., 2022). The m options (max_considered,
default 16) with the highest log pi + Gumbel noise (noise=True; log pi only otherwise) are simulated, budget/R battles
per round over R = ceil(log2 m) rounds (4 per option for 16 options and 256 battles), then the better half by mean
value survives and gets twice as many (8, 16, 32). The chosen option is the last survivor.

Outputs: visits (battles per option), q (mean value; NaN if not simulated), the search policy target
(target="visits": the visit distribution; target="softmax": softmax(q / temperature) over the simulated options),
and value = q of the chosen option.
"""

import math
import random
import time
from types import SimpleNamespace

import numpy as np

from pybattle.backend import Phase
from pybattle.emu.decode import SYMBOLS as S, decode_party
from pybattle.pybattle_native import Gen3Game
from pybattle.view import BattleObserver, SeenMon
from . import determinize as DET
from . import encode
from .envs import decode_action
from .search import run_ctx

try:
    from pybattle.pybattle_native import ObsMemory as NativeObsMemory, has_choice as native_has_choice
except ImportError:                     # pragma: no cover - depends on the build
    NativeObsMemory = native_has_choice = None

NO_HINT = (18, 0)                       # the simulated opponent is not the one the attendant talked about
_PLAYER_PARTY = S.addr("gPlayerParty")
_D = Gen3Game.Decision
_P = Gen3Game.FactoryPhase
_RUN_FRAMES = 400000                    # SimBackend._sync's frame budget per decision


# ---- options ----------------------------------------------------------------------------------------------------

def options_of(kind, obs):
    """Legal options of a tactician decision, as network actions: rental (lead, pair index), swap 0..9."""
    if kind == "rental":
        pm = obs["pair_mask"]
        return [(lead, p) for lead in range(6) for p in range(15) if pm[lead, p]]
    return [a for a in range(10) if obs["mask"][a]]


def option_logp(kind, logits, options):
    """Log pi of each option from the network's logits: rental (lead_logits [6], pair_logits [6, 15]) with
    log pi(lead, pair) = log_softmax(lead)[l] + log_softmax(pair[l])[p]; swap logits [10]."""
    def log_softmax(z):
        z = np.asarray(z, np.float64)
        m = z.max(-1, keepdims=True)
        return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))
    if kind == "rental":
        ll, pl = log_softmax(logits[0]), log_softmax(logits[1])
        return np.array([ll[a] + pl[a, p] for a, p in options])
    lp = log_softmax(logits)
    return np.array([lp[a] for a in options])


def _unseen_view(level):
    return SimpleNamespace(enemy_party=[SeenMon() for _ in range(3)], own_party=[SimpleNamespace(level=level)])


def unseen_team_specs(level, rng, open_level=True):
    """Gen3Game.determinize specs for a whole opposing team nobody has seen yet, from player knowledge only (the
    legal sampler of rl/determinize.py with every slot unseen)."""
    return DET.sample_determinization(_unseen_view(level), None, rng, open_level=open_level)


def start_simulated_battle(backend, kind, option, rng, prior="strict", knowledge=None):
    """A clone of `backend` (at a RENTAL / SWAP decision) with `option` applied and a battle about to start against
    an opponent sampled with the sampler `prior` ("strict": player knowledge; "factory_sets": the Factory's set list,
    training only, with `knowledge` = backend.opponent_knowledge_for(action), computed here if None).
    -> the cloned SimBackend (its game at the battle's start; its opponent_knowledge is the simulated opponent's)."""
    act = decode_action(kind, option)
    if prior != "strict":
        DET.check_prior(prior)
        if knowledge is None:
            knowledge = backend.opponent_knowledge_for(act)
    be = backend.clone()
    g = be.game
    if kind == "rental":
        ok = g.factory_rent(*act)
    else:
        ok = g.factory_swap(-1) if act is None else g.factory_swap(act[0], act[1])
    if not ok:
        raise ValueError(f"{kind} option {option} refused by the game")
    # StartBattle has filled gEnemyParty with the real next opponent; no frame of the battle has run yet
    own = g.read(_PLAYER_PARTY, 300)
    g.write_party(1, own)
    level = decode_party(own)[0].level
    if prior == "strict":
        specs = unseen_team_specs(level, rng, backend.open_level)
    else:
        specs = DET.determinization(_unseen_view(level), knowledge, rng, backend.open_level, prior)
    be.opponent_knowledge = knowledge
    if not g.determinize(specs):
        raise RuntimeError("the simulated opponent team was not fully rebuilt")
    g.set_rng(rng.getrandbits(32))
    be.phase = Phase.BATTLE
    return be


# ---- simulated battles, many at once --------------------------------------------------------------------------------

class _Sim:
    __slots__ = ("be", "g", "obs", "ctx", "ndec", "d", "done", "value", "won", "trunc", "view", "tag")

    def __init__(self, be, tag, cpp):
        self.be, self.g, self.tag = be, be.game, tag
        self.obs = NativeObsMemory(*NO_HINT) if cpp else BattleObserver(*NO_HINT)
        self.ctx = None
        self.ndec, self.done, self.value, self.won, self.trunc, self.view = 0, False, 0.0, False, False, None


class BattleSimulator:
    """Plays simulated battles with the battler network, greedy, all of them in lockstep: one batched evaluator
    call per step. evaluator(batch dict) -> (priors [B, 7], values [B] in [0, 1]) (rl/inference.py)."""

    def __init__(self, evaluator, max_decisions=300, cpp_obs=None):
        self.evaluator = evaluator
        self.max_decisions = max_decisions
        if cpp_obs is None:
            cpp_obs = NativeObsMemory is not None and encode.VERSION in (3, 4)
        if cpp_obs and (NativeObsMemory is None or encode.VERSION not in (3, 4)):
            raise ValueError("the C++ observer needs the extension's ObsMemory and encoding version 3 or 4")
        self.cpp = cpp_obs
        self.steps = self.net_calls = self.errors = 0

    def _to_choice(self, s):
        """Run `s` to its next decision with a choice. -> encoded observation, or None when the battle is over."""
        g = s.g
        while True:
            if s.d == _D.BATTLE_OVER:
                s.done, s.won = True, g.factory_info.last_outcome == 1
                s.value = float(s.won)
                return None
            if s.d not in (_D.ACTION, _D.SWITCH):
                s.done, s.won, s.value = True, False, 0.0           # (the battle stalled: never seen)
                self.errors += 1
                return None
            if s.ctx is None:
                s.ctx = run_ctx(s.be)                               # the streak context of this battle
            forced = s.d == _D.SWITCH
            unusable, can_sw = g.unusable_moves(0), g.can_switch(0)
            if self.cpp:
                s.obs.observe(g, forced, unusable, can_sw)
                choice = forced or native_has_choice(g)
            else:
                s.view = s.obs.observe(g, forced, unusable, can_sw)
                choice = forced or any(s.view.usable_moves) or bool(s.view.switch_targets)
            if not choice and s.ndec < self.max_decisions:
                g.choose_move(0)                                    # nothing to choose (FactoryEnv plays move 0)
                s.ndec += 1
                self._run(s)
                continue
            if s.ndec >= self.max_decisions:
                s.trunc = True                                      # cut: valued by the network below
            return s.obs.encode(g, s.ctx) if self.cpp else encode.battle(s.view, s.ctx)

    @staticmethod
    def _run(s):
        """Run the battle of `s` to its next decision, or to its end: then, as SimBackend does, the observer
        accounts for the last turn (finish) at the moment the battle is decided, before the game winds it down and
        runs on to the next Factory phase."""
        d = s.g.run(_RUN_FRAMES)
        if d == _D.BATTLE_OVER:
            s.obs.finish(s.g)
            d = s.g.factory_run_battle()
        s.d = d

    def run(self, sims):
        """Play every _Sim to the end of its battle (the game then runs on to the next Factory phase)."""
        for s in sims:
            self._run(s)
        active = list(sims)
        while active:
            batch, who = [], []
            for s in active:
                x = self._to_choice(s)
                if x is not None:
                    batch.append(x)
                    who.append(s)
            if not batch:
                break
            pri, val = self.evaluator({k: np.stack([x[k] for x in batch]) for k in batch[0]})
            self.net_calls += 1
            self.steps += len(batch)
            for s, x, p, v in zip(who, batch, pri, val):
                if s.trunc:
                    s.done, s.value = True, float(v)
                    continue
                a = int(np.argmax(np.where(x["mask"], p, -1.0)))
                try:
                    if a < 4:
                        s.g.choose_move(a)
                    else:
                        s.g.choose_switch(a - 4)
                except (RuntimeError, ValueError, IndexError):
                    s.done, s.won, s.value = True, False, 0.0      # refused by the game: a loss
                    self.errors += 1
                    continue
                s.ndec += 1
                self._run(s)
            active = [s for s in who if not s.done]


def observer_memory(obs):
    """A BattleObserver holding what the simulated battle's observer (C++ ObsMemory or Python BattleObserver)
    remembers for the swap screen: the reveals (moves, items, abilities) and the per-opponent records."""
    if isinstance(obs, BattleObserver):
        return obs
    o = BattleObserver()
    o.revealed_moves = {int(i): list(v) for i, v in obs.revealed_moves.items()}
    o.revealed_items = {int(i): int(v) for i, v in obs.revealed_items.items()}
    o.revealed_abilities = {int(i): int(v) for i, v in obs.revealed_abilities.items()}
    o._records = [list(r) for r in obs.records]
    o._team_max_hp = obs.team_max_hp
    o._finished = obs.finished
    return o


def next_tactician_obs(be, obs=None):
    """After a won simulated battle: the next tactician decision's (kind, observation). The swap screen shows the
    defeated team as the player saw it in the simulated battle (`obs`, its observer: species, revealed moves / items
    / abilities, and the v4 defeated-foe records), exactly as SimBackend's swap view after a real battle."""
    fp = be.game.factory_phase
    if fp == _P.SWAP:
        be.phase = Phase.SWAP
        be._observer = observer_memory(obs) if obs is not None else BattleObserver()
        view = be.view()
        return "swap", encode.swap(view, run_ctx(be))
    if fp == _P.RENTAL:
        be.phase = Phase.RENTAL
        view = be.view()
        return "rental", encode.rental(view, run_ctx(be))
    return None, None


# ---- the search -----------------------------------------------------------------------------------------------------------

class TacticianSearch:
    """tactician_values(kind_list, obs_list) -> values in return units (battles won until the run ends), or None
    (bootstrap off)."""

    def __init__(self, battler_evaluator, tactician_values=None, budget=256, max_considered=16, max_decisions=300,
                 bootstrap=True, target="visits", temperature=0.5, seed=0, cpp_obs=None, opponent_prior="strict"):
        """opponent_prior: the simulated opponents' sampler, "strict" (player knowledge) or "factory_sets" (the
        Factory's set list: training only, refused otherwise)."""
        DET.check_prior(opponent_prior)
        self.prior = opponent_prior
        if target not in ("visits", "softmax"):
            raise ValueError(f"target must be 'visits' or 'softmax', not {target!r}")
        if bootstrap and tactician_values is None:
            raise ValueError("bootstrap=True needs tactician_values")
        self.sim = BattleSimulator(battler_evaluator, max_decisions, cpp_obs)
        self.values = tactician_values
        self.budget, self.m, self.bootstrap = budget, max_considered, bootstrap
        self.target, self.temperature = target, temperature
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.last_stats = None

    def search(self, backend, kind, obs, logp, noise=True):
        """backend: a SimBackend at a RENTAL / SWAP decision; obs: its encoded observation; logp: log pi of each
        option of options_of(kind, obs) (option_logp)."""
        t0 = time.perf_counter()
        options = options_of(kind, obs)
        n = len(options)
        logp = np.asarray(logp, np.float64)
        visits, qsum = np.zeros(n), np.zeros(n)
        if n == 0:
            raise ValueError(f"no legal {kind} option")
        g = self.np_rng.gumbel(size=n) if noise else np.zeros(n)
        score0 = logp + g
        sims_done = 0
        steps0, calls0 = self.sim.steps, self.sim.net_calls
        if n == 1:
            chosen, rounds = 0, 0
        else:
            m = min(self.m, n)
            alive = list(np.argsort(-score0, kind="stable")[:m])
            rounds = max(1, math.ceil(math.log2(m)))
            per_round = max(1, self.budget // rounds)
            know = {}
            if self.prior != "strict":
                DET.check_prior(self.prior)
                for i in alive:
                    know[i] = backend.opponent_knowledge_for(decode_action(kind, options[i]))
            for _ in range(rounds):
                k = max(1, per_round // len(alive))
                sims = [_Sim(start_simulated_battle(backend, kind, options[i], self.rng, self.prior, know.get(i)), i,
                             self.sim.cpp)
                        for i in alive for _ in range(k)]
                self.sim.run(sims)
                vals = self._values(sims)
                for s, v in zip(sims, vals):
                    visits[s.tag] += 1
                    qsum[s.tag] += v
                sims_done += len(sims)
                q = qsum[alive] / visits[alive]
                order = sorted(range(len(alive)), key=lambda j: (q[j], score0[alive[j]]), reverse=True)
                alive = [alive[j] for j in order[:max(1, math.ceil(len(alive) / 2))]]
                if len(alive) == 1:
                    break
            chosen = int(alive[0])
        q = np.where(visits > 0, qsum / np.maximum(visits, 1), np.nan)
        policy = self._policy(n, chosen, visits, q)
        value = float(q[chosen]) if visits[chosen] > 0 else float("nan")
        self.last_stats = {"options": n, "sims": sims_done, "rounds": rounds, "ms": (time.perf_counter() - t0) * 1000,
                           "sim_steps": self.sim.steps - steps0, "net_calls": self.sim.net_calls - calls0,
                           "chosen_is_prior_argmax": float(chosen == int(np.argmax(logp))),
                           "value": value}
        return {"options": options, "visits": visits, "q": q, "chosen": chosen, "action": options[chosen],
                "policy": policy, "value": value}

    def _values(self, sims):
        """Each finished simulation's value in return units."""
        vals = np.array([s.value for s in sims], np.float64)
        if not self.bootstrap:
            return vals
        nxt = [(j, *next_tactician_obs(s.be, s.obs)) for j, s in enumerate(sims) if s.won and not s.trunc]
        nxt = [x for x in nxt if x[1] is not None]
        if nxt:
            v = self.values([k for _, k, _ in nxt], [o for _, _, o in nxt])
            for (j, _, _), vj in zip(nxt, v):
                vals[j] += max(float(vj), 0.0)                      # won: 1 + the wins still to come
        return vals

    def _policy(self, n, chosen, visits, q):
        if n == 1 or visits.sum() == 0:
            p = np.zeros(n)
            p[chosen] = 1.0
            return p
        if self.target == "visits":
            return visits / visits.sum()
        z = np.where(visits > 0, q / max(self.temperature, 1e-6), -np.inf)
        z = np.exp(z - z[visits > 0].max())
        return z / z.sum()


def policy_to_action_space(kind, options, policy):
    """The search policy over options -> the network's action space: rental [6, 15] joint (lead, pair), swap [10]."""
    if kind == "rental":
        out = np.zeros((6, 15), np.float32)
        for (lead, p), w in zip(options, policy):
            out[lead, p] = w
        return out
    out = np.zeros(10, np.float32)
    for a, w in zip(options, policy):
        out[a] = w
    return out
