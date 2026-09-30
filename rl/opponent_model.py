"""The opponent model learned from the agent's own experience (docs/RL_DECISIONS.md §22, decided 2026-09-30).

    model = ExperienceModel.load("runs/opp_model_v3/model.pt")      # or ExperienceModel() and add() observations
    specs = model.determinization(view, challenge, rng)              # Gen3Game.determinize specs (search roots)

The legal search needs a guess of what the player does not see of the opponent (rl/determinize.py). The strict
sampler guesses from game knowledge only (any Frontier species, any learnable moves, any useful item): its opponents
are absurd and weak. The Factory's set list (factory_sets) is not player knowledge. This model is the third way: a
player who remembers what the Factory showed him in his own games.

What the player observes (the only inputs; checked in the decomp, see §22.1):
    battle        each opponent Pokemon sent out: species, level, the moves it used, its item if the game named it
                  (Leftovers, a berry eaten, Knock Off / Thief / Trick), its ability if announced, the HP bar;
    swap screen   the defeated team's species (picture and name only: the game offers no Summary of an opponent's
                  Pokemon, battle_factory_screen.c Swap_ActionMon / Swap_Task_OpenSummaryScreen shows gPlayerParty
                  only), so nothing beyond the battle. BUT once one of them is taken it is ours: the summary shows its
                  species, 4 moves, item, nature (and stats);
    rental screen the 6 rentals, fully visible through the Summary (species, 4 moves, item, nature, stats). They are
                  drawn by the same generator from (almost) the same range as the opponents of the round.

Records (collected by ExperienceCollector from views and the player's battle memory, never from the true state):
    key = (source, bucket, species, moves, item, nature, full) with
        source  OPP (an opponent Pokemon: battle, or the one we took at the swap screen) or RENTAL
        bucket  min(challenge, N_BUCKETS - 1) (the round - 1 the player was in)
        moves   sorted tuple: all of its moves (full) or the ones seen being used (partial)
        item    the held item (full; 0 = none) or the one the battle revealed, -1 = not revealed
        nature  full records: the summary's nature; -1 otherwise
        full    1: the whole moveset and item are known (rental, taken at a swap), 0: a battle sighting
    Every opponent Pokemon sent out is one record: partial, or full if we took it at the next swap screen.

The model (all counts, smoothed; "configuration" c = (species, moveset, item)):
    species     P_b(s) = (n_b(s) + beta P(s)) / (N_b + beta),  P(s) = (n(s) + a0 / |F|) / (N + a0)
                over the Frontier species F (the strict sampler's knowledge); opponents count 1, rentals rental_w.
    config      P_b(c | s) = (n_b(s, c) + beta P(c | s)) / (N_b(s) + beta), c in the configurations of s seen in
                full, or "novel" (one the player has never seen: drawn by the strict rules);
                P(c | s) = n(s, c) / (N(s) + a_cfg), P(novel | s) = (n(s, novel) + a_cfg) / (N(s) + a_cfg).
                n(s, c): full records, plus the battle sightings shared out over the configurations consistent with
                what they revealed (expectation-maximization, `em_iters` rounds: each sighting adds its posterior
                to the configurations and to "novel"); a sighting no configuration explains goes to "novel".
    posterior   at a decision, a seen Pokemon (species s, revealed moves R, revealed item I):
                P(c) ∝ P_b(c | s) [R ⊆ moves(c)] [I = item(c) if revealed, else item(c) not held by a teammate],
                P(novel) ∝ P_b(novel | s) L(R, I | s), L = the strict sampler's probability of drawing moves ⊇ R
                (C(L - r, 4 - r) / C(L, 4) over its L learnable moves) and that item (1 / its useful items):
                so nothing seen is impossible, and an unexplained sighting falls back to the strict draw.
                An unseen slot: species from P_b(s) (not repeating a teammate), then P_b(c | s) (not repeating an
                item).
    the rest    as the strict sampler: IVs uniform, a random EV spread, ability a coin flip unless announced, HP
                inside the bar; the nature from the configuration's observed natures (+ 1 pseudo-count spread over
                the 25), uniform for a novel one.
    hint        optional (use_hint): whole teams are drawn until their hint (hint_of_team: species + moves, as the
                game computes it) is the heard one, like the other samplers (max_hint_tries, then the last draw).

Never read here: the true state (gEnemyParty, gFrontierTempParty, the game), the Battle Factory's set list or pool
tables (FRONTIER_MONS, SET_*, pool, fixed_iv). tests/python/test_opponent_model.py scans this source.
"""

import collections
import math
import os
import random

import numpy as np
import torch

from pybattle.backend import Phase
from pybattle.view import SeenMon

from . import determinize as DET

N_BUCKETS = 8                       # challenge 0..6, 7+ together (rounds 1..7, 8+)
OPP, RENTAL = 0, 1
VERSION = 1
DEFAULT_CFG = {"a0": 1.0, "a_cfg": 1.0, "beta": 5.0, "rental_w": 1.0, "a_nat": 1.0, "em_iters": 3,
               "use_hint": False, "max_hint_tries": 512, "level": 100}


def bucket_of(challenge):
    return max(0, min(int(challenge), N_BUCKETS - 1))


def _moves_key(moves):
    return tuple(sorted(int(x) for x in moves if x))


# ---- observations: what the player saw (views and the player's own battle memory) --------------------------------

def full_record(source, challenge, mon):
    """A Pokemon the player sees whole (an OwnMon: a rental candidate, or one of ours)."""
    return (int(source), bucket_of(challenge), int(mon.species), _moves_key(mon.moves), int(mon.item),
            int(mon.nature), 1)


def partial_record(challenge, mon):
    """An opponent Pokemon seen in battle (a SeenMon)."""
    item = int(mon.revealed_item) if mon.revealed_item else -1
    return (OPP, bucket_of(challenge), int(mon.species), _moves_key(mon.revealed_moves), item, -1, 0)


def rental_records(view, challenge):
    """The rental screen (RentalView): the 6 candidates, fully visible through the Summary."""
    return [full_record(RENTAL, challenge, m) for m in view.candidates]


def seen_opponents(observer):
    """The opponent Pokemon the player saw in a battle, from the player's memory of it (BattleObserver: the last
    view's species of the seen slots, and the moves / items revealed up to the end, finish() included).
    -> {party slot: SeenMon}."""
    view = getattr(observer, "_cache_view", None)
    if view is None:
        return {}
    out = {}
    for i, m in enumerate(view.enemy_party[:3]):
        if not m.seen or not m.species:
            continue
        moves = list(observer.revealed_moves.get(i, m.revealed_moves))
        item = observer.revealed_items.get(i, m.revealed_item)
        out[i] = SeenMon(m.species, m.level, m.hp_pixels, m.status, moves, item, m.fainted, True)
    return out


class ExperienceCollector:
    """Turns one FactoryEnv's decisions into records, from what the player sees only. attach(env) wraps env.step /
    env.reset; drain() returns the records since the last drain (a Counter of keys).

    - rental screen: the 6 candidates (full);
    - end of every battle (won, lost or truncated): each opponent Pokemon sent out (partial), kept pending until
      the next decision: if it is a swap screen and we take one of them, that one becomes a full record (it is ours
      now: its summary);
    - never the true opponent: species of seen slots and reveals come from the player's memory of the battle
      (BattleObserver), the taken Pokemon from our own party."""

    def __init__(self):
        self.records = collections.Counter()
        self._rental_done = False
        self._observer = None           # the player's memory of the battle in progress
        self._challenge = 0
        self._pending = {}              # slot -> partial record of the last battle, awaiting the swap screen
        self._pending_species = {}

    def attach(self, env):
        step, reset = env.step, env.reset

        def wrapped_step(action):
            before = env.backend.phase
            swap_view = env.backend.view() if before == Phase.SWAP else None
            ev = step(action)
            self.after(env, before, action, swap_view, ev)
            return ev

        def wrapped_reset():
            self._flush()
            ev = reset()
            self._rental_done = False
            self.after(env, None, None, None, ev)
            return ev

        env.step, env.reset = wrapped_step, wrapped_reset
        self.after(env, None, None, None, None)
        return env

    def _flush(self):
        for rec in self._pending.values():
            self.records[rec] += 1
        self._pending, self._pending_species = {}, {}

    def after(self, env, before, action, swap_view, ev):
        """Called after each env step: before = the phase the action was taken in (None after a reset)."""
        P = Phase
        be = env.backend
        if before == P.SWAP:
            # the swap screen was answered: the Pokemon we took (if any) is ours now and fully visible
            if isinstance(action, tuple) and swap_view is not None:
                sp = swap_view.enemy_party[action[1]].species
                slot = self._pending_species.get(sp)
                if slot is not None:
                    own = self._own_party(be)
                    mon = next((m for m in own if m.species == sp), None)
                    if mon is not None:
                        rec = self._pending.pop(slot)
                        self._pending_species.pop(sp)
                        self.records[(OPP, rec[1], int(mon.species), _moves_key(mon.moves), int(mon.item),
                                      int(mon.nature), 1)] += 1
            self._flush()
        elif before == P.RENTAL:
            self._rental_done = False
        ended = ev is not None and (ev["stats"].get("battle") is not None or ev["truncate"])
        if ended and self._observer is not None:
            self._flush()
            for slot, m in seen_opponents(self._observer).items():
                self._pending[slot] = partial_record(self._challenge, m)
                self._pending_species[m.species] = slot
            self._observer = None
        if be.phase == P.RENTAL:
            self._flush()
            if not self._rental_done:
                ch = be.run_info().challenge_num
                for rec in rental_records(be.view(), ch):
                    self.records[rec] += 1
                self._rental_done = True
        elif be.phase in (P.BATTLE, P.FORCED_SWITCH):
            self._flush()
            self._observer = be._observer
            self._challenge = be.run_info().challenge_num
        elif be.phase != P.SWAP:
            self._flush()

    @staticmethod
    def _own_party(be):
        P = Phase
        if be.phase in (P.BATTLE, P.FORCED_SWITCH, P.SWAP):
            return be.view().own_party[:3]
        return []

    def drain(self):
        out, self.records = self.records, collections.Counter()
        return out


# ---- the model --------------------------------------------------------------------------------------------------------

def _comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def strict_move_prob(species, revealed, level=100):
    """P(a strict draw's moveset contains `revealed`) and the size of its draw space: (L, r) -> C(L-r, 4-r)/C(L, 4)."""
    learn = set(DET.learnable(species, level)) | set(revealed)
    L, r = len(learn), len(revealed)
    k = min(4, L)
    if r > k:
        return 0.0
    return _comb(L - r, k - r) / max(_comb(L, k), 1)


def strict_config_logp(species, moves, item, revealed=(), revealed_item=None, used_items=(), level=100):
    """log P(the strict sampler draws exactly this moveset and item | species and what was revealed)."""
    learn = set(DET.learnable(species, level)) | set(revealed)
    L, r = len(learn), len([x for x in revealed if x])
    k = min(4, L)
    n = len([x for x in moves if x])
    if not set(revealed) <= set(moves) or n != k:
        return -math.inf
    lp = -math.log(max(_comb(L - r, k - r), 1))
    if revealed_item:
        return lp if item == revealed_item else -math.inf
    items = [x for x in DET.plausible_items(species) if x not in set(used_items)]
    return lp - math.log(len(items)) if item in items else -math.inf


class _Compiled:
    """The smoothed tables of one ExperienceModel state (built lazily; see ExperienceModel.compile)."""

    def __init__(self, counts, cfg):
        self.cfg = cfg
        a0, a_cfg, beta = cfg["a0"], cfg["a_cfg"], cfg["beta"]
        rw, level = cfg["rental_w"], cfg["level"]
        self.level = level
        frontier = DET._FRONTIER_SPECIES[True]
        self.frontier = np.array(frontier, np.int64)
        self._fidx = {s: i for i, s in enumerate(frontier)}
        nS = len(frontier)
        spc = np.zeros((N_BUCKETS, nS))
        full = collections.defaultdict(float)              # (b, s, cfg) -> count
        natures = collections.defaultdict(lambda: np.zeros(DET.NUM_NATURES))
        partial = []                                       # (b, s, R, I, count)
        for key in sorted(counts):
            n = float(counts[key])
            src, b, s, moves, item, nature, is_full = key
            if s not in self._fidx or n <= 0:
                continue
            w = n * (rw if src == RENTAL else 1.0)
            spc[b, self._fidx[s]] += w
            if is_full:
                c = (moves, item)
                full[(b, s, c)] += w
                if 0 <= nature < DET.NUM_NATURES:
                    natures[(s, c)][nature] += n
            else:
                partial.append((b, s, frozenset(moves), item, n))
        # species
        tot = spc.sum(0)
        p_all = (tot + a0 / nS) / (tot.sum() + a0)
        self.p_species = (spc + beta * p_all[None]) / (spc.sum(1, keepdims=True) + beta)
        self.p_species_all = p_all
        # configurations of each species (the ones seen in full), sorted for determinism
        cfgs = collections.defaultdict(set)
        for (b, s, c) in full:
            cfgs[s].add(c)
        self.cfgs = {s: sorted(cs) for s, cs in cfgs.items()}
        self.cfg_sets = {s: [frozenset(c[0]) for c in cs] for s, cs in self.cfgs.items()}
        idx = {s: {c: i for i, c in enumerate(cs)} for s, cs in self.cfgs.items()}
        base = {}                                           # s -> [N_BUCKETS, n_cfg + 1] full counts (+ novel)
        for s, cs in self.cfgs.items():
            base[s] = np.zeros((N_BUCKETS, len(cs) + 1))
        for (b, s, c), w in full.items():
            base[s][b, idx[s][c]] += w
        self._lik_novel = {}
        # battle sightings: grouped by (s, R, I) with a count per bucket
        groups = collections.defaultdict(lambda: np.zeros(N_BUCKETS))
        for b, s, R, I, n in partial:
            groups[(s, R, I)][b] += n
        self.n_partial = len(partial)
        em = {s: np.zeros_like(v) for s, v in base.items()}
        novel_only = collections.defaultdict(lambda: np.zeros(N_BUCKETS))
        for it in range(max(1, int(cfg["em_iters"]))):
            tables = self._tables(base, em, novel_only, a_cfg, beta)
            new_em = {s: np.zeros_like(v) for s, v in base.items()}
            new_novel = collections.defaultdict(lambda: np.zeros(N_BUCKETS))
            for (s, R, I), per_b in sorted(groups.items(), key=lambda kv: (kv[0][0], sorted(kv[0][1]), kv[0][2])):
                if s not in self.cfgs:
                    new_novel[s] += per_b
                    continue
                cons = self._consistent(s, R, I, ())
                lik = self._novel_lik(s, R, I, ())
                p = tables[s]                                          # [N_BUCKETS, n_cfg + 1]
                w = np.zeros_like(p)
                w[:, :-1] = p[:, :-1] * cons[None]
                w[:, -1] = p[:, -1] * lik
                z = w.sum(1, keepdims=True)
                z[z <= 0] = 1.0
                new_em[s] += w / z * per_b[:, None]
            em, novel_only = new_em, new_novel
        self.tables = self._tables(base, em, novel_only, a_cfg, beta)
        self.p_novel_unknown = {s: v for s, v in novel_only.items()}       # (species with no known configuration)
        self.natures = {k: v for k, v in natures.items()}
        self.level = level

    @staticmethod
    def _tables(base, em, novel_only, a_cfg, beta):
        out = {}
        for s, cnt in base.items():
            n = cnt + em[s]                                   # [N_BUCKETS, n_cfg + 1] (last: novel)
            tot = n.sum(0)
            p_all = tot.copy()
            p_all[-1] += a_cfg
            p_all /= tot.sum() + a_cfg
            out[s] = (n + beta * p_all[None]) / (n.sum(1, keepdims=True) + beta)
        return out

    def _consistent(self, s, R, I, used_items):
        cs = self.cfgs[s]
        out = np.zeros(len(cs))
        for j, (c, mset) in enumerate(zip(cs, self.cfg_sets[s])):
            if not R <= mset:
                continue
            item = c[1]
            if I is not None and I >= 0:
                if item != I:
                    continue
            elif item and item in used_items:
                continue
            out[j] = 1.0
        return out

    def _novel_lik(self, s, R, I, used_items):
        key = (s, R, I if I is not None and I >= 0 else -1, tuple(sorted(used_items)))
        v = self._lik_novel.get(key)
        if v is None:
            v = strict_move_prob(s, tuple(R), self.level)
            if I is not None and I >= 0:
                items = [x for x in DET.plausible_items(s) if x not in set(used_items)]
                v *= 1.0 / max(len(items), 1)
            self._lik_novel[key] = v
        return v

    # --- the posterior of one slot ------------------------------------------------------------------------------

    def species_probs(self, b, exclude=()):
        p = self.p_species[b].copy()
        if exclude:
            for s in exclude:
                i = self._fidx.get(int(s))
                if i is not None:
                    p[i] = 0.0
        z = p.sum()
        return p / z if z > 0 else p

    def config_posterior(self, b, s, R=frozenset(), I=-1, used_items=()):
        """-> (list of known configurations (moves tuple, item), their probabilities, P(novel)) for species s."""
        R = frozenset(int(x) for x in R if x)
        if s not in self.cfgs:
            return [], np.zeros(0), 1.0
        p = self.tables[s][b]
        cons = self._consistent(s, R, I, used_items)
        w = p[:-1] * cons
        wn = p[-1] * self._novel_lik(s, R, I, used_items)
        z = w.sum() + wn
        if z <= 0:
            return self.cfgs[s], np.zeros(len(w)), 1.0
        return self.cfgs[s], w / z, wn / z

    def nature(self, s, c, rng):
        cnt = self.natures.get((s, c))
        if cnt is None:
            return rng.randrange(DET.NUM_NATURES)
        a = self.cfg["a_nat"]
        w = cnt + a / DET.NUM_NATURES
        return _choice(rng, w)


def _choice(rng, weights):
    """An index drawn with probability ∝ weights, from a random.Random (deterministic for a seed)."""
    w = np.asarray(weights, np.float64)
    c = np.cumsum(w)
    if c[-1] <= 0:
        return rng.randrange(len(w))
    x = rng.random() * c[-1]
    return int(min(np.searchsorted(c, x, side="right"), len(w) - 1))


class ExperienceModel:
    """Counts of what the player saw (records, see the module docstring) and the sampler built on them.

    add(counter)            add records (a Counter of keys, e.g. ExperienceCollector.drain() of every worker)
    determinization(view, challenge, rng, open_level=True, hint=None)
                            Gen3Game.determinize specs (the layout of rl.determinize.sample_determinization)
    slot_logp(view, slot, challenge, species, moves, item)
                            log P(that configuration for that slot | what the view shows) (validation)
    state() / from_state(), save(path) / load(path)"""

    def __init__(self, cfg=None, counts=None):
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.counts = collections.Counter()
        self._compiled = None
        self.n_updates = 0
        if counts:
            self.add(counts)

    # --- data ------------------------------------------------------------------------------------------------------

    def add(self, counts):
        for k, v in dict(counts).items():
            k = (int(k[0]), int(k[1]), int(k[2]), tuple(int(x) for x in k[3]), int(k[4]), int(k[5]), int(k[6]))
            self.counts[k] += float(v)
        self._compiled = None
        self.n_updates += 1
        return self

    def n_records(self, source=None, full=None):
        return float(sum(v for k, v in self.counts.items()
                         if (source is None or k[0] == source) and (full is None or k[6] == int(full))))

    def summary(self):
        return {"records": self.n_records(), "opponent_sightings": self.n_records(OPP),
                "opponent_full": self.n_records(OPP, True), "rentals": self.n_records(RENTAL),
                "keys": len(self.counts), "species": len({k[2] for k in self.counts}),
                "configurations": len({(k[2], k[3], k[4]) for k in self.counts if k[6]})}

    def state(self):
        keys = sorted(self.counts)
        return {"version": VERSION, "cfg": dict(self.cfg), "keys": [list(k[:3]) + [list(k[3])] + list(k[4:])
                                                                  for k in keys],
                "counts": [float(self.counts[k]) for k in keys], "n_updates": int(getattr(self, "n_updates", 0))}

    @classmethod
    def from_state(cls, st, cfg=None):
        m = cls(dict(st.get("cfg") or {}, **(cfg or {})))
        m.counts = collections.Counter({(int(k[0]), int(k[1]), int(k[2]), tuple(k[3]), int(k[4]), int(k[5]),
                                         int(k[6])): float(c) for k, c in zip(st["keys"], st["counts"])})
        m.n_updates = int(st.get("n_updates", 0))
        return m

    def save(self, path):
        tmp = path + ".tmp"
        torch.save({"opponent_model": self.state()}, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, cfg=None):
        """A model file (save()) or a checkpoint that carries one ("opponent_model")."""
        st = torch.load(path, map_location="cpu", weights_only=False)
        st = st.get("opponent_model", st)
        if st is None or "keys" not in st:
            raise ValueError(f"{path}: no opponent model in it")
        return cls.from_state(st, cfg)

    def compile(self):
        if self._compiled is None:
            self._compiled = _Compiled(self.counts, self.cfg)
        return self._compiled

    # --- the sampler ------------------------------------------------------------------------------------------------

    def _draw_team(self, t, b, enemy, level, rng):
        """One team: [(slot, species, moves [4], item, nature, known)] for every slot (fainted seen ones included,
        for the hint), seen slots first."""
        used_species = {m.species for m in enemy if m.seen}
        used_items = {m.revealed_item for m in enemy if m.seen and m.revealed_item}
        out = {}
        for i in sorted(range(len(enemy)), key=lambda j: not enemy[j].seen):
            m = enemy[i]
            if m.seen:
                s = int(m.species)
                R = frozenset(x for x in m.revealed_moves if x)
                I = int(m.revealed_item) if m.revealed_item else -1
            else:
                p = t.species_probs(b, used_species)
                s = int(t.frontier[_choice(rng, p)])
                R, I = frozenset(), -1
                used_species.add(s)
            cs, pc, pn = t.config_posterior(b, s, R, I, used_items)
            k = _choice(rng, np.append(pc, pn)) if len(cs) else len(cs)
            if k < len(cs):
                moves, item = list(cs[k][0]), cs[k][1]
                nature = t.nature(s, cs[k], rng)
                known = True
            else:
                revealed = sorted(R)
                rest = [x for x in DET.learnable(s, level) if x not in R]
                n_more = min(4 - len(revealed), len(rest))
                moves = revealed + (rng.sample(rest, n_more) if n_more > 0 else [])
                if I >= 0:
                    item = I
                else:
                    items = [x for x in DET.plausible_items(s) if x not in used_items]
                    item = items[rng.randrange(len(items))] if items else 0
                nature = rng.randrange(DET.NUM_NATURES)
                known = False
            if item:
                used_items.add(item)
            out[i] = (s, (list(moves) + [0] * 4)[:4], int(item), nature, known)
        return out

    def determinization(self, view, challenge, rng=None, open_level=True, hint=None):
        """Specs for Gen3Game.determinize: [(slot, species | KEEP, [4 moves], item, [6 IVs], [6 EVs], nature,
        ability_bit, hp_fraction)] drawn from the experience (see the module docstring). hint: the heard
        (hint_type, hint_style) of this opponent (used only with cfg use_hint)."""
        rng = rng or random.Random()
        t = self.compile()
        b = bucket_of(challenge)
        level = DET.team_level(view)
        enemy = view.enemy_party[:3]
        team = self._draw_team(t, b, enemy, level, rng)
        if hint is not None and self.cfg["use_hint"] and len(enemy) == 3:
            h = (int(hint[0]), int(hint[1]))
            DET.hint_stats["calls"] += 1
            for _ in range(int(self.cfg["max_hint_tries"]) - 1):
                sp = np.array([[team[i][0] for i in range(3)]])
                mv = np.array([[team[i][1] for i in range(3)]])
                ht, hs = DET.hint_of_arrays(sp, mv)
                DET.hint_stats["draws"] += 1
                if (int(ht[0]), int(hs[0])) == h:
                    break
                team = self._draw_team(t, b, enemy, level, rng)
            else:
                DET.hint_stats["fallback"] += 1
        specs = []
        for i, m in enumerate(enemy):
            if m.seen and m.fainted:
                specs.append((i, DET.KEEP, [0, 0, 0, 0], 0, [0] * 6, [0] * 6, 0, 0, -1.0))
                continue
            s, moves, item, nature, known = team[i]
            ivs = [rng.randrange(32) for _ in range(6)]
            evs = DET._evs(rng)
            bit = DET._ability_bit(s, m if m.seen else None, rng)
            hp = DET._hp_fraction(DET.max_hp_of(s, ivs[0], evs[0], level), m.hp_pixels, rng) if m.seen else 1.0
            specs.append((i, s, moves, item, ivs, evs, nature, bit, hp))
        return specs

    # --- scoring (validation) -----------------------------------------------------------------------------------------

    def slot_posterior(self, view, slot, challenge):
        """What the model believes about one slot, as the sampler would draw it first (seen slots: their species;
        unseen: the species not seen on the team). -> dict with "species" (p over t.frontier, or None when seen)
        and a function config(species) -> (configs, probs, p_novel)."""
        t = self.compile()
        b = bucket_of(challenge)
        enemy = view.enemy_party[:3]
        m = enemy[slot]
        used_items = {x.revealed_item for j, x in enumerate(enemy) if j != slot and x.seen and x.revealed_item}
        if m.seen:
            R = frozenset(x for x in m.revealed_moves if x)
            I = int(m.revealed_item) if m.revealed_item else -1
            return {"species": None, "R": R, "I": I, "used_items": used_items,
                    "config": lambda s: t.config_posterior(b, s, R, I, used_items)}
        seen_sp = {x.species for x in enemy if x.seen}
        return {"species": t.species_probs(b, seen_sp), "frontier": t.frontier, "R": frozenset(), "I": -1,
                "used_items": used_items, "config": lambda s: t.config_posterior(b, s, frozenset(), -1, used_items)}

    def slot_logp(self, view, slot, challenge, species, moves, item):
        """log P(species, moveset, item | view) for one slot (the sampler's distribution for that slot as if drawn
        first among the unseen ones)."""
        post = self.slot_posterior(view, slot, challenge)
        lp = 0.0
        if post["species"] is not None:
            idx = np.flatnonzero(post["frontier"] == species)
            if not len(idx) or post["species"][idx[0]] <= 0:
                return -math.inf
            lp += math.log(post["species"][idx[0]])
        elif species != view.enemy_party[slot].species:
            return -math.inf
        cs, pc, pn = post["config"](species)
        key = (_moves_key(moves), int(item))
        for c, p in zip(cs, pc):
            if c == key:
                return lp + math.log(p) if p > 0 else -math.inf
        if pn <= 0:
            return -math.inf
        return lp + math.log(pn) + strict_config_logp(species, key[0], key[1], tuple(post["R"]),
                                                      post["I"] if post["I"] >= 0 else None, post["used_items"],
                                                      self.cfg["level"])

    def slot_rank(self, view, slot, challenge, species, moves, item, max_rank=1000):
        """Rank (1 = most likely) of the true configuration among the model's known ones for this slot (unseen
        slots: over species x configuration); max_rank + 1 when it is not among them (novel for the model)."""
        post = self.slot_posterior(view, slot, challenge)
        key = (int(species), _moves_key(moves), int(item))
        scored = []
        if post["species"] is None:
            sp_list = [(int(view.enemy_party[slot].species), 1.0)]
        else:
            sp_list = [(int(s), float(p)) for s, p in zip(post["frontier"], post["species"]) if p > 0]
        for s, ps in sp_list:
            cs, pc, _ = post["config"](s)
            scored += [(ps * p, (s,) + c) for c, p in zip(cs, pc) if p > 0]
        scored.sort(key=lambda x: -x[0])
        for r, (_, k) in enumerate(scored[:max_rank]):
            if k == key:
                return r + 1
        return max_rank + 1


# ---- the strict sampler's probabilities (validation baseline) --------------------------------------------------------

def strict_slot_logp(view, slot, species, moves, item, level=100):
    """log P(species, moveset, item | view) under the strict sampler (rl.determinize.sample_determinization) for
    one slot, as if drawn first among the unseen ones."""
    enemy = view.enemy_party[:3]
    m = enemy[slot]
    used_items = {x.revealed_item for j, x in enumerate(enemy) if j != slot and x.seen and x.revealed_item}
    lp = 0.0
    if m.seen:
        if species != m.species:
            return -math.inf
        R = tuple(x for x in m.revealed_moves if x)
        I = m.revealed_item or None
    else:
        seen_sp = {x.species for x in enemy if x.seen}
        avail = [s for s in DET._FRONTIER_SPECIES[True] if s not in seen_sp]
        if species not in avail:
            return -math.inf
        lp -= math.log(len(avail))
        R, I = (), None
    return lp + strict_config_logp(species, _moves_key(moves), int(item), R, I, used_items, level)


def merge(counters):
    """Sum record Counters in the given order (deterministic)."""
    out = collections.Counter()
    for c in counters:
        for k in sorted(c):
            out[k] += c[k]
    return out
