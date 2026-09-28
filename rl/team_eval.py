"""Team evaluator network (alphazero_v2, docs/RL_DECISIONS.md §19.2).

    ev = TeamEvaluator.load("runs/team_eval_v1/model.pt")
    ev.w(teams, round_idx)                         # [n, 3]: W(normal), W(last battle), W(Noland) in [0, 1]
    team_of_option(view, "rental" | "swap", option)                    # the team an option leads to
    option_features(ev, view, kind, ctx, options, v_next)             # [n_opt, 4] = [E_round, W(normal),
                                                                      #   W(last or Noland), Delta]
    phi(ev, view, kind_or_state, ctx, v_next)                         # E_round + P(complete) * v_next

Definitions (§19.2):
    round        7 battles; a team only lives in its round (it is handed back when the round ends)
    k            index 1..7 of the next battle in the round: streak % 7 + 1
    W(team, r, t) probability that our battler (the network it was trained with, greedy) wins ONE battle with
                 `team` against an opponent drawn for round r (the Factory's real sets, built as the game builds
                 them) of battle type t: normal (battles 1-6, IV 3), last (battle 7, IV 6), Noland (battle 7 of
                 rounds 3 and 6: his own IV row, sets excluded instead of species)
    p_i          W of the type of battle i (i < 7 normal; i = 7 last, or Noland when the game schedules him)
    E_round      p_k + p_k p_{k+1} + ... + p_k ... p_7: expected wins in the rest of the round keeping the team
    P(complete)  p_k p_{k+1} ... p_7
    Phi          E_round + P(complete) * v_next, v_next = expected wins from the start of the next round (given by
                 the caller: it does not depend on the team, which is handed back)
    Delta        E_round(resulting team) - E_round(current team) (swap), or - mean over the options (rental)

What the evaluator sees of a team (TeamMon): only what the tactician's view shows. Our own Pokemon (rentals, and
our team at a swap screen) are fully visible on the summary screen: species, moves, held item, ability and stats
(IVs / EVs / nature are not used: they are already in the stats). A Pokemon we would trade in at a swap screen is
known only as the player saw it in battle (SeenMon: species, the moves it used, the item and ability the game
named); the rest is UNKNOWN and its stats unknown. Nothing of any opponent's hidden state is read.

Rounds beyond 6 (open level, the only mode trained): W(team, r) = W(team, 6) exactly, as the game does: the
opponent pool row is min(challenge, 7) and rows 4..7 are the same range at open level, normal / last opponents always
get IV 3 / 6 (GetFactoryMonFixedIV is called with the Battle Tower streak: the "Tower bug"), and Noland's IV row is
31 from round 6 on. So round_idx is clamped to 1..6 (round 5 and 6 share the normal / last opponents too).

Training: `python -m rl.team_eval train --data runs/team_eval_v1/data --out runs/team_eval_v1/model.pt`
(data from rl/team_eval_data.py).

alphazero_v3 (docs §20):
    ev.w(teams, round_idx, hint)                   # W conditioned on the attendant's hint about the battle's
                                                   #   opponent (hint_type 0..18, hint_style 0..8; None = unknown)
    plan_values(ev, view, kind, ctx, options, v_next, hint, planner_cfg)   # Q_plan: the planner (see its section)
    option_features_v3(ev, view, kind, ctx, options, v_next_table, hint, planner_cfg)   # [n_opt, N_OPT_FEATURES]
    phi_v3(ev, view, kind, ctx, v_next_table, hint, planner_cfg); q_eval_v3(features)
A W trained without hints (v1) works everywhere: the hint is ignored.
"""

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pybattle.backend import factory_brain_status, factory_symbols_for_streak
from pybattle.view import possible_abilities
from . import encode as E
from .gamedata import MOVES, N_ABILITIES, N_EFFECTS, N_ITEMS, N_MOVES, N_SPECIES, N_TYPES, SPECIES

ENCODE_VERSION = 1                  # the team featurization below
N_ROUNDS = 6                        # rounds >= 6 are round 6 (see the module docstring)
TYPES = ("normal", "last", "noland")
NORMAL, LAST, NOLAND = 0, 1, 2
UNKNOWN = -1                        # a move / item the player has not seen
MON_TOK = 7                         # species, item, ability, 4 moves
MON_NUM = 10
# the attendant's hint about a battle's opponent (alphazero_v3, docs §20): hint_type 0..18 (18: no single most
# common type), hint_style 0..8 (FACTORY_STYLE_*, 8: more than two styles); plus an "unknown" token each, used for a
# battle whose hint the player has not heard yet (a future battle), for Noland (no hint) and, in training, for a
# fraction of the battles (hint dropout) so that W(team | unknown) is the average over the hints
N_HINT_TYPES, N_HINT_STYLES = 19, 9
HINT_UNK_TYPE, HINT_UNK_STYLE = N_HINT_TYPES, N_HINT_STYLES


def hint_arrays(hint, n):
    """hint -> (hint_type [n], hint_style [n]) int64 token ids. hint: None (unknown for all), one (type, style) pair,
    or per-team pairs / an [n, 2] array (None or negative entries: unknown)."""
    ht = np.full(n, HINT_UNK_TYPE, np.int64)
    hs = np.full(n, HINT_UNK_STYLE, np.int64)
    if hint is None:
        return ht, hs
    if isinstance(hint, tuple) and len(hint) == 2 and np.isscalar(hint[0]):
        hint = [hint] * n
    for i, h in enumerate(hint):
        if h is None:
            continue
        t, st = int(h[0]), int(h[1])
        if t >= 0 and st >= 0:
            ht[i], hs[i] = min(t, HINT_UNK_TYPE), min(st, HINT_UNK_STYLE)
    return ht, hs


# ---- teams ------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TeamMon:
    """One of our Pokemon as the player knows it. moves: 4 ids (0 = no move, UNKNOWN = not seen); item: id (0 =
    none, UNKNOWN = not seen); ability: id (0 = not seen); stats: (max HP, Atk, Def, Spe, SpA, SpD) or None."""
    species: int
    moves: Tuple[int, int, int, int]
    item: int
    ability: int
    stats: Optional[Tuple[int, ...]] = None

    @classmethod
    def from_own(cls, m) -> "TeamMon":
        """A Pokemon the player controls (pybattle.view.OwnMon: summary screen)."""
        mv = (list(m.moves) + [0] * 4)[:4]
        return cls(int(m.species), tuple(int(x) for x in mv), int(m.item), int(m.ability),
                   (int(m.max_hp), *(int(s) for s in m.stats)))

    @classmethod
    def from_seen(cls, m) -> "TeamMon":
        """An opponent Pokemon as seen in battle (pybattle.view.SeenMon), e.g. the one traded in at a swap."""
        rev = [int(x) for x in m.revealed_moves if x][:4]
        mv = rev + [UNKNOWN] * (4 - len(rev))
        item = UNKNOWN if m.revealed_item is None else int(m.revealed_item)
        ab = int(m.revealed_ability or 0)
        pa = possible_abilities(m.species)
        if not ab and len(pa) == 1:
            ab = pa[0]                                  # a one-ability species: known
        return cls(int(m.species), tuple(mv), item, ab, None)


@dataclass(frozen=True)
class Team:
    """Our 3 Pokemon in party order: mons[0] leads."""
    mons: Tuple[TeamMon, TeamMon, TeamMon]

    @property
    def lead(self) -> TeamMon:
        return self.mons[0]

    @property
    def species(self):
        return tuple(m.species for m in self.mons)


def _is_keep(option):
    from .envs import NO_ACTION
    return option is None or option is NO_ACTION or (isinstance(option, (int, np.integer)) and int(option) == 0)


def swap_trade(option):
    """A swap option -> None (keep) or (own_slot, foe_slot). Accepts the network's action index 0..9 (0 keep,
    1 + 3 i + j), None / rl.envs.NO_ACTION (keep) or an (own_slot, foe_slot) pair."""
    if _is_keep(option):
        return None
    if isinstance(option, (int, np.integer)):
        a = int(option)
        if not 1 <= a <= 9:
            raise ValueError(f"swap action {a} out of range")
        return (a - 1) // 3, (a - 1) % 3
    i, j = option
    return int(i), int(j)


def rental_pick(option):
    """A rental option -> (lead, j, k) rental slots in party order. Accepts the network's (lead, pair_index)
    (rl.envs.decode_action) or the backend's (lead, j, k)."""
    if len(option) == 3:
        return tuple(int(x) for x in option)
    lead, pair = int(option[0]), int(option[1])
    j, k = E.PAIRS[pair]
    return lead, j, k


def current_team(view) -> Team:
    """Our team at a swap screen (SwapView.own_party; slot 0 leads)."""
    return Team(tuple(TeamMon.from_own(m) for m in view.own_party[:3]))


def team_of_option(view, kind: str, option) -> Team:
    """The team a tactician option leads to, from the view only. rental: RentalView and (lead, pair_index) as
    rl.tactician_search.options_of gives them (the party is [lead, j, k], PAIRS[pair] = (j, k)); swap: SwapView and
    an action 0..9 / None / NO_ACTION / (own_slot, foe_slot): the traded-in Pokemon takes the slot of the one it
    replaces (Gen3Game.factory_swap) and is known only as it was seen in battle."""
    if kind == "rental":
        lead, j, k = rental_pick(option)
        return Team(tuple(TeamMon.from_own(view.candidates[i]) for i in (lead, j, k)))
    if kind != "swap":
        raise ValueError(f"kind must be 'rental' or 'swap', not {kind!r}")
    team = list(current_team(view).mons)
    t = swap_trade(option)
    if t is not None:
        team[t[0]] = TeamMon.from_seen(view.enemy_party[t[1]])
    return Team(tuple(team))


# ---- featurization -------------------------------------------------------------------------------------------------------

def featurize_arrays(species, moves, item, ability, stats, stats_known):
    """Vectorized team features. species [n,3], moves [n,3,4], item [n,3], ability [n,3] (UNKNOWN / 0 as in
    TeamMon), stats [n,3,6], stats_known [n,3] -> (ids int64 [n,3,7], num float32 [n,3,MON_NUM])."""
    species = np.asarray(species, np.int64)
    moves = np.asarray(moves, np.int64)
    item = np.asarray(item, np.int64)
    ability = np.asarray(ability, np.int64)
    n = species.shape[0]
    mv = np.where(moves == UNKNOWN, E.MOVE_UNK, moves)
    it = np.where(item == UNKNOWN, E.ITEM_UNK, item)
    ab = np.where(ability <= 0, E.ABILITY_UNK, ability)
    ids = np.concatenate([species[..., None], it[..., None], ab[..., None], mv], -1)
    known = np.asarray(stats_known, bool)
    st = np.where(known[..., None], np.asarray(stats, np.float32), 0.0) / 500.0
    num = np.zeros((n, 3, MON_NUM), np.float32)
    num[..., 0] = known
    num[..., 1:7] = st
    num[:, 0, 7] = 1.0                                   # lead
    num[..., 8] = item != UNKNOWN
    num[..., 9] = (moves == UNKNOWN).sum(-1) / 4.0
    return ids, num


def featurize(teams):
    """Teams -> (ids, num) (featurize_arrays)."""
    n = len(teams)
    sp = np.zeros((n, 3), np.int64)
    mv = np.zeros((n, 3, 4), np.int64)
    it = np.zeros((n, 3), np.int64)
    ab = np.zeros((n, 3), np.int64)
    st = np.zeros((n, 3, 6), np.float32)
    kn = np.zeros((n, 3), bool)
    for a, t in enumerate(teams):
        mons = t.mons if isinstance(t, Team) else t
        for b, m in enumerate(mons):
            sp[a, b], it[a, b], ab[a, b] = m.species, m.item, m.ability
            mv[a, b] = m.moves
            if m.stats is not None:
                st[a, b], kn[a, b] = m.stats, True
    return featurize_arrays(sp, mv, it, ab, st, kn)


def round_index(round_idx):
    """round 1.. -> embedding row 0..N_ROUNDS-1 (clamped: see the module docstring)."""
    return np.clip(np.asarray(round_idx, np.int64), 1, N_ROUNDS) - 1


# ---- the network ---------------------------------------------------------------------------------------------------------

def _static_tables():
    """Game knowledge per token: move effect / type / numbers, species types / base stats / abilities."""
    m_eff = np.full(E.N_MOVE_TOK, E.EFFECT_NONE, np.int64)
    m_type = np.full(E.N_MOVE_TOK, E.TYPE_NONE, np.int64)
    m_num = np.zeros((E.N_MOVE_TOK, 6), np.float32)
    for i in range(1, N_MOVES):
        d = MOVES[i]
        m_eff[i], m_type[i] = d["effect"], d["type"]
        m_num[i] = [d["power"] / 255, d["accuracy"] / 100, float(d["accuracy"] == 0), d["priority"] / 6,
                    d["pp"] / 64, 1.0]
    m_eff[E.MOVE_UNK], m_type[E.MOVE_UNK] = E.EFFECT_UNK, E.TYPE_UNK
    s_type = np.full((E.N_SPECIES_TOK, 2), E.TYPE_NONE, np.int64)
    s_base = np.zeros((E.N_SPECIES_TOK, 6), np.float32)
    s_abil = np.zeros((E.N_SPECIES_TOK, 2), np.int64)
    for i in range(1, N_SPECIES):
        d = SPECIES[i]
        t = list(d["types"])
        s_type[i] = [t[0], t[1] if len(t) > 1 and t[1] != t[0] else E.TYPE_NONE]
        s_base[i] = np.array(d["base"], np.float32) / 255
        a = list(d["abilities"]) + [0, 0]
        s_abil[i] = [a[0], a[1] if a[1] != a[0] else 0]
    return m_eff, m_type, m_num, s_type, s_base, s_abil


class TeamNet(nn.Module):
    """Set encoder over our 3 Pokemon (lead = a flag on the first) + round embedding -> 3 logits (normal, last,
    Noland). Pokemon tokens: embeddings of species / types / item / ability / possible abilities, the sum of 4 move
    encodings (move, effect, type embeddings + power / accuracy / priority / PP), base stats and the numbers of
    featurize_arrays; the round embedding is added to every token; a small transformer without positions mixes
    them (permutation-equivariant; order only enters through the lead flag); mean-pool + lead token + round -> MLP.

    hint=True (W_v3, docs §20): the head is also conditioned on the attendant's hint about the battle's opponent:
    an embedding of hint_type (19 + unknown) plus one of hint_style (9 + unknown), and an attention read-out of the 3
    Pokemon tokens with a query made from that hint embedding (which of our Pokemon matters against that kind of
    opponent); head input = [mean, lead, round, hint embedding, read-out]. The team part (encode) does not depend on
    the hint, so a team is encoded once and its head evaluated for every hint (training: once per battle)."""

    def __init__(self, d=128, d_emb=32, layers=2, heads=4, hint=False):
        super().__init__()
        self.cfg = dict(d=d, d_emb=d_emb, layers=layers, heads=heads, hint=bool(hint))
        self.hint = bool(hint)
        for name, arr in zip(("m_eff", "m_type", "m_num", "s_type", "s_base", "s_abil"), _static_tables()):
            self.register_buffer(name, torch.from_numpy(arr), persistent=False)
        self.species = nn.Embedding(E.N_SPECIES_TOK, 2 * d_emb)
        self.move = nn.Embedding(E.N_MOVE_TOK, d_emb)
        self.effect = nn.Embedding(E.N_EFFECT_TOK, d_emb // 2)
        self.type_emb = nn.Embedding(E.N_TYPE_TOK, d_emb // 2)
        self.item = nn.Embedding(E.N_ITEM_TOK, d_emb)
        self.ability = nn.Embedding(E.N_ABILITY_TOK, d_emb)
        self.round = nn.Embedding(N_ROUNDS, d)
        dm = 2 * d_emb
        self.move_enc = nn.Sequential(nn.Linear(d_emb + d_emb // 2 + 6, dm), nn.ReLU(), nn.Linear(dm, dm))
        self.mon_enc = nn.Sequential(nn.Linear(2 * d_emb + d_emb // 2 + 3 * d_emb + dm + 6 + MON_NUM, d), nn.ReLU(),
                                     nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        if self.hint:
            self.hint_type = nn.Embedding(N_HINT_TYPES + 1, d)
            self.hint_style = nn.Embedding(N_HINT_STYLES + 1, d)
            self.hint_q = nn.Linear(d, d)
            self.hint_k = nn.Linear(d, d)
            self.hint_v = nn.Linear(d, d)
            self.head = nn.Sequential(nn.Linear(5 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d), nn.ReLU(),
                                      nn.Linear(d, 3))
        else:
            self.head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 3))

    def forward(self, ids, num, rnd, hint_type=None, hint_style=None):
        """ids int64 [B,3,7], num float [B,3,MON_NUM], rnd int64 [B] (0..N_ROUNDS-1), hint_type / hint_style int64
        [B] (hint model; None: unknown) -> logits [B,3]."""
        h, z = self.encode(ids, num, rnd)
        return self.head_logits(h, z, hint_type, hint_style)

    def head_logits(self, h, z, hint_type=None, hint_style=None):
        """Logits [B,3] from encode's (tokens h [B,3,d], pooled z [B,3d]) and the hint tokens [B]."""
        if not self.hint:
            return self.head(z)
        B = z.shape[0]
        if hint_type is None:
            hint_type = torch.full((B,), HINT_UNK_TYPE, dtype=torch.long, device=z.device)
        if hint_style is None:
            hint_style = torch.full((B,), HINT_UNK_STYLE, dtype=torch.long, device=z.device)
        e = self.hint_type(hint_type) + self.hint_style(hint_style)                  # [B,d]
        att = torch.einsum("bd,btd->bt", self.hint_q(e), self.hint_k(h)) / math.sqrt(h.shape[-1])
        read = torch.einsum("bt,btd->bd", torch.softmax(att, -1), self.hint_v(h))
        return self.head(torch.cat([z, e, read], -1))

    def encode(self, ids, num, rnd):
        """-> (tokens [B,3,d], pooled [B,3d] = [mean, lead token, round embedding])."""
        sp, it, ab, mv = ids[..., 0], ids[..., 1], ids[..., 2], ids[..., 3:7]
        mvec = torch.cat([self.move(mv), self.effect(self.m_eff[mv]) + self.type_emb(self.m_type[mv]),
                          self.m_num[mv]], -1)
        mvec = self.move_enc(mvec).sum(2)                                        # [B,3,dm]
        st, sa = self.s_type[sp], self.s_abil[sp]
        mon = torch.cat([self.species(sp), self.type_emb(st[..., 0]) + self.type_emb(st[..., 1]), self.item(it),
                         self.ability(ab), self.ability(sa[..., 0]) + self.ability(sa[..., 1]), mvec,
                         self.s_base[sp], num], -1)
        r = self.round(rnd)                                                       # [B,d]
        h = self.mon_enc(mon) + r[:, None]
        h = self.norm(self.mix(h))
        return h, torch.cat([h.mean(1), h[:, 0], r], -1)


# ---- the evaluator -------------------------------------------------------------------------------------------------------

def _single_thread_in_children():
    """In a worker process (forked FactoryEnv / pool workers), torch must not enter the intra-op thread pool it
    inherited from a parent that already used it: that deadlocks (seen in rl.eval_rounds' VecEnv workers while
    building the network). A few teams per call need one thread anyway. The main process is left alone."""
    import multiprocessing as mp
    if mp.parent_process() is not None and torch.get_num_threads() != 1:
        torch.set_num_threads(1)


class TeamEvaluator:
    """W(team, round, type) from a trained TeamNet (CPU by default: a few teams per decision)."""

    def __init__(self, net: TeamNet, meta=None, device="cpu"):
        self.net = net.to(device).eval()
        self.meta = dict(meta or {})
        self.device = torch.device(device)

    @classmethod
    def load(cls, path, device="cpu") -> "TeamEvaluator":
        _single_thread_in_children()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("encode_version") != ENCODE_VERSION:
            raise ValueError(f"team encoding version {ck.get('encode_version')} != {ENCODE_VERSION}")
        net = TeamNet(**ck["cfg"])                      # (a v1 checkpoint's cfg has no "hint": hint=False)
        net.load_state_dict(ck["net"])
        return cls(net, ck.get("meta"), device)

    def save(self, path, meta=None):
        m = dict(self.meta, **(meta or {}))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"net": {k: v.detach().cpu() for k, v in self.net.state_dict().items()}, "cfg": self.net.cfg,
                    "encode_version": ENCODE_VERSION, "meta": m}, path + ".tmp")
        os.replace(path + ".tmp", path)

    @property
    def uses_hint(self) -> bool:
        """W conditioned on the attendant's hint (W_v3); a v1 network ignores the hint."""
        return bool(getattr(self.net, "hint", False))

    @torch.no_grad()
    def w_features(self, ids, num, rounds, hint_type=None, hint_style=None):
        """W [n, 3] from featurized teams; hint_type / hint_style: token ids [n] (None: unknown)."""
        _single_thread_in_children()
        rnd = round_index(np.broadcast_to(np.asarray(rounds), (len(ids),)))
        x = [torch.from_numpy(np.ascontiguousarray(a)).to(self.device) for a in (ids, num, rnd)]
        if self.uses_hint and hint_type is not None:
            x += [torch.from_numpy(np.ascontiguousarray(np.asarray(a, np.int64))).to(self.device)
                  for a in (hint_type, hint_style)]
        return torch.sigmoid(self.net(*x).float()).cpu().numpy().astype(np.float64)

    def w(self, teams: list, round_idx, hint=None) -> np.ndarray:
        """[n, 3]: W(normal), W(last battle), W(Noland) in [0, 1] of each team in round `round_idx` (an int, or one
        per team). hint (W_v3): the attendant's (hint_type, hint_style) about the battle's opponent, one pair for all
        or one per team; None = unknown (a future battle, or Noland). Ignored by a v1 network."""
        if len(teams) == 0:
            return np.zeros((0, 3))
        ids, num = featurize(teams)
        if hint is None or not self.uses_hint:
            return self.w_features(ids, num, round_idx)
        ht, hs = hint_arrays(hint, len(teams))
        return self.w_features(ids, num, round_idx, ht, hs)


# ---- rounds: E_round, P(complete), Phi ------------------------------------------------------------------------------------

def round_of(ctx):
    """(round 1.., k 1..7) of the next battle from the streak context (rl.search.run_ctx / FactoryEnv._ctx):
    "challenge" = rounds completed, "battle" = battles already won in this round (the convention rl/alphazero_v2
    and rl/tactician_features.py use); a ctx with only "streak" is read as streak // 7, streak % 7."""
    if "challenge" in ctx and "battle" in ctx:
        return int(ctx["challenge"]) + 1, int(ctx["battle"]) + 1
    s = int(ctx["streak"])
    return s // 7 + 1, s % 7 + 1


def battle_types(round_idx):
    """Type (NORMAL / LAST / NOLAND) of battles 1..7 of a round: battle 7 is Noland when the game schedules him
    (GetFrontierBrainStatus with the symbols a player at that streak holds: rounds 3 and 6, then every 3 rounds)."""
    last = 7 * round_idx - 1                       # wins before battle 7
    brain = factory_brain_status(last, factory_symbols_for_streak(last)) != 0
    return [NORMAL] * 6 + [NOLAND if brain else LAST]


def round_values(W, k, round_idx):
    """W [n, 3] -> (E_round [n], P(complete) [n]) from battle k (1..7) to battle 7."""
    W = np.asarray(W, np.float64)
    types = battle_types(round_idx)
    n = W.shape[0]
    e, run = np.zeros(n), np.ones(n)
    for i in range(k, 8):
        run = run * W[:, types[i - 1]]
        e = e + run
    return e, run


def last_type(round_idx):
    return battle_types(round_idx)[6]


def option_features(ev: TeamEvaluator, view, kind, ctx, options, v_next: float = 0.0) -> np.ndarray:
    """[n_opt, 4] = [E_round, W(normal), W(last or Noland: whichever battle 7 of this round is), Delta] of the team
    each option leads to. Delta: E_round - E_round(current team) (swap), E_round - mean over `options` (rental).
    v_next is not needed here (kept for a uniform signature)."""
    r, k = round_of(ctx)
    teams = [team_of_option(view, kind, o) for o in options]
    if kind == "swap":
        teams.append(current_team(view))
    W = ev.w(teams, r)
    e, _ = round_values(W, k, r)
    if kind == "swap":
        base, W, e = e[-1], W[:-1], e[:-1]
    else:
        base = e.mean() if len(e) else 0.0
    return np.stack([e, W[:, NORMAL], W[:, last_type(r)], e - base], 1).astype(np.float32) if len(e) \
        else np.zeros((0, 4), np.float32)


def phi_teams(ev: TeamEvaluator, teams, ctx, v_next: float) -> np.ndarray:
    """Phi of each team at the state with streak context ctx: E_round + P(complete) * v_next."""
    r, k = round_of(ctx)
    e, p = round_values(ev.w(list(teams), r), k, r)
    return e + p * float(v_next)


def phi_options(ev: TeamEvaluator, view, kind, ctx, options, v_next: float) -> np.ndarray:
    """Phi of the state after each option (the team it leads to, before battle k): Q_eval of §19.3."""
    return phi_teams(ev, [team_of_option(view, kind, o) for o in options], ctx, v_next)


def phi(ev: TeamEvaluator, view, kind_or_state, ctx, v_next: float) -> float:
    """Phi = E_round + P(complete) * v_next at a tactician decision.
        kind_or_state "swap": the CURRENT team (SwapView.own_party);
        "rental": there is no team yet: the mean of Phi over the legal rentals (as Delta's baseline);
        a Team: that team (e.g. the state after the choice; also phi_options);
        ("rental" | "swap", option): the state after that option."""
    if isinstance(kind_or_state, Team):
        return float(phi_teams(ev, [kind_or_state], ctx, v_next)[0])
    if isinstance(kind_or_state, tuple) and len(kind_or_state) == 2 and isinstance(kind_or_state[0], str):
        kind, opt = kind_or_state
        return float(phi_options(ev, view, kind, ctx, [opt], v_next)[0])
    if kind_or_state == "swap":
        return float(phi_teams(ev, [current_team(view)], ctx, v_next)[0])
    if kind_or_state == "rental":
        opts = legal_rentals(view)
        return float(phi_options(ev, view, "rental", ctx, opts, v_next).mean())
    raise ValueError(f"kind_or_state must be 'rental', 'swap', a Team or (kind, option), not {kind_or_state!r}")


def legal_rentals(view):
    """Every legal (lead, pair_index) of a RentalView (3 different species), as rl.encode.rental's pair_mask."""
    sp = [m.species for m in view.candidates]
    return [(lead, p) for lead in range(6) for p, (j, k) in enumerate(E.PAIRS)
            if lead not in (j, k) and len({sp[lead], sp[j], sp[k]}) == 3]


# how a defeated Pokemon looks at the swap screen, measured on 1,500 of them under alphazero_v1 greedy play
# (fraction with 0..4 distinct moves revealed; item named; ability announced, species with two abilities); used to
# hide what the player would not have seen of a swapped-in Pokemon in half of the training samples
REVEAL = {"moves": (0.180, 0.437, 0.243, 0.103, 0.038), "item": 0.187, "ability": 0.044}


# ---- the planner (alphazero_v3, docs/RL_DECISIONS.md §20.2) --------------------------------------------------------------
#
# V(T, k): expected wins from battle k of the round to its end (plus P(complete) * v_next) with team T, when the
# player may still swap after each battle. With depth D (planner_cfg.depth) and the hint h_k of battle k (known at
# the decision; later hints unknown):
#     V_D(T, k) = p_k(T | h_k) * (1 + E_opp[ max_o V_{D-1}(T + o, k + 1) ])     if D > 0 and k < 7
#     V_0(T, k) = E_round(T, k) + P(complete)(T, k) * v_next                      (Phi, the leaf)
# E_opp: over n_opp opponents of battle k drawn from the round's pool as the game draws them (no repeated species /
# item, none of the species on screen), with the hint h_k when it is known (rejection), the same draws for every
# option (common random numbers); o: keep, or trade one of our 3 Pokemon for one of that opponent's 3 (9 swaps), the
# traded-in Pokemon known only as a defeated one is seen at the swap screen (REVEAL: the data generator's masking);
# the swapped-in Pokemon takes the slot of the one it replaces. Deeper levels: the hints are unknown, the opponents
# of battle k + 1 (n_opp2 per level-1 opponent) also exclude that opponent's species. Depth 2 keeps, for each
# (option, opponent), the `beam` best level-1 options by V_0 and expands only those (V_1 >= V_0: a skipped option
# would have been dominated by the beam unless its planning value is larger, which the beam ignores).
# Q_plan(option) = V_D(team after the option, k); depth 0 is Phi (§19.2) with the hint.

@dataclass
class PlannerConfig:
    depth: int = 1            # 0: Phi; 1: one swap ahead; 2: two swaps ahead
    n_opp: int = 8            # opponents of battle k sampled (common to all options)
    n_opp2: int = 4           # opponents of battle k + 1 per level-1 opponent (depth 2)
    beam: int = 3             # depth 2: level-1 options expanded per (option, opponent)
    device: str = "cpu"       # (informational: the network runs where the TeamEvaluator was loaded)
    seed: Optional[int] = None  # None: fresh draws at every call
    chunk: int = 16384        # teams per network call


def planner_config(cfg) -> PlannerConfig:
    if cfg is None:
        return PlannerConfig()
    if isinstance(cfg, PlannerConfig):
        return cfg
    if isinstance(cfg, dict):
        return PlannerConfig(**cfg)
    raise TypeError(f"planner_cfg must be None, a dict or a PlannerConfig, not {type(cfg).__name__}")


def _mon_rows(mons):
    """Per-mon features of featurize_arrays (lead flag unset): ids [M, 7], num [M, MON_NUM]."""
    M = len(mons)
    pad = (-M) % 3
    tm = list(mons) + [mons[0]] * pad
    teams = [tuple(tm[i:i + 3]) for i in range(0, len(tm), 3)]
    ids, num = featurize(teams)
    num[:, 0, 7] = 0.0
    return ids.reshape(-1, MON_TOK)[:M], num.reshape(-1, MON_NUM)[:M]


class _MonTable:
    def __init__(self):
        self.mons, self.index = [], {}

    def add(self, m) -> int:
        i = self.index.get(m)
        if i is None:
            i = self.index[m] = len(self.mons)
            self.mons.append(m)
        return i


def seen_set_mon(set_id: int, g, reveal=REVEAL) -> TeamMon:
    """A Battle Frontier set as a defeated opponent Pokemon is known at the swap screen: its species, a random
    number of its moves (REVEAL distribution), its item with REVEAL's probability, its ability if the species has
    only one or with REVEAL's probability (the species' coin flip), stats unknown (TeamMon.from_seen)."""
    from . import determinize as DET
    sp = int(DET.SET_SPECIES[set_id])
    real = [int(x) for x in DET.SET_MOVES_AS_BUILT[set_id] if x]
    pm = np.asarray(reveal["moves"], np.float64)
    r = int(g.choice(len(pm), p=pm / pm.sum()))
    keep = [real[i] for i in g.permutation(len(real))[:r]] if real else []
    moves = tuple((keep + [UNKNOWN] * 4)[:4])
    item = int(DET.SET_ITEM[set_id]) if g.random() < reveal["item"] else UNKNOWN
    pa = possible_abilities(sp)
    ab = pa[0] if len(pa) == 1 else (int(pa[int(g.integers(2))]) if g.random() < reveal["ability"] else 0)
    return TeamMon(sp, moves, item, int(ab), None)


def leaf_values(W_first, W_rest, j, round_idx, v_next):
    """Phi from battle j: E_round + P(complete) * v_next with p_j from W_first [..., 3] (the battle whose hint may be
    known) and p_i, i > j, from W_rest [..., 3] (hint unknown)."""
    types = battle_types(round_idx)
    run = W_first[..., types[j - 1]]
    e = run.copy()
    for i in range(j + 1, 8):
        run = run * W_rest[..., types[i - 1]]
        e = e + run
    return e + run * float(v_next)


def _visible_species(view, kind):
    if kind == "rental":
        return {int(m.species) for m in view.candidates}
    return {int(m.species) for m in view.own_party[:3]} | {int(m.species) for m in view.enemy_party[:3]}


def _swap_children(teams, opp_mons):
    """teams [..., 3] mon indices, opp_mons [..., 3] (broadcast) -> [..., 10, 3]: keep, then own slot i traded
    for opponent Pokemon j at index 1 + 3 i + j (the network's swap action order)."""
    teams, opp_mons = np.broadcast_arrays(teams, opp_mons)
    out = np.repeat(teams[..., None, :], 10, axis=-2).copy()
    for i in range(3):
        for j in range(3):
            out[..., 1 + 3 * i + j, i] = opp_mons[..., j]
    return out


def plan_values(ev, view, kind, ctx, options, v_next, hint=None, planner_cfg=None, with_current=False):
    """The planner's values of the tactician options (see above). -> dict:
        q_plan [n_opt]       V_D(team after the option, k)
        phi [n_opt]          V_0 = E_round + P(complete) v_next with the hint (Q_eval of §19.3 with the hint)
        e_round [n_opt]      E_round with the hint
        w_hint [n_opt, 3]    W(team | h_k) (unknown if no hint / Noland)
        w_unknown [n_opt, 3] W(team | unknown)
        current              (with_current, swap) the same keys for the current team (keep)
        teams                number of distinct teams evaluated
    hint: the attendant's (hint_type, hint_style) about battle k's opponent, or None; ignored when battle k is
    Noland's."""
    from . import determinize as DET
    cfg = planner_config(planner_cfg)
    r, k = round_of(ctx)
    types = battle_types(r)
    if types[k - 1] == NOLAND:
        hint = None
    g = np.random.default_rng(cfg.seed)
    table = _MonTable()
    root_teams = [team_of_option(view, kind, o) for o in options]
    if with_current and kind == "swap":
        root_teams.append(current_team(view))
    R = len(root_teams)
    if R == 0:
        z = np.zeros(0)
        return {"q_plan": z, "phi": z, "e_round": z, "w_hint": np.zeros((0, 3)), "w_unknown": np.zeros((0, 3)),
                "teams": 0}
    root = np.array([[table.add(m) for m in t.mons] for t in root_teams], np.int64)          # [R, 3]
    depth = cfg.depth if k < 7 else 0
    visible = _visible_species(view, kind)
    challenge = int(ctx.get("challenge", r - 1))
    pool_ids = DET.pool(challenge)
    groups = [root]
    L1 = L2 = None
    if depth >= 1:
        opp1 = DET.sample_pool_teams(pool_ids, cfg.n_opp, g, visible, hint)                   # [N, 3]
        m1 = np.array([[table.add(seen_set_mon(int(x), g)) for x in row] for row in opp1], np.int64)
        L1 = _swap_children(root[:, None, :], m1[None])                                     # [R, N, 10, 3]
        groups.append(L1.reshape(-1, 3))
        if depth >= 2 and k + 1 < 7:
            opp2 = np.stack([DET.sample_pool_teams(pool_ids, cfg.n_opp2, g, visible | set(DET.SET_SPECIES[row]
                                                                                          .tolist()), None)
                             for row in opp1])                                              # [N, N2, 3]
            m2 = np.array([[[table.add(seen_set_mon(int(x), g)) for x in row] for row in per_n] for per_n in opp2],
                          np.int64)
            L2 = m2
    # distinct teams among root + level 1 (level 2 is added after the beam)
    uniq = _TeamValues(ev, table, cfg, r)
    uniq.add(np.concatenate(groups))
    if L2 is not None:
        # beam at level 1 by V_0 (needs level-1 W first)
        uniq.evaluate()
        Wl1 = uniq.w_unknown(L1)                                                           # [R, N, 10, 3]
        v0_l1 = leaf_values(Wl1, Wl1, k + 1, r, v_next)                                    # [R, N, 10]
        B = min(max(1, cfg.beam), 10)
        top = np.argsort(-v0_l1, axis=-1, kind="stable")[..., :B]                          # [R, N, B]
        E = np.take_along_axis(L1, top[..., None], axis=2)                                 # [R, N, B, 3]
        L2t = _swap_children(E[:, :, :, None, :], L2[None, :, None, :, :])                  # [R, N, B, N2, 10, 3]
        uniq.add(L2t.reshape(-1, 3))
    uniq.evaluate()
    W_u_root = uniq.w_unknown(root)
    W_h_root = uniq.w_hint(root, hint) if hint is not None and ev_uses_hint(ev) else W_u_root
    phi_root = leaf_values(W_h_root, W_u_root, k, r, v_next)
    e_root = leaf_values(W_h_root, W_u_root, k, r, 0.0)
    p_k = W_h_root[:, types[k - 1]]
    if depth == 0:
        q = phi_root
    elif L2 is None:
        Wl1 = uniq.w_unknown(L1)
        v1 = leaf_values(Wl1, Wl1, k + 1, r, v_next)                                       # [R, N, 10]
        q = p_k * (1.0 + v1.max(-1).mean(-1))
    else:
        Wl2 = uniq.w_unknown(L2t)                                                          # [R, N, B, N2, 10, 3]
        v2 = leaf_values(Wl2, Wl2, k + 2, r, v_next).max(-1).mean(-1)                      # [R, N, B]
        WE = uniq.w_unknown(E)                                                             # [R, N, B, 3]
        v1 = WE[..., types[k]] * (1.0 + v2)                                                # V_1(E, k + 1)
        q = p_k * (1.0 + v1.max(-1).mean(-1))
    out = {"q_plan": q, "phi": phi_root, "e_round": e_root, "w_hint": W_h_root, "w_unknown": W_u_root,
           "teams": uniq.n}
    if with_current and kind == "swap":
        out["current"] = {key: v[-1] for key, v in out.items() if key != "teams"}
        for key in ("q_plan", "phi", "e_round", "w_hint", "w_unknown"):
            out[key] = out[key][:-1]
    return out


def ev_uses_hint(ev) -> bool:
    return bool(getattr(ev, "uses_hint", False))


def _team_keys(teams):
    t = np.asarray(teams, np.int64)
    return (t[..., 0] << 42) | (t[..., 1] << 21) | t[..., 2]


def _keys_to_teams(keys):
    m = (1 << 21) - 1
    return np.stack([keys >> 42, (keys >> 21) & m, keys & m], -1)


class _TeamValues:
    """W(team | unknown) of the distinct teams (mon-index triples) the planner meets, computed in batched network
    calls (each distinct team once)."""

    def __init__(self, ev, table, cfg, round_idx):
        self.ev, self.table, self.cfg, self.round_idx = ev, table, cfg, round_idx
        self.k = np.zeros(0, np.int64)                  # sorted keys of the evaluated teams
        self.Wu = np.zeros((0, 3))
        self.pending = np.zeros(0, np.int64)
        self._mon_ids = self._mon_num = None

    @property
    def n(self):
        return len(self.k) + len(self.pending)

    def add(self, teams):
        u = np.unique(_team_keys(np.asarray(teams, np.int64).reshape(-1, 3)))
        u = u[~np.isin(u, self.k)]
        self.pending = np.union1d(self.pending, u)

    def _features(self, triples):
        if self._mon_ids is None or len(self._mon_ids) < len(self.table.mons):
            self._mon_ids, self._mon_num = _mon_rows(self.table.mons)
        ids = self._mon_ids[triples]                                                  # [n, 3, 7]
        num = self._mon_num[triples].copy()
        num[:, 0, 7] = 1.0
        return ids, num

    def evaluate(self):
        if not len(self.pending):
            return
        new = self.pending
        self.pending = np.zeros(0, np.int64)
        W = self._forward(_keys_to_teams(new), None)
        k = np.concatenate([self.k, new])
        order = np.argsort(k, kind="stable")
        self.k, self.Wu = k[order], np.concatenate([self.Wu, W])[order]

    def _forward(self, triples, hint):
        ev, cfg = self.ev, self.cfg
        out = []
        for i in range(0, len(triples), cfg.chunk):
            ids, num = self._features(triples[i:i + cfg.chunk])
            if hint is None or not ev_uses_hint(ev):
                out.append(ev.w_features(ids, num, self.round_idx))
            else:
                ht, hs = hint_arrays(hint, len(ids))
                out.append(ev.w_features(ids, num, self.round_idx, ht, hs))
        return np.concatenate(out) if out else np.zeros((0, 3))

    def w_unknown(self, teams):
        return self.Wu[np.searchsorted(self.k, _team_keys(teams))]

    def w_hint(self, teams, hint):
        t = np.asarray(teams, np.int64)
        return self._forward(t.reshape(-1, 3), hint).reshape(t.shape[:-1] + (3,))


# ---- the v3 interface (rl/alphazero_v3.py) ---------------------------------------------------------------------------------
#
#   N_OPT_FEATURES                                                                      columns of option_features_v3
#   option_features_v3(ev, view, kind, ctx, options, v_next_table, hint, planner_cfg)  -> [n_opt, N_OPT_FEATURES]
#   phi_v3(ev, view, kind, ctx, v_next_table, hint, planner_cfg)                         -> float
#   q_eval_v3(features)                                                                  -> [n_opt] (Q_plan)
#
# options: as rl.tactician_search.options_of gives them; ctx: run_ctx ("challenge" = rounds completed, "battle" =
# battles won in the round); v_next_table: per-round table, table[r] = expected wins from the start of round r (as
# rl.tactician_features.v_next_of reads it: v_next = table[challenge + 2], the next round), or None (v_next = 0);
# hint: the attendant's (hint_type, hint_style) about the next battle's opponent (RentalView / SwapView), None if not
# heard (it is ignored before Noland's battle, when the game gives none); planner_cfg: PlannerConfig / dict / None.

OPT_FEATURES = ("e_round", "w_normal", "w_last", "delta", "q_plan", "q_plan_centered", "phi")
N_OPT_FEATURES = len(OPT_FEATURES)
Q_PLAN = OPT_FEATURES.index("q_plan")
# the network gets O(1) inputs: the columns in wins (E_round, delta, Q_plan, centered Q_plan, Phi; up to ~20 with
# v_next) are divided by WIN_SCALE; the W columns are probabilities. q_eval_v3 / phi_v3 stay in wins.
WIN_SCALE = 10.0
WIN_COLUMNS = (0, 3, 4, 5, 6)
_LAST_PLAN = {}                          # the last decision planned (phi_v3 after option_features_v3 reuses it)


def v_next_value(v_next_table, ctx) -> float:
    if v_next_table is None:
        return 0.0
    if np.isscalar(v_next_table):
        return float(v_next_table)
    r = int(ctx["challenge"]) + 2                      # the next round, 1-based
    return float(v_next_table[min(r, len(v_next_table) - 1)])


def _decision_key(ev, view, kind, ctx, v_next, hint, cfg):
    if kind == "rental":
        vis = tuple(int(m.species) for m in view.candidates) + tuple(int(m.item) for m in view.candidates)
    else:
        vis = tuple(int(m.species) for m in view.own_party[:3]) + tuple(int(m.species) for m in view.enemy_party[:3])
    h = None if hint is None else (int(hint[0]), int(hint[1]))
    return (id(ev), id(view), kind, vis, round_of(ctx), round(float(v_next), 9), h, repr(cfg))


def _opt_key(o):
    return tuple(int(x) for x in o) if isinstance(o, (tuple, list)) else (None if o is None else int(o))


def _plan(ev, view, kind, ctx, options, v_next_table, hint, planner_cfg, reuse=None):
    """plan_values with a one-entry cache: the same decision planned again (same view object, context, hint, v_next,
    config) returns the previous result; reuse(cached options) says whether a cached plan with other options
    serves (phi_v3)."""
    cfg = planner_config(planner_cfg)
    vn = v_next_value(v_next_table, ctx)
    options = list(options)
    dkey = _decision_key(ev, view, kind, ctx, vn, hint, cfg)
    okey = tuple(_opt_key(o) for o in options)
    if _LAST_PLAN.get("dkey") == dkey and (_LAST_PLAN["okey"] == okey or (reuse and reuse(_LAST_PLAN["okey"]))):
        return _LAST_PLAN["out"]
    out = plan_values(ev, view, kind, ctx, options, vn, hint, cfg, with_current=(kind == "swap"))
    _LAST_PLAN.clear()
    _LAST_PLAN.update(dkey=dkey, okey=okey, out=out)
    return out


def option_features_v3(ev, view, kind, ctx, options, v_next_table=None, hint=None, planner_cfg=None) -> np.ndarray:
    """[n_opt, N_OPT_FEATURES] per option, from the team it leads to (k = next battle, h = the hint of battle k):
        0 e_round           E_round with p_k = W(. | h), later battles W(. | unknown)            (/ WIN_SCALE)
        1 w_normal          W(normal | h)
        2 w_last            W(last or Noland, whichever battle 7 of this round is | h if k = 7 else unknown)
        3 delta             e_round - that of the current team (swap) / - the mean over the options (rental)  (/)
        4 q_plan            the planner's value V_D(team, k) (plan_values; wins, v_next included)          (/)
        5 q_plan_centered   q_plan - its mean over the options                                           (/)
        6 phi               V_0 = E_round + P(complete) * v_next with the hint (Q_eval of §19.3, with the hint) (/)
    The columns in wins are divided by WIN_SCALE (= 10: O(1) network inputs); q_eval_v3 undoes it."""
    options = list(options)
    if not options:
        return np.zeros((0, N_OPT_FEATURES), np.float32)
    r, k = round_of(ctx)
    out = _plan(ev, view, kind, ctx, options, v_next_table, hint, planner_cfg)
    lt = last_type(r)
    w_last = out["w_hint"][:, lt] if k == 7 else out["w_unknown"][:, lt]
    e = out["e_round"]
    base = out["current"]["e_round"] if kind == "swap" else e.mean()
    q = out["q_plan"]
    f = np.stack([e, out["w_hint"][:, NORMAL], w_last, e - base, q, q - q.mean(), out["phi"]], 1)
    f[:, WIN_COLUMNS] /= WIN_SCALE
    return f.astype(np.float32)


def phi_v3(ev, view, kind, ctx, v_next_table=None, hint=None, planner_cfg=None) -> float:
    """The shaping potential with the planner: swap: Q_plan of the current team (keeping it); rental (no team yet):
    the mean Q_plan over the legal rentals (the same reference as delta). Reuses option_features_v3's plan of the
    same decision when there is one."""
    if kind == "swap":
        out = _plan(ev, view, "swap", ctx, [0], v_next_table, hint, planner_cfg, reuse=lambda okey: True)
        return float(out["current"]["q_plan"])
    opts = legal_rentals(view)
    want = {_opt_key(o) for o in opts}
    out = _plan(ev, view, "rental", ctx, opts, v_next_table, hint, planner_cfg,
                reuse=lambda okey: set(okey) == want)
    return float(np.mean(out["q_plan"]))


def q_eval_v3(features) -> np.ndarray:
    """The expert's option values in wins: Q_plan (column q_plan of option_features_v3, times WIN_SCALE)."""
    f = np.asarray(features, np.float64)
    return f[:, Q_PLAN] * WIN_SCALE if f.size else np.zeros(0)


# ---- W refresh hooks (the v3 trainer calls these when the module has them) ------------------------------------------

def generate_battles(evaluator, n_battles, seed, *args, **kw):
    """rl.team_eval_refresh.generate_battles: simulated battles with the battler `evaluator`, each recording its
    opponent's hint (b_hint_type / b_hint_style; NO_HINT for Noland)."""
    from .team_eval_refresh import generate_battles as gen
    return gen(evaluator, n_battles, seed, *args, **kw)


def probe_teams(n=1024, seed=12345):
    """rl.team_eval_refresh.probe_teams (a fixed set of teams to measure how much W moves; hint unknown)."""
    from .team_eval_refresh import probe_teams as pt
    return pt(n, seed)


# ---- training ---------------------------------------------------------------------------------------------------------------



def load_data(data_dir):
    """Concatenate the worker chunks of rl.team_eval_data."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no data in {data_dir}")
    parts = [dict(np.load(f)) for f in files]
    team_keys = [k for k in parts[0] if not k.startswith("b_")]
    out = {}
    off = 0
    b_team = []
    for p in parts:
        b_team.append(p["b_team"] + off)
        off += len(p["round"])
    for k in team_keys:
        out[k] = np.concatenate([p[k] for p in parts])
    for k in [k for k in parts[0] if k.startswith("b_")]:
        out[k] = np.concatenate([p[k] for p in parts]) if k != "b_team" else np.concatenate(b_team)
    return out


def team_targets(d):
    """Per team: battles played and won (truncated battles excluded)."""
    n = len(d["round"])
    ok = ~d["b_trunc"]
    played = np.bincount(d["b_team"][ok], minlength=n)
    won = np.bincount(d["b_team"][ok], weights=d["b_won"][ok].astype(np.float64), minlength=n)
    return played, won


def hide_swapped(d, idx, rng, p_hide=0.5, reveal=REVEAL):
    """Features of teams `idx`; each swapped-in Pokemon is shown as seen at the swap screen with prob p_hide."""
    sp, mv, it = d["species"][idx], d["moves"][idx].copy(), d["item"][idx].copy()
    ab, st = d["ability"][idx].copy(), d["stats"][idx]
    known = np.ones(sp.shape, bool)
    hide = d["swapped"][idx] & (rng.random(sp.shape) < p_hide)
    if hide.any():
        a, b = np.nonzero(hide)
        pm = np.asarray(reveal["moves"], np.float64)
        n_rev = rng.choice(len(pm), size=len(a), p=pm / pm.sum())
        for x, y, r in zip(a, b, n_rev):
            real = [m for m in mv[x, y] if m]
            keep = list(rng.permutation(real)[:r]) if real else []
            mv[x, y] = (keep + [UNKNOWN] * 4)[:4]
        it[a, b] = np.where(rng.random(len(a)) < reveal["item"], it[a, b], UNKNOWN)
        one = np.array([len(possible_abilities(s)) == 1 for s in sp[a, b]])
        ab[a, b] = np.where(one | (rng.random(len(a)) < reveal["ability"]), ab[a, b], 0)
        known[a, b] = False
    return featurize_arrays(sp, mv, it, ab, st, known)


def auc(y, p):
    y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ps = p[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(p) + 1, dtype=np.float64)
    while i < len(p):
        j = i
        while j + 1 < len(p) and ps[j + 1] == ps[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def battle_csr(d):
    """Battles grouped by team: (order, start) with the battles of team t at order[start[t]:start[t + 1]]
    (cached in d)."""
    if "_csr" not in d:
        order = np.argsort(d["b_team"], kind="stable")
        start = np.searchsorted(d["b_team"][order], np.arange(len(d["round"]) + 1))
        d["_csr"] = (order, start)
    return d["_csr"]


def battles_of(d, idx, include_truncated=False):
    """Battles of teams idx: (battle indices, position of each battle's team in idx)."""
    order, start = battle_csr(d)
    idx = np.asarray(idx, np.int64)
    cnt = start[idx + 1] - start[idx]
    pos = np.repeat(np.arange(len(idx)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    b = order[start[idx][pos] + off]
    if not include_truncated:
        keep = ~d["b_trunc"][b]
        b, pos = b[keep], pos[keep]
    return b, pos


def battle_hints(d, b, rng=None, p_unknown=0.0):
    """Hint tokens of battles b: the recorded hint of the battle's opponent; unknown for Noland (no hint), for
    battles without a recorded hint and, with probability p_unknown each, as training-time hint dropout."""
    n = len(b)
    if "b_hint_type" not in d:
        return np.full(n, HINT_UNK_TYPE, np.int64), np.full(n, HINT_UNK_STYLE, np.int64)
    ht = d["b_hint_type"][b].astype(np.int64)
    hs = d["b_hint_style"][b].astype(np.int64)
    unk = (ht < 0) | (hs < 0) | (d["btype"][d["b_team"][b]] == NOLAND)
    if p_unknown > 0 and rng is not None:
        unk |= rng.random(n) < p_unknown
    return np.where(unk, HINT_UNK_TYPE, ht), np.where(unk, HINT_UNK_STYLE, hs)


def battle_loss(net, d, idx, rng, dev, p_unknown=0.0, p_hide=0.5):
    """Mean per-battle BCE of W(team, round, battle type, hint) over the battles of teams idx (swapped-in Pokemon
    hidden with prob p_hide; hint dropout p_unknown). With one hint for all battles of a team this is the binomial
    log-likelihood of (won of played) of rl.team_eval v1. -> (loss tensor, number of battles)."""
    ids, num = hide_swapped(d, idx, rng, p_hide)
    b, pos = battles_of(d, idx)
    if len(b) == 0:
        return None, 0
    x = [torch.from_numpy(v).to(dev) for v in (ids, num, round_index(d["round"][idx]))]
    h, z = net.encode(*x)
    ht, hs = battle_hints(d, b, rng, p_unknown)
    posT = torch.from_numpy(pos).to(dev)
    logit = net.head_logits(h[posT], z[posT], torch.from_numpy(ht).to(dev), torch.from_numpy(hs).to(dev))
    typ = torch.from_numpy(d["btype"][d["b_team"][b]].astype(np.int64)).to(dev)
    y = torch.from_numpy(d["b_won"][b].astype(np.float32)).to(dev)
    lg = logit.gather(1, typ[:, None])[:, 0]
    return F.binary_cross_entropy_with_logits(lg, y), len(b)


@torch.no_grad()
def predict_battles(ev, d, idx, rng, hint=True, chunk=8192):
    """Predicted W of every non-truncated battle of teams idx (swapped-in Pokemon hidden with prob 0.5, as in
    training). hint: use each battle's recorded hint (unknown for Noland) or, False, the unknown token for all.
    -> (p [n_battles], battle indices)."""
    net, dev = ev.net, ev.device
    ids, num = hide_swapped(d, idx, rng)
    ps, bs = [], []
    for i in range(0, len(idx), chunk):
        sl = idx[i:i + chunk]
        b, pos = battles_of(d, sl)
        x = [torch.from_numpy(np.ascontiguousarray(v)).to(dev)
             for v in (ids[i:i + chunk], num[i:i + chunk], round_index(d["round"][sl]))]
        h, z = net.encode(*x)
        if hint and getattr(net, "hint", False):
            ht, hs = battle_hints(d, b)
            ht, hs = torch.from_numpy(ht).to(dev), torch.from_numpy(hs).to(dev)
        else:
            ht = hs = None
        posT = torch.from_numpy(pos).to(dev)
        lg = net.head_logits(h[posT], z[posT], ht, hs)
        typ = torch.from_numpy(d["btype"][d["b_team"][b]].astype(np.int64)).to(dev)
        ps.append(torch.sigmoid(lg.gather(1, typ[:, None])[:, 0].float()).cpu().numpy())
        bs.append(b)
    return (np.concatenate(ps).astype(np.float64) if ps else np.zeros(0)), \
        (np.concatenate(bs) if bs else np.zeros(0, np.int64))


def _scores(p, y):
    return {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)), "auc": auc(y, p),
            "bce": float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))),
            "brier_const": float(np.mean((y.mean() - y) ** 2)), "win": float(y.mean()), "pred": float(p.mean())}


def evaluate_heldout(ev, d, idx, rng):
    """Per-battle metrics on held-out teams (their swapped-in Pokemon hidden with prob 0.5, as in training): with
    each battle's hint and (hint model) with the hint unknown, overall, by round and type, and by hint."""
    rng_state = rng.bit_generator.state
    p, b = predict_battles(ev, d, idx, rng, hint=True)
    y = d["b_won"][b].astype(bool)
    typ = d["btype"][d["b_team"][b]]
    rnd = d["round"][d["b_team"][b]]
    out = {"n_battles": int(len(b)), "brier": float(np.mean((p - y) ** 2)), "auc": auc(y, p),
           "brier_const": float(np.mean((y.mean() - y) ** 2))}
    rows = []
    edges = np.quantile(p, np.linspace(0, 1, 11))
    for q in range(10):
        m = (p >= edges[q]) & ((p < edges[q + 1]) if q < 9 else (p <= edges[q + 1]))
        rows.append({"decile": q + 1, "n": int(m.sum()), "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    out["reliability"] = rows
    hinted = getattr(ev.net, "hint", False) and "b_hint_type" in d
    if hinted:
        rng.bit_generator.state = rng_state            # the same hidden swapped-in Pokemon
        pu, bu = predict_battles(ev, d, idx, rng, hint=False)
        assert np.array_equal(bu, b)
        out["hint"] = _scores(p, y)
        out["unknown"] = _scores(pu, y)
    per = []
    for r in range(1, N_ROUNDS + 1):
        for t in range(3):
            m = (rnd == r) & (typ == t)
            if m.sum() < 50:
                continue
            base = y[m].mean()
            row = {"round": r, "type": TYPES[t], "n": int(m.sum()), "win": float(base),
                   "pred": float(p[m].mean()), "auc": auc(y[m], p[m]), "brier": float(np.mean((p[m] - y[m]) ** 2)),
                   "brier_const": float(np.mean((base - y[m]) ** 2))}
            if hinted:
                row["auc_unknown"] = auc(y[m], pu[m])
                row["brier_unknown"] = float(np.mean((pu[m] - y[m]) ** 2))
            per.append(row)
    out["per_round_type"] = per
    if hinted:
        ht, hs = d["b_hint_type"][b], d["b_hint_style"][b]
        normal = typ != NOLAND
        by = []
        for name, key, vals in (("type", ht, range(N_HINT_TYPES)), ("style", hs, range(N_HINT_STYLES))):
            for v in vals:
                m = normal & (key == v)
                if m.sum() < 200:
                    continue
                by.append({"by": name, "value": int(v), "n": int(m.sum()), "win": float(y[m].mean()),
                           "brier": float(np.mean((p[m] - y[m]) ** 2)),
                           "brier_unknown": float(np.mean((pu[m] - y[m]) ** 2)),
                           "auc": auc(y[m], p[m]), "auc_unknown": auc(y[m], pu[m]),
                           "pred": float(p[m].mean()), "pred_unknown": float(pu[m].mean())})
        out["by_hint"] = by
        m = normal
        out["normal_battles"] = {"hint": _scores(p[m], y[m]), "unknown": _scores(pu[m], y[m])}
    # team level: how much of the spread of team win rates the model explains (binomial noise removed), with the
    # hint unknown (a team's battles have different opponents and hints)
    played, won = team_targets(d)
    pl, wn = played[idx], won[idx]
    ok = pl >= 4
    Wt = ev.w_features(*hide_swapped(d, idx, np.random.default_rng(7)), d["round"][idx])
    pt = Wt[np.arange(len(idx)), d["btype"][idx]][ok]
    rate = wn[ok] / pl[ok]
    noise = np.mean(rate * (1 - rate) / np.maximum(pl[ok] - 1, 1))
    var_true = max(np.var(rate) - noise, 1e-9)
    out["team_level"] = {"teams": int(ok.sum()), "corr": float(np.corrcoef(pt, rate)[0, 1]),
                         "var_rate": float(np.var(rate)), "binomial_noise": float(noise),
                         "var_true_rate": float(var_true), "var_pred": float(np.var(pt)),
                         "explained_of_true": float(1 - (np.mean((rate - pt) ** 2) - noise) / var_true)}
    return out


def train(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, nargs="+", help="one or more data directories")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=2048, help="teams per step (all their battles)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--d-emb", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hint", type=int, default=1, help="1: W conditioned on the attendant's hint (W_v3)")
    p.add_argument("--p-unknown", type=float, default=0.3, help="hint dropout: fraction of battles trained with "
                   "the hint unknown")
    p.add_argument("--heldout", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    from .team_eval_refresh import merge_arrays
    parts = [load_data(x) for x in a.data]
    d = parts[0] if len(parts) == 1 else merge_arrays(parts)
    played, won = team_targets(d)
    n = len(played)
    perm = rng.permutation(n)
    n_held = int(round(a.heldout * n))
    held, tr = np.sort(perm[:n_held]), perm[n_held:]
    tr = tr[played[tr] > 0]
    dev = torch.device(a.device)
    net = TeamNet(a.d, a.d_emb, a.layers, hint=bool(a.hint)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    mpath = os.path.join(a.data[0], "meta.json")
    meta = json.load(open(mpath)) if os.path.exists(mpath) else {}
    print(f"{n:,} teams ({len(tr):,} train, {n_held:,} held out), {int(played.sum()):,} battles "
          f"({int(d['b_trunc'].sum()):,} truncated, excluded); hint {bool(a.hint)} "
          f"(recorded: {'b_hint_type' in d}), hint dropout {a.p_unknown}", flush=True)
    ev = TeamEvaluator(net, device=dev)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        order = rng.permutation(tr)
        tot, cnt = 0.0, 0
        for i in range(0, len(order), a.batch):
            idx = np.sort(order[i:i + a.batch])
            loss, nb = battle_loss(net, d, idx, rng, dev, a.p_unknown if a.hint else 0.0)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * nb
            cnt += nb
        net.eval()
        ho = evaluate_heldout(ev, d, held, np.random.default_rng(123))
        extra = ""
        if "unknown" in ho:
            extra = f" | unknown hint: brier {ho['unknown']['brier']:.4f} auc {ho['unknown']['auc']:.3f}"
        print(f"epoch {ep + 1}/{a.epochs}: train bce {tot / max(cnt, 1):.4f}  held-out brier {ho['brier']:.4f} "
              f"(const {ho['brier_const']:.4f}) auc {ho['auc']:.3f}{extra} team corr {ho['team_level']['corr']:.3f} "
              f"explained {ho['team_level']['explained_of_true']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    meta = dict(meta, data=a.data, data_teams=int(n), data_battles=int(played.sum()), heldout_teams=int(n_held),
                epochs=a.epochs, hint=bool(a.hint), p_unknown=a.p_unknown, trained=time.strftime("%Y-%m-%d %H:%M"),
                heldout={k: v for k, v in ho.items() if k not in ("per_round_type", "by_hint")})
    ev.net.to("cpu")
    ev.device = torch.device("cpu")
    ev.save(a.out, meta)
    json.dump(ho, open(os.path.splitext(a.out)[0] + "_heldout.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in ho.items() if k not in ("per_round_type", "by_hint")}, indent=1))
    for row in ho["per_round_type"]:
        print(row)
    for row in ho.get("by_hint", []):
        print(row)
    return ev


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train(sys.argv[2:])
    else:
        raise SystemExit("usage: python -m rl.team_eval train --data DIR [DIR ...] --out PATH")
