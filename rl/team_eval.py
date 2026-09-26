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
    them (permutation-equivariant; order only enters through the lead flag); mean-pool + lead token + round -> MLP."""

    def __init__(self, d=128, d_emb=32, layers=2, heads=4):
        super().__init__()
        self.cfg = dict(d=d, d_emb=d_emb, layers=layers, heads=heads)
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
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 3))

    def forward(self, ids, num, rnd):
        """ids int64 [B,3,7], num float [B,3,MON_NUM], rnd int64 [B] (0..N_ROUNDS-1) -> logits [B,3]."""
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
        return self.head(torch.cat([h.mean(1), h[:, 0], r], -1))


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
        net = TeamNet(**ck["cfg"])
        net.load_state_dict(ck["net"])
        return cls(net, ck.get("meta"), device)

    def save(self, path, meta=None):
        m = dict(self.meta, **(meta or {}))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"net": {k: v.detach().cpu() for k, v in self.net.state_dict().items()}, "cfg": self.net.cfg,
                    "encode_version": ENCODE_VERSION, "meta": m}, path + ".tmp")
        os.replace(path + ".tmp", path)

    @torch.no_grad()
    def w_features(self, ids, num, rounds):
        _single_thread_in_children()
        rnd = round_index(np.broadcast_to(np.asarray(rounds), (len(ids),)))
        x = [torch.from_numpy(np.ascontiguousarray(a)).to(self.device) for a in (ids, num, rnd)]
        return torch.sigmoid(self.net(*x).float()).cpu().numpy().astype(np.float64)

    def w(self, teams: list, round_idx) -> np.ndarray:
        """[n, 3]: W(normal), W(last battle), W(Noland) in [0, 1] of each team in round `round_idx` (an int, or one
        per team)."""
        if len(teams) == 0:
            return np.zeros((0, 3))
        ids, num = featurize(teams)
        return self.w_features(ids, num, round_idx)


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


# ---- training ---------------------------------------------------------------------------------------------------------------

# how a defeated Pokemon looks at the swap screen, measured on 1,500 of them under alphazero_v1 greedy play
# (fraction with 0..4 distinct moves revealed; item named; ability announced, species with two abilities); used to
# hide what the player would not have seen of a swapped-in Pokemon in half of the training samples
REVEAL = {"moves": (0.180, 0.437, 0.243, 0.103, 0.038), "item": 0.187, "ability": 0.044}


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


def evaluate_heldout(ev, d, idx, rng):
    """Per-battle metrics on held-out teams (their swapped-in Pokemon hidden with prob 0.5, as in training)."""
    ids, num = hide_swapped(d, idx, rng)
    W = np.concatenate([ev.w_features(ids[i:i + 8192], num[i:i + 8192], d["round"][idx][i:i + 8192])
                        for i in range(0, len(idx), 8192)])
    pos = np.full(len(d["round"]), -1)
    pos[idx] = np.arange(len(idx))
    sel = (pos[d["b_team"]] >= 0) & ~d["b_trunc"]
    bt = d["b_team"][sel]
    y = d["b_won"][sel].astype(bool)
    typ = d["btype"][bt]
    p = W[pos[bt], typ]
    rnd = d["round"][bt]
    out = {"n_battles": int(sel.sum()), "brier": float(np.mean((p - y) ** 2)), "auc": auc(y, p),
           "brier_const": float(np.mean((y.mean() - y) ** 2))}
    rows = []
    edges = np.quantile(p, np.linspace(0, 1, 11))
    for q in range(10):
        m = (p >= edges[q]) & ((p < edges[q + 1]) if q < 9 else (p <= edges[q + 1]))
        rows.append({"decile": q + 1, "n": int(m.sum()), "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    out["reliability"] = rows
    per = []
    for r in range(1, N_ROUNDS + 1):
        for t in range(3):
            m = (rnd == r) & (typ == t)
            if m.sum() < 50:
                continue
            base = y[m].mean()
            per.append({"round": r, "type": TYPES[t], "n": int(m.sum()), "win": float(base),
                        "pred": float(p[m].mean()), "auc": auc(y[m], p[m]), "brier": float(np.mean((p[m] - y[m]) ** 2)),
                        "brier_const": float(np.mean((base - y[m]) ** 2))})
    out["per_round_type"] = per
    # team level: how much of the spread of team win rates the model explains (binomial noise removed)
    played, won = team_targets(d)
    pl, wn = played[idx], won[idx]
    ok = pl >= 4
    pt = W[np.arange(len(idx)), d["btype"][idx]][ok]
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
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--d-emb", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heldout", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = load_data(a.data)
    played, won = team_targets(d)
    n = len(played)
    perm = rng.permutation(n)
    n_held = int(round(a.heldout * n))
    held, tr = np.sort(perm[:n_held]), perm[n_held:]
    tr = tr[played[tr] > 0]
    dev = torch.device(a.device)
    net = TeamNet(a.d, a.d_emb, a.layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    meta = json.load(open(os.path.join(a.data, "meta.json"))) if os.path.exists(os.path.join(a.data, "meta.json")) \
        else {}
    print(f"{n:,} teams ({len(tr):,} train, {n_held:,} held out), {int(played.sum()):,} battles "
          f"({int(d['b_trunc'].sum()):,} truncated, excluded)", flush=True)
    ev = TeamEvaluator(net, device=dev)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        order = rng.permutation(tr)
        tot, cnt = 0.0, 0
        for i in range(0, len(order), a.batch):
            idx = np.sort(order[i:i + a.batch])
            ids, num = hide_swapped(d, idx, rng)
            x = [torch.from_numpy(v).to(dev) for v in (ids, num, round_index(d["round"][idx]))]
            typ = torch.from_numpy(d["btype"][idx].astype(np.int64)).to(dev)
            pl = torch.from_numpy(played[idx].astype(np.float32)).to(dev)
            wn = torch.from_numpy(won[idx].astype(np.float32)).to(dev)
            logit = net(*x).gather(1, typ[:, None])[:, 0]
            # binomial log-likelihood of (won of played) = sum of the per-battle BCEs
            loss = (pl * F.softplus(logit) - wn * logit).sum() / pl.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * float(pl.sum())
            cnt += float(pl.sum())
        net.eval()
        ho = evaluate_heldout(ev, d, held, np.random.default_rng(123))
        print(f"epoch {ep + 1}/{a.epochs}: train bce {tot / cnt:.4f}  held-out brier {ho['brier']:.4f} "
              f"(const {ho['brier_const']:.4f}) auc {ho['auc']:.3f} team corr {ho['team_level']['corr']:.3f} "
              f"explained {ho['team_level']['explained_of_true']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    meta = dict(meta, data_teams=int(n), data_battles=int(played.sum()), heldout_teams=int(n_held),
                epochs=a.epochs, trained=time.strftime("%Y-%m-%d %H:%M"), heldout=ho)
    ev.net.to("cpu")
    ev.device = torch.device("cpu")
    ev.save(a.out, meta)
    json.dump(ho, open(os.path.splitext(a.out)[0] + "_heldout.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in ho.items() if k != "per_round_type"}, indent=1))
    for row in ho["per_round_type"]:
        print(row)
    return ev


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train(sys.argv[2:])
    else:
        raise SystemExit("usage: python -m rl.team_eval train --data DIR --out PATH")
