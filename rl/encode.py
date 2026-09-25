"""Views -> fixed-size arrays (docs/RL_DECISIONS.md §5-6).

Every decision becomes 6 Pokemon tokens + 1 context token:
    battle: own party 0-2, enemy party 0-2, field + streak context
    rental: the 6 candidates, streak context
    swap:   own party 0-2 (slot 0 leads next battle), the defeated team 0-2, streak context

Embedding ids reserve a NONE and an UNKNOWN index per table: an unrevealed enemy item is UNKNOWN, a Pokemon
holding nothing is NONE. Numbers use fixed game-bound scales. IVs, EVs and nature are not given (already in the
stats); nor are the attendant's hints to the tactician. The battler's view includes them (it's in BattleView).
"""

import math
from itertools import combinations

import numpy as np
import torch

from pybattle.view import MAJOR_STATUSES, SEMI_INVULNERABLE, WEATHERS
from . import damage
from .gamedata import MOVES, N_ABILITIES, N_EFFECTS, N_ITEMS, N_MOVES, N_SPECIES, N_TYPES, SPECIES

# ---- embedding vocabularies --------------------------------------------------------------------------------
SPECIES_UNK = N_SPECIES;            N_SPECIES_TOK = N_SPECIES + 1        # 0 = none
MOVE_UNK = N_MOVES;                 N_MOVE_TOK = N_MOVES + 1             # 0 = none
EFFECT_NONE = N_EFFECTS;            EFFECT_UNK = N_EFFECTS + 1;  N_EFFECT_TOK = N_EFFECTS + 2
TYPE_NONE = N_TYPES;                TYPE_UNK = N_TYPES + 1;      N_TYPE_TOK = N_TYPES + 2
ITEM_UNK = N_ITEMS;                 N_ITEM_TOK = N_ITEMS + 1             # 0 = none
ABILITY_UNK = N_ABILITIES;          N_ABILITY_TOK = N_ABILITIES + 1      # 0 = none

PAIRS = list(combinations(range(6), 2))      # 15 unordered pairs of rental slots

# ---- layout ------------------------------------------------------------------------------------------------
# mon ids: species, type1, type2, item, ability, move x4, effect x4, move type x4, possible ability x2
MON_IDS = 19

ACTIVE_BOOLS = ("confused", "infatuated", "substitute", "leech_seeded", "cursed", "nightmare", "trapped",
                "focus_energy", "transformed", "perish_song", "rooted", "yawn", "torment", "taunted",
                "must_recharge", "destiny_bond", "defense_curl", "foresight", "minimized", "charged_up",
                "imprisoning", "grudge", "mud_sport", "water_sport", "rage", "first_turn")
ACTIVE_COUNTERS = (("confusion_turns", 5), ("toxic_counter", 15), ("taunt_turns_left", 2), ("encore_turns", 6),
                   ("disable_turns", 5), ("yawn_turns_left", 2), ("wrapped_turns", 6), ("uproar_turns", 5),
                   ("rampage_turns", 3), ("bide_turns_left", 2), ("rollout_hits_left", 5),
                   ("fury_cutter_count", 5), ("stockpile", 3), ("charge_turns_left", 2),
                   ("locked_on_turns_left", 2), ("protect_uses", 4))
N_ACTIVE = 7 + len(ACTIVE_BOOLS) + len(ACTIVE_COUNTERS) + 2 + len(SEMI_INVULNERABLE) + N_TYPES
N_HINT_TYPES, N_STYLES = N_TYPES + 1, 8
FIELD_NUM = (len(WEATHERS) + 2 + 2 * 8 + 3 + 3 + 4 + 3 + 4 + 3 + 2 + 2 + 2 + N_HINT_TYPES + N_STYLES)
CTX_IDS = 2
N_ROUNDS = 7                                # rounds 1-6, and 7+

# Encoding versions (docs/RL_DECISIONS.md): 2 = ppo_joint_v1/v2; 3 = + damage/speed estimates, round one-hot.
VERSION = None
MON_NUM = MOVE_NUM = CONTEXT_NUM = CTX_NUM = None


def set_version(v):
    """Select the observation layout. Call before building networks or forking environment workers."""
    global VERSION, MON_NUM, MOVE_NUM, CONTEXT_NUM, CTX_NUM
    VERSION = v
    MON_NUM = 6 + 1 + 1 + 5 + 6 + len(MAJOR_STATUSES) + 1 + 1 + 1 + N_ACTIVE + (4 if v >= 3 else 0)
    MOVE_NUM = 14 + (3 if v >= 3 else 0)
    CONTEXT_NUM = (1 + 7 + N_ROUNDS + 6 + 1 + 1) if v >= 3 else (1 + 7 + 1 + 6 + 1 + 1)
    CTX_NUM = FIELD_NUM + CONTEXT_NUM


set_version(3)
RENT_RANKS = (15, 22, 29, 36, 43)          # battle_factory.c GetNumPastRentalsRank


def _onehot(i, n):
    v = np.zeros(n, np.float32)
    if 0 <= i < n:
        v[i] = 1.0
    return v


def _types(types, unknown=False):
    if unknown or not types:
        return TYPE_UNK, TYPE_UNK
    t1, t2 = types[0], types[1] if len(types) > 1 else types[0]
    return t1, (t2 if t2 != t1 else TYPE_NONE)


def _move_ids(moves, unknown_fill):
    """moves -> (move ids, effect ids, move type ids), padded to 4 with UNKNOWN (enemy) or NONE (own)."""
    m, e, t = [], [], []
    for i in range(4):
        mv = moves[i] if i < len(moves) else None
        if mv is None:
            m.append(MOVE_UNK if unknown_fill else 0)
            e.append(EFFECT_UNK if unknown_fill else EFFECT_NONE)
            t.append(TYPE_UNK if unknown_fill else TYPE_NONE)
        elif mv == 0:
            m.append(0); e.append(EFFECT_NONE); t.append(TYPE_NONE)
        else:
            m.append(mv); e.append(MOVES[mv]["effect"]); t.append(MOVES[mv]["type"])
    return m, e, t


def _move_num(move, known, pp=None, usable=False, flags=(False, False, False, False), last=False, est=None):
    if not known or not move:
        return np.array([float(known)] + [0.0] * (MOVE_NUM - 1), np.float32)
    d = MOVES[move]
    acc = d["accuracy"]
    base_pp = max(d["pp"], 1)
    out = [1.0, d["power"] / 255, acc / 100, float(acc == 0), (pp / base_pp) if pp is not None else 1.0,
           base_pp / 64, d["priority"] / 6, d["secondary_chance"] / 100, float(usable), *map(float, flags), float(last)]
    if VERSION >= 3:
        out += list(est) if est is not None else [0.0, 0.0, 0.0]      # damage vs the opponent: min, max, can KO
    return np.array(out, np.float32)


def _log_stage(s):
    # the multiplier of a stage is (2+s)/2 or 2/(2-s) (accuracy/evasion differ slightly); log2 / 2 keeps it in [-1, 1]
    return math.log2((2 + s) / 2 if s >= 0 else 2 / (2 - s)) / 2


def _active_feats(a, max_hp=None):
    if a is None:
        return np.zeros(N_ACTIVE, np.float32)
    f = [_log_stage(s) for s in a.stat_stages]
    f += [float(getattr(a, k)) for k in ACTIVE_BOOLS]
    f += [min(getattr(a, k), n) / n for k, n in ACTIVE_COUNTERS]
    f += [(a.perish_count + 1) / 4 if a.perish_count >= 0 else 0.0]
    f += [a.substitute_hp / max_hp if (max_hp and a.substitute_hp >= 0) else 0.0]
    f = np.array(f, np.float32)
    types = np.zeros(N_TYPES, np.float32)
    for t in a.types:
        if t < N_TYPES:
            types[t] = 1.0
    return np.concatenate([f, _onehot(a.semi_invulnerable, len(SEMI_INVULNERABLE)), types])


def _own_mon(m, *, active=None, is_active=False, slot0=False, usable=None, last_move=0, est=None, threat=None,
             speed=None):
    """A Pokemon the player controls (battle party, rental candidate, swap team)."""
    t1, t2 = _types(m.types)
    mv, ef, mt = _move_ids(m.moves, unknown_fill=False)
    sp = SPECIES[m.species]
    abil = sp["abilities"]
    ids = [m.species, t1, t2, m.item, m.ability, *mv, *ef, *mt, abil[0], abil[1] or 0]
    num = np.concatenate([
        [1.0, 0.0, float(is_active), 1.0, float(m.hp == 0), 1.0],
        [m.hp / m.max_hp if m.max_hp else 0.0], [m.max_hp / 500],
        np.array(m.stats, np.float32) / 500, np.array(sp["base"], np.float32) / 255,
        _onehot(m.status, len(MAJOR_STATUSES)), [min(getattr(m, "sleep_turns", 0), 5) / 5], [float(slot0)],
        [m.level / 100], _active_feats(active, m.max_hp),
        (list(threat or (0.0, 0.0)) + list(speed or (0.0, 0.0))) if VERSION >= 3 else []]).astype(np.float32)
    flags = [(False,) * 4] * 4
    if active is not None:
        flags = [(mv[i] and mv[i] == active.encored_move, mv[i] and mv[i] == active.disabled_move,
                  mv[i] and mv[i] == active.locked_move, mv[i] and mv[i] == active.charging_move) for i in range(4)]
    moves = np.stack([_move_num(m.moves[i] if i < len(m.moves) else 0, True, m.pp[i] if i < len(m.pp) else 0,
                                bool(usable[i]) if usable is not None else False, flags[i],
                                bool(last_move) and m.moves[i] == last_move, est[i] if est else None)
                      for i in range(4)])
    return ids, num, moves


def _seen_mon(m, *, active=None, is_active=False, last_move=0, est=None):
    """An opponent Pokemon as the player knows it. Never-seen Pokemon are all UNKNOWN / zero."""
    if not m.seen:
        ids = [SPECIES_UNK, TYPE_UNK, TYPE_UNK, ITEM_UNK, ABILITY_UNK] + [MOVE_UNK] * 4 + [EFFECT_UNK] * 4 \
              + [TYPE_UNK] * 4 + [ABILITY_UNK, ABILITY_UNK]
        num = np.zeros(MON_NUM, np.float32)
        num[1] = 1.0                                  # is_enemy
        return ids, num, np.zeros((4, MOVE_NUM), np.float32)
    t1, t2 = _types(m.types)
    revealed = list(m.revealed_moves)[:4]
    mv, ef, mt = _move_ids(revealed, unknown_fill=True)
    item = m.revealed_item if m.revealed_item is not None else ITEM_UNK
    ability = m.revealed_ability if m.revealed_ability else ABILITY_UNK
    pa = list(m.possible_abilities) + [0, 0]
    ids = [m.species, t1, t2, item, ability, *mv, *ef, *mt, pa[0], pa[1]]
    hp = max(m.hp_pixels, 0) / 48
    num = np.concatenate([
        [0.0, 1.0, float(is_active), 1.0, float(m.fainted), 0.0], [hp], [0.0],
        np.zeros(5, np.float32), np.array(m.base_stats or [0] * 6, np.float32) / 255,
        _onehot(m.status, len(MAJOR_STATUSES)), [min(m.sleep_turns, 5) / 5], [0.0],
        [m.level / 100], _active_feats(active), [0.0] * 4 if VERSION >= 3 else []]).astype(np.float32)
    moves = []
    for i in range(4):
        known = i < len(revealed)
        mvid = revealed[i] if known else 0
        flags = (False,) * 4
        if active is not None and known:
            flags = (mvid == active.encored_move, mvid == active.disabled_move, mvid == active.locked_move,
                     mvid == active.charging_move)
        moves.append(_move_num(mvid, known, None, False, flags, known and bool(last_move) and mvid == last_move,
                               est[i] if (est and i < len(est)) else None))
    return ids, num, np.stack(moves)


def _side(s):
    return [s.reflect_turns / 5, s.light_screen_turns / 5, s.safeguard_turns / 5, s.mist_turns / 5, s.spikes / 3,
            s.future_sight_turns / 2, float(s.future_sight_move != 0), s.wish_turns / 2]


def _context(ctx):
    streak = ctx["streak"]
    rank = sum(ctx["rents"] >= t for t in RENT_RANKS)
    rnd = _onehot(min(ctx["challenge"], N_ROUNDS - 1), N_ROUNDS) if VERSION >= 3 else [min(ctx["challenge"], 10) / 10]
    return np.concatenate([[math.log1p(streak) / math.log1p(100)], _onehot(ctx["battle"], 7),
                           rnd, _onehot(rank, 6), [min(ctx["rents"], 50) / 50],
                           [float(streak % 7 == 0 and streak > 0)]]).astype(np.float32)


def _pack(mons, ctx_ids, ctx_num, **extra):
    ids, num, mv = zip(*mons)
    out = {"mon_ids": np.array(ids, np.int64), "mon_num": np.stack(num), "move_num": np.stack(mv),
           "ctx_ids": np.array(ctx_ids, np.int64), "ctx_num": ctx_num.astype(np.float32)}
    out.update(extra)
    return out


# ---- public encoders ---------------------------------------------------------------------------------------

def battle(view, ctx):
    lt = view.last_turn
    a_own, a_foe = view.own_active.party_index, view.enemy_active.party_index
    if VERSION >= 3:
        own_est, foe_est, threats, speed = damage.battle_estimates(view, ctx)
    else:
        own_est, foe_est, threats, speed = [None] * 3, None, [None] * 3, [None] * 3
    mons = [_own_mon(m, active=view.own_active if i == a_own else None, is_active=i == a_own,
                     usable=view.usable_moves if i == a_own else None,
                     last_move=lt.own_move if i == a_own else 0, est=own_est[i], threat=threats[i], speed=speed[i])
            for i, m in enumerate(view.own_party[:3])]
    mons += [_seen_mon(m, active=view.enemy_active if i == a_foe else None, is_active=i == a_foe,
                       last_move=lt.enemy_move if i == a_foe else 0, est=foe_est if i == a_foe else None)
             for i, m in enumerate(view.enemy_party[:3])]
    own_max = view.own_party[a_own].max_hp or 1
    field = np.concatenate([
        _onehot(view.weather, len(WEATHERS)), [float(view.weather_permanent), view.weather_turns_left / 5],
        _side(view.own_side), _side(view.enemy_side),
        [min(view.turn, 50) / 50, float(view.forced_switch), float(view.must_struggle)],
        _onehot(lt.first + 1, 3), _onehot(lt.own_action, 4), _onehot(lt.own_crit + 1, 3),
        _onehot(lt.enemy_action, 4), _onehot(lt.enemy_crit + 1, 3),
        [lt.damage_dealt_pixels / 48, min(lt.damage_taken / own_max, 1.0)],
        [float(lt.own_fainted), float(lt.enemy_fainted)], [float(lt.in_progress), min(lt.turns, 2) / 2],
        _onehot(view.hint_type, N_HINT_TYPES), _onehot(view.hint_style, N_STYLES)])
    mask = np.zeros(7, bool)
    for a in view.legal_actions:
        if a[0] == "move":
            mask[a[1]] = True
        elif a[0] == "switch":
            mask[4 + a[1]] = True
    return _pack(mons, [lt.own_move, lt.enemy_move], np.concatenate([field, _context(ctx)]),
                 mask=mask, active=np.int64(a_own))


def rental(view, ctx):
    mons = [_own_mon(m) for m in view.candidates]
    sp = [m.species for m in view.candidates]
    pair_mask = np.zeros((6, len(PAIRS)), bool)
    for lead in range(6):
        for p, (j, k) in enumerate(PAIRS):
            pair_mask[lead, p] = lead not in (j, k) and len({sp[lead], sp[j], sp[k]}) == 3
    lead_mask = pair_mask.any(1)
    return _pack(mons, [0, 0], np.concatenate([np.zeros(FIELD_NUM, np.float32), _context(ctx)]),
                 lead_mask=lead_mask, pair_mask=pair_mask)


def swap(view, ctx, valid=None):
    """`valid(i, j)` (optional) says whether the backend accepts trading own slot i for enemy slot j."""
    mons = [_own_mon(m, slot0=i == 0) for i, m in enumerate(view.own_party[:3])]
    mons += [_seen_mon(m) for m in view.enemy_party[:3]]
    own_sp = [m.species for m in view.own_party[:3]]
    mask = np.zeros(10, bool)
    mask[0] = True
    for i in range(3):
        for j in range(3):
            ok = view.enemy_party[j].species not in [s for k, s in enumerate(own_sp) if k != i]
            if ok and valid is not None:
                ok = valid(i, j)
            mask[1 + 3 * i + j] = ok
    return _pack(mons, [0, 0], np.concatenate([np.zeros(FIELD_NUM, np.float32), _context(ctx)]), mask=mask)


def collate(obs_list, device):
    out = {}
    for k in obs_list[0]:
        arr = np.stack([o[k] for o in obs_list])
        t = torch.from_numpy(arr)
        out[k] = t.to(device, non_blocking=True)
    return out
