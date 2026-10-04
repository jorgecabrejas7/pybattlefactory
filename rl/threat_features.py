"""Threat features: battler inputs computed from the LEARNED opponent model (docs/RL_DECISIONS.md §26, §27.x).

    tf = ThreatFeaturizer(ExperienceModel.load("runs/opp_model_v3/model.pt"))
    feats = tf(view, challenge)          # float32 [6, N_THREAT]: one row per battle token (own 0-2, enemy 0-2)

For every opponent slot (seen or not yet seen) the experience model (rl/opponent_model.py, §22) gives a posterior
over its configuration c = (species, moveset, item) given what the battle showed (species, revealed moves, revealed
item, the teammates' revealed items, the round). It is a mixture of "atoms":
    known atoms   configurations the player has seen in full, weight P(c | what was seen);
    novel atoms   one per species: "a configuration never seen", whose moves are the revealed ones plus a uniform
                  draw (without repetition) from what the species can learn (the strict rule). Their features are
                  computed exactly in closed form (hypergeometric counts), not sampled.
An unseen slot's atoms are species x configuration (species from the model's round distribution minus the team's
seen species; mass below MIN_SPECIES_P dropped and renormalized).

The stats of an opponent are not known: a "typical" value is used (IV 15.5, EV 85 = 510 / 6, the nature from the
model's observed natures of that configuration, uniform for a novel one). Damage is the Gen 3 formula at the mean
roll (0.925) without crits, weather, screens or stat stages (the network sees those), as a fraction of the
defender's max HP. Abilities: our own exactly; the opponent's revealed one, else the average over its species'
possible abilities (Huge Power, Levitate...). Items: ours exactly; the opponent's from the atom (Choice Band, type
boosters, Thick Club, Light Ball); novel atoms: no item.

Per pair (opponent j, own i), PF = 6 numbers:
    0 p_se       P(j has a damaging move super effective on i)
    1 dmg_in     E[max over j's moves of the damage fraction on i], capped at CAP
    2 ko_in      P(that max >= i's current HP fraction)            (a one-hit KO from where i is now)
    3 faster     P(j's speed > i's speed) (+ half the ties), speed stages and paralysis of the actives included
    4 dmg_out    E[max over i's moves of the damage fraction on j], capped at CAP
    5 ko_out     P(that max >= j's current HP fraction)             (unseen: full HP)
Token rows (N_THREAT = 18):
    own i        [PF(active enemy, i), elementwise max over the other alive enemy slots of PF(j, i), 0 x 6]
    enemy j      [PF(j, own active), elementwise max over our other alive Pokemon of PF(j, i),
                  p_status, p_setup, p_recovery, p_priority, entropy / ENT_SCALE, p_novel]
    fainted / empty: zeros.
p_status etc.: P(j has at least one move of that class); entropy: of the atom posterior (nats); p_novel: mass of
the novel atoms.

Information rule (§15, §22.4): inputs are the BattleView (what the player sees), the challenge number and the
ExperienceModel (counts of what the player saw). Never the true state, the set list, the true IVs / EVs / sets.
tests/python/test_threat_features.py scans this source and checks that two states that look the same to the player
give the same features.

Cost: everything that does not move during a battle (posteriors, damage tables) is cached by what the player knows
(LRU); a decision recomputes only the HP / speed dependent parts. See docs §27.x for measured costs.

ThreatNet: FactoryNet whose battler trunk takes the features as N_THREAT more numbers per Pokemon token (the first
layer of the Pokemon encoder widened, new columns zero-initialized: from a FactoryNet state_dict it starts as exactly
that network). The tactician is unchanged (it gets zeros if it shares the trunk).
"""

import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from pybattle.view import NAMES, possible_abilities

from . import damage as DMG
from . import determinize as DET
from . import opponent_model as OM
from . import gamedata as GD
from .model import FactoryNet

MOVES, SPECIES, TYPE_EFFECTIVENESS = GD.MOVES, GD.SPECIES, GD.TYPE_EFFECTIVENESS

PF = 6
N_THREAT = 3 * PF
CAP = 1.5
ROLL = 0.925
TYP_IV, TYP_EV = 15.5, 85.0
ENT_SCALE = 8.0
MIN_SPECIES_P = 1e-4
CACHE_SIZE = 20000
TOPW = 64                                    # novel atoms: the 64 strongest non-revealed moves are kept

# ---- static move tables ---------------------------------------------------------------------------------------------

_E = {v: int(k) for k, v in NAMES["effects"].items()}
NM = len(MOVES)
M_TYPE = np.array([m["type"] for m in MOVES], np.int64)
M_EFF = np.array([m["effect"] for m in MOVES], np.int64)
M_PRIO = np.array([m["priority"] for m in MOVES], np.int64)
_pow = []
for m in MOVES:
    p = float(m["power"])
    if p == 1:
        p = float(DMG.VARIABLE_POWER.get(m["effect"], 60))
    if m["effect"] == DMG.EFFECT_FLAIL:
        p = 20.0                                         # at full HP (static tables)
    if m["effect"] == DMG.EFFECT_ERUPTION:
        p = 150.0
    if m["effect"] in DMG.NO_ESTIMATE or m["effect"] == DMG.EFFECT_SUPER_FANG or m["effect"] in DMG.FIXED_DAMAGE:
        p = 0.0
    _pow.append(p)
M_POW = np.array(_pow, np.float64)
M_FIXED = np.zeros(NM, np.float64)                       # fixed damage (HP); -1: the user's level
for i, m in enumerate(MOVES):
    e = m["effect"]
    if e in DMG.FIXED_DAMAGE:
        M_FIXED[i] = -1.0 if e in (_E["EFFECT_LEVEL_DAMAGE"], _E["EFFECT_PSYWAVE"]) else DMG.FIXED_DAMAGE[e]
M_HITS = np.array([DMG.HITS.get(m["effect"], 1) for m in MOVES], np.float64)
M_PHYS = M_TYPE < 9
M_DAMAGING = (M_POW > 0) | (M_FIXED != 0)
M_DAMAGING[0] = False


def _flag(names):
    ids = {_E[n] for n in names if n in _E}
    f = np.array([m["effect"] in ids for m in MOVES], bool)
    f[0] = False
    return f


F_STATUS = _flag(["EFFECT_SLEEP", "EFFECT_TOXIC", "EFFECT_POISON", "EFFECT_PARALYZE", "EFFECT_WILL_O_WISP",
                  "EFFECT_CONFUSE", "EFFECT_YAWN", "EFFECT_ATTRACT", "EFFECT_LEECH_SEED", "EFFECT_SWAGGER",
                  "EFFECT_FLATTER", "EFFECT_TEETER_DANCE"])
F_SETUP = _flag(["EFFECT_ATTACK_UP", "EFFECT_ATTACK_UP_2", "EFFECT_DEFENSE_UP", "EFFECT_DEFENSE_UP_2",
                 "EFFECT_SPECIAL_ATTACK_UP", "EFFECT_SPECIAL_ATTACK_UP_2", "EFFECT_SPECIAL_DEFENSE_UP",
                 "EFFECT_SPECIAL_DEFENSE_UP_2", "EFFECT_SPEED_UP", "EFFECT_SPEED_UP_2", "EFFECT_EVASION_UP",
                 "EFFECT_EVASION_UP_2", "EFFECT_BULK_UP", "EFFECT_CALM_MIND", "EFFECT_DRAGON_DANCE", "EFFECT_CURSE",
                 "EFFECT_BELLY_DRUM", "EFFECT_COSMIC_POWER", "EFFECT_MINIMIZE"])
F_RECOV = _flag(["EFFECT_RESTORE_HP", "EFFECT_SOFTBOILED", "EFFECT_MORNING_SUN", "EFFECT_SYNTHESIS",
                 "EFFECT_MOONLIGHT", "EFFECT_REST", "EFFECT_WISH", "EFFECT_PAIN_SPLIT"])
F_PRIO = (M_PRIO > 0) & M_DAMAGING
CLASSES = (F_STATUS, F_SETUP, F_RECOV, F_PRIO)

# ---- items, abilities, natures ---------------------------------------------------------------------------------------

_HOLD = np.array([it["hold_effect"] for it in DMG.ITEMS], np.int64)
_HOLD_PARAM = np.array([it.get("hold_effect_param", 0) for it in DMG.ITEMS], np.float64)
N_ITEMS_ALL = len(DMG.ITEMS)
ITEM_BAND = _HOLD == DMG.CHOICE_BAND
ITEM_BOOST_TYPE = np.full(N_ITEMS_ALL, -1, np.int64)        # the type an item powers up (-1: none)
for i, h in enumerate(_HOLD):
    if int(h) in DMG.TYPE_POWER_ITEM:
        ITEM_BOOST_TYPE[i] = DMG.TYPE_POWER_ITEM[int(h)]
ITEM_CLUB = _HOLD == DMG.THICK_CLUB
ITEM_BALL = _HOLD == DMG.LIGHT_BALL

# nature n raises stat n // 5 and lowers stat n % 5 (Atk Def Spe SpA SpD), neutral when equal
NAT_MULT = np.ones((25, 5))
for n in range(25):
    if n // 5 != n % 5:
        NAT_MULT[n, n // 5] = 1.1
        NAT_MULT[n, n % 5] = 0.9
NAT_UNIFORM = NAT_MULT.mean(0)                              # [5] expected multiplier, uniform nature
SPE = 2


def _species_arr(key):
    return np.array([s.get(key) for s in SPECIES], dtype=object)


BASE = np.array([s["base"] for s in SPECIES], np.float64)          # [n_species, 6] HP Atk Def Spe SpA SpD
TYPES = [tuple(s["types"]) for s in SPECIES]


def typical_stat(base, level):
    """A non-HP stat at IV 15.5, EV 85, neutral nature (float)."""
    return (2 * base + TYP_IV + TYP_EV / 4) * level / 100 + 5


def typical_hp(base, level):
    if np.ndim(base) == 0 and base == 1:
        return 1.0
    hp = (2 * base + TYP_IV + TYP_EV / 4) * level / 100 + level + 10
    return np.where(base == 1, 1.0, hp) if np.ndim(base) else hp


def _ability_atk_mult(abilities):
    """Expected physical attack multiplier from an attacker's possible abilities (Huge / Pure Power, Hustle)."""
    if not abilities:
        return 1.0
    out = []
    for a in abilities:
        m = 1.0
        if a in (DMG.HUGE_POWER, DMG.PURE_POWER):
            m = 2.0
        elif a == DMG.HUSTLE:
            m = 1.5
        out.append(m)
    return float(np.mean(out))


ABM_SPECIES = np.array([_ability_atk_mult(possible_abilities(s)) if s else 1.0 for s in range(len(SPECIES))])
TYPES2 = np.array([(list(t) + list(t))[:2] if t else [18, 18] for t in TYPES], np.int64)
_DEF_CACHE = {}


def defender_mult(def_types, abilities):
    key = (tuple(def_types), tuple(abilities))
    v = _DEF_CACHE.get(key)
    if v is None:
        v = _DEF_CACHE[key] = _defender_type_mult(def_types, abilities)
    return v


def _defender_type_mult(def_types, abilities):
    """[18] expected damage multiplier of each attacking type on a defender (type chart x ability), and [18] the
    probability that a hit of that type is super effective (eff > 1 and not blocked)."""
    eff = np.ones(18)
    for t in set(def_types):
        if t < 18:
            eff *= TYPE_EFFECTIVENESS[:, t].astype(np.float64)
    abilities = list(abilities) or [0]
    mult = np.zeros(18)
    se = np.zeros(18)
    for a in abilities:
        f = eff.copy()
        if a == DMG.LEVITATE:
            f[DMG.GROUND] = 0
        if a == DMG.WATER_ABSORB:
            f[DMG.WATER] = 0
        if a == DMG.VOLT_ABSORB:
            f[DMG.ELECTRIC] = 0
        if a == DMG.FLASH_FIRE:
            f[DMG.FIRE] = 0
        if a == DMG.WONDER_GUARD:
            f = np.where(f > 1, f, 0)
        if a == DMG.THICK_FAT:
            f[DMG.FIRE] *= 0.5
            f[DMG.ICE] *= 0.5
        mult += f
        se += (f > 1)
    return mult / len(abilities), se / len(abilities)


IS_CLUB_SP = np.zeros(len(SPECIES), bool)
IS_CLUB_SP[list(DMG.THICK_CLUB_SPECIES)] = True
IS_BALL_SP = np.zeros(len(SPECIES), bool)
IS_BALL_SP[list(DMG.LIGHT_BALL_SPECIES)] = True


def move_fracs(moves, *, level, A_phys, A_spec, att_types, item, species, D_phys, D_spec, def_mult, def_hp):
    """Damage fraction of each move (array of move ids, any shape) on a defender. Attacker values (A_*, att_types
    [..., 2], item, species) and defender values (D_*, def_hp, def_mult [..., 18]) broadcast against `moves`."""
    moves = np.asarray(moves, np.int64)
    t = M_TYPE[moves]
    phys = M_PHYS[moves]
    p = M_POW[moves]
    item = np.asarray(item, np.int64)
    species = np.asarray(species, np.int64)
    A = np.where(phys, A_phys, A_spec) * np.where(phys & ITEM_BAND[item], 1.5, 1.0)
    A = A * np.where(phys & ITEM_CLUB[item] & IS_CLUB_SP[species], 2.0, 1.0)
    A = A * np.where(~phys & ITEM_BALL[item] & IS_BALL_SP[species], 2.0, 1.0)
    p = np.where(ITEM_BOOST_TYPE[item] == t, p * (100 + _HOLD_PARAM[item]) / 100, p)
    D = np.where(phys, D_phys, D_spec)
    att_types = np.asarray(att_types, np.int64)
    stab = (t == att_types[..., 0]) | (t == att_types[..., 1])
    def_mult = np.asarray(def_mult, np.float64)
    if def_mult.ndim == 1:
        dm = def_mult[t]
    else:
        dm = np.take_along_axis(np.broadcast_to(def_mult, t.shape[:-1] + (18,)), t, -1)
    dmg = (42 * p * A / np.maximum(D, 1) / 50 + 2) * np.where(stab, 1.5, 1.0) * dm * ROLL * M_HITS[moves]
    dmg = np.where(p > 0, dmg, 0.0)
    fx = M_FIXED[moves]
    fixed = np.where(fx < 0, float(level), fx) * (dm > 0)
    dmg = np.where(fx != 0, fixed, dmg)
    return dmg / np.maximum(def_hp, 1.0)


# ---- speed distributions ---------------------------------------------------------------------------------------------

_SPEED = {}
NAT_SPE = (1.1, 0.9, 1.0)
MAX_SPEED = 1024


def speed_table(level):
    """[n_species, 3, MAX_SPEED + 1] P(speed > v) and P(speed == v) per species and nature (+, -, neutral) over
    IV 0..31 x EV 0..252 (step 4), all equally likely (the stat formula with its floors)."""
    v = _SPEED.get(level)
    if v is None:
        iv = np.arange(32)[:, None]
        ev = np.arange(0, 253, 4)[None]
        gt = np.zeros((len(SPECIES), 3, MAX_SPEED + 1), np.float32)
        eq = np.zeros_like(gt)
        for sp in range(1, len(SPECIES)):
            raw = (np.floor((2 * BASE[sp, 3] + iv + ev // 4) * level / 100) + 5).ravel()
            for k, m in enumerate(NAT_SPE):
                st = np.minimum(np.floor(raw * m), MAX_SPEED).astype(np.int64)
                h = np.bincount(st, minlength=MAX_SPEED + 1) / len(st)
                eq[sp, k] = h
                gt[sp, k] = 1.0 - np.cumsum(h)
        v = _SPEED[level] = (gt, eq)
    return v


def p_faster_species(sp, level, mult_foe, x, plus, minus):
    """P(foe speed x mult_foe > x) + half the ties, per species sp [U] with P(+Spe) plus [U], P(-Spe) minus [U]."""
    gt, eq = speed_table(level)
    y = x / mult_foe
    fy = int(min(max(math.floor(y), 0), MAX_SPEED))
    tie = 0.5 * eq[sp, :, fy] if float(y).is_integer() else 0.0
    p = gt[sp, :, fy] + tie                                   # [U, 3]
    return plus * p[:, 0] + minus * p[:, 1] + (1 - plus - minus) * p[:, 2]


# ---- hypergeometric helpers (novel atoms) ---------------------------------------------------------------------------

def _lcomb(n, k):
    from scipy.special import gammaln
    n = np.asarray(n, np.float64)
    k = np.asarray(k, np.float64)
    ok = (k >= 0) & (k <= n)
    out = gammaln(n + 1) - gammaln(k + 1) - gammaln(np.maximum(n - k, 0) + 1)
    return np.where(ok, out, -np.inf)


def p_none(n, q, k):
    """P(a uniform k-subset of n items avoids q given ones) = C(n - q, k) / C(n, k)."""
    n, q, k = np.asarray(n), np.asarray(q), np.asarray(k)
    return np.where(k <= 0, 1.0, np.exp(_lcomb(n - q, k) - _lcomb(n, k)))


def expected_max(sorted_desc, n, k, m0):
    """E[max(m0, max of a uniform k-subset)] per row: sorted_desc [S, W] values (desc, padded at the end), n [S]
    real lengths, k [S] subset sizes (<= n), m0 [S] the revealed moves' max."""
    S, W = sorted_desc.shape
    i = np.arange(W)[None]
    n_ = n[:, None]
    k_ = k[:, None]
    # P(item i is the largest chosen) = C(n - 1 - i, k - 1) / C(n, k)
    w = np.where((i < n_) & (k_ > 0), np.exp(_lcomb(n_ - 1 - i, k_ - 1) - _lcomb(n_, k_)), 0.0)
    val = np.maximum(sorted_desc, m0[:, None])
    e = (w * val).sum(1)
    if W:                                   # rows truncated to their W largest values: the rest at the W-th value
        e = e + np.clip(1.0 - w.sum(1), 0.0, 1.0) * np.maximum(sorted_desc[:, -1], m0) * (n > W)
    return np.where(k > 0, e, m0)


# ---- beliefs -------------------------------------------------------------------------------------------------------

class Belief:
    """The model's posterior of one opponent slot, as arrays (see the module docstring)."""

    def __init__(self, sp, w, moves, item, nat, nov_sp, nov_w, R, level, ab_known):
        self.sp, self.w, self.moves, self.item, self.nat = sp, w, moves, item, nat     # known atoms
        self.nov_sp, self.nov_w = nov_sp, nov_w                                         # novel atoms
        self.R = R
        self.level = level
        z = w.sum() + nov_w.sum()
        if z > 0:
            self.w, self.nov_w = w / z, nov_w / z
        # novel: the non-revealed learnable moves per species, padded
        lists = [[m for m in DET.learnable(int(s), level) if m not in R] for s in nov_sp]
        width = max([len(x) for x in lists] + [1])
        self.nov_moves = np.zeros((len(nov_sp), width), np.int64)
        self.nov_n = np.array([len(x) for x in lists], np.int64)
        for r, x in enumerate(lists):
            self.nov_moves[r, :len(x)] = x
        self.nov_k = np.minimum(np.maximum(4 - len(R), 0), self.nov_n)
        self.R_arr = np.array(sorted(R), np.int64)
        # attacker multipliers per atom: ability (expected unless revealed), nature
        self.ab = ab_known                                   # revealed ability or None
        # misc static features
        feats = []
        has_rev = [bool(len(self.R_arr)) and bool(f[self.R_arr].any()) for f in CLASSES]
        for f, hr in zip(CLASSES, has_rev):
            known = f[moves].any(1) if len(moves) else np.zeros(0, bool)
            q = (f[self.nov_moves] & (self.nov_moves > 0)).sum(1) if len(nov_sp) else np.zeros(0)
            pn = 1.0 - p_none(self.nov_n, q, self.nov_k) if len(nov_sp) else np.zeros(0)
            if hr:
                pn = np.ones_like(pn)
                known = np.ones_like(known)
            feats.append(float((self.w * known).sum() + (self.nov_w * pn).sum()))
        allw = np.concatenate([self.w, self.nov_w])
        allw = allw[allw > 0]
        ent = float(-(allw * np.log(allw)).sum())
        feats += [min(ent / ENT_SCALE, 1.0), float(self.nov_w.sum())]
        self.static = np.array(feats, np.float32)
        # species marginal (for damage dealt and speed)
        self.u_sp, inv = np.unique(np.concatenate([sp, nov_sp]).astype(np.int64), return_inverse=True)
        wa = np.concatenate([self.w, self.nov_w])
        self.u_w = np.bincount(inv, weights=wa, minlength=len(self.u_sp))
        # speed nature mix per species: P(+Spe), P(-Spe) (known atoms: observed natures; novel: uniform)
        plus = np.concatenate([nat[:, 5], np.full(len(nov_sp), 4 / 25)])
        minus = np.concatenate([nat[:, 6], np.full(len(nov_sp), 4 / 25)])
        uz = np.maximum(self.u_w, 1e-300)
        self.u_plus = np.bincount(inv, weights=wa * plus, minlength=len(self.u_sp)) / uz
        self.u_minus = np.bincount(inv, weights=wa * minus, minlength=len(self.u_sp)) / uz


def _nat_probs(model_t, s, c):
    """[5] expected nature multiplier and (P(+Spe), P(-Spe)) of a known configuration."""
    cnt = model_t.natures.get((s, c))
    if cnt is None:
        p = np.full(25, 1 / 25)
    else:
        a = model_t.cfg["a_nat"]
        p = (cnt + a / 25) / (cnt.sum() + a)
    return np.concatenate([p @ NAT_MULT, [p[_PLUS_SPE].sum(), p[_MINUS_SPE].sum()]])


_PLUS_SPE = [n for n in range(25) if n // 5 == SPE and n % 5 != SPE]
_MINUS_SPE = [n for n in range(25) if n % 5 == SPE and n // 5 != SPE]
NAT_DEFAULT = np.concatenate([NAT_UNIFORM, [4 / 25, 4 / 25]])


class ThreatFeaturizer:
    """tf(view, challenge) -> float32 [6, N_THREAT]. Caches keyed by what the player knows only."""

    def __init__(self, model, cache_size=CACHE_SIZE):
        self.model = model
        self.t = model.compile()
        self._beliefs = OrderedDict()
        self._pairs = OrderedDict()
        self.cache_size = cache_size
        self.stats = {"belief_miss": 0, "pair_miss": 0, "calls": 0}

    # --- caches ---
    def _get(self, cache, key, fn):
        v = cache.get(key)
        if v is None:
            v = fn()
            cache[key] = v
            if len(cache) > self.cache_size:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return v

    # --- beliefs ---
    def belief(self, view, j, challenge):
        enemy = view.enemy_party[:3]
        m = enemy[j]
        b = OM.bucket_of(challenge)
        level = DET.team_level(view)
        used = tuple(sorted({int(x.revealed_item) for k, x in enumerate(enemy)
                             if k != j and x.seen and x.revealed_item}))
        if m.seen:
            R = tuple(sorted({int(x) for x in m.revealed_moves if x}))
            I = int(m.revealed_item) if m.revealed_item else -1
            key = ("seen", b, int(m.species), R, I, used, int(m.revealed_ability or 0), int(m.level or level))
        else:
            excl = tuple(sorted({int(x.species) for x in enemy if x.seen}))
            key = ("unseen", b, excl, used, level)
        return key, self._get(self._beliefs, key, lambda: self._make_belief(key))

    def _make_belief(self, key):
        self.stats["belief_miss"] += 1
        t = self.t
        if key[0] == "seen":
            _, b, s, R, I, used, ab, level = key
            species = [(s, 1.0)]
            Rset = frozenset(R)
        else:
            _, b, excl, used, level = key
            p = t.species_probs(b, excl)
            keep = np.flatnonzero(p >= MIN_SPECIES_P)
            species = [(int(t.frontier[i]), float(p[i])) for i in keep]
            Rset, I, ab = frozenset(), -1, 0
        parts, nov_sp, nov_w = [], [], []
        for s, ps in species:
            q, mv, it, nat, pn = self._configs(b, s, Rset, I, used)
            if len(q):
                parts.append((np.full(len(q), s, np.int64), ps * q, mv, it, nat))
            if pn > 0:
                nov_sp.append(s)
                nov_w.append(ps * pn)
        if parts:
            sp, w, moves, item, nat = (np.concatenate(x) for x in zip(*parts))
        else:
            sp, w, moves, item, nat = (np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 4), np.int64),
                                       np.zeros(0, np.int64), np.zeros((0, 7)))
        return Belief(sp, w, moves, item, nat, np.array(nov_sp, np.int64), np.array(nov_w, np.float64), Rset, level,
                      ab or None)

    def _configs(self, b, s, R, I, used):
        """The model's configuration posterior of species s as arrays (cached): probs, moves [K, 4], items,
        nature multipliers [K, 7], P(novel)."""
        key = ("cfg", b, s, R, I, used)

        def make():
            cs, pc, pn = self.t.config_posterior(b, s, R, I, used)
            keep = [k for k, q in enumerate(pc) if q > 0]
            return (np.array([pc[k] for k in keep], np.float64),
                    np.array([(list(cs[k][0]) + [0] * 4)[:4] for k in keep], np.int64).reshape(-1, 4),
                    np.array([cs[k][1] for k in keep], np.int64),
                    np.array([_nat_probs(self.t, s, cs[k]) for k in keep], np.float64).reshape(-1, 7), float(pn))
        return self._get(self._beliefs, key, make)

    # --- pair tables (opponent belief x one of our Pokemon) ---
    @staticmethod
    def own_key(me):
        return (int(me.species), tuple(int(x) for x in me.moves), int(me.item), int(me.ability),
                tuple(int(x) for x in me.stats), int(me.max_hp), int(me.level))

    def pair(self, bkey, bel, me):
        key = (bkey, self.own_key(me))
        return self._get(self._pairs, key, lambda: self._make_pair(bel, me))

    def _att_arrays(self, bel, sp, nat):
        """Attack values of atoms (species sp, nature multipliers nat [., 7])."""
        L = bel.level
        if bel.ab is not None:
            abm = np.full(len(sp), _ability_atk_mult([bel.ab]))
        else:
            abm = ABM_SPECIES[sp]
        A_phys = typical_stat(BASE[sp, 1], L) * nat[:, 0] * abm
        A_spec = typical_stat(BASE[sp, 4], L) * nat[:, 3]
        return A_phys, A_spec, TYPES2[sp]

    def _make_pair(self, bel, me):
        self.stats["pair_miss"] += 1
        L = bel.level
        def_mult, def_se = defender_mult(me.types, [me.ability])
        D_phys, D_spec, hp = float(me.stats[1]), float(me.stats[4]), float(me.max_hp or 1)
        out = {}
        # known atoms: [K, 4]
        K = len(bel.sp)
        if K:
            A_phys, A_spec, types = self._att_arrays(bel, bel.sp, bel.nat)
            fr = move_fracs(bel.moves, level=L, A_phys=A_phys[:, None], A_spec=A_spec[:, None],
                            att_types=types[:, None, :], item=bel.item[:, None], species=bel.sp[:, None],
                            D_phys=D_phys, D_spec=D_spec, def_mult=def_mult, def_hp=hp)
            fr = np.where(bel.moves > 0, fr, 0.0)
            out["k_max"] = fr.max(1)
            se_mv = (def_se[M_TYPE[bel.moves]] > 0) & M_DAMAGING[bel.moves] & (M_FIXED[bel.moves] == 0)
            out["k_se"] = (se_mv * def_se[M_TYPE[bel.moves]]).max(1)
        else:
            out["k_max"] = out["k_se"] = np.zeros(0)
        # novel atoms: the revealed moves' max m0 and the sorted fractions of the rest
        S = len(bel.nov_sp)
        if S:
            nat = np.repeat(NAT_DEFAULT[None], S, 0)
            A_phys, A_spec, types = self._att_arrays(bel, bel.nov_sp, nat)
            kw = dict(level=L, A_phys=A_phys[:, None], A_spec=A_spec[:, None], att_types=types[:, None, :], item=0,
                      species=bel.nov_sp[:, None], D_phys=D_phys, D_spec=D_spec, def_mult=def_mult, def_hp=hp)
            fr = np.where(bel.nov_moves > 0, move_fracs(bel.nov_moves, **kw), -1.0)
            se = np.where(bel.nov_moves > 0, (def_se[M_TYPE[bel.nov_moves]] > 0) & M_DAMAGING[bel.nov_moves]
                          & (M_FIXED[bel.nov_moves] == 0), False)
            if len(bel.R_arr):
                rm = np.broadcast_to(bel.R_arr[None], (S, len(bel.R_arr)))
                m0 = move_fracs(rm, **kw).max(1)
                se_rev = ((def_se[M_TYPE[bel.R_arr]] > 0) & M_DAMAGING[bel.R_arr] & (M_FIXED[bel.R_arr] == 0)).any()
            else:
                m0 = np.zeros(S)
                se_rev = False
            srt = (-np.sort(-fr, 1))[:, :TOPW].astype(np.float32)
            out["n_sorted"] = srt
            out["n_m0"] = m0
            out["n_emax"] = expected_max(np.minimum(srt, CAP), bel.nov_n, bel.nov_k, np.minimum(m0, CAP))
            p_se = 1.0 - p_none(bel.nov_n, se.sum(1), bel.nov_k)
            out["n_se"] = np.ones(S) if se_rev else p_se
        # static aggregates
        out["p_se"] = float((bel.w * out["k_se"]).sum() + ((bel.nov_w * out["n_se"]).sum() if S else 0.0))
        out["dmg_in"] = float((bel.w * np.minimum(out["k_max"], CAP)).sum()
                              + ((bel.nov_w * out["n_emax"]).sum() if S else 0.0))
        # our damage on each species of the belief: best over our moves
        u = bel.u_sp
        own_moves = np.array([x for x in me.moves if x] or [0], np.int64)
        if bel.ab is not None:
            dms = np.stack([defender_mult(TYPES[s], [bel.ab])[0] for s in u]) if len(u) else np.zeros((0, 18))
        else:
            dms = np.stack([defender_mult(TYPES[s], possible_abilities(int(s)))[0] for s in u]) if len(u) \
                else np.zeros((0, 18))
        if len(u):
            f = move_fracs(np.broadcast_to(own_moves[None], (len(u), len(own_moves))), level=me.level,
                           A_phys=float(me.stats[0]) * _ability_atk_mult([me.ability]), A_spec=float(me.stats[3]),
                           att_types=TYPES2[int(me.species)] if len(me.types) == 0 else
                           np.array((list(me.types) + list(me.types))[:2]),
                           item=int(me.item), species=int(me.species), D_phys=typical_stat(BASE[u, 2], L)[:, None],
                           D_spec=typical_stat(BASE[u, 5], L)[:, None], def_mult=dms,
                           def_hp=typical_hp(BASE[u, 0], L)[:, None])
            best = np.where(own_moves[None] > 0, f, 0.0).max(1)
        else:
            best = np.zeros(0)
        out["u_best"] = best
        out["dmg_out"] = float((bel.u_w * np.minimum(best, CAP)).sum())
        return out

    # --- dynamic parts ---
    @staticmethod
    def _ko_in(bel, pt, h):
        if h <= 0:
            return 0.0
        p = float((bel.w * (pt["k_max"] >= h)).sum())
        if len(bel.nov_sp):
            hit0 = pt["n_m0"] >= h
            q = (pt["n_sorted"] >= h).sum(1)
            pn = np.where(hit0, 1.0, 1.0 - p_none(bel.nov_n, q, bel.nov_k))
            p += float((bel.nov_w * pn).sum())
        return p

    def _faster(self, bel, x, mult_foe):
        """P(foe faster than our speed x)."""
        if x <= 0:
            return 1.0
        if not len(bel.u_sp):
            return 0.0
        p = p_faster_species(bel.u_sp, bel.level, mult_foe, x, bel.u_plus, bel.u_minus)
        return float((bel.u_w * p).sum())

    def _faster_cached(self, bkey, bel, x, mult_foe):
        key = ("spd", bkey, round(float(x), 3), round(float(mult_foe), 4))
        return self._get(self._pairs, key, lambda: self._faster(bel, x, mult_foe))

    def __call__(self, view, challenge):
        self.stats["calls"] += 1
        out = np.zeros((6, N_THREAT), np.float32)
        own = view.own_party[:3]
        enemy = view.enemy_party[:3]
        a_own = view.own_active.party_index
        a_foe = view.enemy_active.party_index
        own_alive = [i for i, m in enumerate(own) if m.hp > 0]
        foe_alive = [j for j, m in enumerate(enemy) if not (m.seen and m.fainted)]
        pf = {}
        beliefs = {}
        for j in foe_alive:
            bkey, bel = self.belief(view, j, challenge)
            beliefs[j] = (bkey, bel)
            m = enemy[j]
            foe_h = max(m.hp_pixels, 0) / 48 if m.seen else 1.0
            foe_mult = 1.0
            if j == a_foe:
                foe_mult = DMG.stage_mult(view.enemy_active.stat_stages[SPE]) * (0.25 if m.status == 5 else 1.0)
            for i in own_alive:
                me = own[i]
                pt = self.pair(bkey, bel, me)
                x = float(me.stats[SPE])
                if i == a_own:
                    x *= DMG.stage_mult(view.own_active.stat_stages[SPE])
                if me.status == 5:
                    x *= 0.25
                best = pt["u_best"]
                ko_out = float((bel.u_w * (best >= foe_h)).sum()) if foe_h > 0 else 0.0
                pf[(j, i)] = np.array([pt["p_se"], pt["dmg_in"], self._ko_in(bel, pt, me.hp / max(me.max_hp, 1)),
                                       self._faster_cached(bkey, bel, x, foe_mult), pt["dmg_out"], ko_out],
                                      np.float32)
        z = np.zeros(PF, np.float32)
        for i in own_alive:
            out[i, :PF] = pf.get((a_foe, i), z)
            others = [pf[(j, i)] for j in foe_alive if j != a_foe]
            if others:
                out[i, PF:2 * PF] = np.max(others, 0)
        for j in foe_alive:
            out[3 + j, :PF] = pf.get((j, a_own), z)
            others = [pf[(j, i)] for i in own_alive if i != a_own]
            if others:
                out[3 + j, PF:2 * PF] = np.max(others, 0)
            out[3 + j, 2 * PF:] = beliefs[j][1].static
        return out


# ---- the network --------------------------------------------------------------------------------------------------

class ThreatNet(FactoryNet):
    """FactoryNet + N_THREAT numbers per Pokemon token in the battler (x["threat"] [B, 6, N_THREAT]). The battler
    trunk's Pokemon encoder takes [mon_num, threat]; its first layer is widened with zero-initialized columns, so a
    FactoryNet state_dict loads into a ThreatNet that computes exactly the same function until trained. Without
    x["threat"] the features are zeros. `threat_dim` 0 = a plain FactoryNet (control runs share this class)."""

    def __init__(self, *a, threat_dim=N_THREAT, **kw):
        super().__init__(*a, **kw)
        self.threat_dim = int(threat_dim)
        if self.threat_dim:
            old = self.trunk.mon_enc[0]
            new = nn.Linear(old.in_features + self.threat_dim, old.out_features)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :old.in_features] = old.weight
                new.bias.copy_(old.bias)
            self.trunk.mon_enc[0] = new

    def _with_threat(self, x, battler):
        if not self.threat_dim:
            return x
        num = x["mon_num"]
        th = x.get("threat") if battler else None
        if th is None:
            th = num.new_zeros(num.shape[:-1] + (self.threat_dim,))
        return dict(x, mon_num=torch.cat([num, th.to(num.dtype)], -1))

    def battler(self, x):
        return super().battler(self._with_threat(x, True))

    def _shared(self):
        return self.share == "all" and self.threat_dim

    def rental(self, x, lead=None):
        return super().rental(self._with_threat(x, False) if self._shared() else x, lead)

    def rental_joint(self, x):
        return super().rental_joint(self._with_threat(x, False) if self._shared() else x)

    def swap(self, x):
        return super().swap(self._with_threat(x, False) if self._shared() else x)

    def load_state_dict(self, state_dict, strict=True, **kw):
        """Accepts a FactoryNet state_dict (narrow first layer): the threat columns stay zero."""
        k = "trunk.mon_enc.0.weight"
        own = self.state_dict()
        if self.threat_dim and k in state_dict and state_dict[k].shape != own[k].shape:
            w = own[k].clone().zero_()
            w[:, :state_dict[k].shape[1]] = state_dict[k]
            state_dict = dict(state_dict)
            state_dict[k] = w
        return super().load_state_dict(state_dict, strict=strict, **kw)


# ---- offline dataset features, benchmark B, CLI ----------------------------------------------------------------------

V4_CKPT = "runs/alphazero_v4/stopped_it1111.pt"          # the dataset's recorder: its opponent model is the one used
THREAT_DIR = "runs/v5_offline/threat"


def load_model(path=V4_CKPT):
    """The experience model carried by a checkpoint (or a model file)."""
    return OM.ExperienceModel.load(path)


def _shard_of(run, procs):
    import hashlib
    return int(hashlib.sha1(str(int(run)).encode()).hexdigest()[:8], 16) % procs


def _compute_job(c, p, procs):
    import time
    from .offline_data import Dataset, replay_positions
    torch.set_num_threads(1)
    ds = Dataset(c["data"])
    feats = np.load(os.path.join(c["out"], "threat.npy"), mmap_mode="r+")
    done = np.load(os.path.join(c["out"], "done.npy"), mmap_mode="r+")
    run = np.asarray(ds["run"])
    uniq = np.unique(run)
    mine = np.array([r for r in uniq if _shard_of(r, procs) == p], np.int64)
    rows = np.flatnonzero(np.isin(run, mine) & ~done)
    tf = ThreatFeaturizer(load_model(c["model"]))
    t0, n = time.time(), 0
    runs_all = ds.runs
    # chunks of whole runs (each run replayed once), flushed after each chunk
    order = rows[np.argsort(run[rows], kind="stable")]
    rr = run[order]
    cuts = np.flatnonzero(np.diff(rr)) + 1
    groups = np.split(order, cuts) if len(order) else []
    chunk = []
    for gi, g in enumerate(groups):
        chunk.append(g)
        if len(chunk) < 50 and gi != len(groups) - 1:
            continue
        idx = np.concatenate(chunk)
        chunk = []
        out_idx, out_f = [], []
        tt = time.perf_counter()
        for rp in replay_positions(idx.tolist(), c["data"], runs=runs_all, clone=False):
            out_idx.append(rp.key)
            out_f.append(tf(rp.view, rp.ctx["challenge"]))
        feats[np.array(out_idx)] = np.stack(out_f).astype(feats.dtype)
        feats.flush()
        done[np.array(out_idx)] = True
        done.flush()
        n += len(out_idx)
        if p == 0:
            el = time.time() - t0
            print(f"[threat p0] {n} rows in {el:.0f}s ({n / max(el, 1e-9):.0f}/s per proc), "
                  f"chunk {len(out_idx)} rows {time.perf_counter() - tt:.1f}s, {tf.stats}", flush=True)
    return {"rows": n, "seconds": time.time() - t0, "stats": tf.stats}


def compute(a):
    """Features of every row of an offline dataset (runs/v5_offline/CONTRACT.md) -> <out>/threat.npy (float16
    [N, 6, N_THREAT], row-aligned), resumable (<out>/done.npy)."""
    import json
    import multiprocessing as mp
    from .offline_data import Dataset
    ds = Dataset(a.data)
    N = len(ds)
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, "threat.npy")
    dp = os.path.join(a.out, "done.npy")
    if os.path.exists(fp) and np.load(fp, mmap_mode="r").shape[0] != N:
        raise RuntimeError(f"{fp}: another dataset size; remove it")
    if not os.path.exists(fp):
        np.lib.format.open_memmap(fp, "w+", np.float16, (N, 6, N_THREAT)).flush()
        np.lib.format.open_memmap(dp, "w+", np.bool_, (N,)).flush()
    c = {"data": a.data, "out": a.out, "model": a.model}
    if a.procs <= 1:
        res = [_compute_job(c, 0, 1)]
    else:
        with mp.get_context("spawn").Pool(a.procs) as pool:
            res = pool.starmap(_compute_job, [(c, p, a.procs) for p in range(a.procs)])
    done = np.load(dp)
    meta = {"rows": int(N), "done": int(done.sum()), "data": a.data, "model": a.model, "n_threat": N_THREAT,
            "jobs": res}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in meta.items() if k != "jobs"}))
    return meta


def _shard_job(c, p, procs):
    """Worker p: features of every finished recording shard assigned to it (hash of the shard name), polling until
    the recording is over (c["until_meta"]: the dataset's meta.json exists) and nothing is left."""
    import glob
    import gzip
    import json
    import time
    from .offline_data import replay_positions
    torch.set_num_threads(1)
    tf = ThreatFeaturizer(load_model(c["model"]))
    os.makedirs(c["out"], exist_ok=True)
    n, t0 = 0, time.time()
    while True:
        names = sorted(q[:-4] for q in glob.glob(os.path.join(c["shards"], "*.npz")))
        todo = [nm for nm in names if hash_name(nm) % procs == p
                and not os.path.exists(os.path.join(c["out"], os.path.basename(nm) + ".npz"))]
        if not todo:
            if not c["watch"] or os.path.exists(c["until_meta"]):
                break
            time.sleep(30)
            continue
        for nm in todo:
            with np.load(nm + ".npz") as z:
                run, li = z["run"].astype(np.int64), z["li"].astype(np.int64)
            runs = {}
            with gzip.open(nm + ".runs.jsonl.gz", "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    runs[r["run"]] = r
            pos = {(int(r), int(l)): i for i, (r, l) in enumerate(zip(run, li))}
            feats = np.zeros((len(run), 6, N_THREAT), np.float16)
            tt = time.perf_counter()
            for rp in replay_positions(list(pos), runs=runs, clone=False):
                feats[pos[rp.key]] = tf(rp.view, rp.ctx["challenge"])
            dst = os.path.join(c["out"], os.path.basename(nm))
            np.savez(dst + ".tmp.npz", threat=feats, run=run, li=li)
            os.replace(dst + ".tmp.npz", dst + ".npz")
            n += len(run)
            print(f"[threat p{p}] {os.path.basename(nm)}: {len(run)} rows {time.perf_counter() - tt:.1f}s; total {n} "
                  f"({n / max(time.time() - t0, 1e-9):.0f}/s) {tf.stats}", flush=True)
    return {"rows": n, "seconds": time.time() - t0, "stats": tf.stats}


def hash_name(nm):
    import hashlib
    return int(hashlib.sha1(os.path.basename(nm).encode()).hexdigest()[:8], 16)


def compute_shards(a):
    """Features per recording shard, while the dataset is being recorded (--watch), into <out>/shards/."""
    import multiprocessing as mp
    c = {"shards": os.path.join(a.data, "shards"), "out": os.path.join(a.out, "shards"), "model": a.model,
         "watch": a.watch, "until_meta": os.path.join(a.data, "meta.json")}
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = pool.starmap(_shard_job, [(c, p, a.procs) for p in range(a.procs)])
    print(res)
    return res


def assemble(a):
    """<out>/shards/*.npz -> <out>/threat.npy, done.npy, meta.json, row-aligned with the consolidated dataset
    (consolidate concatenates the shards in sorted name order; checked against run.npy / li.npy)."""
    import glob
    import json
    from .offline_data import Dataset
    ds = Dataset(a.data)
    names = sorted(q[:-4] for q in glob.glob(os.path.join(a.data, "shards", "*.npz")))
    N = len(ds)
    th = np.lib.format.open_memmap(os.path.join(a.out, "threat.npy"), "w+", np.float16, (N, 6, N_THREAT))
    run, li = np.asarray(ds["run"]), np.asarray(ds["li"])
    i = 0
    for nm in names:
        with np.load(os.path.join(a.out, "shards", os.path.basename(nm) + ".npz")) as z:
            s = len(z["run"])
            if not (np.array_equal(z["run"], run[i:i + s]) and np.array_equal(z["li"], li[i:i + s])):
                raise RuntimeError(f"{nm}: rows do not match the consolidated dataset")
            th[i:i + s] = z["threat"]
        i += s
    assert i == N
    th.flush()
    np.save(os.path.join(a.out, "done.npy"), np.ones(N, bool))
    meta = {"rows": int(N), "done": int(N), "data": a.data, "model": a.model, "n_threat": N_THREAT,
            "shards": len(names)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))
    return meta


# --- benchmark B with a ThreatNet (rl.agent_probe's B: same states, seeds, protocol) ---

def load_net(ckpt, device="cpu"):
    """(net, value (mean, std), threat model path or None) of a checkpoint (FactoryNet or ThreatNet)."""
    from . import encode
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    encode.set_version(a.get("encode_version", 5))
    th = (a.get("offline") or {}).get("threat") or a.get("threat")
    net = ThreatNet(a["d_emb"], a["d"], a["layers"], a["heads"], share=a.get("share", "all"),
                    opt_feat=a.get("opt_feat", 0), threat_dim=N_THREAT if th else 0)
    net.load_state_dict(ck["net"])
    net.to(device).eval()
    norm = (ck.get("value_norm") or {}).get("battler")
    vn = (norm["mean"], max(norm["var"], 1e-4) ** 0.5) if norm else (0.0, 1.0)
    model = None
    if th:                                   # the features' directory: its meta.json names the opponent model
        import json
        mp = os.path.join(th, "meta.json") if isinstance(th, str) else ""
        model = json.load(open(mp))["model"] if os.path.exists(mp) else V4_CKPT
    return net, vn, model


class ThreatBattler:
    """rl.agent_probe.Battler-compatible (act_many(games)): the network greedy, with the threat features computed
    from each game's view when the checkpoint has them. Records the featurizer's time."""

    kind = "net"

    def __init__(self, ckpt, device="cuda"):
        self.net, self.vn, model = load_net(ckpt, device)
        self.device = device
        self.tf = ThreatFeaturizer(load_model(model)) if model else None
        self.t_feat = 0.0
        self.n_feat = 0

    def act_many(self, games):
        import time
        from . import encode
        from .search import run_ctx, to_backend_action
        if not games:
            return []
        obs = []
        for g in games:
            v = g.be.view()
            ctx = run_ctx(g.be)
            o = encode.battle(v, ctx)
            if self.tf is not None:
                t = time.perf_counter()
                o["threat"] = self.tf(v, ctx["challenge"])
                self.t_feat += time.perf_counter() - t
                self.n_feat += 1
            obs.append(o)
        x = encode.collate(obs, self.device)
        with torch.no_grad():
            logits, _ = self.net.battler(x)
        logits = logits.float().cpu().numpy()
        return [to_backend_action(int(np.argmax(np.where(o["mask"], l, -1e30)))) for o, l in zip(obs, logits)]


def bench_prefix(name):
    return os.path.join(THREAT_DIR, "bench", name)


def _bench_job(c, p, procs):
    import time
    from . import agent_probe as AP
    torch.set_num_threads(1)
    bench = AP.Bench(c["bench"])
    battler = ThreatBattler(c["ckpt"], c["device"])
    states = AP._shard(AP.b_states(bench, c["subset"]), p, procs, lambda s: s["run"])
    out = f"{c['prefix']}.p{p}.jsonl"
    done = {(r["id"], r["seed"]) for r in AP.read_shards(c["prefix"])}
    k = c["seeds"]
    todo = [s for s in states if any((s["id"], sd) not in done for sd in s["seeds"][:k])]
    t0, n_dec, buf = time.time(), [0], []

    def feeder():
        for s, pre in AP.replay_states(bench, todo):
            action = AP._dec(s["action"])
            AP.check_opponent(s, pre, action)
            for sd in s["seeds"][:k]:
                if (s["id"], sd) in done:
                    continue
                yield AP.Game((s["id"], sd), AP.battle_start(pre, action, sd), "battle", {"id": s["id"], "seed": sd})

    def on_done(g):
        n_dec[0] += g.decisions
        buf.append({"id": g.meta["id"], "seed": g.meta["seed"], "won": int(g.wins), "dec": g.decisions,
                    "trunc": int(g.truncated)})
        if len(buf) >= 64:
            AP.write_jsonl(out, buf)
            buf.clear()

    AP.play_lockstep(feeder(), battler, on_done, width=c["width"])
    AP.write_jsonl(out, buf)
    return {"seconds": time.time() - t0, "decisions": n_dec[0], "feat_seconds": battler.t_feat,
            "feat_calls": battler.n_feat}


def run_bench(ckpt, name, procs=2, seeds=4, device="cuda", width=256, subset=None, bench_name="battles_v1"):
    import json
    import multiprocessing as mp
    os.makedirs(os.path.dirname(bench_prefix(name)), exist_ok=True)
    c = {"ckpt": ckpt, "prefix": bench_prefix(name), "seeds": seeds, "device": device, "width": width,
         "subset": subset, "bench": bench_name}
    if procs <= 1:
        res = [_bench_job(c, 0, 1)]
    else:
        with mp.get_context("spawn").Pool(procs) as pool:
            res = pool.starmap(_bench_job, [(c, p, procs) for p in range(procs)])
    with open(bench_prefix(name) + ".timing.json", "a") as fh:
        fh.write(json.dumps({"ckpt": ckpt, "jobs": res}) + "\n")
    return res


def bench_per_state(name, bench_name="battles_v1"):
    from collections import defaultdict
    from . import agent_probe as AP
    per = defaultdict(list)
    for r in AP.read_shards(bench_prefix(name)):
        per[r["id"]].append(r["won"])
    return {k: float(np.mean(v)) for k, v in per.items()}


def paired_table(names, ref, bench_name="battles_v1", B=2000, seed=0):
    """Win rate per stratum (mean, bootstrap 95 % CI over states) and the paired difference with `ref`."""
    from . import agent_probe as AP
    bench = AP.Bench(bench_name)
    def per_of(n):              # "a+b+c": the per-state mean over several runs (e.g. training seeds)
        ps = [bench_per_state(x, bench_name) for x in n.split("+")]
        ids = set.intersection(*(set(p) for p in ps))
        return {i: float(np.mean([p[i] for p in ps])) for i in ids}
    pers = {n: per_of(n) for n in set(names) | {ref}}
    rng = np.random.default_rng(seed)
    out = {}
    for n in names:
        row = {}
        for st in (None, "r5", "r6", "r3", "noland"):
            ids = sorted(set(pers[n]) & set(pers[ref]))
            if st is not None:
                ids = [i for i in ids if bench.by_id[i]["stratum"] == st]
            if not ids:
                continue
            a = np.array([pers[n][i] for i in ids])
            d = a - np.array([pers[ref][i] for i in ids])
            bs = rng.integers(0, len(ids), (B, len(ids)))
            ci_a = np.percentile(a[bs].mean(1), [2.5, 97.5])
            ci_d = np.percentile(d[bs].mean(1), [2.5, 97.5])
            row[st or "total"] = {"n": len(ids), "win": float(a.mean()), "win_ci": ci_a.tolist(),
                                  "diff": float(d.mean()), "diff_ci": ci_d.tolist()}
        out[n] = row
    return out


def main(argv=None):
    import argparse
    import json
    p = argparse.ArgumentParser(description="threat features (docs/RL_DECISIONS.md §27.x)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--data", default="runs/v5_offline/main")
    c.add_argument("--out", default=THREAT_DIR)
    c.add_argument("--model", default=V4_CKPT)
    c.add_argument("--procs", type=int, default=3)
    cs = sub.add_parser("compute-shards")
    cs.add_argument("--data", default="runs/v5_offline/main")
    cs.add_argument("--out", default=THREAT_DIR)
    cs.add_argument("--model", default=V4_CKPT)
    cs.add_argument("--procs", type=int, default=3)
    cs.add_argument("--watch", action="store_true")
    asm = sub.add_parser("assemble")
    asm.add_argument("--data", default="runs/v5_offline/main")
    asm.add_argument("--out", default=THREAT_DIR)
    asm.add_argument("--model", default=V4_CKPT)
    b = sub.add_parser("bench")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--procs", type=int, default=2)
    b.add_argument("--seeds", type=int, default=4)
    b.add_argument("--subset", type=int, default=None)
    b.add_argument("--device", default="cuda")
    t = sub.add_parser("table")
    t.add_argument("--names", required=True)
    t.add_argument("--ref", required=True)
    a = p.parse_args(argv)
    if a.cmd == "compute":
        return compute(a)
    if a.cmd == "compute-shards":
        return compute_shards(a)
    if a.cmd == "assemble":
        return assemble(a)
    if a.cmd == "bench":
        r = run_bench(a.ckpt, a.name, a.procs, a.seeds, a.device, subset=a.subset)
        print(json.dumps(r))
        return r
    r = paired_table(a.names.split(","), a.ref)
    print(json.dumps(r, indent=1))
    return r


if __name__ == "__main__":
    main()
