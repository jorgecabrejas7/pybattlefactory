"""Approximate damage and speed estimates, as an experienced player would work them out (docs/RL_DECISIONS.md §14).

The Gen 3 damage formula is public knowledge. Our own Pokemon's stats are known exactly. The opponent's are not:
its species' base stats, level 100 and the Factory's fixed IVs for this round are known, but its EVs and nature
are not. No Factory set is used; instead each hidden stat is bounded by the possible EVs (0-252) and natures
(0.9-1.1), which gives a [min, max] range the way a player bounds a calculation. The random roll (85-100%)
widens it further. Abilities and items are applied only when the player knows them (own Pokemon; the opponent's
if revealed, or its species' only possible ability).
"""

import math

from pybattle.view import NAMES
from .gamedata import MOVES, SPECIES, TYPE_EFFECTIVENESS, _DATA

ITEMS = _DATA["items"]
STAGE = _DATA["stat_stage_ratios"]            # [num, den] per stage -6..+6
FIXED_IVS = _DATA["factory_fixed_ivs"]        # per challenge: [normal battles, 7th battle]

_E = {v: int(k) for k, v in NAMES["effects"].items()}
_A = {v: int(k) for k, v in NAMES["abilities"].items()}
_H = {v: int(k) for k, v in NAMES["hold_effects"].items()}
TYPE_POWER_ITEM = {_H[f"HOLD_EFFECT_{t}_POWER"]: i for i, t in enumerate(
    ("NORMAL", "FIGHTING", "FLYING", "POISON", "GROUND", "ROCK", "BUG", "GHOST", "STEEL", None, "FIRE", "WATER",
     "GRASS", "ELECTRIC", "PSYCHIC", "ICE", "DRAGON", "DARK")) if t}
FIXED_DAMAGE = {_E["EFFECT_LEVEL_DAMAGE"]: 100, _E["EFFECT_DRAGON_RAGE"]: 40, _E["EFFECT_SONICBOOM"]: 20,
                _E["EFFECT_PSYWAVE"]: 100}
HITS = {_E["EFFECT_DOUBLE_HIT"]: 2, _E["EFFECT_TWINEEDLE"]: 2, _E["EFFECT_MULTI_HIT"]: 3,
        _E["EFFECT_TRIPLE_KICK"]: 3}
VARIABLE_POWER = {_E["EFFECT_RETURN"]: 102, _E["EFFECT_FRUSTRATION"]: 102, _E["EFFECT_LOW_KICK"]: 60,
                  _E["EFFECT_MAGNITUDE"]: 71, _E["EFFECT_PRESENT"]: 52, _E["EFFECT_HIDDEN_POWER"]: 70}
NO_ESTIMATE = {_E["EFFECT_COUNTER"], _E["EFFECT_MIRROR_COAT"], _E["EFFECT_OHKO"], _E["EFFECT_ENDEAVOR"]}
EFFECT_FLAIL, EFFECT_ERUPTION, EFFECT_SUPER_FANG = _E["EFFECT_FLAIL"], _E["EFFECT_ERUPTION"], _E["EFFECT_SUPER_FANG"]
EFFECT_FACADE = _E["EFFECT_FACADE"]
LEVITATE, WATER_ABSORB, VOLT_ABSORB, FLASH_FIRE, WONDER_GUARD, THICK_FAT, HUGE_POWER, PURE_POWER, HUSTLE, GUTS = (
    _A[f"ABILITY_{a}"] for a in ("LEVITATE", "WATER_ABSORB", "VOLT_ABSORB", "FLASH_FIRE", "WONDER_GUARD",
                                 "THICK_FAT", "HUGE_POWER", "PURE_POWER", "HUSTLE", "GUTS"))
CHOICE_BAND, THICK_CLUB, LIGHT_BALL = _H["HOLD_EFFECT_CHOICE_BAND"], _H["HOLD_EFFECT_THICK_CLUB"], _H["HOLD_EFFECT_LIGHT_BALL"]
SP_CUBONE, SP_MAROWAK, SP_PIKACHU = (int(k) for k, v in NAMES["species"].items()
                                     if v in ("SPECIES_CUBONE", "SPECIES_MAROWAK", "SPECIES_PIKACHU"))
WEATHER_RAIN, WEATHER_SUN = 1, 2
STATUS_BURN = 3
FIRE, WATER, ELECTRIC, GROUND, ICE = 10, 11, 13, 4, 15

# base-stat index for (physical, special) attack and defense; stats order HP Atk Def Spe SpA SpD
ATK_IDX, DEF_IDX = (1, 4), (2, 5)
OWN_ATK, OWN_DEF = (0, 3), (1, 4)             # OwnMon.stats order Atk Def Spe SpA SpD


def is_physical(t):
    return t < 9


def enemy_ivs(challenge, battle_in_challenge):
    row = FIXED_IVS[min(challenge, 7)]
    return row[1] if battle_in_challenge == 6 else row[0]


def stat_range(base, iv, level=100):
    """[min, max] of a non-HP stat over EVs 0-252 and natures 0.9-1.1."""
    lo = math.floor((math.floor((2 * base + iv) * level / 100) + 5) * 0.9)
    hi = math.floor((math.floor((2 * base + iv + 63) * level / 100) + 5) * 1.1)
    return lo, hi


def hp_range(base, iv, level=100):
    if base == 1:                              # Shedinja
        return 1, 1
    return (math.floor((2 * base + iv) * level / 100) + level + 10,
            math.floor((2 * base + iv + 63) * level / 100) + level + 10)


def stage_mult(stage):
    n, d = STAGE[stage + 6]
    return n / d


def known_ability(possible, revealed):
    if revealed:
        return {revealed}
    ab = [a for a in possible if a]
    return {ab[0]} if len(set(ab)) == 1 else set()


def effectiveness(move_type, def_types):
    m = 1.0
    for t in set(def_types):
        if t < 18:
            m *= float(TYPE_EFFECTIVENESS[move_type, t])     # Python float: double math with any NumPy (NEP 50)
    return m


def move_damage(move, atk, dfn, *, atk_types, def_types, atk_stage=0, def_stage=0, atk_item=0, atk_ability=(),
                def_ability=(), atk_species=0, atk_burned=False, screen=False, weather=0, atk_hp_frac=1.0,
                def_hp=(1, 1)):
    """Damage range (HP) of `move`. atk / dfn are (min, max) stat ranges; def_hp the defender's max HP range."""
    if not move:
        return 0.0, 0.0
    m = MOVES[move]
    eff_id, mtype, power = m["effect"], m["type"], m["power"]
    if eff_id in NO_ESTIMATE:
        return 0.0, 0.0
    type_mult = effectiveness(mtype, def_types)
    if type_mult == 0:
        return 0.0, 0.0
    if (mtype == GROUND and LEVITATE in def_ability) or (mtype == WATER and WATER_ABSORB in def_ability) or \
       (mtype == ELECTRIC and VOLT_ABSORB in def_ability) or (mtype == FIRE and FLASH_FIRE in def_ability) or \
       (WONDER_GUARD in def_ability and type_mult <= 1):
        return 0.0, 0.0
    if eff_id in FIXED_DAMAGE:
        d = float(FIXED_DAMAGE[eff_id])
        return d, d
    if eff_id == EFFECT_SUPER_FANG:
        return def_hp[0] / 2, def_hp[1] / 2
    if power == 0:
        return 0.0, 0.0
    if power == 1:
        power = VARIABLE_POWER.get(eff_id, 60)
    if eff_id == EFFECT_FLAIL:
        power = 200 if atk_hp_frac < 0.04 else 150 if atk_hp_frac < 0.1 else 100 if atk_hp_frac < 0.2 else \
            80 if atk_hp_frac < 0.35 else 40 if atk_hp_frac < 0.69 else 20
    if eff_id == EFFECT_ERUPTION:
        power = max(1, int(150 * atk_hp_frac))
    if eff_id == EFFECT_FACADE and atk_burned:
        power *= 2
    phys = is_physical(mtype)
    a_mult = stage_mult(atk_stage)
    if phys and (HUGE_POWER in atk_ability or PURE_POWER in atk_ability):
        a_mult *= 2
    if phys and HUSTLE in atk_ability:
        a_mult *= 1.5
    if phys and ITEMS[atk_item]["hold_effect"] == CHOICE_BAND:
        a_mult *= 1.5
    if phys and ITEMS[atk_item]["hold_effect"] == THICK_CLUB and atk_species in (SP_CUBONE, SP_MAROWAK):
        a_mult *= 2
    if not phys and ITEMS[atk_item]["hold_effect"] == LIGHT_BALL and atk_species == SP_PIKACHU:
        a_mult *= 2
    if ITEMS[atk_item]["hold_effect"] in TYPE_POWER_ITEM and TYPE_POWER_ITEM[ITEMS[atk_item]["hold_effect"]] == mtype:
        power = power * (100 + ITEMS[atk_item]["hold_effect_param"]) / 100
    if THICK_FAT in def_ability and mtype in (FIRE, ICE):
        power /= 2
    d_mult = stage_mult(def_stage)
    out = []
    for a, d, roll in ((atk[0], dfn[1], 0.85), (atk[1], dfn[0], 1.0)):
        dmg = math.floor(math.floor(42 * power * a * a_mult / max(1, d * d_mult)) / 50)
        if phys and atk_burned and GUTS not in atk_ability:
            dmg //= 2
        if screen:
            dmg //= 2
        dmg += 2
        if weather == WEATHER_RAIN:
            dmg *= 1.5 if mtype == WATER else 0.5 if mtype == FIRE else 1
        elif weather == WEATHER_SUN:
            dmg *= 1.5 if mtype == FIRE else 0.5 if mtype == WATER else 1
        if mtype in atk_types:
            dmg *= 1.5
        dmg *= type_mult * roll * HITS.get(eff_id, 1)
        out.append(dmg)
    return out[0], out[1]


# ---- the features the battler receives ----------------------------------------------------------------------

def battle_estimates(view, ctx):
    """Per own Pokemon i and move j: damage vs the enemy's active Pokemon as fractions of its max HP
    (min, max, can-KO-now); per revealed enemy move j: damage vs our active Pokemon (min, max, can-KO-now);
    per own Pokemon: worst revealed threat (max fraction, can-KO) and whether it outspeeds (sure, maybe)."""
    foe = view.enemy_party[view.enemy_active.party_index]
    fa = view.enemy_active
    iv = enemy_ivs(ctx["challenge"], ctx["battle"])
    base = SPECIES[foe.species]["base"] if foe.species else [0] * 6
    f_hp = hp_range(base[0], iv)
    f_atk = [stat_range(base[i], iv) for i in ATK_IDX]
    f_def = [stat_range(base[i], iv) for i in DEF_IDX]
    f_spe = stat_range(base[3], iv)
    f_ab = known_ability(foe.possible_abilities, foe.revealed_ability)
    f_item = foe.revealed_item or 0
    f_frac = max(foe.hp_pixels, 0) / 48
    a_own = view.own_active.party_index
    own_moves, threats, speed = [], [], []
    for i, me in enumerate(view.own_party):
        active = i == a_own
        st = view.own_active if active else None
        rows = []
        for mv in me.moves:
            phys = bool(mv) and is_physical(MOVES[mv]["type"])
            k = 0 if phys else 1
            a = me.stats[OWN_ATK[k]]
            lo, hi = move_damage(mv, (a, a), f_def[k], atk_types=me.types, def_types=fa.types,
                                 atk_stage=st.stat_stages[0 if phys else 3] if st else 0,
                                 def_stage=fa.stat_stages[1 if phys else 4], atk_item=me.item,
                                 atk_ability={me.ability}, def_ability=f_ab, atk_species=me.species,
                                 atk_burned=me.status == STATUS_BURN,
                                 screen=(view.enemy_side.reflect_turns if phys else view.enemy_side.light_screen_turns) > 0,
                                 weather=view.weather, atk_hp_frac=me.hp / me.max_hp if me.max_hp else 0,
                                 def_hp=f_hp)
            lo_f, hi_f = lo / f_hp[1], hi / f_hp[0]
            rows.append((min(lo_f, 1.5), min(hi_f, 1.5), float(hi_f >= f_frac > 0)))
        own_moves.append(rows)
        worst, ko = 0.0, 0.0
        for mv in foe.revealed_moves:
            phys = is_physical(MOVES[mv]["type"])
            k = 0 if phys else 1
            d = me.stats[OWN_DEF[k]]
            lo, hi = move_damage(mv, f_atk[k], (d, d), atk_types=fa.types if foe.seen else [], def_types=me.types,
                                 atk_stage=fa.stat_stages[0 if phys else 3],
                                 def_stage=st.stat_stages[1 if phys else 4] if st else 0, atk_item=f_item,
                                 atk_ability=f_ab, def_ability={me.ability}, atk_species=foe.species,
                                 atk_burned=foe.status == STATUS_BURN,
                                 screen=(view.own_side.reflect_turns if phys else view.own_side.light_screen_turns) > 0,
                                 weather=view.weather, atk_hp_frac=f_frac, def_hp=(me.max_hp, me.max_hp))
            frac = hi / me.max_hp if me.max_hp else 0
            worst = max(worst, min(frac, 1.5))
            ko = max(ko, float(me.hp > 0 and hi >= me.hp))
        threats.append((worst, ko))
        spe = me.stats[2] * (stage_mult(view.own_active.stat_stages[2]) if active else 1)
        f_lo = f_spe[0] * stage_mult(fa.stat_stages[2])
        f_hi = f_spe[1] * stage_mult(fa.stat_stages[2])
        speed.append((float(spe > f_hi), float(f_lo < spe <= f_hi)))
    # the enemy's revealed moves vs our active Pokemon, per enemy move slot
    me = view.own_party[a_own]
    st = view.own_active
    foe_moves = []
    for mv in foe.revealed_moves[:4]:
        phys = is_physical(MOVES[mv]["type"])
        k = 0 if phys else 1
        d = me.stats[OWN_DEF[k]]
        lo, hi = move_damage(mv, f_atk[k], (d, d), atk_types=fa.types, def_types=me.types,
                             atk_stage=fa.stat_stages[0 if phys else 3], def_stage=st.stat_stages[1 if phys else 4],
                             atk_item=f_item, atk_ability=f_ab, def_ability={me.ability}, atk_species=foe.species,
                             atk_burned=foe.status == STATUS_BURN,
                             screen=(view.own_side.reflect_turns if phys else view.own_side.light_screen_turns) > 0,
                             weather=view.weather, atk_hp_frac=f_frac, def_hp=(me.max_hp, me.max_hp))
        foe_moves.append((min(lo / me.max_hp, 1.5), min(hi / me.max_hp, 1.5), float(me.hp > 0 and hi >= me.hp)))
    return own_moves, foe_moves, threats, speed
