"""Game knowledge for the play study (docs/RL_DECISIONS.md §23): Pokemon / team descriptors, roles, type coverage,
synergies and move classes, all from what a player knows (species data, move data, the type chart, and for our own
Pokemon their full summary). Used by rl/playstats.py.

A "mon" here is a plain dict (from a logged view):
    species, moves (non-zero ids), item (0 none / -1 unknown), ability (0 unknown), types, base [HP Atk Def Spe SpA
    SpD], stats [Atk Def Spe SpA SpD] or None (unknown: an opponent's), max_hp or None, ivs or None, full (bool: the
    whole set is known: ours, a rental; False: an opponent seen in battle, moves = the ones it used).
"""

from itertools import combinations

from pybattle.view import NAMES, SPECIES, MOVES
from .gamedata import TYPE_EFFECTIVENESS

_EFF = {v: int(k) for k, v in NAMES["effects"].items()}
_AB = {v: int(k) for k, v in NAMES["abilities"].items()}
_MV = {v: int(k) for k, v in NAMES["moves"].items()}

TYPE_NAMES = {int(k): v.replace("TYPE_", "").lower() for k, v in NAMES["types"].items()}
TYPE_MYSTERY = 9
TYPES17 = [t for t in range(18) if t != TYPE_MYSTERY]
PHYS_TYPES = [t for t in TYPES17 if t < 9]          # Gen 3: the move's category is its type's
SPEC_TYPES = [t for t in TYPES17 if t > 9]
NORMAL, WATER, FIRE, GRASS, ELECTRIC, ICE, GROUND, ROCK, STEEL, POISON, GHOST, DRAGON = 0, 11, 10, 12, 13, 15, 4, 5, 8, 3, 7, 16


def eff_id(name):
    return _EFF[f"EFFECT_{name}"]


def ability_id(name):
    return _AB[f"ABILITY_{name}"]


def move_id(name):
    return _MV[f"MOVE_{name}"]


def species_name(s):
    return NAMES["species"].get(str(s), str(s)).replace("SPECIES_", "").title()


def move_name(m):
    return NAMES["moves"].get(str(m), str(m)).replace("MOVE_", "").replace("_", " ").title()


def item_name(i):
    if i is None or i < 0:
        return "?"
    return NAMES["items"].get(str(i), str(i)).replace("ITEM_", "").replace("_", " ").title() if i else "-"


def effect_name(m):
    return NAMES["effects"].get(str(MOVES[m]["effect"]), "?").replace("EFFECT_", "")


# ---- move classes ---------------------------------------------------------------------------------------------------

E = eff_id
# moves whose damage ignores the type chart (only immunity) or that need special handling: not "typed attacks"
UNTYPED_DAMAGE = {E("LEVEL_DAMAGE"), E("DRAGON_RAGE"), E("SONICBOOM"), E("PSYWAVE"), E("SUPER_FANG"), E("ENDEAVOR"),
                  E("COUNTER"), E("MIRROR_COAT"), E("OHKO"), E("BIDE")}
HIDDEN_POWER, WEATHER_BALL = move_id("HIDDEN_POWER"), move_id("WEATHER_BALL")
HP_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]     # Hidden Power: fighting .. dark

STATUS_GROUPS = {          # the status strategies of §23 E, by move effect
    "toxic": {E("TOXIC")}, "poison": {E("POISON")}, "paralysis": {E("PARALYZE")}, "sleep": {E("SLEEP")},
    "yawn": {E("YAWN")}, "confusion": {E("CONFUSE"), E("SWAGGER"), E("FLATTER"), E("TEETER_DANCE")},
    "burn": {E("WILL_O_WISP")}, "leech_seed": {E("LEECH_SEED")},
}
MAJOR_STATUS_GROUPS = ("toxic", "poison", "paralysis", "sleep", "burn")
OFFENSIVE_BOOST = {E("ATTACK_UP"), E("ATTACK_UP_2"), E("SPEED_UP"), E("SPEED_UP_2"), E("SPECIAL_ATTACK_UP"),
                   E("SPECIAL_ATTACK_UP_2"), E("CALM_MIND"), E("DRAGON_DANCE"), E("BULK_UP"), E("CURSE"),
                   E("BELLY_DRUM")}
DEFENSIVE_BOOST = {E("DEFENSE_UP"), E("DEFENSE_UP_2"), E("SPECIAL_DEFENSE_UP"), E("SPECIAL_DEFENSE_UP_2"),
                   E("EVASION_UP"), E("EVASION_UP_2"), E("ACCURACY_UP"), E("ACCURACY_UP_2"), E("COSMIC_POWER"),
                   E("MINIMIZE"), E("DEFENSE_CURL"), E("STOCKPILE")}
BOOST = OFFENSIVE_BOOST | DEFENSIVE_BOOST
HEAL = {E("RESTORE_HP"), E("SOFTBOILED"), E("MOONLIGHT"), E("MORNING_SUN"), E("SYNTHESIS"), E("REST"), E("WISH"),
        E("SWALLOW")}
OHKO_EFFECT, EXPLOSION_EFFECT, HIGH_CRIT_EFFECT = E("OHKO"), E("EXPLOSION"), E("HIGH_CRITICAL")
PROTECT_EFFECT, SUBSTITUTE_EFFECT, BATON_PASS_EFFECT = E("PROTECT"), E("SUBSTITUTE"), E("BATON_PASS")
FLAG_CONTACT = 1
CONTACT_PUNISH = {ability_id(a) for a in ("STATIC", "POISON_POINT", "EFFECT_SPORE", "FLAME_BODY", "ROUGH_SKIN",
                                          "CUTE_CHARM")}
SERENE_GRACE = ability_id("SERENE_GRACE")


def minfo(m):
    return MOVES[m]


def is_attack(m):
    """A damaging move (power > 0; power 1 = variable power)."""
    return bool(m) and MOVES[m]["power"] > 0


def is_typed_attack(m):
    """A damaging move whose damage follows the type chart."""
    return is_attack(m) and MOVES[m]["effect"] not in UNTYPED_DAMAGE


def is_status_move(m):
    return bool(m) and MOVES[m]["power"] == 0


def hidden_power_type(ivs):
    """Gen 3 Hidden Power type from the IVs [HP Atk Def Spe SpA SpD]."""
    bits = [ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5]]
    v = sum((b & 1) << i for i, b in enumerate(bits))
    return HP_TYPES[v * 15 // 63]


def move_type(m, mon=None):
    if m == HIDDEN_POWER and mon is not None and mon.get("ivs"):
        return hidden_power_type(mon["ivs"])
    return MOVES[m]["type"]


def category(t):
    """'phys' / 'spec' / None (mystery) for a move type: Gen 3 splits by type."""
    return None if t == TYPE_MYSTERY else ("phys" if t < 9 else "spec")


def effectiveness(atk_type, def_types, ability=0):
    m = 1.0
    for t in set(def_types):
        if t < 18 and t != TYPE_MYSTERY:
            m *= float(TYPE_EFFECTIVENESS[atk_type, t])
    if ability:
        if (atk_type == GROUND and ability == ability_id("LEVITATE")) or \
           (atk_type == WATER and ability == ability_id("WATER_ABSORB")) or \
           (atk_type == ELECTRIC and ability == ability_id("VOLT_ABSORB")) or \
           (atk_type == FIRE and ability == ability_id("FLASH_FIRE")) or \
           (ability == ability_id("WONDER_GUARD") and m <= 1):
            return 0.0
        if ability == ability_id("THICK_FAT") and atk_type in (FIRE, ICE):
            m *= 0.5
    return m


def known_ability_of(species, revealed=0):
    """The ability when the player knows it: revealed, or the species' only possible one; else 0."""
    if revealed:
        return revealed
    ab = [a for a in SPECIES[species]["abilities"] if a]
    return ab[0] if len(set(ab)) == 1 else 0


# ---- a mon from a logged view ---------------------------------------------------------------------------------------

def own_mon(d):
    """An OwnMon dict (ours, a rental) -> mon."""
    sp = d["species"]
    return {"species": sp, "moves": [m for m in d["moves"] if m], "item": d["item"], "ability": d["ability"],
            "types": list(SPECIES[sp]["types"]), "base": list(SPECIES[sp]["base"]), "stats": list(d["stats"]),
            "max_hp": d["max_hp"], "ivs": list(d.get("ivs") or []) or None, "full": True}


def seen_mon(d):
    """A SeenMon dict (an opponent as the player saw it) -> mon (moves: the ones it used)."""
    sp = d["species"]
    item = d.get("revealed_item")
    return {"species": sp, "moves": [m for m in d.get("revealed_moves") or [] if m],
            "item": -1 if item is None else item, "ability": known_ability_of(sp, d.get("revealed_ability", 0)) if sp else 0,
            "types": list(SPECIES[sp]["types"]) if sp else [], "base": list(SPECIES[sp]["base"]) if sp else [0] * 6,
            "stats": None, "max_hp": None, "ivs": None, "full": False}


def mon_key(m):
    """The identity of a set: (species, sorted moves, item)."""
    return (m["species"], tuple(sorted(m["moves"])), m["item"])


# ---- per-mon descriptors ----------------------------------------------------------------------------------------------

def bst(m):
    return sum(m["base"])


def base_bulk(m):
    """(physical, special) bulk from base stats: HP x Def, HP x SpD (in thousands)."""
    b = m["base"]
    return b[0] * b[2] / 1000.0, b[0] * b[5] / 1000.0


def stat_bulk(m):
    """(physical, special) bulk from the real stats (known for ours): max HP x Def, max HP x SpD (in thousands)."""
    if not m.get("stats") or not m.get("max_hp"):
        return None
    return m["max_hp"] * m["stats"][1] / 1000.0, m["max_hp"] * m["stats"][4] / 1000.0


def n_status_moves(m):
    return sum(is_status_move(x) for x in m["moves"])


def has_effect(m, effects):
    return any(MOVES[x]["effect"] in effects for x in m["moves"])


ROLES = ("setup", "support", "fast", "wall", "slow_attacker")
ROLE_NAMES_ES = {"setup": "barredor con mejora", "support": "apoyo / estados", "fast": "barredor rápido",
                 "wall": "muro / tanque", "slow_attacker": "atacante lento / robusto"}


def role_of(m):
    """The role of a set (§23 A3), first that applies:
        setup          an offensive boost move (Swords Dance, Dragon Dance, Calm Mind, Bulk Up, Curse, Agility,
                       Belly Drum, Growth, Howl...) and >= 2 typed attacking moves
        support        >= 2 status (non-damaging) moves
        fast           base Speed >= 90
        wall           base HP + (base Def + base SpD) / 2 >= 170 and max(base Atk, base SpA) < 100
        slow_attacker  the rest (attackers with base Speed < 90)
    For an opponent seen in battle (moves = the ones it used) the role is a lower bound of what it is."""
    b = m["base"]
    n_att = sum(is_typed_attack(x) for x in m["moves"])
    if has_effect(m, OFFENSIVE_BOOST) and n_att >= 2:
        return "setup"
    if n_status_moves(m) >= 2:
        return "support"
    if b[3] >= 90:
        return "fast"
    if b[0] + (b[2] + b[5]) / 2 >= 170 and max(b[1], b[4]) < 100:
        return "wall"
    return "slow_attacker"


def attack_types(m):
    """[(type, category)] of its typed attacking moves."""
    out = []
    for x in m["moves"]:
        if is_typed_attack(x):
            t = move_type(x, m)
            if t != TYPE_MYSTERY:
                out.append((t, category(t)))
    return out


def stat_move_fit(m):
    """Fraction of its typed attacking moves whose category (physical / special) matches its higher attacking base
    stat (Atk >= SpA: physical); None without attacking moves."""
    at = attack_types(m)
    if not at:
        return None
    want = "phys" if m["base"][1] >= m["base"][4] else "spec"
    return sum(c == want for _, c in at) / len(at)


# ---- team descriptors -------------------------------------------------------------------------------------------------

def def_coverage(team):
    """Defensive coverage over the 17 attacking types (abilities the player knows count: Levitate...):
        resisted       types at least one member resists (x <= 0.5, immunities included)
        unresisted     17 - resisted (types nobody resists)
        shared_weak    types >= 2 members are weak to (x >= 2)
    each also split by the attacking type's category: _phys (9 types) / _spec (8 types)."""
    out = {}
    for name, types in (("", TYPES17), ("_phys", PHYS_TYPES), ("_spec", SPEC_TYPES)):
        res = weak2 = 0
        for a in types:
            ms = [effectiveness(a, m["types"], m.get("ability", 0)) for m in team]
            res += any(x < 1 for x in ms)
            weak2 += sum(x > 1 for x in ms) >= 2
        out["resisted" + name] = res
        out["unresisted" + name] = len(types) - res
        out["shared_weak" + name] = weak2
    return out


def off_coverage(team):
    """Offensive coverage: of the 17 single types, how many some member hits super-effectively (x >= 2) with a typed
    attacking move; also counting only physical moves (_phys) or only special moves (_spec)."""
    at = [t for m in team for t in attack_types(m)]
    out = {}
    for name, cats in (("", ("phys", "spec")), ("_phys", ("phys",)), ("_spec", ("spec",))):
        mts = {t for t, c in at if c in cats}
        out["se" + name] = sum(any(float(TYPE_EFFECTIVENESS[t, d]) > 1 for t in mts) for d in TYPES17)
    return out


def team_profile(team):
    """Every team descriptor used in §23 B / C / H (means over members unless noted)."""
    d = {}
    d.update(def_coverage(team))
    d.update(off_coverage(team))
    bb = [base_bulk(m) for m in team]
    d["phys_bulk"] = sum(x[0] for x in bb) / len(team)
    d["spec_bulk"] = sum(x[1] for x in bb) / len(team)
    d["base_speed"] = sum(m["base"][3] for m in team) / len(team)
    d["max_base_speed"] = max(m["base"][3] for m in team)
    d["min_base_speed"] = min(m["base"][3] for m in team)
    d["base_hp"] = sum(m["base"][0] for m in team) / len(team)
    d["base_offense"] = sum(max(m["base"][1], m["base"][4]) for m in team) / len(team)
    d["bst"] = sum(bst(m) for m in team) / len(team)
    d["status_moves"] = sum(n_status_moves(m) for m in team)
    at = [c for m in team for _, c in attack_types(m)]
    d["phys_moves"] = sum(c == "phys" for c in at)
    d["spec_moves"] = sum(c == "spec" for c in at)
    d["phys_share"] = d["phys_moves"] / len(at) if at else None
    fits = [f for f in (stat_move_fit(m) for m in team) if f is not None]
    d["fit"] = sum(fits) / len(fits) if fits else None
    d["misfits"] = sum(f < 0.5 for f in fits)
    roles = [role_of(m) for m in team]
    for r in ROLES:
        d["role_" + r] = roles.count(r)
    d["archetype"] = archetype(team)
    d.update({"syn_" + k: v for k, v in synergies(team).items()})
    return d


def archetype(team):
    """Team speed archetype (§23 B5), from base stats (means over the 3):
        fast_frail   mean base Speed >= 85 and mean base HP x (Def + SpD) / 2 < 7.5 (thousands)
        bulky_slow   mean base Speed < 70 and mean bulk >= 7.5
        mixed        the rest"""
    spe = sum(m["base"][3] for m in team) / len(team)
    bulk = sum(m["base"][0] * (m["base"][2] + m["base"][5]) / 2 for m in team) / len(team) / 1000.0
    if spe >= 85 and bulk < 7.5:
        return "fast_frail"
    if spe < 70 and bulk >= 7.5:
        return "bulky_slow"
    return "mixed"


# ---- synergies (§23 C) ------------------------------------------------------------------------------------------------

def _moves_with(team, effects):
    return [(i, x) for i, m in enumerate(team) for x in m["moves"] if MOVES[x]["effect"] in effects]


SYNERGIES = ("rain", "sun", "sand", "hail", "baton_pass", "screens", "spikes", "cleric", "core_fwg", "core_pair")
SYNERGY_NAMES_ES = {
    "rain": "lluvia (Danza Lluvia + ataque de agua / Trueno / Nado Rápido)",
    "sun": "sol (Día Soleado + ataque de fuego / Rayo Solar / Clorofila)",
    "sand": "arena (Tormenta Arena o Chorro Arena + roca / tierra / acero)",
    "hail": "granizo (Granizo + tipo hielo)",
    "baton_pass": "Relevo + mejora en el mismo Pokémon",
    "screens": "Reflejo / Pantalla de Luz",
    "spikes": "Púas",
    "cleric": "Cascabel Cura / Aromaterapia",
    "core_fwg": "núcleo fuego / agua / planta",
    "core_pair": "pareja defensiva (cada uno resiste las debilidades del otro)",
}


def synergies(team):
    """Which synergies a team has (booleans). Definitions:
        rain        a Rain Dance user and (a water attacking move, Thunder or Swift Swim) on the team
        sun         a Sunny Day user and (a fire attacking move, Solar Beam, Weather Ball or Chlorophyll)
        sand        (Sandstorm or Sand Stream) and >= 2 members immune to sand (rock / ground / steel types)
        hail        Hail and >= 2 ice-type members
        baton_pass  a Pokemon with Baton Pass and a boosting move or Substitute
        screens     Reflect or Light Screen
        spikes      Spikes
        cleric      Heal Bell / Aromatherapy
        core_fwg    a fire-, a water- and a grass-type member
        core_pair   two members each resisting every weakness of the other (both have at least one weakness)"""
    swift, chloro, sstream = ability_id("SWIFT_SWIM"), ability_id("CHLOROPHYLL"), ability_id("SAND_STREAM")
    s = {}
    rain_payoff = any(is_typed_attack(x) and move_type(x, m) == WATER or MOVES[x]["effect"] == E("THUNDER")
                      for m in team for x in m["moves"]) or any(m.get("ability") == swift for m in team)
    s["rain"] = bool(_moves_with(team, {E("RAIN_DANCE")})) and rain_payoff
    sun_payoff = any(is_typed_attack(x) and move_type(x, m) == FIRE or MOVES[x]["effect"] == E("SOLAR_BEAM")
                     or x == WEATHER_BALL for m in team for x in m["moves"]) or any(m.get("ability") == chloro
                                                                                   for m in team)
    s["sun"] = bool(_moves_with(team, {E("SUNNY_DAY")})) and sun_payoff
    sand_setter = bool(_moves_with(team, {E("SANDSTORM")})) or any(m.get("ability") == sstream for m in team)
    s["sand"] = sand_setter and sum(any(t in (ROCK, GROUND, STEEL) for t in m["types"]) for m in team) >= 2
    s["hail"] = bool(_moves_with(team, {E("HAIL")})) and sum(ICE in m["types"] for m in team) >= 2
    s["baton_pass"] = any(has_effect(m, {BATON_PASS_EFFECT}) and has_effect(m, BOOST | {SUBSTITUTE_EFFECT})
                          for m in team)
    s["screens"] = bool(_moves_with(team, {E("REFLECT"), E("LIGHT_SCREEN")}))
    s["spikes"] = bool(_moves_with(team, {E("SPIKES")}))
    s["cleric"] = bool(_moves_with(team, {E("HEAL_BELL")}))
    s["core_fwg"] = all(any(t in m["types"] for m in team) for t in (FIRE, WATER, GRASS))
    s["core_pair"] = any(_covers(a, b) and _covers(b, a) for a, b in combinations(team, 2))
    return s


def weaknesses(m):
    return [a for a in TYPES17 if effectiveness(a, m["types"], m.get("ability", 0)) > 1]


def _covers(a, b):
    """b resists every weakness of a (a has at least one)."""
    w = weaknesses(a)
    return bool(w) and all(effectiveness(t, b["types"], b.get("ability", 0)) < 1 for t in w)


def resist_count(team, t):
    """Members resisting attacking type t (x <= 0.5)."""
    return sum(effectiveness(t, m["types"], m.get("ability", 0)) < 1 for m in team)


def se_count(team, t):
    """Members with a typed attacking move super-effective against the single type t."""
    return sum(any(float(TYPE_EFFECTIVENESS[a, t]) > 1 for a, _ in attack_types(m)) for m in team)


# ---- status move legality (§23 E: errors) --------------------------------------------------------------------------------

def status_group(m):
    e = MOVES[m]["effect"]
    for g, effs in STATUS_GROUPS.items():
        if e in effs:
            return g
    return None


def status_failure(move, foe_types, foe_status, foe_ability, foe_confused, foe_seeded, foe_substitute,
                   safeguard=False):
    """Why a status move cannot work on the opponent's active Pokemon, from what the player sees (None if it can):
    'substitute' (behind a Substitute), 'safeguard', 'already_status' (a major status already), 'already_confused',
    'already_seeded', 'immune_type' (Thunder Wave vs ground / Glare vs ghost: the type chart; Toxic / poison vs
    poison / steel; Will-O-Wisp vs fire; Leech Seed vs grass), 'immune_ability' (a known Limber, Immunity, Insomnia /
    Vital Spirit, Own Tempo, Water Veil)."""
    g = status_group(move)
    if g is None:
        return None
    if foe_substitute:
        return "substitute"
    if safeguard and g in MAJOR_STATUS_GROUPS + ("confusion", "yawn"):
        return "safeguard"
    if g in MAJOR_STATUS_GROUPS + ("yawn",) and foe_status:
        return "already_status"
    if g == "confusion" and foe_confused:
        return "already_confused"
    if g == "leech_seed" and foe_seeded:
        return "already_seeded"
    mt = MOVES[move]["type"]
    if g == "paralysis" and effectiveness(mt, foe_types) == 0:
        return "immune_type"
    if g in ("toxic", "poison") and any(t in (POISON, STEEL) for t in foe_types):
        return "immune_type"
    if g == "burn" and FIRE in foe_types:
        return "immune_type"
    if g == "leech_seed" and GRASS in foe_types:
        return "immune_type"
    ab = foe_ability
    if ab and ((g == "paralysis" and ab == ability_id("LIMBER")) or
               (g in ("toxic", "poison") and ab == ability_id("IMMUNITY")) or
               (g in ("sleep", "yawn") and ab in (ability_id("INSOMNIA"), ability_id("VITAL_SPIRIT"))) or
               (g == "confusion" and ab == ability_id("OWN_TEMPO")) or
               (g == "burn" and ab == ability_id("WATER_VEIL"))):
        return "immune_ability"
    return None
