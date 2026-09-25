"""Hidden information of a Factory battle, sampled from what a player knows (search determinizations).

Strict player-knowledge rule (docs/RL_DECISIONS.md §15, decided 2026-09-25): no draw uses anything a real player
would not have. In particular the Battle Factory's list of sets (gBattleFrontierMons), the round's pool, the fixed
IVs of the Factory's opponents and the true state are never read. A player knows:
    - what the battle showed: species seen, moves used, items and abilities the game named, the HP bar, statuses,
      how many turns / move attempts a random-length effect has lasted (BattleView);
    - the game itself: which species can appear in the Battle Frontier (the Factory's species list; level 50
      excludes the high-tier ones), each species' base stats, abilities and learnable moves (level-up, TM/HM,
      move tutor, egg moves, inherited from pre-evolutions: pybattle/data/learnsets.json), the hold items that do
      something in battle, the stat formula, and the visible rules: no repeated species and no repeated held item
      in a team.

Per enemy slot (Gen3Game.determinize specs):
    species   seen: its species; unseen: uniform over the Frontier species, minus the team's other species
    moves     the revealed ones, the rest uniform without repetition over what the species can learn at its level
    item      the revealed one; else uniform over the battle hold items (species-specific ones only for their
              species) not held by another team member; a consumed / knocked-off item stays gone (Gen3Game keeps it)
    IVs       uniform 0-31 per stat
    EVs       a random spread of up to 510 (at most 255 per stat, drawn in steps of 4 up to 252)
    nature    uniform over the 25
    ability   the revealed one; else a fair coin between the species' two, except that a seen Pokemon whose
              species may have Intimidate and never announced it does not have it
    HP        seen and alive: an exact HP drawn uniformly among those showing the same 48-pixel bar for the drawn
              max HP (passed as hp / max_hp); unseen: full; fainted: kept as they are
Hidden counters (the elapsed part is what the player counted, the rest is resampled in Gen3Game.determinize):
sleep and confusion attempts (both sides), wrap / uproar / rampage turns.

Not modelled (documented simplifications): negative evidence is not used (e.g. no Leftovers message while damaged
does not rule Leftovers out; an unused move is not unlikely); Rest's fixed 3-turn sleep is not told apart from a
random one; moves only obtainable by trading from other games (26 of the 3,528 Factory set moves) are not in the
learnsets, and Smeargle (Sketch) may know any move.

The Factory's own rules (pool, fixed IVs, set-built max HP) stay below for tests and analysis only: the legal
sampler never calls them.

Training-time sampler "factory_sets" (sample_factory_sets, docs/RL_DECISIONS.md §18, decided 2026-09-25): for the
searches DURING TRAINING only (the battler's determinizations and the tactician's simulated opponents), the opponent
Pokemon are drawn from the Battle Factory's set list, the way the game draws them: the round's pool
(sInitialRentalMonRanges; level 50 without the high tier; no Unown), no repeated species or held item in the team,
the species on screen when the team was generated excluded (pybattle.backend.OpponentKnowledge; Noland: set ids
excluded). A set gives species, moves and held item; the ability is the species' coin flip, as in the game; IVs
(uniform 0-31), EVs (the random spread of _evs: the v4 assumption, any 0-252 per stat) and nature (uniform) are
never taken from the set. Everything the player saw is kept: a seen Pokemon's set must contain its revealed moves
and item; HP inside its bar, statuses and hidden counters as above; a seen Pokemon no consistent set explains falls
back to the strict draw. A player learns the sets by playing; the network input never contains them. Refused
outside training (rl.search.in_training): evaluation and inference keep the strict sampler.
"""

import json
import os
import random

import numpy as np

from pybattle import view as _view
from pybattle.view import _DATA, HP_BAR_PIXELS, NAMES, SPECIES, hp_bar_pixels, possible_abilities

KEEP = -1
NUM_NATURES = 25
PRIORS = ("strict", "factory_sets")     # opponent samplers: player knowledge / the Factory's set list (training)
MAX_EVS, MAX_EV_PER_STAT = 510, 252

_SPECIES_ID = {v: int(k) for k, v in NAMES["species"].items()}
_MOVE_ID = {v: int(k) for k, v in NAMES["moves"].items()}
_ITEM_ID = {v: int(k) for k, v in NAMES["items"].items()}
_ABILITY_ID = {v: int(k) for k, v in NAMES["abilities"].items()}
SPECIES_UNOWN = _SPECIES_ID["SPECIES_UNOWN"]
SPECIES_SHEDINJA = _SPECIES_ID["SPECIES_SHEDINJA"]
SPECIES_SMEARGLE = _SPECIES_ID["SPECIES_SMEARGLE"]
ABILITY_INTIMIDATE = _ABILITY_ID["ABILITY_INTIMIDATE"]

# ---- game knowledge -----------------------------------------------------------------------------------------------

_LEARNSETS = {int(k): v for k, v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(_view.__file__)),
                                                                "data", "learnsets.json"))).items()}
_ALL_MOVES = tuple(m for m in range(1, len(_DATA["moves"])) if m not in (_MOVE_ID["MOVE_STRUGGLE"],))
_learn_cache = {}


def learnable(species: int, level: int = 100):
    """Moves `species` can know at `level` (tuple of move ids, sorted)."""
    key = (species, level)
    out = _learn_cache.get(key)
    if out is None:
        if species == SPECIES_SMEARGLE:
            out = _ALL_MOVES
        else:
            ls = _LEARNSETS.get(species, {"level": [], "other": []})
            out = tuple(sorted({m for lv, m in ls["level"] if lv <= level} | set(ls["other"])))
        _learn_cache[key] = out
    return out


# hold effects with no battle effect (Exp. Share, Soothe Bell, Amulet Coin, Cleanse Tag, Smoke Ball, Everstone,
# Lucky Egg, Up-Grade, Dragon Scale): a player would not bring them
_NO_BATTLE_EFFECT = {"HOLD_EFFECT_EXP_SHARE", "HOLD_EFFECT_FRIENDSHIP_UP", "HOLD_EFFECT_DOUBLE_PRIZE",
                     "HOLD_EFFECT_REPEL", "HOLD_EFFECT_CAN_ALWAYS_RUN", "HOLD_EFFECT_PREVENT_EVOLVE",
                     "HOLD_EFFECT_LUCKY_EGG", "HOLD_EFFECT_UP_GRADE", "HOLD_EFFECT_DRAGON_SCALE"}
# species-specific items: only useful to these species
_ITEM_SPECIES = {
    "ITEM_SOUL_DEW": ("SPECIES_LATIAS", "SPECIES_LATIOS"),
    "ITEM_DEEP_SEA_TOOTH": ("SPECIES_CLAMPERL",), "ITEM_DEEP_SEA_SCALE": ("SPECIES_CLAMPERL",),
    "ITEM_LIGHT_BALL": ("SPECIES_PIKACHU",), "ITEM_LUCKY_PUNCH": ("SPECIES_CHANSEY",),
    "ITEM_METAL_POWDER": ("SPECIES_DITTO",), "ITEM_THICK_CLUB": ("SPECIES_CUBONE", "SPECIES_MAROWAK"),
    "ITEM_STICK": ("SPECIES_FARFETCHD",),
}
_HOLD_NAMES = NAMES["hold_effects"]
_GENERAL_ITEMS, _SPECIFIC_ITEMS = [], {}
for _i, _it in enumerate(_DATA["items"]):
    _he = _it["hold_effect"]
    if not _he or _HOLD_NAMES.get(str(_he)) in _NO_BATTLE_EFFECT:
        continue
    _name = NAMES["items"].get(str(_i))
    if _name in _ITEM_SPECIES:
        for _sp in _ITEM_SPECIES[_name]:
            _SPECIFIC_ITEMS.setdefault(_SPECIES_ID[_sp], []).append(_i)
    else:
        _GENERAL_ITEMS.append(_i)
_GENERAL_ITEMS = tuple(_GENERAL_ITEMS)


def plausible_items(species: int):
    """Hold items a player could expect on `species` (every item that does something in battle for it)."""
    return _GENERAL_ITEMS + tuple(_SPECIFIC_ITEMS.get(species, ()))


# ---- the Battle Factory's own rules (reference for tests and analysis; the legal sampler never uses them) -----------

FRONTIER_MONS = _DATA["frontier_mons"]
HELD_ITEMS = _DATA["frontier_held_items"]
RANGES = _DATA["factory_rental_ranges"]            # sInitialRentalMonRanges: 8 level-50 rows, then 8 open level
FIXED_IVS = _DATA["factory_fixed_ivs"]              # sFixedIVTable, plus the row read out of bounds at index 8
HIGH_TIER = 849                                     # FRONTIER_MONS_HIGH_TIER
SET_SPECIES = np.array([m["species"] for m in FRONTIER_MONS], np.int32)
SET_ITEM = np.array([HELD_ITEMS[m["item_table_id"]] for m in FRONTIER_MONS], np.int32)
SET_MOVES = [frozenset(x for x in m["moves"] if x) for m in FRONTIER_MONS]


def frontier_species(open_level: bool = True):
    """The species that can appear in the Battle Frontier's rental / opponent lists (a player knows the list;
    level 50 excludes the high-tier ones). Unown never appears."""
    ids = np.arange(len(FRONTIER_MONS)) if open_level else np.arange(HIGH_TIER + 1)
    return tuple(sorted(set(int(s) for s in SET_SPECIES[ids]) - {SPECIES_UNOWN}))


_FRONTIER_SPECIES = {True: frontier_species(True), False: frontier_species(False)}


def pool(challenge: int, open_level: bool = True) -> np.ndarray:
    """Set ids GetFactoryMonId can return for an opponent (useBetterRange FALSE), Unown removed."""
    row = (8 if open_level else 0) + min(challenge, 7)
    lo, hi = RANGES[row]
    ids = np.arange(lo, hi + 1)
    if not open_level:
        ids = ids[ids <= HIGH_TIER]
    return ids[SET_SPECIES[ids] != SPECIES_UNOWN]


def fixed_iv(challenge: int, last_battle: bool) -> int:
    """GetFactoryMonFixedIV without BUGFIX (index 8 reads past the table)."""
    i = len(FIXED_IVS) - 2 if challenge > len(FIXED_IVS) - 1 else challenge
    return FIXED_IVS[i][int(bool(last_battle))]


def enemy_iv(ctx) -> int:
    """The IV of every stat of every opponent Pokemon in a battle (the game's rule; hidden from the player)."""
    if ctx.get("brain"):
        return fixed_iv(ctx["challenge"] + 2, False)
    return fixed_iv(ctx.get("tower_challenge", 0), ctx["battle"] == 6)


def set_spec(slot: int, set_id: int, iv: int, ability_bit: int = 0, hp_fraction: float = -1.0):
    """The Gen3Game.determinize spec that rebuilds Battle Frontier set `set_id` exactly as the game does
    (CreateMonWithEVSpreadNatureOTID + SetMonMoveAvoidReturn + the set's item). Tests only."""
    m = FRONTIER_MONS[set_id]
    spread = m["ev_spread"]
    n = bin(spread & 0x3F).count("1")
    ev = MAX_EVS // n if n else 0
    evs = [ev if spread >> i & 1 else 0 for i in range(6)]
    moves = [_MOVE_ID["MOVE_FRUSTRATION"] if x == _MOVE_ID["MOVE_RETURN"] else x for x in m["moves"]]
    return (slot, m["species"], moves, HELD_ITEMS[m["item_table_id"]], [iv] * 6, evs, m["nature"], ability_bit,
            hp_fraction)


# ---- the stat formula -------------------------------------------------------------------------------------------------

def max_hp_of(species: int, iv: int, ev: int, level: int = 100) -> int:
    if species == SPECIES_SHEDINJA:
        return 1
    base = SPECIES[species]["base"][0]
    return (2 * base + iv + ev // 4) * level // 100 + level + 10


def max_hp(set_id: int, iv: int, level: int = 100) -> int:
    """A Battle Frontier set's max HP (tests only)."""
    s = set_spec(0, set_id, iv)
    return max_hp_of(s[1], iv, s[5][0], level)


# ---- the exclusions the game applies when generating an opponent (game rule, reference only) -----------------------

class ExclusionTracker:
    """What frontier.rentalMons held when the current opponent was generated, from the screens the player saw.
    Call on_rental(view) at each RENTAL decision, on_swap(view) at each SWAP decision and, after the swap is made,
    set_own_ids with the set ids of the team that goes into battle. The strict legal sampler does not use it (the
    generation rules are not player knowledge); it is kept for analysis."""

    def __init__(self):
        self.species = set()
        self.own_ids = set()

    def on_rental(self, view):
        self.species = {m.species for m in view.candidates}
        self.own_ids = set()          # our picks' ids are known after renting: set them with set_own_ids

    def on_swap(self, view, own_ids=()):
        self.species = {m.species for m in view.own_party[:3]} | {m.species for m in view.enemy_party[:3]}
        self.own_ids = set(own_ids)

    def set_own_ids(self, ids):
        self.own_ids = set(ids)


# ---- the legal sampler --------------------------------------------------------------------------------------------------

def _evs(rng: random.Random):
    """A random EV spread: up to 510 in steps of 4, at most 252 per stat."""
    ev = [0] * 6
    budget = rng.randint(0, MAX_EVS // 4)
    for _ in range(budget):
        open_stats = [i for i in range(6) if ev[i] + 4 <= MAX_EV_PER_STAT]
        if not open_stats:
            break
        ev[rng.choice(open_stats)] += 4
    return ev


def _hp_fraction(mh: int, pixels: int, rng: random.Random) -> float:
    if pixels >= HP_BAR_PIXELS:
        return 1.0
    hs = [h for h in range(1, mh + 1) if hp_bar_pixels(h, mh) == pixels]
    if not hs:
        return max(pixels, 1) / HP_BAR_PIXELS
    return hs[rng.randrange(len(hs))] / mh          # determinize: HP = round(max_hp * f)


def _ability_bit(species: int, seen_mon, rng: random.Random) -> int:
    a = SPECIES[species]["abilities"]
    if len(possible_abilities(species)) < 2:
        return 0
    if seen_mon is not None and seen_mon.revealed_ability:
        return 1 if (a[1] and seen_mon.revealed_ability == a[1] and a[1] != a[0]) else 0
    if seen_mon is not None and ABILITY_INTIMIDATE in a:
        # seen on the field and never announced Intimidate (the game announces it on every switch-in)
        return 0 if a[1] == ABILITY_INTIMIDATE else 1
    return rng.randrange(2)


def hidden_counters(view):
    """The elapsed counts the player saw, in Gen3Game.determinize's `hidden` layout (14 ints)."""
    own = [m.sleep_turns for m in view.own_party[:3]] + [0] * (3 - len(view.own_party[:3]))
    foe = [m.sleep_turns if m.seen else 0 for m in view.enemy_party[:3]]
    oa, ea = view.own_active, view.enemy_active
    return (own + foe + [oa.confusion_turns, ea.confusion_turns] + [oa.wrapped_turns, ea.wrapped_turns]
            + [oa.uproar_turns, ea.uproar_turns] + [oa.rampage_turns, ea.rampage_turns])


def team_level(view) -> int:
    """All Pokemon of a Factory battle have the same level (open level 100, else 50): a visible rule."""
    for m in view.enemy_party:
        if m.seen and m.level:
            return m.level
    return view.own_party[0].level


def check_prior(prior):
    """The opponent sampler `prior` may be used in this process: "factory_sets" only in training."""
    if prior not in PRIORS:
        raise ValueError(f"opponent prior must be one of {PRIORS}, not {prior!r}")
    if prior == "factory_sets":
        from .search import in_training
        if not in_training():
            raise PermissionError("the factory_sets opponent sampler is for training-time search only "
                                  "(evaluation and inference use the strict sampler)")


def determinization(view, knowledge=None, rng=None, open_level=True, prior="strict"):
    """Gen3Game.determinize specs drawn with the sampler `prior`: "strict" (sample_determinization) or
    "factory_sets" (sample_factory_sets; training only, needs the opponent's OpponentKnowledge)."""
    if prior == "strict":
        return sample_determinization(view, None, rng, open_level=open_level)
    check_prior(prior)
    return sample_factory_sets(view, knowledge, rng, open_level=open_level)


def sample_determinization(view, ctx=None, rng=None, open_level=True):
    """Specs for Gen3Game.determinize drawn from player knowledge (see the module docstring):
    [(party_slot, species | KEEP, [4 moves], item, [6 IVs], [6 EVs], nature, ability_bit, hp_fraction)].
    `ctx` is accepted for compatibility and not used (nothing of the streak context is needed)."""
    rng = rng or random.Random()
    level = team_level(view)
    enemy = view.enemy_party[:3]
    used_species = {m.species for m in enemy if m.seen}
    used_items = {m.revealed_item for m in enemy if m.seen and m.revealed_item}
    specs = []
    for i, m in enumerate(enemy):
        if m.seen and m.fainted:
            specs.append((i, KEEP, [0, 0, 0, 0], 0, [0] * 6, [0] * 6, 0, 0, -1.0))
            continue
        if m.seen:
            species = m.species
        else:
            choices = [s for s in _FRONTIER_SPECIES[bool(open_level)] if s not in used_species]
            species = choices[rng.randrange(len(choices))]
            used_species.add(species)
        revealed = [x for x in (m.revealed_moves if m.seen else []) if x][:4]
        rest = [x for x in learnable(species, level) if x not in revealed]
        n_more = min(4 - len(revealed), len(rest))
        moves = revealed + (rng.sample(rest, n_more) if n_more > 0 else [])
        moves += [0] * (4 - len(moves))
        if m.seen and m.revealed_item:
            item = m.revealed_item
        else:
            items = [x for x in plausible_items(species) if x not in used_items]
            item = items[rng.randrange(len(items))] if items else 0
            if item:
                used_items.add(item)
        ivs = [rng.randrange(32) for _ in range(6)]
        evs = _evs(rng)
        nature = rng.randrange(NUM_NATURES)
        bit = _ability_bit(species, m if m.seen else None, rng)
        hp = _hp_fraction(max_hp_of(species, ivs[0], evs[0], level), m.hp_pixels, rng) if m.seen else 1.0
        specs.append((i, species, moves, item, ivs, evs, nature, bit, hp))
    return specs


# ---- the training-time sampler: the Factory's set list -------------------------------------------------------------

# SetMonMoveAvoidReturn: the moves the opponent really knows (Return is given as Frustration)
SET_MOVES_AS_BUILT = [tuple(_MOVE_ID["MOVE_FRUSTRATION"] if x == _MOVE_ID["MOVE_RETURN"] else x for x in m["moves"])
                      for m in FRONTIER_MONS]
_SET_MOVESET = [frozenset(x for x in mv if x) for mv in SET_MOVES_AS_BUILT]
factory_stats = {"seen_slots": 0, "fallback": 0}           # seen Pokemon no Factory set explains (strict draw)


def factory_pool(knowledge, open_level=True) -> np.ndarray:
    """Set ids the opponent team described by `knowledge` (pybattle.backend.OpponentKnowledge) is drawn from."""
    ids = pool(knowledge.challenge, open_level)
    if knowledge.noland:
        if knowledge.set_ids:
            ids = ids[~np.isin(ids, np.array(sorted(knowledge.set_ids), np.int64))]
    elif knowledge.species:
        ids = ids[~np.isin(SET_SPECIES[ids], np.array(sorted(knowledge.species), np.int64))]
    return ids


def _free(ids, used_species, used_items):
    """Sets of `ids` whose species is not in the team and whose held item (if any) is not held by a team member."""
    ok = ~np.isin(SET_SPECIES[ids], np.array(sorted(used_species), np.int64))
    items = SET_ITEM[ids]
    ok &= (items == 0) | ~np.isin(items, np.array(sorted(used_items), np.int64))
    return ids[ok]


def _strict_moves_item(m, level, used_items, rng):
    """A seen Pokemon's moves and item by the strict rules (no Factory set explains what it showed)."""
    revealed = [x for x in m.revealed_moves if x][:4]
    rest = [x for x in learnable(m.species, level) if x not in revealed]
    n_more = min(4 - len(revealed), len(rest))
    moves = revealed + (rng.sample(rest, n_more) if n_more > 0 else [])
    moves += [0] * (4 - len(moves))
    if m.revealed_item:
        return moves, m.revealed_item
    items = [x for x in plausible_items(m.species) if x not in used_items]
    return moves, (items[rng.randrange(len(items))] if items else 0)


def sample_factory_sets(view, knowledge, rng=None, open_level=True):
    """Specs for Gen3Game.determinize (sample_determinization's layout) with the opponent's Pokemon drawn from the
    Factory's set list (training-time search only; see the module docstring). knowledge: the opponent's
    pybattle.backend.OpponentKnowledge (the round, Noland, the exclusions of its generation)."""
    if knowledge is None:
        raise ValueError("factory_sets needs the opponent's OpponentKnowledge (SimBackend.opponent_knowledge)")
    rng = rng or random.Random()
    level = team_level(view)
    enemy = view.enemy_party[:3]
    ids = factory_pool(knowledge, open_level)
    used_species = {m.species for m in enemy if m.seen}
    used_items = {m.revealed_item for m in enemy if m.seen and m.revealed_item}
    drawn = {}
    # the seen Pokemon first (their species are fixed), then the unseen ones; each uniform over the sets still legal
    # (the game draws set ids uniformly from the pool and rejects repeated species / items)
    for i in sorted(range(len(enemy)), key=lambda j: not enemy[j].seen):
        m = enemy[i]
        if m.seen and m.fainted:
            drawn[i] = None
            continue
        if m.seen:
            factory_stats["seen_slots"] += 1
            revealed = frozenset(x for x in m.revealed_moves if x)
            same = ids[SET_SPECIES[ids] == m.species]
            if m.revealed_item:
                same = same[SET_ITEM[same] == m.revealed_item]
            else:
                same = _free(same, set(), used_items)
            cand = [int(k) for k in same if revealed <= _SET_MOVESET[k]]
            if not cand:
                factory_stats["fallback"] += 1
                moves, item = _strict_moves_item(m, level, used_items, rng)
                if item:
                    used_items.add(item)
                drawn[i] = (m.species, moves, item)
                continue
        else:
            cand = _free(ids, used_species, used_items)
            if len(cand) == 0:                  # (never: a pool is far larger than a team)
                raise RuntimeError("no Factory set left for an unseen opponent")
        k = int(cand[rng.randrange(len(cand))])
        species, item = int(SET_SPECIES[k]), int(SET_ITEM[k])
        used_species.add(species)
        if item:
            used_items.add(item)
        drawn[i] = (species, list(SET_MOVES_AS_BUILT[k]), item)
    specs = []
    for i, m in enumerate(enemy):
        d = drawn[i]
        if d is None:
            specs.append((i, KEEP, [0, 0, 0, 0], 0, [0] * 6, [0] * 6, 0, 0, -1.0))
            continue
        species, moves, item = d
        ivs = [rng.randrange(32) for _ in range(6)]
        evs = _evs(rng)
        nature = rng.randrange(NUM_NATURES)
        bit = _ability_bit(species, m if m.seen else None, rng)
        hp = _hp_fraction(max_hp_of(species, ivs[0], evs[0], level), m.hp_pixels, rng) if m.seen else 1.0
        specs.append((i, species, moves, item, ivs, evs, nature, bit, hp))
    return specs
