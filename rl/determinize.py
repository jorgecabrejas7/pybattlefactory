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
excluded). A drawn set is built exactly as the game builds it: species, moves, held item, EV spread and nature from
the set, every IV the game's value for that battle (fixed_iv; Noland's own row), the ability the species' coin flip.
These are the search's model of the world only (decided 2026-09-25: the simulated opponents are real Factory
opponents); they never enter the network's input. Everything the player saw is kept: a seen Pokemon's set must contain its revealed moves
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


# ---- the attendant's hint about the next opponent (battle_factory.c) ------------------------------------------------
# Before a normal battle the attendant names the opponent's most common type and its battle style (RentalView /
# SwapView / BattleView .hint_type / .hint_style). The game computes both from the opponent's 3 sets
# (GetOpponentMostCommonMonType, GetOpponentBattleStyle); hint_of_team replicates it, tie rules included. Before
# Noland's battle there is no hint (NO_HINT = (18, 0), which a normal opponent can also get).

N_HINT_TYPES = 19                   # type ids 0..17 (TYPE_MYSTERY 9 never occurs), 18 = no single most common type
HINT_TYPE_NONE = 18                 # NUMBER_OF_MON_TYPES
N_HINT_STYLES = 9                   # FACTORY_STYLE_NONE 0, the 7 styles 1..7, FACTORY_NUM_STYLES 8 = more than 2
HINT_STYLE_MANY = 8
NO_HINT = (HINT_TYPE_NONE, 0)
# sMoveStyles, in FACTORY_STYLE_* order (1 PREPARATION .. 7 WEATHER); a move in two lists (Snatch) counts for the
# first (GetMoveBattleStyle returns the first match)
_STYLE_MOVES = (
    ("MOVE_SWORDS_DANCE", "MOVE_GROWTH", "MOVE_MEDITATE", "MOVE_AGILITY", "MOVE_DOUBLE_TEAM", "MOVE_HARDEN",
     "MOVE_MINIMIZE", "MOVE_WITHDRAW", "MOVE_DEFENSE_CURL", "MOVE_BARRIER", "MOVE_FOCUS_ENERGY", "MOVE_AMNESIA",
     "MOVE_ACID_ARMOR", "MOVE_SHARPEN", "MOVE_CONVERSION", "MOVE_CONVERSION_2", "MOVE_BELLY_DRUM", "MOVE_PSYCH_UP",
     "MOVE_CHARGE", "MOVE_SNATCH", "MOVE_TAIL_GLOW", "MOVE_COSMIC_POWER", "MOVE_IRON_DEFENSE", "MOVE_HOWL",
     "MOVE_BULK_UP", "MOVE_CALM_MIND", "MOVE_DRAGON_DANCE"),
    ("MOVE_SING", "MOVE_SUPERSONIC", "MOVE_DISABLE", "MOVE_LEECH_SEED", "MOVE_POISON_POWDER", "MOVE_STUN_SPORE",
     "MOVE_SLEEP_POWDER", "MOVE_THUNDER_WAVE", "MOVE_TOXIC", "MOVE_HYPNOSIS", "MOVE_CONFUSE_RAY", "MOVE_GLARE",
     "MOVE_POISON_GAS", "MOVE_LOVELY_KISS", "MOVE_SPORE", "MOVE_SPIDER_WEB", "MOVE_SWEET_KISS", "MOVE_SPIKES",
     "MOVE_SWAGGER", "MOVE_MEAN_LOOK", "MOVE_ATTRACT", "MOVE_ENCORE", "MOVE_TORMENT", "MOVE_FLATTER",
     "MOVE_WILL_O_WISP", "MOVE_TAUNT", "MOVE_YAWN", "MOVE_IMPRISON", "MOVE_SNATCH", "MOVE_TEETER_DANCE",
     "MOVE_GRASS_WHISTLE", "MOVE_BLOCK"),
    ("MOVE_MIST", "MOVE_RECOVER", "MOVE_LIGHT_SCREEN", "MOVE_HAZE", "MOVE_REFLECT", "MOVE_SOFT_BOILED", "MOVE_REST",
     "MOVE_PROTECT", "MOVE_DETECT", "MOVE_ENDURE", "MOVE_MILK_DRINK", "MOVE_HEAL_BELL", "MOVE_SAFEGUARD",
     "MOVE_BATON_PASS", "MOVE_MORNING_SUN", "MOVE_SYNTHESIS", "MOVE_MOONLIGHT", "MOVE_SWALLOW", "MOVE_WISH",
     "MOVE_INGRAIN", "MOVE_MAGIC_COAT", "MOVE_RECYCLE", "MOVE_REFRESH", "MOVE_MUD_SPORT", "MOVE_SLACK_OFF",
     "MOVE_AROMATHERAPY", "MOVE_WATER_SPORT"),
    ("MOVE_GUILLOTINE", "MOVE_HORN_DRILL", "MOVE_DOUBLE_EDGE", "MOVE_HYPER_BEAM", "MOVE_COUNTER", "MOVE_FISSURE",
     "MOVE_BIDE", "MOVE_SELF_DESTRUCT", "MOVE_SKY_ATTACK", "MOVE_EXPLOSION", "MOVE_FLAIL", "MOVE_REVERSAL",
     "MOVE_DESTINY_BOND", "MOVE_PERISH_SONG", "MOVE_PAIN_SPLIT", "MOVE_MIRROR_COAT", "MOVE_MEMENTO", "MOVE_GRUDGE",
     "MOVE_FACADE", "MOVE_FOCUS_PUNCH", "MOVE_BLAST_BURN", "MOVE_HYDRO_CANNON", "MOVE_OVERHEAT",
     "MOVE_FRENZY_PLANT", "MOVE_PSYCHO_BOOST", "MOVE_VOLT_TACKLE"),
    ("MOVE_SAND_ATTACK", "MOVE_TAIL_WHIP", "MOVE_LEER", "MOVE_GROWL", "MOVE_STRING_SHOT", "MOVE_SCREECH",
     "MOVE_SMOKESCREEN", "MOVE_KINESIS", "MOVE_FLASH", "MOVE_COTTON_SPORE", "MOVE_SPITE", "MOVE_SCARY_FACE",
     "MOVE_CHARM", "MOVE_KNOCK_OFF", "MOVE_SWEET_SCENT", "MOVE_FEATHER_DANCE", "MOVE_FAKE_TEARS",
     "MOVE_METAL_SOUND", "MOVE_TICKLE"),
    ("MOVE_MIMIC", "MOVE_METRONOME", "MOVE_MIRROR_MOVE", "MOVE_TRANSFORM", "MOVE_SUBSTITUTE", "MOVE_SKETCH",
     "MOVE_CURSE", "MOVE_PRESENT", "MOVE_FOLLOW_ME", "MOVE_TRICK", "MOVE_ROLE_PLAY", "MOVE_ASSIST", "MOVE_SKILL_SWAP",
     "MOVE_CAMOUFLAGE"),
    ("MOVE_SANDSTORM", "MOVE_RAIN_DANCE", "MOVE_SUNNY_DAY", "MOVE_HAIL", "MOVE_WEATHER_BALL"),
)
STYLE_REQUIRED = (3, 3, 3, 2, 2, 2, 2)                    # sRequiredMoveCounts, styles 1..7
MOVE_STYLE = np.zeros(len(_DATA["moves"]) + 1, np.int8)   # GetMoveBattleStyle: move id -> style 0..7
for _s in range(len(_STYLE_MOVES) - 1, -1, -1):           # (reverse: the first list wins for Snatch)
    for _nm in _STYLE_MOVES[_s]:
        MOVE_STYLE[_MOVE_ID[_nm]] = _s + 1
SPECIES_TYPES = np.array([list(s["types"])[:2] + list(s["types"])[:1] * (2 - len(s["types"]))
                          for s in SPECIES], np.int64)


def _hint_from_counts(tc, sp):
    """Vectorized hint from type counts [n, 18] and style points [n, 8] -> (type [n], style [n])."""
    tc = np.asarray(tc)
    n = tc.shape[0]
    mc0, mc1 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    ar = np.arange(n)
    for i in range(1, 18):
        c0, ci = tc[ar, mc0], tc[:, i]
        up = c0 < ci
        eq = ~up & (c0 == ci)
        mc0 = np.where(up, i, mc0)
        mc1 = np.where(eq, i, mc1)
    c0, c1 = tc[ar, mc0], tc[ar, mc1]
    t = np.where(c0 == 0, HINT_TYPE_NONE,
                 np.where((c0 > c1) | (mc0 == mc1), mc0, HINT_TYPE_NONE))
    sp = np.asarray(sp)
    ok = sp[:, 1:] >= np.array(STYLE_REQUIRED)[None]
    count = ok.sum(1)
    last = np.where(ok.any(1), 7 - np.argmax(ok[:, ::-1], 1), 0)
    style = np.where(count > 2, HINT_STYLE_MANY, last)
    return t.astype(np.int64), style.astype(np.int64)


def hint_counts(species, moves):
    """species [n, 3], moves [n, 3, 4] -> (type counts [n, 18], style points [n, 8]) as the game counts them."""
    species = np.asarray(species, np.int64)
    moves = np.asarray(moves, np.int64)
    n = species.shape[0]
    tc = np.zeros((n, 18), np.int64)
    ty = SPECIES_TYPES[species]                                    # [n, 3, 2]
    for j in range(species.shape[1]):
        np.add.at(tc, (np.arange(n), ty[:, j, 0]), 1)
        second = ty[:, j, 1] != ty[:, j, 0]
        np.add.at(tc, (np.arange(n)[second], ty[second, j, 1]), 1)
    st = MOVE_STYLE[np.clip(moves, 0, len(MOVE_STYLE) - 1)].reshape(n, -1)
    sp = np.zeros((n, 8), np.int64)
    for s in range(8):
        sp[:, s] = (st == s).sum(1)
    return tc, sp


def hint_of_arrays(species, moves):
    """Vectorized hint_of_team: species [n, 3], moves [n, 3, 4] -> (hint_type [n], hint_style [n])."""
    return _hint_from_counts(*hint_counts(species, moves))


def _as_species_moves(team):
    out_sp, out_mv = [], []
    for m in team:
        if isinstance(m, (int, np.integer)):                       # a Battle Frontier set id
            fm = FRONTIER_MONS[int(m)]
            out_sp.append(int(fm["species"]))
            out_mv.append([int(x) for x in fm["moves"]])
        elif isinstance(m, dict):
            out_sp.append(int(m["species"]))
            out_mv.append(list(m["moves"]))
        elif len(m) == 2:                                          # (species, moves)
            out_sp.append(int(m[0]))
            out_mv.append(list(m[1]))
        else:                                                      # a Gen3Game.determinize spec
            out_sp.append(int(m[1]))
            out_mv.append(list(m[2]))
    out_mv = [(list(mv) + [0] * 4)[:4] for mv in out_mv]
    return out_sp, out_mv


def hint_of_team(team):
    """(hint_type, hint_style) the attendant gives for an opponent team, exactly as the game computes it
    (battle_factory.c): team = 3 Battle Frontier set ids, or 3 (species, moves), or 3 determinize specs.
        type: count each species' types (a dual type counts both, a single type once); the most common one if it is
              strictly more common than the runner-up the game tracks, else 18 (none). The game's scan keeps
              mostCommonTypes[1] = the last type whose count equalled the leader's count at the time it was
              seen, so a type that later overtakes the leader can still "tie" with an earlier, smaller one: this
              function reproduces the scan literally.
        style: per style (FACTORY_STYLE_*), the moves of the 12 in its list (the first list a move is in);
              styles reaching their threshold (3 for preparation / slow & steady / endurance, 2 for the rest):
              none -> 0, one or two -> the last of them in style order, more than two -> 8 (no single style)."""
    sp, mv = _as_species_moves(team)
    t, s = hint_of_arrays(np.array([sp]), np.array([mv]))
    return int(t[0]), int(s[0])


def hint_matches(team, hint):
    return hint is None or hint_of_team(team) == tuple(int(x) for x in hint)


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
        import os
        from .search import in_training
        if not in_training() and os.environ.get("PYBATTLE_DIAGNOSTIC_PRIOR") != "factory_sets":
            raise PermissionError("the factory_sets opponent sampler is for training-time search only "
                                  "(evaluation and inference use the strict sampler)")


def determinization(view, knowledge=None, rng=None, open_level=True, prior="strict", hint=None):
    """Gen3Game.determinize specs drawn with the sampler `prior`: "strict" (sample_determinization) or
    "factory_sets" (sample_factory_sets; training only, needs the opponent's OpponentKnowledge). hint: the
    attendant's (hint_type, hint_style) about this opponent (None: not used, e.g. Noland's battle): only teams the
    game would describe with that hint are drawn (hint_of_team; rejection, see the samplers)."""
    if prior == "strict":
        return sample_determinization(view, None, rng, open_level=open_level, hint=hint)
    check_prior(prior)
    return sample_factory_sets(view, knowledge, rng, open_level=open_level, hint=hint)


def battle_hint(view, noland: bool):
    """The hint a player heard about the opponent of the battle in `view` (BattleView / SwapView / RentalView
    .hint_type / .hint_style), or None before Noland's battle (the attendant says nothing; NO_HINT is also a
    possible hint for a normal opponent, so it cannot mean "unknown" by itself)."""
    if noland:
        return None
    return int(view.hint_type), int(view.hint_style)


# rejection sampling on the hint: a whole team is drawn, its hint computed as the game does, and it is kept only if
# it equals the heard one; after HINT_TRIES failed draws the last draw is used (counted in hint_stats["fallback"]).
# Exact: the accepted teams follow the sampler's own distribution conditioned on the hint.
HINT_TRIES_STRICT = 4096
HINT_TRIES_FACTORY = 4096
hint_stats = {"calls": 0, "draws": 0, "fallback": 0}


def _batches(max_tries):
    """Batch sizes of the vectorized rejection draws: 64, 256, then 1024 at a time, max_tries in total."""
    left, size = max(1, int(max_tries)), 64
    while left > 0:
        yield min(size, left)
        left -= size
        size = min(size * 4, 1024)


def sample_determinization(view, ctx=None, rng=None, open_level=True, hint=None,
                            max_tries=HINT_TRIES_STRICT):
    """Specs for Gen3Game.determinize drawn from player knowledge (see the module docstring):
    [(party_slot, species | KEEP, [4 moves], item, [6 IVs], [6 EVs], nature, ability_bit, hp_fraction)].
    `ctx` is accepted for compatibility and not used (nothing of the streak context is needed).
    hint: the heard (hint_type, hint_style) of this opponent, or None. With a hint, the species and moves of all 3
    slots (the fainted seen ones too: their unrevealed moves count for the style) are redrawn until the team's hint
    (hint_of_team) is the heard one (at most max_tries draws, then the last one: hint_stats["fallback"]); items,
    IVs, EVs, natures, abilities and HP are drawn afterwards (they do not enter the hint)."""
    rng = rng or random.Random()
    if hint is not None:
        return _strict_with_hint(view, rng, open_level, tuple(int(x) for x in hint), max_tries)
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


def _strict_species_moves(enemy, level, open_level, rng):
    """Species and 4 moves of every slot by the strict rules (seen: its species and revealed moves)."""
    used_species = {m.species for m in enemy if m.seen}
    sp, mv = [], []
    for m in enemy:
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
        sp.append(species)
        mv.append(moves + [0] * (4 - len(moves)))
    return sp, mv


_LEARN_PAD = {}


def _learn_table(level):
    """Learnable moves of every species as a padded matrix [n_species, max_len] and the lengths (cached)."""
    t = _LEARN_PAD.get(level)
    if t is None:
        lists = [learnable(sp, level) if sp else () for sp in range(len(SPECIES))]
        width = max(len(x) for x in lists)
        mat = np.zeros((len(lists), width), np.int64)
        for sp, x in enumerate(lists):
            mat[sp, :len(x)] = x
        t = _LEARN_PAD[level] = (mat, np.array([len(x) for x in lists], np.int64))
    return t


def _strict_draw_batch(enemy, level, open_level, g, M):
    """M independent draws of _strict_species_moves (vectorized): species uniform over the Frontier species not
    used by another slot (seen slots: their species), then the revealed moves plus a uniform subset of the other
    learnable ones. -> species [M, 3], moves [M, 3, 4]."""
    front = np.array(_FRONTIER_SPECIES[bool(open_level)], np.int64)
    seen_sp = [m.species for m in enemy if m.seen]
    avail = front[~np.isin(front, np.array(seen_sp, np.int64))]
    mat, lens = _learn_table(level)
    sp = np.zeros((M, 3), np.int64)
    mv = np.zeros((M, 3, 4), np.int64)
    drawn_unseen = []
    for i, m in enumerate(enemy):
        if m.seen:
            sp[:, i] = m.species
            continue
        x = avail[g.integers(len(avail), size=M)]
        for _ in range(64):
            bad = np.zeros(M, bool)
            for j in drawn_unseen:
                bad |= x == sp[:, j]
            if not bad.any():
                break
            x = np.where(bad, avail[g.integers(len(avail), size=M)], x)
        sp[:, i] = x
        drawn_unseen.append(i)
    for i, m in enumerate(enemy):
        revealed = [x for x in (m.revealed_moves if m.seen else []) if x][:4]
        cand = mat[sp[:, i]]                                        # [M, width]
        valid = np.arange(mat.shape[1])[None] < lens[sp[:, i]][:, None]
        if revealed:
            valid &= ~np.isin(cand, np.array(revealed, np.int64))
        keys = np.where(valid, g.random(cand.shape), np.inf)
        n_more = 4 - len(revealed)
        pick = np.argsort(keys, 1)[:, :n_more]
        chosen = np.take_along_axis(cand, pick, 1)
        ok = np.take_along_axis(valid, pick, 1)
        mv[:, i, :len(revealed)] = revealed
        mv[:, i, len(revealed):] = np.where(ok, chosen, 0)
    return sp, mv


def _strict_with_hint(view, rng, open_level, hint, max_tries):
    level = team_level(view)
    enemy = view.enemy_party[:3]
    hint_stats["calls"] += 1
    if len(enemy) < 3:
        sp, mv = _strict_species_moves(enemy, level, open_level, rng)
    else:
        g = np.random.default_rng(rng.getrandbits(64))
        found = False
        for M in _batches(max_tries):
            S_, V_ = _strict_draw_batch(enemy, level, open_level, g, M)
            ht, hs = hint_of_arrays(S_, V_)
            ok = np.nonzero((ht == hint[0]) & (hs == hint[1]))[0]
            j = int(ok[0]) if len(ok) else M - 1
            hint_stats["draws"] += (j + 1) if len(ok) else M
            sp = [int(x) for x in S_[j]]
            mv = [[int(y) for y in row] for row in V_[j]]
            if len(ok):
                found = True
                break
        if not found:
            hint_stats["fallback"] += 1
    used_items = {m.revealed_item for m in enemy if m.seen and m.revealed_item}
    specs = []
    for i, m in enumerate(enemy):
        if m.seen and m.fainted:
            specs.append((i, KEEP, [0, 0, 0, 0], 0, [0] * 6, [0] * 6, 0, 0, -1.0))
            continue
        species, moves = sp[i], mv[i]
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


def sample_factory_sets(view, knowledge, rng=None, open_level=True, hint=None,
                        max_tries=HINT_TRIES_FACTORY):
    """Specs for Gen3Game.determinize (sample_determinization's layout) with the opponent's Pokemon drawn from the
    Factory's set list (training-time search only; see the module docstring). knowledge: the opponent's
    pybattle.backend.OpponentKnowledge (the round, Noland, the exclusions of its generation).
    hint: the heard (hint_type, hint_style) of this opponent, or None (ignored for Noland: no hint is given). With
    a hint, whole teams of set ids (the fainted seen Pokemon too: their sets count for the hint) are drawn the way
    the game draws them and kept only if hint_of_team equals the heard hint (_draw_ids_with_hint: vectorized
    batches, at most max_tries teams, then the last draw is used and counted in hint_stats["fallback"])."""
    if knowledge is None:
        raise ValueError("factory_sets needs the opponent's OpponentKnowledge (SimBackend.opponent_knowledge)")
    rng = rng or random.Random()
    if hint is not None and not knowledge.noland:
        drawn = _draw_ids_with_hint(view, knowledge, rng, open_level, tuple(int(x) for x in hint), max_tries)
        if drawn is not None:
            return _factory_specs(view, knowledge, drawn, rng, team_level(view))
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
                drawn[i] = (m.species, moves, item, None)
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
        drawn[i] = (species, list(SET_MOVES_AS_BUILT[k]), item, k)
    return _factory_specs(view, knowledge, drawn, rng, level)


def _factory_specs(view, knowledge, drawn, rng, level):
    """Specs of the drawn sets (drawn[slot] = (species, moves, item, set id | None) or None: fainted, kept)."""
    enemy = view.enemy_party[:3]
    # the drawn sets are built as the game builds them: the set's EVs and nature, the game's IV for this battle
    iv = fixed_iv(knowledge.challenge + 2, False) if knowledge.noland else fixed_iv(0, knowledge.battle == 6)
    specs = []
    for i, m in enumerate(enemy):
        d = drawn[i]
        if d is None:
            specs.append((i, KEEP, [0, 0, 0, 0], 0, [0] * 6, [0] * 6, 0, 0, -1.0))
            continue
        species, moves, item, k = d
        if k is None:                           # strict fallback (no set explains it): nothing to take from a set
            ivs, evs, nature = [rng.randrange(32) for _ in range(6)], _evs(rng), rng.randrange(NUM_NATURES)
        else:
            _, _, _, _, ivs, evs, nature, _, _ = set_spec(i, k, iv)
        bit = _ability_bit(species, m if m.seen else None, rng)
        hp = _hp_fraction(max_hp_of(species, ivs[0], evs[0], level), m.hp_pixels, rng) if m.seen else 1.0
        specs.append((i, species, moves, item, ivs, evs, nature, bit, hp))
    return specs


FRONTIER_MOVES_RAW = np.array([(list(m["moves"]) + [0] * 4)[:4] for m in FRONTIER_MONS], np.int64)


def _draw_ids_with_hint(view, knowledge, rng, open_level, hint, max_tries):
    """Set ids for the 3 slots drawn as sample_factory_sets draws them, in vectorized batches, until a team's hint
    is `hint`. Per slot (seen ones first) a candidate array (seen: sets of its species containing its revealed moves
    and item; unseen: the pool); a slot's draw is uniform over its candidates that do not repeat a species or held
    item of the slots drawn before (redrawn until it does not: the same distribution as the sequential draw).
    -> drawn dict for _factory_specs (fainted seen slots None), or None when a seen Pokemon has no consistent set
    (the caller then uses the sequential sampler with its strict fallback)."""
    enemy = view.enemy_party[:3]
    if len(enemy) < 3:
        return None
    ids = factory_pool(knowledge, open_level)
    seen_species = np.array(sorted({m.species for m in enemy if m.seen}), np.int64)
    ri = np.array(sorted({m.revealed_item for m in enemy if m.seen and m.revealed_item}), np.int64)
    order = sorted(range(3), key=lambda j: not enemy[j].seen)
    cands = []
    for i in order:
        m = enemy[i]
        if m.seen:
            revealed = frozenset(x for x in m.revealed_moves if x)
            same = ids[SET_SPECIES[ids] == m.species]
            if m.revealed_item:
                same = same[SET_ITEM[same] == m.revealed_item]
            else:
                same = same[(SET_ITEM[same] == 0) | ~np.isin(SET_ITEM[same], ri)]
            c = np.array([int(k) for k in same if revealed <= _SET_MOVESET[k]], np.int64)
            if len(c) == 0:
                return None
        else:
            c = ids[~np.isin(SET_SPECIES[ids], seen_species)]
            c = c[(SET_ITEM[c] == 0) | ~np.isin(SET_ITEM[c], ri)]
        cands.append(c)
    g = np.random.default_rng(rng.getrandbits(64))
    hint_stats["calls"] += 1
    last = None
    found = False
    for M in _batches(max_tries):
        out, valid = _vector_draw(cands, M, g)
        teams = out[valid]
        if len(teams) == 0:
            continue
        slot_ids = np.zeros_like(teams)
        for s, i in enumerate(order):
            slot_ids[:, i] = teams[:, s]
        ht, hs = hint_of_arrays(SET_SPECIES[slot_ids], FRONTIER_MOVES_RAW[slot_ids])
        ok = np.nonzero((ht == hint[0]) & (hs == hint[1]))[0]
        last = slot_ids[ok[0]] if len(ok) else slot_ids[-1]
        hint_stats["draws"] += int(np.nonzero(valid)[0][ok[0]]) + 1 if len(ok) else M
        if len(ok):
            found = True
            break
    if not found:
        hint_stats["fallback"] += 1
    if last is None:
        return None
    drawn = {}
    for i, m in enumerate(enemy):
        if m.seen and m.fainted:
            drawn[i] = None
            continue
        k = int(last[i])
        drawn[i] = (int(SET_SPECIES[k]), list(SET_MOVES_AS_BUILT[k]), int(SET_ITEM[k]), k)
    return drawn


def _vector_draw(cands, M, g):
    """M teams drawn slot by slot: slot s uniform over cands[s] minus the species and held items (non-zero) of
    slots 0..s-1 of the same team (redrawn until it clashes with none: the game's sequential rejection).
    -> (set ids [M, len(cands)], valid [M]: False for a team whose slot could not be completed in 64 redraws)."""
    out = np.zeros((M, len(cands)), np.int64)
    valid = np.ones(M, bool)
    for s, c in enumerate(cands):
        x = c[g.integers(len(c), size=M)]
        for _ in range(64):
            bad = np.zeros(M, bool)
            for q in range(s):
                prev = out[:, q]
                bad |= SET_SPECIES[x] == SET_SPECIES[prev]
                bad |= (SET_ITEM[x] != 0) & (SET_ITEM[x] == SET_ITEM[prev])
            bad &= valid
            if not bad.any():
                break
            x = np.where(bad, c[g.integers(len(c), size=M)], x)
        else:
            valid &= ~bad                         # (a slot whose only candidates clash: that team is dropped)
        out[:, s] = x
    return out, valid


def sample_pool_teams(ids, n, g, excluded_species=(), hint=None, max_tries=HINT_TRIES_FACTORY):
    """n opponent teams (set ids [n, 3]) drawn from the set ids `ids` as the game draws a team (uniform set ids, no
    repeated species or held item), none of `excluded_species`, each (independently) with the attendant's hint
    `hint` if given (rejection; a team still unmatched after max_tries draws is the last draw: counted in
    hint_stats["fallback"]). g: numpy Generator. Used by the tactician's planner (rl.team_eval.plan_values)."""
    ids = np.asarray(ids, np.int64)
    if len(excluded_species):
        ids = ids[~np.isin(SET_SPECIES[ids], np.array(sorted(set(int(x) for x in excluded_species)), np.int64))]
    cands = [ids, ids, ids]
    if hint is None:
        out = np.zeros((0, 3), np.int64)
        while len(out) < n:
            t, v = _vector_draw(cands, max(n - len(out), 1), g)
            out = np.concatenate([out, t[v]])
        return out[:n]
    hint = (int(hint[0]), int(hint[1]))
    got, last = [], None
    tried = 0
    hint_stats["calls"] += 1
    M = max(64, 4 * n)
    while len(got) < n and tried < max_tries * n:
        t, v = _vector_draw(cands, M, g)
        t = t[v]
        tried += M
        hint_stats["draws"] += M
        if not len(t):
            continue
        last = t
        ht, hs = hint_of_arrays(SET_SPECIES[t], FRONTIER_MOVES_RAW[t])
        got.extend(t[(ht == hint[0]) & (hs == hint[1])][:n - len(got)])
        M = min(M * 4, 8192)
    if len(got) < n:
        hint_stats["fallback"] += n - len(got)
        got.extend(last[:n - len(got)])
    return np.array(got, np.int64).reshape(n, 3)
