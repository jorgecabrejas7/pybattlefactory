"""What the player can see, built identically for the simulator and the emulator.

Everything here is read from game RAM by GBA address through a `RamReader` (Gen3Game.read or
the emulator's read), so both backends produce the same views from the same game state. The
only other input is `BattleObserver`'s memory of earlier decisions in the same battle (what has
been revealed, when a status began) and the opponent hints heard before the battle.

The rule: a view holds exactly what a human player could know from the screen and the battle
messages, with perfect memory of the battle so far and knowledge of the game (species data, move
data, fixed durations). Never: the opponent's exact HP, stats, unrevealed moves / item / ability,
PP, the remaining turns of random-length effects, the RNG, or what the AI thinks.

Hidden RAM (the opponent's exact HP, ability, speed, sleep counter...) is read *internally* to
work out what the player just watched happen -- who moved first, which ability was announced,
how far the HP bar dropped -- but only those visible outcomes end up in a view. Each field's
docstring says why it is legal.

Timing conventions (see TurnEvents): a BATTLE decision describes the turn that has just ended;
a FORCED_SWITCH decision describes the turn in progress (after a faint the whole turn, including
end-of-turn effects, is over; after our Baton Pass it is not: `in_progress` is True).
"""

import copy
import functools
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from .emu.decode import SYMBOLS as S, BattleMon, PartyMon, decode_battle_mon, decode_party

_DATA = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "game_data.json")))
SPECIES = _DATA["species"]
MOVES = _DATA["moves"]
ITEMS = _DATA["items"]
NAMES = _DATA["names"]
_TYPE_CHART = _DATA["type_effectiveness"]          # [attacking type, defending type, x10]


def _ids(table: str, prefix: str) -> Dict[str, int]:
    return {v[len(prefix):]: int(k) for k, v in NAMES[table].items() if v.startswith(prefix)}


MOVE = _ids("moves", "MOVE_")
EFFECT = _ids("effects", "EFFECT_")
ABILITY = _ids("abilities", "ABILITY_")
HOLD = _ids("hold_effects", "HOLD_EFFECT_")
TYPE = _ids("types", "TYPE_")

HP_BAR_PIXELS = 48          # B_HEALTHBAR_PIXELS
UNSEEN_HP = -1              # SeenMon.hp_pixels of an opponent never sent out
NO_HINT_TYPE = 18           # RentalView / BattleView hint_type when there is no hint
MOVE_UNAVAILABLE = 0xFFFF

# status1
STATUS1_SLEEP = 0x7
STATUS1_POISON = 1 << 3
STATUS1_BURN = 1 << 4
STATUS1_FREEZE = 1 << 5
STATUS1_PARALYSIS = 1 << 6
STATUS1_TOXIC = 1 << 7
STATUS1_TOXIC_COUNTER = 0xF << 8
STATUS1_ANY = STATUS1_SLEEP | STATUS1_POISON | STATUS1_BURN | STATUS1_FREEZE | STATUS1_PARALYSIS | STATUS1_TOXIC
# status2
STATUS2_CONFUSION = 0x7
STATUS2_UPROAR = 0x7 << 4
STATUS2_BIDE = 0x3 << 8
STATUS2_LOCK_CONFUSE = 0x3 << 10
STATUS2_MULTIPLETURNS = 1 << 12
STATUS2_WRAPPED = 0x7 << 13
STATUS2_INFATUATION = 0xF << 16
STATUS2_FOCUS_ENERGY = 1 << 20
STATUS2_TRANSFORMED = 1 << 21
STATUS2_RECHARGE = 1 << 22
STATUS2_RAGE = 1 << 23
STATUS2_SUBSTITUTE = 1 << 24
STATUS2_DESTINY_BOND = 1 << 25
STATUS2_ESCAPE_PREVENTION = 1 << 26
STATUS2_NIGHTMARE = 1 << 27
STATUS2_CURSED = 1 << 28
STATUS2_FORESIGHT = 1 << 29
STATUS2_DEFENSE_CURL = 1 << 30
STATUS2_TORMENT = 1 << 31
# status3
STATUS3_LEECHSEED_BATTLER = 0x3
STATUS3_LEECHSEED = 1 << 2
STATUS3_ALWAYS_HITS = 0x3 << 3
STATUS3_PERISH_SONG = 1 << 5
STATUS3_ON_AIR = 1 << 6
STATUS3_UNDERGROUND = 1 << 7
STATUS3_MINIMIZED = 1 << 8
STATUS3_CHARGED_UP = 1 << 9
STATUS3_ROOTED = 1 << 10
STATUS3_YAWN = 0x3 << 11
STATUS3_IMPRISONED_OTHERS = 1 << 13
STATUS3_GRUDGE = 1 << 14
STATUS3_MUDSPORT = 1 << 16
STATUS3_WATERSPORT = 1 << 17
STATUS3_UNDERWATER = 1 << 18
STATUS3_INTIMIDATE_POKES = 1 << 19
STATUS3_TRACE = 1 << 20
STATUS3_SEMI_INVULNERABLE = STATUS3_ON_AIR | STATUS3_UNDERGROUND | STATUS3_UNDERWATER
# side status
SIDE_STATUS_REFLECT = 1 << 0
SIDE_STATUS_LIGHTSCREEN = 1 << 1
SIDE_STATUS_SPIKES = 1 << 4
SIDE_STATUS_SAFEGUARD = 1 << 5
SIDE_STATUS_MIST = 1 << 8
# gBattleWeather
B_WEATHER_RAIN = 0x7
B_WEATHER_RAIN_PERMANENT = 1 << 2
B_WEATHER_SANDSTORM = 0x3 << 3
B_WEATHER_SANDSTORM_PERMANENT = 1 << 4
B_WEATHER_SUN = 0x3 << 5
B_WEATHER_SUN_PERMANENT = 1 << 6
B_WEATHER_HAIL = 1 << 7
FLAG_MAKES_CONTACT = 1 << 0

MAJOR_STATUSES = ("none", "sleep", "poison", "burn", "freeze", "paralysis", "toxic")
WEATHERS = ("none", "rain", "sun", "sandstorm", "hail")
WEATHER_NONE, WEATHER_RAIN, WEATHER_SUN, WEATHER_SANDSTORM, WEATHER_HAIL = range(5)
# TurnEvents.own_action / enemy_action
ACTIONS = ("none", "move", "switch", "cant_move")
ACTION_NONE, ACTION_MOVE, ACTION_SWITCH, ACTION_CANT_MOVE = range(4)
# ActiveState.semi_invulnerable
SEMI_INVULNERABLE = ("none", "in_air", "underground", "underwater")


class RamReader(Protocol):
    def read(self, addr: int, size: int) -> bytes: ...


def major_status(status1: int) -> int:
    """Index into MAJOR_STATUSES."""
    if status1 & STATUS1_SLEEP:
        return 1
    if status1 & STATUS1_TOXIC:
        return 6
    for i, flag in ((2, STATUS1_POISON), (3, STATUS1_BURN), (4, STATUS1_FREEZE), (5, STATUS1_PARALYSIS)):
        if status1 & flag:
            return i
    return 0


def hp_bar_pixels(hp: int, max_hp: int) -> int:
    """GetScaledHPFraction(hp, maxHP, 48): what the enemy HP bar shows."""
    if max_hp <= 0:
        return 0
    v = hp * HP_BAR_PIXELS // max_hp
    return 1 if v == 0 and hp > 0 else v


def ability_of(species: int, ability_num: int) -> int:
    a = SPECIES[species]["abilities"]
    return a[1] if ability_num and a[1] else a[0]


def possible_abilities(species: int) -> List[int]:
    """The species' abilities (game knowledge): one or two candidates."""
    a = SPECIES[species]["abilities"]
    return [a[0]] + ([a[1]] if a[1] and a[1] != a[0] else [])


def hold_effect(item: int) -> Tuple[int, int]:
    return (ITEMS[item]["hold_effect"], ITEMS[item]["hold_effect_param"]) if 0 < item < len(ITEMS) else (0, 0)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@dataclass
class OwnMon:
    """A Pokemon the player controls: everything the summary screen shows."""
    species: int
    level: int
    hp: int
    max_hp: int
    stats: List[int]            # Atk Def Spe SpA SpD
    moves: List[int]
    pp: List[int]
    item: int
    ability: int
    nature: int
    ivs: List[int]
    evs: List[int]
    status: int                 # MAJOR_STATUSES index
    types: List[int]
    sleep_turns: int = 0        # asleep: times it has tried to move while asleep so far (never "turns left")

    @classmethod
    def from_party(cls, m: PartyMon) -> "OwnMon":
        return cls(m.species, m.level, m.hp, m.max_hp, list(m.stats), list(m.moves), list(m.pp), m.held_item,
                   ability_of(m.species, m.ability_num), m.nature, list(m.ivs), list(m.evs), major_status(m.status),
                   list(SPECIES[m.species]["types"]))


@dataclass
class SeenMon:
    """An opponent Pokemon as the player sees it.

    A slot never sent out (`seen` False) reveals nothing: species 0, level 0, hp_pixels -1
    (UNSEEN_HP), status 0, empty lists. That there are 3 is visible (the party balls).
    """
    species: int = 0
    level: int = 0
    hp_pixels: int = UNSEEN_HP  # 0..48: the HP bar (the bar of a benched Pokemon does not move)
    status: int = 0             # MAJOR_STATUSES index, as last seen on the field
    revealed_moves: List[int] = field(default_factory=list)     # moves seen being used
    revealed_item: Optional[int] = None     # item named in a message (Leftovers) or consumed / knocked off
    fainted: bool = False
    seen: bool = False          # has been sent out at least once
    types: List[int] = field(default_factory=list)              # species types (game knowledge)
    base_stats: List[int] = field(default_factory=list)         # species base HP Atk Def Spe SpA SpD
    possible_abilities: List[int] = field(default_factory=list)  # the species' 1-2 abilities
    revealed_ability: int = 0   # ability announced in battle (0 = not yet; see BattleObserver)
    sleep_turns: int = 0        # asleep: times seen trying to move while asleep (never "turns left")


def _species_knowledge(m: SeenMon) -> SeenMon:
    sp = SPECIES[m.species]
    m.types, m.base_stats, m.possible_abilities = list(sp["types"]), list(sp["base"]), possible_abilities(m.species)
    return m


@dataclass
class ActiveState:
    """Field state of one active battler (visible for both sides unless noted).

    Counters follow what the player can know: effects of fixed length give the turns left
    (`*_turns_left`); effects of random length give only the turns elapsed (`*_turns`), counted
    by the observer. "Turns elapsed" of end-of-turn effects counts completed turns since the
    effect started (1 at the first decision after the turn it began).
    """
    party_index: int
    stat_stages: List[int]      # Atk Def Spe SpA SpD Acc Eva, -6..+6 (announced)
    types: List[int]            # current types (Color Change / Transform are announced)
    confused: bool
    infatuated: bool
    substitute: bool
    leech_seeded: bool
    cursed: bool
    nightmare: bool
    trapped: bool               # Mean Look / Spider Block / Block
    focus_energy: bool
    transformed: bool
    perish_song: bool
    rooted: bool
    yawn: bool
    torment: bool
    taunted: bool
    encored_move: int
    disabled_move: int
    protect_uses: int           # consecutive successful Protect/Detect/Endure
    # --- counters ---------------------------------------------------------------------
    confusion_turns: int = 0    # random 2-5: times it tried to move while confused so far
    toxic_counter: int = 0      # badly poisoned: next Toxic damage is (counter+1)/16 (deterministic)
    perish_count: int = -1      # the Perish count shown at the end of this turn (0 = faints then); -1 none
    taunt_turns_left: int = 0   # fixed (2 end-of-turns)
    encore_turns: int = 0       # random 3-6: elapsed
    disable_turns: int = 0      # random 2-5: elapsed
    yawn_turns_left: int = 0    # fixed: falls asleep at the end of the turn this reaches 0 from 1
    wrapped_turns: int = 0      # Wrap/Bind/Fire Spin/Clamp/Whirlpool/Sand Tomb on it, random 3-6: elapsed
    uproar_turns: int = 0       # random 2-5: elapsed
    rampage_turns: int = 0      # Outrage/Thrash/Petal Dance lock, random 2-3: elapsed
    locked_move: int = 0        # move it is locked into (rampage, Uproar, Rollout, Bide, charging, recharge)
    charging_move: int = 0      # first turn of a two-turn move done (Solar Beam, Fly, Dig, Dive, Bounce, ...)
    semi_invulnerable: int = 0  # SEMI_INVULNERABLE index (Fly/Bounce, Dig, Dive)
    must_recharge: bool = False  # Hyper Beam & co: loses its next turn
    bide_turns_left: int = 0    # fixed (2)
    rollout_hits_left: int = 0  # Rollout / Ice Ball: fixed 5-hit sequence
    fury_cutter_count: int = 0  # consecutive Fury Cutter hits
    stockpile: int = 0          # announced ("stockpiled 2")
    charge_turns_left: int = 0  # Charge: next Electric move doubled (fixed)
    locked_on_turns_left: int = 0  # the opponent's Lock-On / Mind Reader: its next move surely hits this battler
    destiny_bond: bool = False
    substitute_hp: int = -1     # own Substitute's HP; -1 for the opponent's (not shown) or no substitute
    defense_curl: bool = False
    foresight: bool = False     # identified by Foresight / Odor Sleuth
    minimized: bool = False
    charged_up: bool = False
    imprisoning: bool = False
    grudge: bool = False
    mud_sport: bool = False
    water_sport: bool = False
    rage: bool = False
    first_turn: bool = False    # came in during the previous turn (Fake Out works)


@dataclass
class SideState:
    """One side of the field. All durations here are fixed by the game, so "turns left" is legal."""
    reflect_turns: int = 0          # 5 on use; counts end-of-turns left
    light_screen_turns: int = 0
    safeguard_turns: int = 0
    mist_turns: int = 0
    spikes: int = 0                 # layers 0-3
    future_sight_turns: int = 0     # Future Sight / Doom Desire lands on this side after this many end-of-turns
    future_sight_move: int = 0
    wish_turns: int = 0             # Wish heals this side's active Pokemon after this many end-of-turns


@dataclass
class TurnEvents:
    """What happened in the most recent turn, as the player watched it.

    At a BATTLE decision: the turn that just ended. At a FORCED_SWITCH decision: the turn in
    progress (after a faint it is over; after our Baton Pass `in_progress` is True). On the first
    decision of a battle `turn` is -1 and everything is empty. When the game skipped our
    decisions (locked into Outrage, recharging...), `turns` > 1 and the events cover all of them
    (moves: the last ones; damage: the total).

    Actions (ACTIONS): none (did nothing: fainted first, dragged out by Roar), move, switch
    (voluntary), cant_move (tried but could not: asleep, frozen, fully paralyzed, flinched,
    recharging, hurt itself in confusion, infatuated, loafing...). A move is only named when it
    was actually used ("X used Y!"); a Pokemon that could not move does not reveal its choice.
    """
    turn: int = -1              # battle turn described (-1: none yet)
    turns: int = 0              # game turns covered (1 normally)
    in_progress: bool = False
    first: int = -1             # side that acted first: 0 us, 1 opponent, -1 unknown / nobody acted
    own_action: int = ACTION_NONE
    own_move: int = 0
    own_crit: int = -1          # our move was a critical hit: 1 yes, 0 no, -1 unknown (see GUIDE)
    enemy_action: int = ACTION_NONE
    enemy_move: int = 0
    enemy_crit: int = -1
    damage_dealt_pixels: int = 0  # the opponent's HP bar drop from direct hits this turn (pixels of 48)
    damage_taken: int = 0         # HP our Pokemon lost to direct hits this turn (exact: it's ours)
    own_fainted: bool = False
    enemy_fainted: bool = False


@dataclass
class BattleView:
    own_party: List[OwnMon]
    own_active: ActiveState
    enemy_party: List[SeenMon]      # by enemy party slot; unseen members reveal nothing (seen=False)
    enemy_active: ActiveState
    weather: int                    # WEATHERS index
    weather_permanent: bool         # from an ability (Drizzle/Drought/Sand Stream): lasts until replaced
    weather_turns_left: int         # from a move: end-of-turns left (fixed 5); 0 when permanent or none
    own_side: SideState
    enemy_side: SideState
    turn: int                       # gBattleResults.battleTurnCounter (0 on the first turn, caps at 255)
    forced_switch: bool
    usable_moves: List[bool]        # the game accepts these move slots
    switch_targets: List[int]       # party indices that can be sent out ([] while trapped)
    last_turn: TurnEvents = field(default_factory=TurnEvents)
    hint_type: int = NO_HINT_TYPE   # what the attendant said about this opponent before the battle
    hint_style: int = 0             # FACTORY_STYLE_*
    must_struggle: bool = False     # every move is blocked (no PP / Disable / Taunt...): the game uses Struggle
    legal_actions: List[tuple] = field(default_factory=list)   # every action the backend accepts now

    # compatibility with the first version of the view
    @property
    def own_spikes(self) -> int:
        return self.own_side.spikes

    @property
    def enemy_spikes(self) -> int:
        return self.enemy_side.spikes


@dataclass
class RentalView:
    candidates: List[OwnMon]        # the 6 rentals, fully visible via Summary
    frontier_ids: List[int]         # gBattleFrontierMons indices
    hint_type: int                  # most common type of the first opponent (18 = none)
    hint_style: int                 # FACTORY_STYLE_*


@dataclass
class SwapView:
    own_party: List[OwnMon]
    enemy_party: List[SeenMon]      # the defeated team: species (+ what was revealed in battle)
    hint_type: int                  # next opponent
    hint_style: int


@dataclass
class RunInfo:
    win_streak: int
    battle_in_challenge: int        # 0..6
    challenge_num: int
    open_level: bool
    wins: int
    rents: int = 0                  # the Factory's rental counter (shown in the records; better rentals later)
    noland: bool = False            # the current / next battle is against the Factory Head (Noland)


# ---------------------------------------------------------------------------
# RAM snapshot at a decision
# ---------------------------------------------------------------------------

class _Snap:
    """The battle RAM the observer uses, read once per decision (hidden values included:
    they are only used to work out visible events)."""

    def __init__(self, ram: RamReader, forced: bool):
        r = lambda sym, n: ram.read(S.addr(sym), n)
        raw = r("gBattleMons", 0x58 * 2)
        self.mons: List[BattleMon] = [decode_battle_mon(raw[i * 0x58:(i + 1) * 0x58]) for i in range(2)]
        idx = r("gBattlerPartyIndexes", 4)
        self.idx = struct.unpack("<2H", idx)
        self.parties = [decode_party(r("gPlayerParty", 300)), decode_party(r("gEnemyParty", 300))]
        dis = r("gDisableStructs", 0x1C * 2)
        self.dis = [dis[:0x1C], dis[0x1C:]]
        self.status3 = struct.unpack("<2I", r("gStatuses3", 8))
        self.sides = struct.unpack("<2H", r("gSideStatuses", 4))
        self.timers = r("gSideTimers", 24)
        self.weather = struct.unpack("<H", r("gBattleWeather", 2))[0]
        self.wfk = r("gWishFutureKnock", 44)
        last = r("gLastMoves", 4)
        self.last_moves = struct.unpack("<2H", last)
        self.last_printed = struct.unpack("<2H", r("gLastPrintedMoves", 4))
        self.bide = struct.unpack("<2i", r("gBideDmg", 8))
        self.crit = r("gCritMultiplier", 1)[0]
        self.locked = struct.unpack("<2H", r("gLockedMoves", 4))
        res = r("gBattleResults", 0x26)
        self.turn = res[0x13]
        self.last_used = struct.unpack_from("<2H", res, 0x22)      # lastUsedMovePlayer / Opponent
        self.player_switches = res[2]                               # playerSwitchesCounter
        self.rnd_turn = struct.unpack("<H", r("gRandomTurnNumber", 2))[0]
        self.order = r("gBattlerByTurnOrder", 2)
        self.forced = forced
        self.key = (forced, raw, idx, self.turn, last, self.rnd_turn)

    # disable-struct fields
    def disabled_move(self, b): return struct.unpack_from("<H", self.dis[b], 4)[0]
    def encored_move(self, b): return struct.unpack_from("<H", self.dis[b], 6)[0]
    def protect_uses(self, b): return self.dis[b][8]
    def stockpile(self, b): return self.dis[b][9]
    def substitute_hp(self, b): return self.dis[b][0x0A]
    def disable_timer(self, b): return self.dis[b][0x0B] & 0xF
    def encore_timer(self, b): return self.dis[b][0x0E] & 0xF
    def perish_timer(self, b): return self.dis[b][0x0F] & 0xF
    def fury_cutter(self, b): return self.dis[b][0x10]
    def rollout_timer(self, b): return self.dis[b][0x11] & 0xF
    def charge_timer(self, b): return self.dis[b][0x12] & 0xF
    def taunt_timer(self, b): return self.dis[b][0x13] & 0xF
    def sure_hit_by(self, b): return self.dis[b][0x15]
    def is_first_turn(self, b): return self.dis[b][0x16]

    def party_mon(self, side, i=None) -> PartyMon:
        return self.parties[side][self.idx[side] if i is None else i]

    def spikes(self, side): return self.timers[12 * side + 10]

    def weather_has_effect(self) -> bool:
        return not any(m.ability in (ABILITY["CLOUD_NINE"], ABILITY["AIR_LOCK"]) for m in self.mons)


def _shift(new, start, now):
    """`new` (a rewritten Pokemon) as it was at the turn start: HP and PP moved back by what the replaced
    Pokemon lost between `start` and `now` (HP scaled to the new max HP; PP matched by move)."""
    out = copy.copy(new)
    lost = start.hp - now.hp
    if lost and now.max_hp:
        out.hp = max(0, min(new.max_hp, new.hp + round(lost * new.max_hp / now.max_hp)))
    pp = list(new.pp)
    for k, mv in enumerate(new.moves):
        if mv and mv in now.moves:
            j = now.moves.index(mv)
            if start.moves[j] == mv:
                pp[k] = max(0, pp[k] + start.pp[j] - now.pp[j])
    out.pp = pp
    return out


_FAINTED = -1          # provisional action: fainted during the turn and replaced


def _priority(move: int) -> int:
    return MOVES[move]["priority"] if 0 < move < len(MOVES) else 0


def _effect(move: int) -> int:
    return MOVES[move]["effect"] if 0 < move < len(MOVES) else 0


@functools.lru_cache(maxsize=None)
def _effect_set(effects: Tuple[str, ...]) -> frozenset:
    return frozenset(EFFECT[e] for e in effects)


def _is(move: int, *effects: str) -> bool:
    return 0 < move < len(MOVES) and MOVES[move]["effect"] in _effect_set(effects)


def _move_type(move: int, mon: BattleMon, weather: int) -> int:
    """Type of `move` used by `mon` (Hidden Power and Weather Ball are dynamic)."""
    if _is(move, "HIDDEN_POWER"):
        iv = mon.ivs
        bits = (iv[0] & 1) | (iv[1] & 1) << 1 | (iv[2] & 1) << 2 | (iv[3] & 1) << 3 | (iv[4] & 1) << 4 | (iv[5] & 1) << 5
        t = 15 * bits // 63 + 1
        return t + 1 if t >= TYPE["MYSTERY"] else t
    if _is(move, "WEATHER_BALL") and weather:
        for flag, t in ((B_WEATHER_RAIN, "WATER"), (B_WEATHER_SUN, "FIRE"), (B_WEATHER_SANDSTORM, "ROCK"),
                        (B_WEATHER_HAIL, "ICE")):
            if weather & flag:
                return TYPE[t]
    return MOVES[move]["type"]


def _type_immune(move_type: int, types: List[int]) -> bool:
    return any(a == move_type and d in types and m == 0 for a, d, m in _TYPE_CHART)


_TWO_TURN = ("SOLAR_BEAM", "RAZOR_WIND", "SKY_ATTACK", "SKULL_BASH", "SEMI_INVULNERABLE")
_STAT_RATIOS = _DATA["stat_stage_ratios"]
_ACC_RATIOS = [(33, 100), (36, 100), (43, 100), (50, 100), (60, 100), (75, 100), (1, 1), (133, 100), (166, 100),
               (2, 1), (233, 100), (133, 50), (3, 1)]

# Status each move effect can give its target (for telling an ability's effect from the move's)
_CALLERS = ("METRONOME", "SLEEP_TALK", "ASSIST", "MIRROR_MOVE", "NATURE_POWER", "SECRET_POWER")
_INFLICTS = {
    STATUS1_PARALYSIS: ("PARALYZE", "PARALYZE_HIT", "TRI_ATTACK", "THUNDER") + _CALLERS,
    STATUS1_BURN: ("BURN_HIT", "WILL_O_WISP", "TRI_ATTACK", "BLAZE_KICK") + _CALLERS,
    STATUS1_POISON: ("POISON", "POISON_HIT", "TOXIC", "POISON_FANG", "TWINEEDLE", "POISON_TAIL") + _CALLERS,
    STATUS1_SLEEP: ("SLEEP", "YAWN") + _CALLERS,
}
_STAT_DOWN = {  # primary stat-lowering status moves -> stat index in stat_stages (1 Atk .. 7 Eva)
    "ATTACK_DOWN": 1, "DEFENSE_DOWN": 2, "SPEED_DOWN": 3, "SPECIAL_ATTACK_DOWN": 4, "SPECIAL_DEFENSE_DOWN": 5,
    "ACCURACY_DOWN": 6, "EVASION_DOWN": 7,
}
_STAT_DOWN.update({k + "_2": v for k, v in list(_STAT_DOWN.items())})


def _status_class(status1: int) -> int:
    if status1 & STATUS1_SLEEP:
        return STATUS1_SLEEP
    if status1 & (STATUS1_POISON | STATUS1_TOXIC):
        return STATUS1_POISON
    return status1 & (STATUS1_BURN | STATUS1_FREEZE | STATUS1_PARALYSIS)


# ---------------------------------------------------------------------------
# Battle view builder with memory of the battle
# ---------------------------------------------------------------------------

class BattleObserver:
    """Builds BattleViews and remembers what the battle has shown so far.

    Memory (all copied by SimBackend.clone()): revealed moves / items / abilities, which enemy
    Pokemon were seen, the RAM snapshot at the start of the current turn, elapsed-turn counters,
    and the opponent hints heard before the battle.

    `observe()` must see every decision of the battle in order (the backends make sure of it);
    calling it again at the same decision returns the same view.
    """

    def __init__(self, hint_type: int = NO_HINT_TYPE, hint_style: int = 0):
        self.hint_type = hint_type
        self.hint_style = hint_style
        self.revealed_moves: Dict[int, List[int]] = {}
        self.revealed_items: Dict[int, int] = {}
        self.revealed_abilities: Dict[int, int] = {}
        self.seen: set = set()
        self._last_enemy_item: Dict[int, int] = {}
        self._bench_status: Dict[int, int] = {}      # Natural Cure mons: status when last on the field
        self._sleep: Dict[Tuple[int, int], List[int]] = {}   # (side, party idx) -> [attempts, raw counter]
        self._vol: List[Dict] = [{}, {}]             # per battler: counters of the Pokemon now there
        self._turns_done = 0                        # completed turns seen (does not cap like the game's)
        self._start: Optional[_Snap] = None         # snapshot at the last BATTLE decision
        self._prev: Optional[_Snap] = None          # snapshot at the previous decision
        self._events = TurnEvents()
        self._cache_key = None
        self._cache_view: Optional[BattleView] = None

    def copy(self) -> "BattleObserver":
        return copy.deepcopy(self)

    def fast_copy(self) -> "BattleObserver":
        """Same result as copy(), ~20x cheaper: copies only the containers observe() mutates in place.

        Shared by reference (never mutated after they are made): the _Snap snapshots, TurnEvents (replaced, not
        edited), the cached view and key, _enemy_species (reassigned each observe). Copied: the reveal / seen /
        item / status dicts and set, the sleep records ([attempts, raw] lists edited in place) and the per-battler
        volatile dicts (their "confusion" record is such a list too)."""
        o = object.__new__(BattleObserver)
        d = o.__dict__
        d.update(self.__dict__)
        d["revealed_moves"] = {k: v[:] for k, v in self.revealed_moves.items()}
        d["revealed_items"] = self.revealed_items.copy()
        d["revealed_abilities"] = self.revealed_abilities.copy()
        d["seen"] = self.seen.copy()
        d["_last_enemy_item"] = self._last_enemy_item.copy()
        d["_bench_status"] = self._bench_status.copy()
        d["_sleep"] = {k: v[:] for k, v in self._sleep.items()}
        d["_vol"] = [{k: (v[:] if type(v) is list else v) for k, v in vol.items()} for vol in self._vol]
        return o

    def rebase(self, ram: RamReader, forced_switch: bool) -> None:
        """The hidden state of the opponent was rewritten in place (a search determinization: other sets for its
        Pokemon, another exact HP inside the same HP-bar bucket, other hidden counters). Re-anchor what the
        observer compares against so the next observe() only sees what happens from here on, and nothing of the
        replaced hidden state can leak into later views:
          - the snapshot at this decision (and at the turn start at a BATTLE decision) becomes the new state;
          - at a FORCED_SWITCH decision the turn-start snapshot keeps our side and takes the opponent's side of the
            new state, its active Pokemon's HP / PP shifted by what the true state lost since the turn start;
          - the remembered item of the opponent's active Pokemon and the raw sleep / confusion counters of both
            sides (whose changes count attempts) are those of the new state; the counts of attempts are kept.
        """
        s = _Snap(ram, forced_switch)
        old_prev, a = self._prev, self._start
        if not forced_switch or a is None:
            self._start = s
        elif old_prev is not None:
            a2 = copy.copy(a)
            parties1 = [_shift(n, st, now) for n, st, now in zip(s.parties[1], a.parties[1], old_prev.parties[1])]
            # the opponent that was out at the turn start: if another one is out now, its battle data stays as it
            # was (it is only used to replay this turn's order and damage)
            m1 = _shift(s.mons[1], a.mons[1], old_prev.mons[1]) if a.idx[1] == s.idx[1] else a.mons[1]
            a2.parties = [a.parties[0], parties1]
            a2.mons = [a.mons[0], m1]
            self._start = a2
        self._prev = s
        e = s.idx[1]
        if e in self._last_enemy_item:
            self._last_enemy_item[e] = s.mons[1].item
        for (side, i), rec in self._sleep.items():        # (hidden counters may be resampled on both sides)
            st = s.mons[side].status1 if i == s.idx[side] else s.parties[side][i].status
            if st & STATUS1_SLEEP:
                rec[1] = st & STATUS1_SLEEP
        for b in (0, 1):
            conf = self._vol[b].get("confusion")
            if conf is not None and s.mons[b].status2 & STATUS2_CONFUSION:
                conf[1] = s.mons[b].status2 & STATUS2_CONFUSION
        if self._cache_key is not None:
            # observe() again at this decision returns the player's view of it (unchanged by the rewrite)
            self._cache_key = (s.key,) + tuple(self._cache_key[1:])

    # --- the view -------------------------------------------------------------------------

    def observe(self, ram: RamReader, forced_switch: bool, unusable_moves: int, can_switch: bool = True) -> BattleView:
        s = _Snap(ram, forced_switch)
        key = (s.key, unusable_moves, can_switch)
        if key == self._cache_key:
            return self._cache_view
        a = self._start
        new_turn_done = a is not None and not forced_switch
        # game turns since the last BATTLE decision (more than one when the game did not ask us:
        # locked into Outrage, recharging, charging Fly...); a FORCED_SWITCH is inside its turn
        n_turns = max(1, s.turn - a.turn + (1 if forced_switch else 0)) if a is not None else 0
        if a is not None:
            ev = self._turn_events(a, s, n_turns)
            self._events = ev
        else:
            ev = self._events = TurnEvents()
        if new_turn_done:
            self._turns_done += max(1, s.turn - a.turn)

        self._update_seen(s)
        self._update_reveals(a, s, ev, can_switch)
        self._update_counters(a, s, ev)
        view = self._build(s, ev, forced_switch, unusable_moves, can_switch)

        if not forced_switch:
            self._start = s
        self._prev = s
        self._cache_key, self._cache_view = key, view
        return view

    def swap_candidates(self, ram: RamReader) -> List[SeenMon]:
        """The defeated team as the swap screen shows it (species) plus what the battle revealed."""
        enemy = decode_party(ram.read(S.addr("gEnemyParty"), 300))
        return [_species_knowledge(SeenMon(m.species, m.level, HP_BAR_PIXELS, 0, list(self.revealed_moves.get(i, [])),
                                           self.revealed_items.get(i), False, True,
                                           revealed_ability=self.revealed_abilities.get(i, 0)))
                for i, m in enumerate(enemy[:3])]

    # --- turn events ----------------------------------------------------------------------

    def _side_action(self, a: _Snap, s: _Snap, side: int, complete: bool) -> Tuple[int, int]:
        m0, mn = a.idx[side], s.idx[side]
        before, after = a.parties[side][m0], s.parties[side][m0]
        used = [k for k in range(4) if after.moves[k] == before.moves[k] and after.pp[k] < before.pp[k]]
        if used:                                      # "X used Y!" (the party copy's PP went down)
            last = s.last_moves[side] if mn == m0 else 0
            moves = [before.moves[k] for k in used]
            return ACTION_MOVE, last if last in moves else moves[-1]
        if mn == m0:
            lm = s.last_moves[side]
            # still there and alive at the end of the turn: it acted (its last move may repeat)
            if lm != a.last_moves[side] or (complete and s.mons[side].hp > 0):
                if lm == MOVE_UNAVAILABLE:
                    return ACTION_CANT_MOVE, 0
                if lm:
                    return ACTION_MOVE, lm            # no PP used: locked / Struggle / Transformed
            if complete and s.mons[side].hp == 0:
                return _FAINTED, 0                    # fainted, not replaced yet: acted or not (resolved later)
            return ACTION_NONE, 0
        if after.hp == 0:
            return _FAINTED, 0                        # fainted this turn: before or after acting (resolved later)
        return ACTION_SWITCH, 0                       # (dragged out by Roar: fixed by the caller)

    def _turn_events(self, a: _Snap, s: _Snap, n_turns: int) -> TurnEvents:
        ev = TurnEvents(turn=a.turn, turns=n_turns)
        ev.in_progress = s.forced and s.mons[0].hp > 0 and s.idx[0] == a.idx[0]     # our Baton Pass
        acts = [list(self._side_action(a, s, side, not ev.in_progress)) for side in (0, 1)]
        raw = [s.bide[b] - a.bide[b] if s.bide[b] >= a.bide[b] else s.bide[b] for b in (0, 1)]
        for side in (0, 1):
            if acts[side][0] != _FAINTED:
                continue
            # it fainted and was replaced: it never acted if a faster opponent's hit knocked it out
            oact, omove = acts[1 - side]
            m0 = a.idx[side]
            ko_by_hit = raw[side] >= a.parties[side][m0].hp
            # the other side got hit while its own action could not have hurt itself: this one struck
            struck = raw[1 - side] > 0 and oact in (ACTION_MOVE, ACTION_SWITCH) \
                and not _is(omove, "RECOIL", "DOUBLE_EDGE", "SUBSTITUTE", "BELLY_DRUM", "CURSE", "PAIN_SPLIT")
            prio = [0, 0]
            prio[side] = _priority(s.last_used[side])
            prio[1 - side] = _priority(omove)
            opp_first = oact == ACTION_SWITCH or (oact == ACTION_MOVE and self._order(a, prio) == 1 - side)
            if ko_by_hit and oact == ACTION_MOVE and MOVES[omove]["power"] and opp_first and not struck:
                acts[side] = [ACTION_NONE, 0]
            elif a.mons[side].status2 & STATUS2_MULTIPLETURNS and a.locked[side]:
                acts[side] = [ACTION_MOVE, a.locked[side]]
            elif not any(a.mons[side].pp[k] for k in range(4) if a.mons[side].moves[k]):
                acts[side] = [ACTION_MOVE, MOVE["STRUGGLE"]]
            else:
                acts[side] = [ACTION_CANT_MOVE, 0]
        (ev.own_action, ev.own_move), (ev.enemy_action, ev.enemy_move) = acts
        # dragged out by Roar / Whirlwind: not a switch of its own (the game counts our switches)
        if ev.own_action == ACTION_SWITCH and s.player_switches == a.player_switches:
            ev.own_action = ACTION_NONE
        if ev.enemy_action == ACTION_SWITCH and ev.own_action == ACTION_MOVE and _is(ev.own_move, "ROAR"):
            ev.enemy_action = ACTION_NONE
        ev.own_fainted, ev.enemy_fainted = (
            any(m.hp == 0 and a.parties[side][i].hp > 0 for i, m in enumerate(s.parties[side][:3])) for side in (0, 1))
        ev.first = self._first(a, s, ev, n_turns)

        # critical hits: gCritMultiplier holds the last move of the turn (each move resets it)
        movers = [side for side, act in ((0, ev.own_action), (1, ev.enemy_action))
                  if act in (ACTION_MOVE, ACTION_CANT_MOVE)]
        last_mover = None
        if len(movers) == 1:
            last_mover = movers[0]
        elif len(movers) == 2 and ev.first in (0, 1):
            last_mover = 1 - ev.first
        dmg = self._direct_damage(a, s, ev)
        for side in (0, 1):
            act, move = (ev.own_action, ev.own_move) if side == 0 else (ev.enemy_action, ev.enemy_move)
            crit = -1
            if act != ACTION_MOVE or MOVES[move]["power"] == 0:
                crit = 0
            elif n_turns == 1 and side == last_mover and s.crit == 2 and (
                    s.bide[1 - side] != a.bide[1 - side] or a.mons[1 - side].status2 & STATUS2_SUBSTITUTE):
                crit = 1
            elif n_turns == 1 and side == last_mover and s.forced:
                crit = 0        # before the opponent's AI has run: gCritMultiplier is still the move's
            if side == 0:
                ev.own_crit = crit
            else:
                ev.enemy_crit = crit
        ev.damage_taken = dmg[0][0]
        h0, lost = dmg[1][1], dmg[1][0]
        ev.damage_dealt_pixels = hp_bar_pixels(h0, dmg[1][2]) - hp_bar_pixels(max(0, h0 - lost), dmg[1][2])
        return ev

    def _first(self, a: _Snap, s: _Snap, ev: TurnEvents, n_turns: int) -> int:
        own, en = ev.own_action, ev.enemy_action
        if own == ACTION_NONE and en == ACTION_NONE:
            return -1
        if own == ACTION_NONE:
            return 1
        if en == ACTION_NONE:
            return 0
        if own == ACTION_SWITCH:
            return 0                                          # switches go first, ours before theirs
        if en == ACTION_SWITCH:
            return 1
        if n_turns > 1:
            return s.order[0] if s.order[0] in (0, 1) else -1    # the game's own end-of-turn order
        return self._order(a, [_priority(m) for m in s.last_used])

    def _order(self, a: _Snap, prio) -> int:
        """GetWhoStrikesFirst at the start of the turn: 0 / 1, or -1 for a speed tie (the RNG decides)."""
        if prio[0] != prio[1]:
            return 0 if prio[0] > prio[1] else 1
        sp = [self._speed(a, b) for b in (0, 1)]
        if sp[0] == sp[1]:
            return -1
        return 0 if sp[0] > sp[1] else 1

    @staticmethod
    def _speed(a: _Snap, b: int) -> int:
        """GetWhoStrikesFirst's speed (battle_main.c) at the start of the turn."""
        m = a.mons[b]
        mult = 1
        if a.weather_has_effect() and ((m.ability == ABILITY["SWIFT_SWIM"] and a.weather & B_WEATHER_RAIN)
                                       or (m.ability == ABILITY["CHLOROPHYLL"] and a.weather & B_WEATHER_SUN)):
            mult = 2
        num, den = _STAT_RATIOS[m.stat_stages[3]]
        spd = m.stats[2] * mult * num // den
        he, param = hold_effect(m.item)
        if he == HOLD["MACHO_BRACE"]:
            spd //= 2
        if m.status1 & STATUS1_PARALYSIS:
            spd //= 4
        if he == HOLD["QUICK_CLAW"] and a.rnd_turn < (0xFFFF * param) // 100:
            spd = 0xFFFFFFFF
        return spd

    def _direct_damage(self, a: _Snap, s: _Snap, ev: TurnEvents):
        """Per side: (HP lost to the opponent's hits this turn, HP before them, max HP) of the
        Pokemon that was hit. gBideDmg[battler] adds up every HP loss of that battler slot outside
        the end-of-turn effects (the game's own Bide counter); the Pokemon's own recoil, HP costs
        (Substitute, Belly Drum, Curse, Pain Split) and Spikes on entry are taken out."""
        raw = [max(0, s.bide[b] - a.bide[b]) if s.bide[b] >= a.bide[b] else max(0, s.bide[b]) for b in (0, 1)]
        h0, max_hp, costs, recoil_div, damaged = [0, 0], [0, 0], [0, 0], [0, 0], [True, True]
        for side in (0, 1):
            act = ev.own_action if side == 0 else ev.enemy_action
            move = ev.own_move if side == 0 else ev.enemy_move
            # whoever was in the slot when the opponent struck: a switch goes first; after our own
            # Baton Pass the newcomer takes the hits of a slower opponent
            passed = act == ACTION_MOVE and _is(move, "BATON_PASS") and ev.first == side
            victim = s.idx[side] if act == ACTION_SWITCH or (passed and s.idx[side] != a.idx[side]) else a.idx[side]
            pm = a.parties[side][victim]
            max_hp[side] = pm.max_hp
            h0[side] = a.mons[side].hp if victim == a.idx[side] else pm.hp
            user = a.mons[side]
            if act == ACTION_MOVE:
                if _is(move, "SUBSTITUTE") and s.mons[side].status2 & STATUS2_SUBSTITUTE \
                        and not user.status2 & STATUS2_SUBSTITUTE:
                    costs[side] += max(1, user.max_hp // 4)
                elif _is(move, "BELLY_DRUM") and s.mons[side].stat_stages[1] == 12 and user.stat_stages[1] < 12:
                    costs[side] += user.max_hp // 2
                elif _is(move, "CURSE") and TYPE["GHOST"] in user.types and \
                        s.mons[1 - side].status2 & STATUS2_CURSED and not a.mons[1 - side].status2 & STATUS2_CURSED:
                    costs[side] += user.max_hp // 2
                elif _is(move, "PAIN_SPLIT"):
                    costs[side] += max(0, user.hp - (user.hp + a.mons[1 - side].hp) // 2)
                elif _is(move, "RECOIL", "DOUBLE_EDGE") and user.ability != ABILITY["ROCK_HEAD"]:
                    recoil_div[side] = 3 if _is(move, "DOUBLE_EDGE") else 4
            if victim != a.idx[side]:                       # switched in: Spikes on entry
                layers = a.spikes(side)
                if layers and TYPE["FLYING"] not in SPECIES[pm.species]["types"] and \
                        ability_of(pm.species, pm.ability_num) != ABILITY["LEVITATE"]:
                    spk = max(1, pm.max_hp // (8, 6, 4)[min(layers, 3) - 1])
                    costs[side] += spk
                    h0[side] = max(0, h0[side] - spk)
            oact = ev.enemy_action if side == 0 else ev.own_action
            omove = ev.enemy_move if side == 0 else ev.own_move
            charging = oact == ACTION_MOVE and s.mons[1 - side].status2 & STATUS2_MULTIPLETURNS \
                and s.locked[1 - side] == omove and _is(omove, *_TWO_TURN) and s.idx[1 - side] == a.idx[1 - side] \
                and not a.mons[1 - side].status2 & STATUS2_MULTIPLETURNS     # first turn: it only charged
            damaged[side] = ev.turns != 1 or (oact == ACTION_MOVE and not charging and bool(
                MOVES[omove]["power"] or _is(omove, "PAIN_SPLIT", *_CALLERS)))
        # recoil is a fraction of the damage the other side took: settle both sides together
        hit = [max(0, raw[b] - costs[b]) for b in (0, 1)]
        for _ in range(8):
            rec = [max(1, min(hit[1 - b], h0[1 - b]) // recoil_div[b]) if recoil_div[b] and min(hit[1 - b], h0[1 - b]) > 0
                   else 0 for b in (0, 1)]
            hit = [max(0, raw[b] - costs[b] - rec[b]) for b in (0, 1)]
        # a Pokemon that fainted lost what it had when hit (gBideDmg counts overkill): its HP at the
        # start, plus what its own earlier move gave back (drain, Shell Bell, recovery)
        for b in (0, 1):
            act = ev.own_action if b == 0 else ev.enemy_action
            move = ev.own_move if b == 0 else ev.enemy_move
            if act == ACTION_MOVE and ev.first == b:
                dealt = min(hit[1 - b], h0[1 - b])
                heal = 0
                if _is(move, "ABSORB", "DREAM_EATER") and dealt:
                    heal += max(1, dealt // 2)
                elif _is(move, "RESTORE_HP", "SOFTBOILED", "MORNING_SUN", "SYNTHESIS", "MOONLIGHT"):
                    heal += max_hp[b] // 2
                elif _is(move, "REST"):
                    heal += max_hp[b]
                if hold_effect(a.mons[b].item)[0] == HOLD["SHELL_BELL"] and dealt and MOVES[move]["power"]:
                    heal += max(1, dealt // hold_effect(a.mons[b].item)[1])
                h0[b] = min(max_hp[b], h0[b] + heal)
        return [(min(hit[b], h0[b]) if damaged[b] else 0, h0[b], max_hp[b]) for b in (0, 1)]

    # --- memory updates -------------------------------------------------------------------

    def _update_seen(self, s: _Snap):
        self.seen.add(s.idx[1])
        prev = self._prev
        enemy = s.parties[1]
        for i, m in enumerate(enemy[:3]):
            if prev is not None:
                pm = prev.parties[1][i]
                if (m.hp, m.pp) != (pm.hp, pm.pp):
                    self.seen.add(i)                      # it was on the field between two decisions
                # moves whose PP went down were used in front of the player
                for k in range(4):
                    if m.moves[k] and m.moves[k] == pm.moves[k] and m.pp[k] < pm.pp[k]:
                        self._reveal_move(i, m.moves[k])
        e = s.idx[1]
        printed = s.last_printed[1]
        if printed and printed != MOVE_UNAVAILABLE and printed in enemy[e].moves:
            self._reveal_move(e, printed)
        prev_item = self._last_enemy_item.get(e)
        if prev_item and s.mons[1].item != prev_item:
            self.revealed_items[e] = prev_item            # consumed / knocked off / tricked: now known
        self._last_enemy_item[e] = s.mons[1].item
        # status of benched Pokemon: what it had when it left (Natural Cure cures it silently)
        if prev is not None and prev.idx[1] != e:
            i = prev.idx[1]
            self._bench_status[i] = major_status(prev.mons[1].status1)

    def _reveal_move(self, i: int, move: int):
        lst = self.revealed_moves.setdefault(i, [])
        if move not in lst:
            lst.append(move)

    def _reveal_ability(self, i: int, ability: int):
        if ability and ability in possible_abilities(self._enemy_species[i]):
            self.revealed_abilities[i] = ability

    def _update_reveals(self, a: Optional[_Snap], s: _Snap, ev: TurnEvents, can_switch: bool):
        """Abilities the game announced, and items named in messages (see GUIDE section 7)."""
        self._enemy_species = [m.species for m in s.parties[1]]
        e = s.idx[1]
        base = lambda i: ability_of(s.parties[1][i].species, s.parties[1][i].ability_num)
        em = s.mons[1]
        # on the field now: Intimidate fired / Trace copied an ability
        if em.ability == ABILITY["INTIMIDATE"] == base(e) and not s.status3[1] & STATUS3_INTIMIDATE_POKES:
            self._reveal_ability(e, ABILITY["INTIMIDATE"])
        if base(e) == ABILITY["TRACE"] and em.ability != ABILITY["TRACE"] and not s.status3[1] & STATUS3_TRACE:
            self._reveal_ability(e, ABILITY["TRACE"])
        # we are trapped by its ability: trying to switch shows "X prevents escape with Y!"
        if not s.forced and not can_switch and not s.mons[0].status2 & (STATUS2_WRAPPED | STATUS2_ESCAPE_PREVENTION) \
                and not s.status3[0] & STATUS3_ROOTED and em.ability == base(e) \
                and em.ability in (ABILITY["SHADOW_TAG"], ABILITY["ARENA_TRAP"], ABILITY["MAGNET_PULL"]):
            self._reveal_ability(e, em.ability)
        # our Intimidate just came in and was stopped by Clear Body / Hyper Cutter / White Smoke
        prev = self._prev
        if s.mons[0].ability == ABILITY["INTIMIDATE"] and (prev is None or prev.idx[0] != s.idx[0]) \
                and (prev is None or prev.idx[1] == e) and em.ability == base(e) \
                and em.ability in (ABILITY["CLEAR_BODY"], ABILITY["HYPER_CUTTER"], ABILITY["WHITE_SMOKE"]):
            self._reveal_ability(e, em.ability)
        if a is None:
            return
        self._reveal_by_our_move(a, s, ev)
        self._reveal_end_of_turn(a, s, ev)

    def _our_move_surely_hit(self, a: _Snap, s: _Snap, ev: TurnEvents, move: int, target_start: bool) -> bool:
        """The accuracy check of our move could not fail (so no "missed" message)."""
        ours, foe = a.mons[0], a.mons[1] if target_start else None
        if ev.enemy_action == ACTION_MOVE and _is(ev.enemy_move, "PROTECT") and s.protect_uses(1) > 0:
            return False
        if a.status3[1] & STATUS3_SEMI_INVULNERABLE or s.status3[1] & STATUS3_SEMI_INVULNERABLE:
            return False
        if target_start and a.status3[1] & STATUS3_ALWAYS_HITS and a.sure_hit_by(1) == 0:
            return True
        if _is(move, "ALWAYS_HIT", "VITAL_THROW"):
            return True
        acc = MOVES[move]["accuracy"]
        if _is(move, "THUNDER") and a.weather_has_effect() and a.weather & B_WEATHER_SUN:
            acc = 50
        if _is(move, "THUNDER") and a.weather_has_effect() and a.weather & B_WEATHER_RAIN:
            return True
        eva = foe.stat_stages[7] if foe is not None else 6
        buff = ours.stat_stages[6] if foe is not None and foe.status2 & STATUS2_FORESIGHT else ours.stat_stages[6] + 6 - eva
        num, den = _ACC_RATIOS[max(0, min(12, buff))]
        calc = num * acc // den
        if ours.ability == ABILITY["COMPOUND_EYES"]:
            calc = calc * 130 // 100
        target = foe if foe is not None else s.mons[1]
        if a.weather_has_effect() and target.ability == ABILITY["SAND_VEIL"] and a.weather & B_WEATHER_SANDSTORM:
            calc = calc * 80 // 100
        if ours.ability == ABILITY["HUSTLE"] and _move_type(move, ours, a.weather) < TYPE["MYSTERY"]:
            calc = calc * 80 // 100
        he, param = hold_effect(target.item)
        if he == HOLD["EVASION_UP"]:
            calc = calc * (100 - param) // 100
        return calc >= 100

    def _reveal_by_our_move(self, a: _Snap, s: _Snap, ev: TurnEvents):
        if ev.own_action != ACTION_MOVE or ev.turns != 1:
            return
        move = ev.own_move
        eff = _effect(move)
        E = lambda *names: eff in {EFFECT[n] for n in names}
        switched = ev.enemy_action == ACTION_SWITCH
        t = s.idx[1] if switched else a.idx[1]          # the Pokemon our move was aimed at
        tmon = (s if switched else a).mons[1]            # its battle state (at the start, or on entry)
        tm = s.parties[1][t]
        ab = ability_of(tm.species, tm.ability_num)
        if tmon.ability != ab:
            return                                       # Trace / Skill Swap: not its own ability
        ours = a.mons[0]
        mtype = _move_type(move, ours, a.weather)
        sure = lambda: self._our_move_surely_hit(a, s, ev, move, not switched)
        dealt = s.bide[1] - a.bide[1] > 0
        sub = tmon.status2 & STATUS2_SUBSTITUTE
        reveal = lambda: self._reveal_ability(t, ab)
        power = MOVES[move]["power"]
        # absorbed: "X restored HP using Water Absorb!" / Flash Fire
        if (ab == ABILITY["WATER_ABSORB"] and mtype == TYPE["WATER"] and power) or \
                (ab == ABILITY["VOLT_ABSORB"] and mtype == TYPE["ELECTRIC"] and power) or \
                (ab == ABILITY["FLASH_FIRE"] and mtype == TYPE["FIRE"] and not tmon.status1 & STATUS1_FREEZE):
            if sure():
                reveal()
            return
        # blocked by an ability before any accuracy check
        if (ab == ABILITY["LIMBER"] and E("PARALYZE")) or (ab == ABILITY["IMMUNITY"] and E("TOXIC", "POISON")) or \
                (ab == ABILITY["OWN_TEMPO"] and E("CONFUSE")) or (ab == ABILITY["SUCTION_CUPS"] and E("ROAR")) or \
                (ab in (ABILITY["INSOMNIA"], ABILITY["VITAL_SPIRIT"]) and E("YAWN")):
            reveal()
            return
        if ab in (ABILITY["INSOMNIA"], ABILITY["VITAL_SPIRIT"]) and E("SLEEP") and not sub \
                and not tmon.status1 & STATUS1_SLEEP:
            reveal()
            return
        if ab == ABILITY["WATER_VEIL"] and E("WILL_O_WISP") and not sub and not tmon.status1 & STATUS1_BURN \
                and TYPE["FIRE"] not in tmon.types:
            reveal()
            return
        if ab == ABILITY["DAMP"] and E("EXPLOSION") and s.mons[0].hp > 0 and s.idx[0] == a.idx[0]:
            reveal()
            return
        if ab == ABILITY["STURDY"] and E("OHKO") and not _type_immune(mtype, tmon.types) \
                and not (ev.enemy_action == ACTION_MOVE and _is(ev.enemy_move, "PROTECT") and s.protect_uses(1) > 0) \
                and not (a.status3[1] | s.status3[1]) & STATUS3_SEMI_INVULNERABLE:
            reveal()
            return
        if ab == ABILITY["OBLIVIOUS"] and E("ATTRACT") and sure():
            reveal()
            return
        stat = next((v for k, v in _STAT_DOWN.items() if eff == EFFECT[k]), None)
        if stat is not None and not sub and not s.mons[1].status2 & STATUS2_SUBSTITUTE \
                and not s.sides[1] & SIDE_STATUS_MIST and sure():
            if ab in (ABILITY["CLEAR_BODY"], ABILITY["WHITE_SMOKE"]) or (ab == ABILITY["KEEN_EYE"] and stat == 6) \
                    or (ab == ABILITY["HYPER_CUTTER"] and stat == 1):
                reveal()
            return
        if ab == ABILITY["LIQUID_OOZE"] and E("ABSORB", "DREAM_EATER") and dealt:
            reveal()
            return
        if ab == ABILITY["STICKY_HOLD"] and tmon.item and s.parties[1][t].held_item == tmon.item and (
                (E("THIEF", "KNOCK_OFF") and dealt) or (E("TRICK") and not sub and sure())):
            reveal()
            return
        if ab == ABILITY["INNER_FOCUS"] and E("FAKE_OUT") and dealt:
            reveal()
            return
        # contact abilities that gave us a status: "X's Static paralyzed Y!"
        if MOVES[move]["flags"] & FLAG_MAKES_CONTACT and power and dealt and s.idx[0] == a.idx[0] \
                and not a.mons[0].status1 & STATUS1_ANY:
            got = _status_class(s.mons[0].status1)
            causes = {ABILITY["STATIC"]: (STATUS1_PARALYSIS,), ABILITY["FLAME_BODY"]: (STATUS1_BURN,),
                      ABILITY["POISON_POINT"]: (STATUS1_POISON,),
                      ABILITY["EFFECT_SPORE"]: (STATUS1_POISON, STATUS1_PARALYSIS, STATUS1_SLEEP)}
            if got and got in causes.get(ab, ()) and not self._enemy_could_inflict(a, ev, got) \
                    and not (got == STATUS1_SLEEP and a.status3[0] & STATUS3_YAWN):
                reveal()
                return
        # Synchronize: we gave it a status and got the same one back
        if ab == ABILITY["SYNCHRONIZE"] and s.idx[0] == a.idx[0] and not a.mons[0].status1 & STATUS1_ANY \
                and not tmon.status1 & STATUS1_ANY and t == s.idx[1]:
            got_them = _status_class(s.mons[1].status1)
            if got_them in (STATUS1_POISON, STATUS1_BURN, STATUS1_PARALYSIS) \
                    and _status_class(s.mons[0].status1) == got_them and not self._enemy_could_inflict(a, ev, got_them):
                reveal()

    @staticmethod
    def _enemy_could_inflict(a: _Snap, ev: TurnEvents, status: int) -> bool:
        if ev.enemy_action != ACTION_MOVE:
            return False
        return _effect(ev.enemy_move) in {EFFECT[n] for n in _INFLICTS.get(status, ())}

    def _reveal_end_of_turn(self, a: _Snap, s: _Snap, ev: TurnEvents):
        """Announcements at the end of the turn by a Pokemon that stayed in all turn."""
        after_end = not ev.in_progress
        e = a.idx[1]
        if not after_end or s.idx[1] != e or ev.enemy_action == ACTION_SWITCH or s.mons[1].hp == 0:
            return
        em, m0 = s.mons[1], a.mons[1]
        pm = s.parties[1][e]
        ab = ability_of(pm.species, pm.ability_num)
        if em.ability == ab == ABILITY["SPEED_BOOST"] and m0.stat_stages[3] < 12:
            self._reveal_ability(e, ab)                   # "X's Speed Boost raised its Speed!"
        if em.ability == ab == ABILITY["LIQUID_OOZE"] and s.status3[1] & STATUS3_LEECHSEED \
                and (s.status3[1] & STATUS3_LEECHSEED_BATTLER) == 0 and s.mons[0].hp > 0:
            self._reveal_ability(e, ab)                   # our Leech Seed sucked up the ooze
        if em.ability == ab == ABILITY["SHED_SKIN"] and ev.turns == 1:
            had = m0.status1 & (STATUS1_PARALYSIS | STATUS1_BURN | STATUS1_POISON | STATUS1_TOXIC)
            cured_by_move = ev.enemy_action == ACTION_MOVE and _is(ev.enemy_move, "REFRESH", "REST", "HEAL_BELL")
            if had and not em.status1 & STATUS1_ANY and em.item == m0.item and not cured_by_move:
                self._reveal_ability(e, ab)
        # "X restored a little HP using its Leftovers!"
        he, _ = hold_effect(em.item)
        if he == HOLD["LEFTOVERS"] and em.hp < em.max_hp and not em.status1 & (STATUS1_POISON | STATUS1_TOXIC | STATUS1_BURN) \
                and not s.status3[1] & STATUS3_LEECHSEED \
                and not em.status2 & (STATUS2_CURSED | STATUS2_NIGHTMARE | STATUS2_WRAPPED):
            self.revealed_items[e] = em.item

    def _update_counters(self, a: Optional[_Snap], s: _Snap, ev: TurnEvents):
        """Elapsed-turn counters of random-length effects."""
        # sleep, per Pokemon (it survives switching): +1 each time the hidden counter goes down. Only
        # the Pokemon on the field are updated: nothing visible happens to a benched one (Natural
        # Cure heals silently, so its record stays as the player last saw it)
        for side in (0, 1):
            for i in {s.idx[side], a.idx[side] if a is not None else s.idx[side]}:
                if i != s.idx[side] and s.parties[side][i].hp > 0:
                    continue                             # left the field: frozen
                st = s.mons[side].status1 if i == s.idx[side] else s.parties[side][i].status
                raw = st & STATUS1_SLEEP
                k = (side, i)
                rec = self._sleep.get(k)
                if not raw:
                    self._sleep.pop(k, None)
                    continue
                if rec is None or raw > rec[1]:          # fell asleep since the last decision
                    tried = 0
                    if a is not None and i == s.idx[side] == a.idx[side]:
                        act = ev.own_action if side == 0 else ev.enemy_action
                        if act == ACTION_CANT_MOVE and ev.first == 1 - side and not a.status3[side] & STATUS3_YAWN \
                                and not a.mons[side].status1 & STATUS1_ANY:
                            tried = 1                    # put to sleep first, then "fast asleep" this turn
                    self._sleep[k] = [tried, raw]
                elif raw < rec[1]:
                    rec[0] += 1
                    rec[1] = raw
        # volatile effects, per battler (reset when the Pokemon there changes)
        for b in (0, 1):
            v = self._vol[b]
            if v.get("mon") != s.idx[b] or (self._prev is not None and self._prev.idx[b] != s.idx[b]):
                v.clear()
                v["mon"] = s.idx[b]
            m = s.mons[b]
            conf = m.status2 & STATUS2_CONFUSION
            rec = v.get("confusion")
            if not conf:
                v.pop("confusion", None)
            elif rec is None or conf > rec[1]:
                tried = 0
                if a is not None and s.idx[b] == a.idx[b]:
                    act = ev.own_action if b == 0 else ev.enemy_action
                    rampage_ended = a.mons[b].status2 & STATUS2_LOCK_CONFUSE and not m.status2 & STATUS2_LOCK_CONFUSE
                    if act in (ACTION_MOVE, ACTION_CANT_MOVE) and ev.first == 1 - b and not rampage_ended:
                        tried = 1
                v["confusion"] = [tried, conf]
            elif conf < rec[1]:
                rec[0] += 1
                rec[1] = conf
            since = self._turns_done - (0 if s.forced else 1)
            for name, on in (("encore", s.encored_move(b) and s.encore_timer(b)), ("disable", s.disabled_move(b)),
                             ("wrapped", m.status2 & STATUS2_WRAPPED), ("uproar", m.status2 & STATUS2_UPROAR),
                             ("rampage", m.status2 & STATUS2_LOCK_CONFUSE)):
                if not on:
                    v.pop(name, None)
                elif name not in v:
                    v[name] = since

    def _elapsed(self, b: int, name: str) -> int:
        start = self._vol[b].get(name)
        return 0 if start is None else max(0, self._turns_done - start)

    # --- assembling the view --------------------------------------------------------------

    def _build(self, s: _Snap, ev: TurnEvents, forced: bool, unusable: int, can_switch: bool) -> BattleView:
        own, enemy = s.parties[0], s.parties[1]
        own_views = [OwnMon.from_party(m) for m in own[:3]]
        a0 = s.idx[0]
        # the active mon's battle copy is authoritative for HP/PP/status mid-battle
        own_views[a0].hp, own_views[a0].pp = s.mons[0].hp, list(s.mons[0].pp)
        own_views[a0].status = major_status(s.mons[0].status1)
        for i, o in enumerate(own_views):
            rec = self._sleep.get((0, i))
            o.sleep_turns = rec[0] if rec and o.status == 1 else 0

        e = s.idx[1]
        enemy_views = []
        for i, m in enumerate(enemy[:3]):
            if i not in self.seen:
                enemy_views.append(SeenMon())
                continue
            active = i == e
            hp = s.mons[1].hp if active else m.hp
            status = major_status(s.mons[1].status1 if active else m.status)
            if not active and ability_of(m.species, m.ability_num) == ABILITY["NATURAL_CURE"] and hp > 0:
                status = self._bench_status.get(i, status)
            if hp == 0:
                status = 0
            rec = self._sleep.get((1, i))
            sv = SeenMon(m.species, m.level, hp_bar_pixels(hp, m.max_hp), status, list(self.revealed_moves.get(i, [])),
                         self.revealed_items.get(i), hp == 0, True, revealed_ability=self.revealed_abilities.get(i, 0),
                         sleep_turns=rec[0] if rec and status == 1 else 0)
            enemy_views.append(_species_knowledge(sv))

        w = s.weather
        kind = WEATHER_NONE
        for flag, k in ((B_WEATHER_RAIN, WEATHER_RAIN), (B_WEATHER_SUN, WEATHER_SUN),
                        (B_WEATHER_SANDSTORM, WEATHER_SANDSTORM), (B_WEATHER_HAIL, WEATHER_HAIL)):
            if w & flag:
                kind = k
                break
        permanent = bool(w & (B_WEATHER_RAIN_PERMANENT | B_WEATHER_SUN_PERMANENT | B_WEATHER_SANDSTORM_PERMANENT))
        weather_left = s.wfk[40] if kind and not permanent else 0

        # voluntary switches are refused while trapped; replacements after a faint are not
        switch_targets = [i for i, m in enumerate(own[:3]) if m.hp > 0 and i != a0] \
            if (forced or can_switch) else []
        usable = [not forced and bool(s.mons[0].moves[i]) and not (unusable >> i) & 1 for i in range(4)]
        must_struggle = not forced and not any(usable)
        if forced:
            legal = [("switch", i) for i in switch_targets]
        else:
            legal = ([("move", i) for i in range(4) if usable[i]] or [("move", 0)]) \
                + [("switch", i) for i in switch_targets] + [("forfeit",)]
        return BattleView(
            own_party=own_views,
            own_active=self._active(s, 0),
            enemy_party=enemy_views,
            enemy_active=self._active(s, 1),
            weather=kind, weather_permanent=permanent, weather_turns_left=weather_left,
            own_side=self._side(s, 0), enemy_side=self._side(s, 1),
            turn=s.turn,
            forced_switch=forced,
            usable_moves=usable,
            switch_targets=switch_targets,
            last_turn=copy.copy(ev),
            hint_type=self.hint_type, hint_style=self.hint_style,
            must_struggle=must_struggle,
            legal_actions=legal,
        )

    @staticmethod
    def _side(s: _Snap, side: int) -> SideState:
        t = s.timers[12 * side:12 * side + 12]
        st = s.sides[side]
        fs_counter, fs_move = s.wfk[side], struct.unpack_from("<H", s.wfk, 24 + 2 * side)[0]
        return SideState(
            reflect_turns=t[0] if st & SIDE_STATUS_REFLECT else 0,
            light_screen_turns=t[2] if st & SIDE_STATUS_LIGHTSCREEN else 0,
            mist_turns=t[4] if st & SIDE_STATUS_MIST else 0,
            safeguard_turns=t[6] if st & SIDE_STATUS_SAFEGUARD else 0,
            spikes=t[10],
            future_sight_turns=fs_counter,
            future_sight_move=fs_move if fs_counter else 0,
            wish_turns=s.wfk[32 + side],
        )

    def _active(self, s: _Snap, b: int) -> ActiveState:
        m = s.mons[b]
        st2, st3 = m.status2, s.status3[b]
        locked = st2 & (STATUS2_MULTIPLETURNS | STATUS2_RECHARGE)
        locked_move = s.locked[b] if locked else 0
        two_turn = locked_move and st2 & STATUS2_MULTIPLETURNS and \
            _is(locked_move, *_TWO_TURN)
        semi = 1 if st3 & STATUS3_ON_AIR else 2 if st3 & STATUS3_UNDERGROUND else 3 if st3 & STATUS3_UNDERWATER else 0
        conf = self._vol[b].get("confusion")
        return ActiveState(
            party_index=s.idx[b],
            stat_stages=[x - 6 for x in m.stat_stages[1:8]],
            types=list(m.types),
            confused=bool(st2 & STATUS2_CONFUSION),
            infatuated=bool(st2 & STATUS2_INFATUATION),
            substitute=bool(st2 & STATUS2_SUBSTITUTE),
            leech_seeded=bool(st3 & STATUS3_LEECHSEED),
            cursed=bool(st2 & STATUS2_CURSED),
            nightmare=bool(st2 & STATUS2_NIGHTMARE),
            trapped=bool(st2 & STATUS2_ESCAPE_PREVENTION),
            focus_energy=bool(st2 & STATUS2_FOCUS_ENERGY),
            transformed=bool(st2 & STATUS2_TRANSFORMED),
            perish_song=bool(st3 & STATUS3_PERISH_SONG),
            rooted=bool(st3 & STATUS3_ROOTED),
            yawn=bool(st3 & STATUS3_YAWN),
            torment=bool(st2 & STATUS2_TORMENT),
            taunted=s.taunt_timer(b) > 0,
            encored_move=s.encored_move(b),
            disabled_move=s.disabled_move(b),
            protect_uses=s.protect_uses(b),
            confusion_turns=conf[0] if conf and st2 & STATUS2_CONFUSION else 0,
            toxic_counter=(m.status1 & STATUS1_TOXIC_COUNTER) >> 8 if m.status1 & STATUS1_TOXIC else 0,
            perish_count=s.perish_timer(b) if st3 & STATUS3_PERISH_SONG else -1,
            taunt_turns_left=s.taunt_timer(b),
            encore_turns=self._elapsed(b, "encore"),
            disable_turns=self._elapsed(b, "disable"),
            yawn_turns_left=(st3 & STATUS3_YAWN) >> 11,
            wrapped_turns=self._elapsed(b, "wrapped"),
            uproar_turns=self._elapsed(b, "uproar"),
            rampage_turns=self._elapsed(b, "rampage"),
            locked_move=locked_move,
            charging_move=locked_move if two_turn else 0,
            semi_invulnerable=semi,
            must_recharge=bool(st2 & STATUS2_RECHARGE),
            bide_turns_left=(st2 & STATUS2_BIDE) >> 8,
            rollout_hits_left=s.rollout_timer(b),
            fury_cutter_count=s.fury_cutter(b),
            stockpile=s.stockpile(b),
            charge_turns_left=s.charge_timer(b) if st3 & STATUS3_CHARGED_UP else 0,
            locked_on_turns_left=(st3 & STATUS3_ALWAYS_HITS) >> 3,
            destiny_bond=bool(st2 & STATUS2_DESTINY_BOND),
            substitute_hp=s.substitute_hp(b) if b == 0 and st2 & STATUS2_SUBSTITUTE else -1,
            defense_curl=bool(st2 & STATUS2_DEFENSE_CURL),
            foresight=bool(st2 & STATUS2_FORESIGHT),
            minimized=bool(st3 & STATUS3_MINIMIZED),
            charged_up=bool(st3 & STATUS3_CHARGED_UP),
            imprisoning=bool(st3 & STATUS3_IMPRISONED_OTHERS),
            grudge=bool(st3 & STATUS3_GRUDGE),
            mud_sport=bool(st3 & STATUS3_MUDSPORT),
            water_sport=bool(st3 & STATUS3_WATERSPORT),
            rage=bool(st2 & STATUS2_RAGE),
            first_turn=s.is_first_turn(b) > 0,
        )
