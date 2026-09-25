"""Decoders for pokeemerald RAM structures (layouts from pokeemerald/include)."""

import json
import os
import struct
from dataclasses import dataclass, field
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))


class Symbols:
    """RAM object and function addresses exported from pokeemerald.elf."""

    def __init__(self, path=os.path.join(_HERE, "symbols.json")):
        with open(path) as f:
            data = json.load(f)
        self.objects = data["objects"]
        self.functions = data["functions"]

    def addr(self, name):
        return self.objects[name][0]

    def size(self, name):
        return self.objects[name][1]

    def func(self, name):
        """Function address as stored in function pointers (Thumb bit set)."""
        return self.functions[name]


SYMBOLS = Symbols()

# ---------------------------------------------------------------------------
# struct Pokemon (100 bytes) / BoxPokemon (80 bytes)
# ---------------------------------------------------------------------------

POKEMON_SIZE = 100
BOX_SIZE = 80

# GetSubstruct: SUBSTRUCT_ORDER[personality % 24][type] = slot holding that substruct type
SUBSTRUCT_ORDER = [
    (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 3, 1, 2), (0, 2, 3, 1), (0, 3, 2, 1),
    (1, 0, 2, 3), (1, 0, 3, 2), (2, 0, 1, 3), (3, 0, 1, 2), (2, 0, 3, 1), (3, 0, 2, 1),
    (1, 2, 0, 3), (1, 3, 0, 2), (2, 1, 0, 3), (3, 1, 0, 2), (2, 3, 0, 1), (3, 2, 0, 1),
    (1, 2, 3, 0), (1, 3, 2, 0), (2, 1, 3, 0), (3, 1, 2, 0), (2, 3, 1, 0), (3, 2, 1, 0),
]


@dataclass
class PartyMon:
    personality: int
    ot_id: int
    species: int
    held_item: int
    experience: int
    pp_bonuses: int
    friendship: int
    moves: List[int]
    pp: List[int]
    evs: List[int]          # HP Atk Def Spe SpA SpD
    ivs: List[int]          # HP Atk Def Spe SpA SpD
    ability_num: int
    is_egg: bool
    status: int
    level: int
    hp: int
    max_hp: int
    stats: List[int]        # Atk Def Spe SpA SpD
    checksum_ok: bool = True

    @property
    def nature(self):
        return self.personality % 25

    @property
    def is_empty(self):
        return self.species == 0


def _decrypt_secure(raw, personality, ot_id):
    key = personality ^ ot_id
    words = struct.unpack("<12I", raw)
    return struct.pack("<12I", *(w ^ key for w in words))


def _checksum(secure):
    return sum(struct.unpack("<24H", secure)) & 0xFFFF


def decode_pokemon(data: bytes) -> PartyMon:
    assert len(data) == POKEMON_SIZE
    personality, ot_id = struct.unpack_from("<II", data, 0)
    (checksum,) = struct.unpack_from("<H", data, 0x1C)
    secure = _decrypt_secure(data[0x20:0x50], personality, ot_id)
    order = SUBSTRUCT_ORDER[personality % 24]
    sub = [secure[order[t] * 12:order[t] * 12 + 12] for t in range(4)]

    species, item, exp, pp_bonuses, friendship = struct.unpack_from("<HHIBB", sub[0])
    moves = list(struct.unpack_from("<4H", sub[1]))
    pp = list(sub[1][8:12])
    evs = list(sub[2][0:6])
    (iv_word,) = struct.unpack_from("<I", sub[3], 4)
    ivs = [(iv_word >> (5 * i)) & 31 for i in range(6)]

    status, level, _mail, hp, max_hp = struct.unpack_from("<IBBHH", data, 0x50)
    stats = list(struct.unpack_from("<5H", data, 0x5A))
    return PartyMon(personality, ot_id, species, item, exp, pp_bonuses, friendship, moves, pp, evs, ivs,
                    (iv_word >> 31) & 1, bool((iv_word >> 30) & 1), status, level, hp, max_hp, stats,
                    checksum_ok=(_checksum(secure) == checksum))


def decode_party(data: bytes) -> List[PartyMon]:
    return [decode_pokemon(data[i:i + POKEMON_SIZE]) for i in range(0, len(data), POKEMON_SIZE)]


# ---------------------------------------------------------------------------
# struct BattlePokemon (0x58 bytes, unencrypted) — gBattleMons
# ---------------------------------------------------------------------------

BATTLE_MON_SIZE = 0x58


@dataclass
class BattleMon:
    species: int
    stats: List[int]            # Atk Def Spe SpA SpD
    moves: List[int]
    ivs: List[int]
    ability_num: int
    stat_stages: List[int]      # HP Atk Def Spe SpA SpD Acc Eva, 6 = neutral
    ability: int
    types: List[int]
    pp: List[int]
    hp: int
    level: int
    friendship: int
    max_hp: int
    item: int
    pp_bonuses: int
    experience: int
    personality: int
    status1: int
    status2: int
    ot_id: int


def decode_battle_mon(data: bytes) -> BattleMon:
    assert len(data) == BATTLE_MON_SIZE
    species, *stats = struct.unpack_from("<6H", data, 0)
    moves = list(struct.unpack_from("<4H", data, 0x0C))
    (iv_word,) = struct.unpack_from("<I", data, 0x14)
    stages = list(data[0x18:0x20])
    ability, t1, t2 = data[0x20], data[0x21], data[0x22]
    pp = list(data[0x24:0x28])
    hp, level, friendship, max_hp, item = struct.unpack_from("<HBBHH", data, 0x28)
    pp_bonuses = data[0x3B]
    exp, personality, status1, status2, ot_id = struct.unpack_from("<5I", data, 0x44)
    return BattleMon(species, stats, moves, [(iv_word >> (5 * i)) & 31 for i in range(6)], (iv_word >> 31) & 1,
                     stages, ability, [t1, t2], pp, hp, level, friendship, max_hp, item, pp_bonuses, exp,
                     personality, status1, status2, ot_id)


# ---------------------------------------------------------------------------
# SaveBlock2 frontier fields (offsets from *gSaveBlock2Ptr; include/global.h)
# ---------------------------------------------------------------------------

SB2_FRONTIER_CHALLENGE_STATUS = 0xCA8
SB2_FRONTIER_LVL_MODE = 0xCA9            # bits 0-1: lvlMode, bits 2-?: see global.h
SB2_FRONTIER_CUR_BATTLE_NUM = 0xCB2
SB2_FRONTIER_TRAINER_IDS = 0xCB4         # u16[20]
SB2_FRONTIER_WIN_STREAK_ACTIVE = 0xCDC
SB2_TOWER_WIN_STREAKS = 0xCE0            # u16[4][2]
SB2_FACTORY_WIN_STREAKS = 0xDE2          # u16[2][2] [battleMode][lvlMode]
SB2_FACTORY_RECORD_WIN_STREAKS = 0xDEA
SB2_FACTORY_RENTS_COUNT = 0xDF2          # u16[2][2] (global.h comment says 0xDF6; the real offset is 0xDF2)
SB2_RENTAL_MONS = 0xE70                  # struct RentalMon[6], 12 bytes


@dataclass
class RentalMon:
    mon_id: int          # index into gBattleFrontierMons (0xFFFF = empty)
    personality: int
    ivs: int
    ability_num: int


def decode_rental_mons(data: bytes) -> List[RentalMon]:
    out = []
    for i in range(6):
        mon_id, _pad, personality, ivs, ability_num = struct.unpack_from("<HHIBB", data, i * 12)
        out.append(RentalMon(mon_id, personality, ivs, ability_num))
    return out
