"""Hidden information of a Factory battle, sampled from what the player knows (search determinizations).

The opponent's team comes from GenerateOpponentMons (battle_factory.c): three draws with rejection from the round's
pool of Battle Frontier sets,
    pool      GetFactoryMonId(lvlMode, challengeNum): open level uses sInitialRentalMonRanges[8 + min(c, 7)]:
              c = 0..3 (rounds 1-4) the 96-set ranges 372-467 / 468-563 / 564-659 / 660-755 (exactly one set per
              species, so seeing the species reveals the set); c >= 4 (round 5 on) ids 372-881 (109 species,
              up to 10 sets each). Level 50 uses sInitialRentalMonRanges[c] and ids <= FRONTIER_MONS_HIGH_TIER.
    rejects   Unown; a species of any of the 6 Pokemon in frontier.rentalMons when the opponent was generated;
              a species already on the team; a held item (not none) already on the team.
and filled by FillFactoryFrontierTrainerParty (battle_tower.c) with the fixed IV of GetFactoryMonFixedIV at
Battle Tower's level-50 challenge number (a retail bug; 0 in the simulator): 3 in battles 1-6, 6 in the 7th.

What frontier.rentalMons holds when the opponent is generated (src/gen3/factory_run.c follows the map scripts):
    battle 1 of a round   the 6 rental candidates (the opponent is generated right after them; hint shown)
    later battles         our team before the swap + the team we just beat (the swap screen shows both)
so the species to exclude are exactly what the player saw on the rental / swap screen: `ExclusionTracker` keeps
them. Without it, pass at least the species of our current team (a subset: the candidates stay a superset).

Noland (factory_info.brain_status != 0; battles 21 and 42, then every 21) is FillFactoryBrainParty: the same pool
and draws, but it rejects rental *set ids* (not species), and the IV is GetFactoryMonFixedIV(c + 2, FALSE) (15 at
battle 21, 31 at 42 and later). It is sampled with those rules; the ids to reject are only those of our own
Pokemon that we know (`own_ids`), not those of the team beaten before (their exact sets are not known), so
Noland's candidates are a slight superset of the true ones. The draw is otherwise the same as the game's: nothing
of his team is taken from the true state in legal mode.

Per enemy slot:
    seen      sets of its species in the pool whose moves include every revealed move and whose item is the
              revealed item (if one was revealed), and whose item is not another team member's revealed item
    unseen    any set of the pool passing the rejections above against our exclusions and the rest of the team
Teams are drawn slot by slot (0, 1, 2, the game's order), each slot uniform over the sets passing both the
game's rejections given the earlier slots and the observations; `sample_teams` importance-resamples several such
draws with the weight prod_i |valid and consistent_i| / |valid_i| so the team follows the game's posterior
given the observations (up to the AI's move choices, which are not used as evidence).
    ability   the revealed one, else a fair coin (personality bit)
    HP        seen and alive: an exact HP drawn uniformly among those that show the same 48-pixel HP bar for the
              drawn set's max HP (passed as hp / max_hp; Gen3Game.determinize rounds max_hp * f);
              unseen: full; fainted: kept as they are
"""

import math
import random

import numpy as np

from pybattle.view import _DATA, HP_BAR_PIXELS, NAMES, SPECIES, hp_bar_pixels, possible_abilities

FRONTIER_MONS = _DATA["frontier_mons"]
HELD_ITEMS = _DATA["frontier_held_items"]
RANGES = _DATA["factory_rental_ranges"]            # sInitialRentalMonRanges: 8 level-50 rows, then 8 open level
FIXED_IVS = _DATA["factory_fixed_ivs"]              # sFixedIVTable, plus the row read out of bounds at index 8
HIGH_TIER = 849                                     # FRONTIER_MONS_HIGH_TIER
_SPECIES_ID = {v: int(k) for k, v in NAMES["species"].items()}
SPECIES_UNOWN = _SPECIES_ID["SPECIES_UNOWN"]
SPECIES_SHEDINJA = _SPECIES_ID["SPECIES_SHEDINJA"]
KEEP = -1

SET_SPECIES = np.array([m["species"] for m in FRONTIER_MONS], np.int32)
SET_ITEM = np.array([HELD_ITEMS[m["item_table_id"]] for m in FRONTIER_MONS], np.int32)
SET_MOVES = [frozenset(x for x in m["moves"] if x) for m in FRONTIER_MONS]


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
    """The IV of every stat of every opponent Pokemon in this battle (see the module docstring)."""
    if ctx.get("brain"):
        return fixed_iv(ctx["challenge"] + 2, False)
    return fixed_iv(ctx.get("tower_challenge", 0), ctx["battle"] == 6)


def max_hp(set_id: int, iv: int, level: int = 100) -> int:
    """CreateMonWithEVSpreadNatureOTID's HP: the set's EV spread splits 510 EVs evenly."""
    m = FRONTIER_MONS[set_id]
    sp = m["species"]
    if sp == SPECIES_SHEDINJA:
        return 1
    spread = m["ev_spread"]
    n = bin(spread & 0x3F).count("1")
    ev = (510 // n) if (n and spread & 1) else 0
    base = SPECIES[sp]["base"][0]
    return (2 * base + iv + ev // 4) * level // 100 + level + 10


class ExclusionTracker:
    """What frontier.rentalMons held when the current opponent was generated, from the screens the player saw.
    Call on_rental(view) at each RENTAL decision and on_swap(view, own_ids) at each SWAP decision."""

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


def _consistent(slot_view, ids, other_revealed_items):
    """Mask over `ids`: sets consistent with what was seen of this slot."""
    m = SET_SPECIES[ids] == slot_view.species
    if slot_view.revealed_item is not None:
        m &= SET_ITEM[ids] == slot_view.revealed_item
    for it in other_revealed_items:
        if it:
            m &= SET_ITEM[ids] != it
    rev = set(x for x in slot_view.revealed_moves if x)
    if rev:
        m &= np.fromiter((rev <= SET_MOVES[i] for i in ids), bool, len(ids))
    return m


def candidates(view, ctx, own_species=(), own_ids=(), open_level=True):
    """Per enemy slot 0-2: the set ids consistent with what the player knows, before the team constraints between
    slots (species / items all different), which sample_teams applies. Seen slots: sets of its species; unseen:
    the pool minus the excluded species (or our known ids for Noland) and the seen species."""
    ids = pool(ctx["challenge"], open_level)
    brain = bool(ctx.get("brain"))
    enemy = view.enemy_party[:3]
    revealed_items = [m.revealed_item if m.seen else None for m in enemy]
    seen_species = {m.species for m in enemy if m.seen}
    out = []
    for i, m in enumerate(enemy):
        if m.seen:
            others = [it for j, it in enumerate(revealed_items) if j != i and it is not None]
            c = ids[_consistent(m, ids, others)]
            if brain and len(own_ids):
                c = c[~np.isin(c, list(own_ids))]
            if len(c) == 0:                            # should not happen: relax the item, then the moves
                c = ids[SET_SPECIES[ids] == m.species]
            if len(c) == 0:
                c = np.flatnonzero(SET_SPECIES == m.species)
            out.append(c)
        else:
            mask = ~np.isin(SET_SPECIES[ids], list(seen_species))
            if brain:
                mask &= ~np.isin(ids, list(own_ids))
            else:
                mask &= ~np.isin(SET_SPECIES[ids], list(own_species))
            for it in revealed_items:
                if it:
                    mask &= SET_ITEM[ids] != it
            out.append(ids[mask])
    return out


def _draw_team(cands, prior_pool, excl_species, excl_ids, brain, rng):
    """One team drawn slot by slot; returns (ids, log weight)."""
    team, logw = [], 0.0
    used_sp, used_it = [], []
    for c in cands:
        # the game's valid set at this step (prior), and its part consistent with the observations
        def ok(arr):
            m = np.ones(len(arr), bool)
            if used_sp:
                sp, it = SET_SPECIES[arr], SET_ITEM[arr]
                for x in used_sp:
                    m &= sp != x
                for x in used_it:
                    if x:
                        m &= it != x
            return m
        valid_c = c[ok(c)]
        if len(valid_c) == 0:
            valid_c = c                                # inconsistent earlier draw: keep going, weight it down
            logw -= 50.0
        pm = ok(prior_pool)
        n_prior = int(pm.sum())
        pick = int(valid_c[rng.randrange(len(valid_c))])
        logw += math.log(len(valid_c)) - math.log(max(n_prior, 1))
        team.append(pick)
        used_sp.append(int(SET_SPECIES[pick]))
        used_it.append(int(SET_ITEM[pick]))
    return team, logw


def sample_team(view, ctx, own_species=(), rng=None, own_ids=(), open_level=True, n_draws=16):
    """Set ids for enemy slots 0-2 drawn from the posterior given the view (see the module docstring)."""
    rng = rng or random.Random()
    cands = candidates(view, ctx, own_species, own_ids, open_level)
    brain = bool(ctx.get("brain"))
    base = pool(ctx["challenge"], open_level)
    if brain:
        prior_pool = base[~np.isin(base, list(own_ids))]
    else:
        prior_pool = base[~np.isin(SET_SPECIES[base], list(own_species))]
    draws = [_draw_team(cands, prior_pool, own_species, own_ids, brain, rng) for _ in range(n_draws)]
    lw = np.array([d[1] for d in draws])
    w = np.exp(lw - lw.max())
    r = rng.random() * w.sum()
    k = int(np.searchsorted(np.cumsum(w), r, side="right"))
    return draws[min(k, n_draws - 1)][0]


def _hp_fraction(set_id, iv, pixels, rng):
    mh = max_hp(set_id, iv)
    if pixels >= HP_BAR_PIXELS:
        return 1.0
    hs = [h for h in range(1, mh + 1) if hp_bar_pixels(h, mh) == pixels]
    if not hs:
        return max(pixels, 1) / HP_BAR_PIXELS
    return hs[rng.randrange(len(hs))] / mh          # determinize: HP = round(max_hp * f)


def sample_determinization(view, ctx, own_species=(), rng=None, own_ids=(), open_level=True):
    """Slot specs for Gen3Game.determinize: [(party_slot, frontier_set_id or -1, iv, ability_bit, hp_fraction or -1)].

    `ctx`: {"challenge": streak // 7, "battle": 0..6, "brain": Noland?}; `own_species`: the species excluded by
    GenerateOpponentMons (ExclusionTracker.species; at least our team's); `own_ids`: our known set ids (Noland)."""
    rng = rng or random.Random()
    own_species = set(own_species) | {m.species for m in view.own_party[:3]}
    team = sample_team(view, ctx, own_species, rng, own_ids, open_level)
    iv = enemy_iv(ctx)
    specs = []
    for i, m in enumerate(view.enemy_party[:3]):
        sid = team[i]
        if m.seen and m.fainted:
            specs.append((i, KEEP, iv, 0, KEEP))
            continue
        sp = int(SET_SPECIES[sid])
        abil = possible_abilities(sp)
        a = SPECIES[sp]["abilities"]
        if m.seen and m.revealed_ability:
            bit = 1 if (a[1] and m.revealed_ability == a[1] and a[1] != a[0]) else 0
        else:
            bit = rng.randrange(2) if len(abil) > 1 else 0
        hp = _hp_fraction(sid, iv, m.hp_pixels, rng) if m.seen else 1.0
        specs.append((i, sid, iv, bit, hp))
    return specs
