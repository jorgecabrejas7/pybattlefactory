#pragma once

#include "types.hpp"
#include "constants.hpp"
#include <cstddef>

namespace pkmn {

// ============================================================================
// Static Data Access
// ============================================================================

/// Get species base stats by ID
const SpeciesData& getSpeciesData(uint16_t speciesId);

/// Get move data by ID
const MoveData& getMoveData(uint16_t moveId);

/// Get type effectiveness: returns 0, 25, 50, 100, 200, or 400 (x100 to avoid floats)
int getTypeEffectiveness(Type attackType, Type defendType);

/// Get combined type effectiveness for dual-type defender
int getTypeEffectivenessDual(Type attackType, Type defType1, Type defType2);

// ============================================================================
// Frontier Mon Data
//
// Data extracted from pokeemerald/src/data/battle_frontier/battle_frontier_mons.h
// These represent the 882 pre-defined Pokemon used in Battle Factory.
// ============================================================================

/// Frontier mon entry (maps to FacilityMon in decompilation)
/// Each frontier mon has pre-set moves, nature, EVs, and held item.
struct FrontierMon {
    uint16_t species;     // Species ID (SPECIES_BULBASAUR = 1, etc)
    uint16_t moves[4];    // Move IDs (MOVE_NONE = 0 for empty slots)
    uint8_t itemTableId;  // Index into frontier item table (0-62)
                          // Use getFrontierItem() to get actual ITEM_* constant
                          // Common values: 0=None, 2=Sitrus, 6=Focus Band, 25=Leftovers
    uint8_t evSpread;     // Bitfield: which stats get EVs (510 total, divided evenly)
                          //   Bit 0 (0x01): HP
                          //   Bit 1 (0x02): Attack
                          //   Bit 2 (0x04): Defense
                          //   Bit 3 (0x08): Speed
                          //   Bit 4 (0x10): Sp. Attack
                          //   Bit 5 (0x20): Sp. Defense
                          // Example: 0x12 = Attack + Sp.Atk = 255 EVs each
    Nature nature;        // Nature (affects stat growth by ±10%)
};

/// Get frontier mon by ID (0-881)
const FrontierMon& getFrontierMon(uint16_t frontierMonId);

/// Get held item from frontier item table
uint16_t getFrontierItem(uint8_t itemTableId);

/// Number of frontier mons
constexpr size_t NUM_FRONTIER_MONS = 882;

// ============================================================================
// EV Spread Flags - defined in constants.hpp
// ============================================================================

/// Convert evSpread bitfield to actual EV array (510 total, distributed evenly)
void calculateEVsFromSpread(uint8_t evSpread, uint8_t* outEVs);

// ============================================================================
// Nature Modifiers
// ============================================================================

/// Nature modifier table access
/// Returns 90, 100, or 110 (x100 to avoid floats)
int getNatureModifier(Nature nature, Stat stat);

}  // namespace pkmn
