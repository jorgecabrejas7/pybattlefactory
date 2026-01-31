#pragma once

#include "types.hpp"
#include "data.hpp"
#include <vector>
#include <cstdint>

namespace pkmn {

class FactoryGenerator {
public:
    // Generate 6 rental mons for the player to choose from at the start of a challenge
    // Returns indices into the FRONTIER_MONS array
    static std::vector<uint16_t> generateRentalPool(uint32_t& rngSeed, int challengeNum, bool isOpenLevel);

    // Generate 3 mons for an opponent team
    // Returns indices into the FRONTIER_MONS array
    static std::vector<uint16_t> generateOpponentTeam(uint32_t& rngSeed, int challengeNum, int battleNum, bool isOpenLevel,
                                                      const std::vector<uint16_t>& playerExcludes = {});

    // Convert a FrontierMon ID to a full Pokemon instance
    static Pokemon createPokemon(uint16_t frontierMonId, int level, uint8_t fixedIV = 31);
    
    // Helper to get ranges for debugging/UI
    static void getChallengeRanges(int challengeNum, bool isOpenLevel, uint16_t& outStart, uint16_t& outEnd);
};

} // namespace pkmn
