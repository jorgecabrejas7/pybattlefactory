#include "factory.hpp"
#include <algorithm>
#include <set>

namespace pkmn {

// Tier ranges by challenge number (from battle_factory.c:sInitialRentalMonRanges)
// [challengeNum][0] = start, [challengeNum][1] = end (inclusive)
static constexpr uint16_t INITIAL_RENTAL_RANGES_LVL50[8][2] = {
    {0, 161},     // Challenge 0
    {162, 266},   // Challenge 1
    {162, 371},   // Challenge 2
    {267, 467},   // Challenge 3
    {372, 563},   // Challenge 4
    {468, 659},   // Challenge 5
    {468, 659},   // Challenge 6
    {564, 881},   // Challenge 7+ (includes Noland)
};

static constexpr uint16_t INITIAL_RENTAL_RANGES_OPEN[8][2] = {
    {102, 267},   // Challenge 0
    {162, 371},   // Challenge 1
    {267, 467},   // Challenge 2
    {372, 563},   // Challenge 3
    {468, 659},   // Challenge 4
    {564, 755},   // Challenge 5
    {564, 755},   // Challenge 6
    {660, 881},   // Challenge 7+
};

// IVs for each challenge number (approximate based on Gen 3 mechanic)
// 0=0, 1=31? No, actually:
// Chal 1: 3, Chal 2: 6, Chal 3: 9, Chal 4: 12, Chal 5: 15, Chal 6: 21, Chal 7: 31
// We will just use a simplified lookup or param.
static const uint8_t CHALLENGE_IVS[] = {3, 6, 9, 12, 15, 21, 31, 31};

// Simple LCG for internal use
static uint32_t nextRandom(uint32_t& seed) {
    seed = seed * 1103515245 + 12345;
    return (seed / 65536) % 32768;
}

void FactoryGenerator::getChallengeRanges(int challengeNum, bool isOpenLevel, uint16_t& outStart, uint16_t& outEnd) {
    int idx = std::min(challengeNum, 7);
    if (isOpenLevel) {
        outStart = INITIAL_RENTAL_RANGES_OPEN[idx][0];
        outEnd = INITIAL_RENTAL_RANGES_OPEN[idx][1];
    } else {
        outStart = INITIAL_RENTAL_RANGES_LVL50[idx][0];
        outEnd = INITIAL_RENTAL_RANGES_LVL50[idx][1];
    }
}

std::vector<uint16_t> FactoryGenerator::generateRentalPool(uint32_t& rngSeed, int challengeNum, bool isOpenLevel) {
    uint16_t start, end;
    getChallengeRanges(challengeNum, isOpenLevel, start, end);
    
    std::vector<uint16_t> pool;
    std::set<uint16_t> usedSpecies;
    
    // We need 6 valid mons
    int attempts = 0;
    while (pool.size() < 6 && attempts < 1000) {
        attempts++;
        uint16_t range = end - start + 1;
        uint16_t id = start + (nextRandom(rngSeed) % range);
        
        // Check uniqueness by ID (allow same species? Gen 3 prevents same species in rental?)
        // Let's enforce unique species for rentals to be safe
        const FrontierMon& mon = getFrontierMon(id);
        if (usedSpecies.count(mon.species)) continue;
        
        pool.push_back(id);
        usedSpecies.insert(mon.species);
    }
    
    return pool;
}

std::vector<uint16_t> FactoryGenerator::generateOpponentTeam(uint32_t& rngSeed, int challengeNum, int /*battleNum*/, bool isOpenLevel,
                                                  const std::vector<uint16_t>& playerExcludes) {
    uint16_t start, end;
    getChallengeRanges(challengeNum, isOpenLevel, start, end);
    
    std::vector<uint16_t> team;
    std::set<uint16_t> usedSpecies;
    
    // Add excludes to used set
    for (uint16_t id : playerExcludes) {
        usedSpecies.insert(getFrontierMon(id).species);
    }
    
    int attempts = 0;
    while (team.size() < 3 && attempts < 1000) {
        attempts++;
        uint16_t range = end - start + 1;
        uint16_t id = start + (nextRandom(rngSeed) % range);
        
        const FrontierMon& mon = getFrontierMon(id);
        if (usedSpecies.count(mon.species)) continue;
        
        // Also check if id is clearly duplicate in team
        bool idExists = false;
        for (uint16_t existing : team) if (existing == id) idExists = true;
        if (idExists) continue;

        team.push_back(id);
        usedSpecies.insert(mon.species);
    }
    
    return team;
}

Pokemon FactoryGenerator::createPokemon(uint16_t frontierMonId, int level, uint8_t fixedIV) {
    const FrontierMon& fm = getFrontierMon(frontierMonId);
    Pokemon mon{};
    mon.species = fm.species;
    mon.level = level;
    mon.nature = fm.nature;
    
    // Copy moves and get PPs
    for (int i = 0; i < 4; i++) {
        mon.moves[i] = fm.moves[i];
        if (fm.moves[i] != MOVE_NONE) {
            const MoveData& md = getMoveData(fm.moves[i]);
            mon.pp[i] = md.pp;
        } else {
            mon.pp[i] = 0;
        }
    }
    
    // Set IVs
    for (int i = 0; i < 6; i++) mon.ivs[i] = fixedIV;
    
    // Set EVs
    calculateEVsFromSpread(fm.evSpread, mon.evs);
    
    // Set Item
    mon.heldItem = getFrontierItem(fm.itemTableId);
    
    // Ability: Use first ability for now, or maybe random?
    // Let's stick to first ability for determinism unless we want to use PID
    const SpeciesData& sd = getSpeciesData(fm.species);
    mon.ability = sd.abilities[0]; 
    if (mon.ability == ABILITY_NONE) mon.ability = sd.abilities[1]; // Should not happen for slot 1
    
    // Calculate Stats
    mon.calculateStats(sd);
    
    return mon;
}

} // namespace pkmn
