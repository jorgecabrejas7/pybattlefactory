#include <iostream>
#include <cassert>
#include <vector>
#include "factory.hpp"
#include "data.hpp"
#include "constants.hpp"

// Simple test runner
#define ASSERT(cond, msg) \
    if (!(cond)) { \
        std::cerr << "Test failed: " << msg << std::endl; \
        std::exit(1); \
    }

void test_challenge_ranges() {
    std::cout << "Testing challenge ranges..." << std::endl;
    uint16_t start, end;
    
    // Level 50, Chal 0
    pkmn::FactoryGenerator::getChallengeRanges(0, false, start, end);
    ASSERT(start == 0 && end == 161, "Chal 0 Lvl 50 range incorrect");
    
    // Open Level, Chal 7
    pkmn::FactoryGenerator::getChallengeRanges(7, true, start, end);
    ASSERT(start == 660 && end == 881, "Chal 7 Open range incorrect");
    
    std::cout << "Range tests passed!" << std::endl;
}

void test_rental_generation() {
    std::cout << "Testing rental generation..." << std::endl;
    uint32_t seed = 42;
    auto pool = pkmn::FactoryGenerator::generateRentalPool(seed, 0, false);
    
    ASSERT(pool.size() == 6, "Rental pool must have 6 mons");
    
    std::cout << "Rental Pool: ";
    for (uint16_t id : pool) {
        ASSERT(id <= 161, "Rental mon ID out of range for Chal 0");
        std::cout << id << " ";
        const auto& mon = pkmn::getFrontierMon(id);
        const auto& spec = pkmn::getSpeciesData(mon.species);
        // Basic check that species is valid
        ASSERT(mon.species > 0 && mon.species < 387, "Invalid species ID");
    }
    std::cout << std::endl;
}

void test_opponent_generation() {
    std::cout << "Testing opponent generation..." << std::endl;
    uint32_t seed = 123;
    std::vector<uint16_t> excludes = {0, 1}; // Exclude first two mons
    auto team = pkmn::FactoryGenerator::generateOpponentTeam(seed, 0, 0, false, excludes);
    
    ASSERT(team.size() == 3, "Opponent team must have 3 mons");
    for (uint16_t id : team) {
        ASSERT(id != 0 && id != 1, "Exclusion failed");
    }
}

void test_pokemon_conversion() {
    std::cout << "Testing pokemon conversion..." << std::endl;
    // Test with Sunkern (ID 0)
    // {SPECIES_SUNKERN, {MOVE_MEGA_DRAIN, MOVE_HELPING_HAND, MOVE_SUNNY_DAY, MOVE_LIGHT_SCREEN}, 58, 17, Nature::Relaxed}
    // Item 58 = LAX_INCENSE ? Wait, check constants.
    // 17 = 00010001(binary)? No, 17 = 16 + 1 = 0x11. EV bits?
    // Let's just check stats are calculated.
    
    pkmn::Pokemon p = pkmn::FactoryGenerator::createPokemon(0, 50, 31);
    
    ASSERT(p.species == pkmn::SPECIES_SUNKERN, "Species should be Sunkern"); 
    // Wait, enum is sequential. SPECIES_SUNKERN is around 190? 
    // Check types.hpp/constants.hpp. 
    // Actually, I can check against getFrontierMon(0).species.
    
    const auto& fm = pkmn::getFrontierMon(0);
    ASSERT(p.species == fm.species, "Species mismatch");
    ASSERT(p.level == 50, "Level mismatch");
    ASSERT(p.moves[0] == fm.moves[0], "Move 0 mismatch");
    ASSERT(p.pp[0] > 0, "PP not set");
    ASSERT(p.ivs[0] == 31, "IV mismatch");
    
    std::cout << "Stats for species " << p.species << ": HP=" << p.stats[0] << std::endl;
    ASSERT(p.stats[0] > 0, "HP calculation failed");
}

int main() {
    test_challenge_ranges();
    test_rental_generation();
    test_opponent_generation();
    test_pokemon_conversion();
    std::cout << "All factory tests passed!" << std::endl;
    return 0;
}
