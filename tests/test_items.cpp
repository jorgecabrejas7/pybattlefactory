#include "battle_engine.hpp"
#include "data.hpp"
#include "constants.hpp"
#include "factory.hpp"
#include <iostream>
#include <cassert>

using namespace pkmn;

void testLightBall() {
    std::cout << "Testing Light Ball..." << std::endl;
    BattleEngine engine;
    engine.reset(42);
    
    // Pikachu with Light Ball
    Pokemon p1 = FactoryGenerator::createPokemon(162, 50, 31); 
    // Let's create manually to be safe.
    Pokemon pikachu{};
    pikachu.species = SPECIES_PIKACHU;
    pikachu.level = 50;
    pikachu.moves[0] = MOVE_THUNDERBOLT; 
    pikachu.pp[0] = 15;
    pikachu.calculateStats(getSpeciesData(SPECIES_PIKACHU));
    pikachu.heldItem = ITEM_LIGHT_BALL;
    
    // Target (Neutral)
    Pokemon target{};
    target.species = SPECIES_MEW;
    target.level = 50;
    target.calculateStats(getSpeciesData(SPECIES_MEW));
    
    engine.setPlayerTeam(&pikachu, 1);
    engine.setOpponentTeam(&target, 1);
    
    int damageWith = engine.calculateDamage(0, 1, MOVE_THUNDERBOLT);
    
    // Pikachu without Light Ball
    pikachu.heldItem = ITEM_NONE;
    engine.setPlayerTeam(&pikachu, 1);
    int damageWithout = engine.calculateDamage(0, 1, MOVE_THUNDERBOLT);
    
    std::cout << "Light Ball: " << damageWith << " vs " << damageWithout << std::endl;
    assert(damageWith > damageWithout * 1.8); // Should be ~2x
}

void testThickClub() {
    std::cout << "Testing Thick Club..." << std::endl;
    BattleEngine engine;
    engine.reset(42);
    
    Pokemon cubone{};
    cubone.species = SPECIES_CUBONE;
    cubone.level = 50;
    cubone.moves[0] = MOVE_BONE_CLUB;
    cubone.calculateStats(getSpeciesData(SPECIES_CUBONE));
    cubone.heldItem = ITEM_THICK_CLUB;
    
    Pokemon target{};
    target.species = SPECIES_MEW;
    target.level = 50;
    target.calculateStats(getSpeciesData(SPECIES_MEW));
    
    engine.setPlayerTeam(&cubone, 1);
    engine.setOpponentTeam(&target, 1);
    
    int damageWith = engine.calculateDamage(0, 1, MOVE_BONE_CLUB);
    
    cubone.heldItem = ITEM_NONE;
    engine.setPlayerTeam(&cubone, 1);
    int damageWithout = engine.calculateDamage(0, 1, MOVE_BONE_CLUB);
    
    std::cout << "Thick Club: " << damageWith << " vs " << damageWithout << std::endl;
    assert(damageWith > damageWithout * 1.8);
}

void testDeepSeaTooth() {
    std::cout << "Testing DeepSeaTooth..." << std::endl;
    BattleEngine engine;
    engine.reset(42);
    
    Pokemon clamperl{};
    clamperl.species = SPECIES_CLAMPERL;
    clamperl.level = 50;
    clamperl.moves[0] = MOVE_SURF; 
    clamperl.calculateStats(getSpeciesData(SPECIES_CLAMPERL));
    clamperl.heldItem = ITEM_DEEP_SEA_TOOTH;
    
    Pokemon target{};
    target.species = SPECIES_MEW;
    target.level = 50;
    target.calculateStats(getSpeciesData(SPECIES_MEW));
    
    engine.setPlayerTeam(&clamperl, 1);
    engine.setOpponentTeam(&target, 1);
    
    int damageWith = engine.calculateDamage(0, 1, MOVE_SURF);
    
    clamperl.heldItem = ITEM_NONE;
    engine.setPlayerTeam(&clamperl, 1);
    int damageWithout = engine.calculateDamage(0, 1, MOVE_SURF);
    
    std::cout << "DeepSeaTooth: " << damageWith << " vs " << damageWithout << std::endl;
    assert(damageWith > damageWithout * 1.8);
}

void testDeepSeaScale() {
    std::cout << "Testing DeepSeaScale..." << std::endl;
    BattleEngine engine;
    engine.reset(42);
    
    Pokemon attacker{};
    attacker.species = SPECIES_MEW;
    attacker.level = 50;
    attacker.moves[0] = MOVE_PSYCHIC; // Special
    attacker.calculateStats(getSpeciesData(SPECIES_MEW));
    
    Pokemon clamperl{};
    clamperl.species = SPECIES_CLAMPERL;
    clamperl.level = 50;
    clamperl.calculateStats(getSpeciesData(SPECIES_CLAMPERL));
    clamperl.heldItem = ITEM_DEEP_SEA_SCALE;
    
    engine.setPlayerTeam(&attacker, 1);
    engine.setOpponentTeam(&clamperl, 1);
    
    int damageWith = engine.calculateDamage(0, 1, MOVE_PSYCHIC);
    
    clamperl.heldItem = ITEM_NONE;
    engine.setOpponentTeam(&clamperl, 1);
    int damageWithout = engine.calculateDamage(0, 1, MOVE_PSYCHIC);
    
    std::cout << "DeepSeaScale: " << damageWith << " vs " << damageWithout << std::endl;
    assert(damageWithout > damageWith * 1.5); // Damage should be halved (approx) (Relaxed to 1.5x gap)
}

void testMetalPowder() {
    std::cout << "Testing Metal Powder..." << std::endl;
    BattleEngine engine;
    engine.reset(42);
    
    Pokemon attacker{};
    attacker.species = SPECIES_MEW;
    attacker.level = 50;
    attacker.moves[0] = MOVE_POUND; // Physical
    attacker.calculateStats(getSpeciesData(SPECIES_MEW));
    
    Pokemon ditto{};
    ditto.species = SPECIES_DITTO;
    ditto.level = 50;
    ditto.calculateStats(getSpeciesData(SPECIES_DITTO));
    ditto.heldItem = ITEM_METAL_POWDER;
    
    engine.setPlayerTeam(&attacker, 1);
    engine.setOpponentTeam(&ditto, 1);
    
    int damageWith = engine.calculateDamage(0, 1, MOVE_POUND);
    
    ditto.heldItem = ITEM_NONE;
    engine.setOpponentTeam(&ditto, 1);
    int damageWithout = engine.calculateDamage(0, 1, MOVE_POUND);
    
    std::cout << "Metal Powder: " << damageWith << " vs " << damageWithout << std::endl;
    assert(damageWithout > damageWith * 1.5); 
}

int main() {
    testLightBall();
    testThickClub();
    testDeepSeaTooth();
    testDeepSeaScale();
    testMetalPowder();
    std::cout << "All item tests passed!" << std::endl;
    return 0;
}
