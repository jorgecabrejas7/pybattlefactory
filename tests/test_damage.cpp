#include "battle_engine.hpp"
#include "data.hpp"
#include <iostream>
#include <cassert>

using namespace pkmn;

void testDamageCalculation() {
    std::cout << "Testing damage calculation...\n";
    
    BattleEngine engine;
    engine.reset(12345);
    
    // Create a simple Charizard vs Blastoise battle
    Pokemon charizard{};
    charizard.species = SPECIES_CHARIZARD;
    charizard.level = 50;
    charizard.moves[0] = MOVE_FIRE_PUNCH;
    charizard.moves[1] = MOVE_THUNDER_PUNCH;
    charizard.pp[0] = 15;
    charizard.pp[1] = 15;
    for (int i = 0; i < 6; i++) charizard.ivs[i] = 31;
    charizard.nature = Nature::Adamant;
    charizard.calculateStats(getSpeciesData(SPECIES_CHARIZARD));
    
    Pokemon blastoise{};
    blastoise.species = SPECIES_BLASTOISE;
    blastoise.level = 50;
    blastoise.moves[0] = MOVE_POUND;  // Placeholder
    blastoise.pp[0] = 35;
    for (int i = 0; i < 6; i++) blastoise.ivs[i] = 31;
    blastoise.nature = Nature::Modest;
    blastoise.calculateStats(getSpeciesData(SPECIES_BLASTOISE));
    
    engine.setPlayerTeam(&charizard, 1);
    engine.setOpponentTeam(&blastoise, 1);
    
    // Test damage calculation
    int damage = engine.calculateDamage(0, 1, MOVE_FIRE_PUNCH);
    std::cout << "Fire Punch damage: " << damage << "\n";
    
    // Fire should be not very effective against Water
    assert(damage > 0 && damage < 50);  // Should be low due to type disadvantage
    
    damage = engine.calculateDamage(0, 1, MOVE_THUNDER_PUNCH);
    std::cout << "Thunder Punch damage: " << damage << "\n";
    
    // Electric should be super effective against Water
    assert(damage > 50);  // Should be higher
    
    std::cout << "Damage tests passed!\n";
}

void testTypeEffectiveness() {
    std::cout << "Testing type effectiveness...\n";
    
    // Fire vs Grass = 2x
    assert(getTypeEffectiveness(Type::Fire, Type::Grass) == 200);
    
    // Fire vs Water = 0.5x
    assert(getTypeEffectiveness(Type::Fire, Type::Water) == 50);
    
    // Normal vs Ghost = 0x
    assert(getTypeEffectiveness(Type::Normal, Type::Ghost) == 0);
    
    // Fighting vs Normal = 2x
    assert(getTypeEffectiveness(Type::Fighting, Type::Normal) == 200);
    
    // Electric vs Ground = 0x
    assert(getTypeEffectiveness(Type::Electric, Type::Ground) == 0);
    
    // Dual type: Electric vs Water/Flying = 4x
    assert(getTypeEffectivenessDual(Type::Electric, Type::Water, Type::Flying) == 400);
    
    // Dual type: Ground vs Fire/Flying = 0x (Flying immune)
    assert(getTypeEffectivenessDual(Type::Ground, Type::Fire, Type::Flying) == 0);
    
    std::cout << "Type effectiveness tests passed!\n";
}

void testStatCalculation() {
    std::cout << "Testing stat calculation...\n";
    
    Pokemon charizard{};
    charizard.species = SPECIES_CHARIZARD;
    charizard.level = 50;
    for (int i = 0; i < 6; i++) {
        charizard.ivs[i] = 31;
        charizard.evs[i] = 0;
    }
    charizard.nature = Nature::Adamant;  // +Atk, -SpA
    charizard.calculateStats(getSpeciesData(SPECIES_CHARIZARD));
    
    std::cout << "Charizard Lv50 stats:\n";
    std::cout << "  HP: " << charizard.stats[HP] << "\n";
    std::cout << "  Attack: " << charizard.stats[Attack] << "\n";
    std::cout << "  Defense: " << charizard.stats[Defense] << "\n";
    std::cout << "  Speed: " << charizard.stats[Speed] << "\n";
    std::cout << "  Sp.Atk: " << charizard.stats[SpAttack] << "\n";
    std::cout << "  Sp.Def: " << charizard.stats[SpDefense] << "\n";
    
    // Verify HP formula: floor((2*78 + 31) * 50/100) + 50 + 10 = floor(187*0.5) + 60 = 93 + 60 = 153
    assert(charizard.stats[HP] == 153);
    
    std::cout << "Stat calculation tests passed!\n";
}

void testBattleFlow() {
    std::cout << "Testing battle flow...\n";
    
    BattleEngine engine;
    engine.reset(42);
    
    Pokemon attacker{};
    attacker.species = SPECIES_CHARIZARD;
    attacker.level = 50;
    attacker.moves[0] = MOVE_FIRE_PUNCH;
    attacker.pp[0] = 15;
    for (int i = 0; i < 6; i++) attacker.ivs[i] = 31;
    attacker.nature = Nature::Adamant;
    attacker.calculateStats(getSpeciesData(SPECIES_CHARIZARD));
    
    Pokemon defender{};
    defender.species = SPECIES_BULBASAUR;
    defender.level = 50;
    defender.moves[0] = MOVE_POUND;
    defender.pp[0] = 35;
    for (int i = 0; i < 6; i++) defender.ivs[i] = 31;
    defender.nature = Nature::Bold;
    defender.calculateStats(getSpeciesData(SPECIES_BULBASAUR));
    
    engine.setPlayerTeam(&attacker, 1);
    engine.setOpponentTeam(&defender, 1);
    
    // Get legal actions
    auto actions = engine.getLegalActions();
    std::cout << "Legal actions: " << actions.size() << "\n";
    assert(!actions.empty());
    
    // Step the battle
    StepResult result = engine.step(actions[0]);
    std::cout << "After step - done: " << result.done << ", reward: " << result.reward << "\n";
    
    // Fire should destroy Bulbasaur quickly (4x effective)
    if (result.done) {
        assert(result.winner == 0);  // Player wins
        std::cout << "Battle ended - player wins!\n";
    }
    
    std::cout << "Battle flow tests passed!\n";
}

int main() {
    std::cout << "=== Pokemon Battle Simulator Tests ===\n\n";
    
    testTypeEffectiveness();
    testStatCalculation();
    testDamageCalculation();
    testBattleFlow();
    
    std::cout << "\nAll tests passed!\n";
    return 0;
}
