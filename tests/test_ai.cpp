#include "battle_engine.hpp"
#include "ai.hpp"
#include "factory.hpp"
#include <iostream>
#include <cassert>

using namespace pkmn;

void test_ai_execution() {
    std::cout << "Testing AI Execution..." << std::endl;

    // 1. Setup Engine
    BattleEngine engine;
    
    // 2. Generate Random Teams
    // We rely on the factory we verified earlier
    engine.reset(12345);
    
    // 2. Generate Teams
    std::cout << "Generating teams..." << std::endl;
    // Create 3 mons for each side
    // We use simple IDs from Chal 0 pool
    std::vector<Pokemon> team1;
    std::vector<Pokemon> team2;
    
    // FactoryGenerator usage
    for(int i=0; i<3; ++i) {
        team1.push_back(FactoryGenerator::createPokemon(i+1, 50, 31)); // IDs 1,2,3
        team2.push_back(FactoryGenerator::createPokemon(i+10, 50, 31)); // IDs 10,11,12
    }
    
    engine.setPlayerTeam(team1.data(), 3);
    engine.setOpponentTeam(team2.data(), 3);
    
    // No start() needed, state is ready after setTeam
    
    // 4. Test AI for Opponent (Battler 1)
    std::cout << "Running chooseAIAction(engine, 1)..." << std::endl;
    
    Action action = chooseAIAction(engine, 1);
    
    std::cout << "AI chose Action Type: " << (int)action.type << std::endl;
    
    if (action.isMove()) {
        std::cout << "AI Selected a Move." << std::endl;
        uint8_t moveIdx = action.getMoveIndex(); // 0-3
        
        const auto& active = engine.getState().getActivePokemon(1);
        
        if (action.type <= ActionType::Move4) {
            std::cout << "Move Index: " << (int)moveIdx << " (ID: " << active.moves[moveIdx] << ")" << std::endl;
        } else {
             std::cout << "Move: Struggle" << std::endl;
        }
    } else if (action.isSwitch()) {
        std::cout << "AI Selected a Switch." << std::endl;
    } else {
        std::cout << "AI Selected Unknown Action." << std::endl;
        exit(1);
    }
    
    std::cout << "Success!" << std::endl;
}

int main() {
    try {
        test_ai_execution();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
