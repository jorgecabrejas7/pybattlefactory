#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "battle_engine.hpp"
#include "factory.hpp"
#include "types.hpp"
#include "constants.hpp"

namespace py = pybind11;
using namespace pkmn;

PYBIND11_MODULE(pybattle_native, m) {
    m.doc() = "Pokemon Emerald Battle Simulator";

    // Enums
    py::enum_<ActionType>(m, "ActionType")
        .value("Move1", ActionType::Move1)
        .value("Move2", ActionType::Move2)
        .value("Move3", ActionType::Move3)
        .value("Move4", ActionType::Move4)
        .value("Switch1", ActionType::Switch1)
        .value("Switch2", ActionType::Switch2)
        .value("Switch3", ActionType::Switch3)
        .value("Switch4", ActionType::Switch4)
        .value("Switch5", ActionType::Switch5)
        .value("Struggle", ActionType::Struggle)
        .export_values();
    
    // Helper to allow implicit conversion from int to Action
    py::class_<Action>(m, "Action")
        .def(py::init<ActionType>())
        .def(py::init([](int type) { return Action{static_cast<ActionType>(type)}; }))
        .def_readwrite("type", &Action::type)
        .def("__repr__", [](const Action& a) {
            return "<Action type=" + std::to_string((int)a.type) + ">";
        });
        
    // Pokemon
    py::class_<Pokemon>(m, "Pokemon")
        .def(py::init<>()) // Empty constructor
        .def_readonly("species", &Pokemon::species)
        .def_readonly("current_hp", &Pokemon::currentHP)
        .def_readonly("max_hp", &Pokemon::maxHP)
        .def_readonly("level", &Pokemon::level)
        .def_readonly("ability", &Pokemon::ability)
        .def_readonly("held_item", &Pokemon::heldItem)
        .def_property_readonly("moves", [](const Pokemon& p) {
            std::vector<uint16_t> moves;
            for(int i=0; i<4; ++i) moves.push_back(p.moves[i]);
            return moves;
        })
        .def_property_readonly("pp", [](const Pokemon& p) {
            std::vector<uint8_t> pp;
            for(int i=0; i<4; ++i) pp.push_back(p.pp[i]);
            return pp;
        });

    py::class_<StepResult>(m, "StepResult")
        .def_readonly("done", &StepResult::done)
        .def_readonly("winner", &StepResult::winner)
        .def_readonly("reward", &StepResult::reward);
        
    // ActiveMon
    py::class_<ActiveMon>(m, "ActiveMon")
        .def_property_readonly("stat_stages", [](const ActiveMon& am) {
            std::vector<int8_t> stages;
            for(int i=0; i<BATTLE_STAT_COUNT; ++i) stages.push_back(am.statStages[i]);
            return stages;
        })
        .def_readonly("is_confused", &ActiveMon::isConfused)
        .def_readonly("is_taunted", &ActiveMon::isTaunted)
        .def_readonly("is_seeded", &ActiveMon::isSeeded)
        .def_readonly("has_substitute", &ActiveMon::hasSubstitute)
        .def_readonly("protected_this_turn", &ActiveMon::protectedThisTurn);
        
    // BattleState
    py::class_<BattleState>(m, "BattleState")
        .def("get_active_pokemon", [](const BattleState& s, int side) -> const Pokemon& { 
            return s.getActivePokemon(side); 
        }, py::return_value_policy::reference)
        .def("get_active", [](const BattleState& s, int side) -> const ActiveMon& {
            return s.active[side];
        }, py::return_value_policy::reference)
        .def_readonly("turn_number", &BattleState::turnNumber)
        .def("is_terminal", &BattleState::isTerminal)
        .def("get_winner", &BattleState::getWinner)
        // Access teams?
        .def("get_player_party_count", [](const BattleState& s) { return s.countRemaining(0); })
        .def("get_opponent_party_count", [](const BattleState& s) { return s.countRemaining(1); });

    // BattleEngine
    py::class_<BattleEngine>(m, "BattleEngine")
        .def(py::init<>())
        .def("reset", &BattleEngine::reset)
        .def("set_player_team", [](BattleEngine& self, const std::vector<Pokemon>& mons) {
            self.setPlayerTeam(mons.data(), mons.size());
        })
        .def("set_opponent_team", [](BattleEngine& self, const std::vector<Pokemon>& mons) {
            self.setOpponentTeam(mons.data(), mons.size());
        })
        .def("get_state", &BattleEngine::getState, py::return_value_policy::reference)
        .def("step", &BattleEngine::step)
        .def("get_legal_actions", &BattleEngine::getLegalActions);

    // Factory Helper
    struct FactoryHelper {
        uint32_t seed;
        FactoryHelper(uint32_t s) : seed(s) {}

        std::vector<Pokemon> generate_opponent_team(int challengeNum, int battleNum, bool isOpenLevel) {
            auto monIds = FactoryGenerator::generateOpponentTeam(seed, challengeNum, battleNum, isOpenLevel);
            std::vector<Pokemon> team;
            int level = isOpenLevel ? 100 : 50; 
            for (uint16_t id : monIds) {
                team.push_back(FactoryGenerator::createPokemon(id, level));
            }
            return team;
        }

        std::vector<Pokemon> generate_player_team(int challengeNum, bool isOpenLevel) {
            auto monIds = FactoryGenerator::generateRentalPool(seed, challengeNum, isOpenLevel);
            std::vector<Pokemon> pool;
            int level = isOpenLevel ? 100 : 50; 
            for (uint16_t id : monIds) {
                pool.push_back(FactoryGenerator::createPokemon(id, level));
            }
            return pool;
        }
    };

    py::class_<FactoryHelper>(m, "FactoryGenerator")
        .def(py::init<uint32_t>()) // seed
        .def("generate_opponent_team", &FactoryHelper::generate_opponent_team)
        .def("generate_player_team", &FactoryHelper::generate_player_team);

    // VecBattleEnv
    py::class_<VecBattleEnv>(m, "VecBattleEnv")
        .def(py::init<size_t>())
        .def("reset", [](VecBattleEnv& self, py::array_t<uint32_t> seeds) {
            py::buffer_info buf = seeds.request();
            if (buf.ndim != 1) throw std::runtime_error("Seeds must be 1D array");
            
            self.reset(static_cast<uint32_t*>(buf.ptr), buf.size);
        })
        .def("step", [](VecBattleEnv& self, py::array_t<uint8_t> actions) {
            // Buffer inputs
            py::buffer_info act_buf = actions.request();
            if (act_buf.ndim != 1) throw std::runtime_error("Actions must be 1D array");
            
            size_t count = act_buf.size;
            
            // Assume contiguous uint8_t actions mapping to Action struct
            const Action* action_ptr = reinterpret_cast<const Action*>(act_buf.ptr);
            
            // Allocate outputs
            auto rewards = py::array_t<float>(count);
            auto dones = py::array_t<bool>(count);
            
            self.step(action_ptr, 
                      static_cast<float*>(rewards.request().ptr),
                      static_cast<bool*>(dones.request().ptr),
                      count);
                      
            return py::make_tuple(rewards, dones);
        })
        .def("set_player_team", [](VecBattleEnv& self, size_t idx, const std::vector<Pokemon>& mons) {
            self.setPlayerTeam(idx, mons.data(), mons.size());
        })
        .def("set_opponent_team", [](VecBattleEnv& self, size_t idx, const std::vector<Pokemon>& mons) {
            self.setOpponentTeam(idx, mons.data(), mons.size());
        })
        .def("get_legal_actions", &VecBattleEnv::getLegalActions)
        .def("get_state", &VecBattleEnv::getState, py::return_value_policy::reference)
        .def("size", &VecBattleEnv::size);
}
