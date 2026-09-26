#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "battle_engine.hpp"
#include "gen3_game.hpp"
#include "gen3_mcts.hpp"
#include "factory.hpp"
#include "types.hpp"
#include "constants.hpp"

namespace py = pybind11;
using namespace pkmn;

void bind_search(py::module_& m);   // src/python/search_bindings.cpp
void bind_observer(py::module_& m); // src/python/observer_bindings.cpp

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

        std::vector<Pokemon> generate_opponent_team(int challengeNum, int battleNum, bool isOpenLevel, const std::vector<uint16_t>& player_team = {}) {
            // Convert input vector (assumed Species IDs from Python) to Set
            std::set<uint16_t> excludedSpecies(player_team.begin(), player_team.end());
            
            auto monIds = FactoryGenerator::generateOpponentTeam(seed, challengeNum, battleNum, isOpenLevel, excludedSpecies);
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
        .def("generate_opponent_team", &FactoryHelper::generate_opponent_team,
             py::arg("challenge_num"), py::arg("battle_num"), py::arg("is_open_level"),
             py::arg("player_species") = std::vector<uint16_t>{})
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

    // Game-exact battle core (pokeemerald battle code, headless)
    py::class_<Gen3Game> gen3(m, "Gen3Game");
    py::enum_<Gen3Game::FactoryPhase>(gen3, "FactoryPhase")
        .value("NONE", Gen3Game::F_NONE)
        .value("RENTAL", Gen3Game::F_RENTAL)
        .value("BATTLE", Gen3Game::F_BATTLE)
        .value("SWAP", Gen3Game::F_SWAP)
        .value("RUN_OVER", Gen3Game::F_RUN_OVER);
    py::class_<Gen3Game::FactoryInfo>(gen3, "FactoryInfo")
        .def_readonly("phase", &Gen3Game::FactoryInfo::phase)
        .def_readonly("lvl_mode", &Gen3Game::FactoryInfo::lvlMode)
        .def_readonly("hint_type", &Gen3Game::FactoryInfo::hintType)
        .def_readonly("hint_style", &Gen3Game::FactoryInfo::hintStyle)
        .def_readonly("brain_status", &Gen3Game::FactoryInfo::brainStatus)
        .def_readonly("last_outcome", &Gen3Game::FactoryInfo::lastOutcome)
        .def_readonly("trainer_id", &Gen3Game::FactoryInfo::trainerId)
        .def_readonly("wins", &Gen3Game::FactoryInfo::wins)
        .def_readonly("swaps", &Gen3Game::FactoryInfo::swaps)
        .def_readonly("challenges_won", &Gen3Game::FactoryInfo::challengesWon);
    py::enum_<Gen3Game::Decision>(gen3, "Decision")
        .value("NONE", Gen3Game::NONE)
        .value("ACTION", Gen3Game::ACTION)
        .value("SWITCH", Gen3Game::SWITCH)
        .value("BATTLE_OVER", Gen3Game::BATTLE_OVER)
        .value("TIMEOUT", Gen3Game::TIMEOUT);
    gen3.def(py::init<>())
        .def("clone", [](const Gen3Game& b) { return Gen3Game(b); })
        .def("set_battle", &Gen3Game::setBattle, py::arg("battle_type_flags"), py::arg("trainer_id"))
        .def("write_party", [](Gen3Game& b, int side, py::bytes raw) { b.writeParty(side, raw); })
        .def("write_saveblock2", [](Gen3Game& b, uint32_t off, py::bytes data) { b.writeSaveBlock2(off, data); })
        .def("read_saveblock2", [](Gen3Game& b, uint32_t off, size_t n) { return py::bytes(b.readSaveBlock2(off, n)); })
        .def("set_var", &Gen3Game::setVar)
        .def("set_flag", &Gen3Game::setFlag)
        .def("start", &Gen3Game::start, py::arg("rng"))
        .def("run", &Gen3Game::run, py::arg("max_frames") = 200000)
        .def("choose_move", &Gen3Game::chooseMove)
        .def("unusable_moves", &Gen3Game::unusableMoves, py::arg("battler") = 0)
        .def("can_switch", &Gen3Game::canSwitch, py::arg("battler") = 0)
        .def("choiced_move", &Gen3Game::choicedMove, py::arg("battler") = 0)
        .def("choose_switch", &Gen3Game::chooseSwitch)
        .def("forfeit", &Gen3Game::forfeit)
        .def_property_readonly("rng", &Gen3Game::rng)
        .def_property_readonly("frames", &Gen3Game::frames)
        .def_static("trace_random", &Gen3Game::traceRandom)
        .def("read", [](Gen3Game& b, uint32_t addr, size_t n) { return py::bytes(b.readRam(addr, n)); })
        .def("write", [](Gen3Game& b, uint32_t addr, py::bytes data) { b.writeRam(addr, data); })
        .def("factory_begin", &Gen3Game::factoryBegin, py::arg("open_level") = true, py::arg("win_streak") = 0,
             py::arg("rents_count") = 0, py::arg("seed") = 0)
        .def_property_readonly("factory_phase", &Gen3Game::factoryPhase)
        .def("factory_rental", [](Gen3Game& g, int i) { return py::bytes(g.factoryRental(i)); })
        .def("factory_rental_mon_id", &Gen3Game::factoryRentalMonId)
        .def("factory_rent", &Gen3Game::factoryRent)
        .def("factory_swap", &Gen3Game::factorySwap, py::arg("player_slot"), py::arg("enemy_slot") = 0)
        .def("factory_run_battle", &Gen3Game::factoryRunBattle, py::arg("max_frames") = 400000)
        .def_property_readonly("factory_info", &Gen3Game::factoryInfo)
        // Search: never advance the Factory phase (see src/gen3/search_host.c)
        .def("sim_step", [](Gen3Game& g, int kind, int index) {
                auto r = g.simStep(kind, index);
                return py::make_tuple(r.first, r.second);
            }, py::arg("kind"), py::arg("index"),
            "kind 0 = move slot, 1 = switch to party index; run to the next decision or the battle's end. "
            "Returns (decision, gBattleOutcome).")
        .def("set_rng", &Gen3Game::setRng, py::arg("seed"))
        .def_property_readonly("battle_outcome", &Gen3Game::battleOutcome)
        .def("determinize", [](Gen3Game& g, const py::list& mons, py::object hidden, int64_t hiddenSeed) {
                std::vector<Gen3Game::DetMon> v;
                for (const py::handle& item : mons) {
                    py::tuple t = py::reinterpret_borrow<py::tuple>(item);
                    if (t.size() != 9)
                        throw std::invalid_argument("mon spec: (slot, species, moves[4], item, ivs[6], evs[6], "
                                                    "nature, ability_bit, hp_fraction)");
                    Gen3Game::DetMon d{};
                    d.partySlot = t[0].cast<int>();
                    d.species = t[1].cast<int>();
                    auto mv = t[2].cast<std::vector<int>>();
                    auto iv = t[4].cast<std::vector<int>>();
                    auto ev = t[5].cast<std::vector<int>>();
                    if (mv.size() != 4 || iv.size() != 6 || ev.size() != 6)
                        throw std::invalid_argument("mon spec: 4 moves, 6 IVs and 6 EVs");
                    for (int i = 0; i < 4; i++) d.moves[i] = mv[i];
                    d.item = t[3].cast<int>();
                    for (int i = 0; i < 6; i++) {
                        d.ivs[i] = iv[i];
                        d.evs[i] = ev[i];
                    }
                    d.nature = t[6].cast<int>();
                    d.abilityBit = t[7].cast<int>();
                    d.hpFraction = t[8].cast<float>();
                    v.push_back(d);
                }
                Gen3Game::DetHidden h{};
                bool haveHidden = !hidden.is_none();
                if (haveHidden) {
                    auto x = hidden.cast<std::vector<int>>();
                    if (x.size() != 14)
                        throw std::invalid_argument("hidden: 14 ints (sleep[2][3], confusion[2], wrap[2], "
                                                    "uproar[2], rampage[2])");
                    for (int s = 0; s < 2; s++) {
                        for (int i = 0; i < 3; i++) h.sleepElapsed[s][i] = x[s * 3 + i];
                        h.confusionElapsed[s] = x[6 + s];
                        h.wrapElapsed[s] = x[8 + s];
                        h.uproarElapsed[s] = x[10 + s];
                        h.rampageElapsed[s] = x[12 + s];
                    }
                }
                return g.determinize(v, haveHidden ? &h : nullptr, hiddenSeed);
            }, py::arg("mons"), py::arg("hidden") = py::none(), py::arg("hidden_seed") = -1,
            "mons: [(party_slot, species (<= 0: keep), [4 moves], item, [6 IVs], [6 EVs], nature, ability_bit, "
            "hp_fraction (< 0: keep))], drawn from player knowledge (rl/determinize.py). hidden: the elapsed "
            "counts the player saw, 14 ints: sleep attempts [side][party slot], confusion attempts, wrap, uproar "
            "and rampage turns [battler]. hidden_seed >= 0 also resamples every hidden counter; < 0 leaves them. "
            "Returns True when every opponent slot was rebuilt or has fainted.")
        .def("redraw_turn", &Gen3Game::redrawTurn, py::arg("seed"),
             "A search root's fresh random turn: gRngValue = seed, this turn's Quick Claw roll redrawn, and the "
             "opponent's choice for this turn (made by its AI while the player decides) undone so it chooses again.")
        .def("_state_bytes", [](Gen3Game& g) { return py::bytes(g.stateBytes()); }, "tests / debugging")
        .def("_state_offset", &Gen3Game::stateOffset, py::arg("gba_address"), py::arg("deref") = false,
             "tests / debugging: offset in _state_bytes() of a GBA RAM symbol, or of what a pointer symbol "
             "points to (deref); -1 outside the state")
        .def_property_readonly("determinized", &Gen3Game::determinized,
                               "True once a full determinization replaced the opponent's hidden data (clones too).");

    py::class_<MctsTree>(m, "MctsTree")
        .def(py::init<int, float, float>(), py::arg("n_actions") = 7, py::arg("c_puct") = 1.5f,
             py::arg("virtual_loss") = 1.0f)
        .def("reset", &MctsTree::reset)
        .def("expand", &MctsTree::expand, py::arg("node"), py::arg("priors"), py::arg("legal"))
        .def("set_terminal", &MctsTree::setTerminal, py::arg("node"), py::arg("value"))
        .def("select", &MctsTree::select, py::arg("k"))
        .def("select_forced", &MctsTree::selectForced, py::arg("action"),
             "Gumbel mode: one simulation through root action `action` ([] on a collision)")
        .def("set_gumbel_rule", &MctsTree::setGumbelRule, py::arg("gumbel"), py::arg("c_visit") = 50.0f,
             py::arg("c_scale") = 0.1f, py::arg("rescale") = true)
        .def("net_value", &MctsTree::netValue, py::arg("node"))
        .def("backup", &MctsTree::backup, py::arg("node"), py::arg("value"))
        .def("root_visits", &MctsTree::rootVisits)
        .def("root_q", &MctsTree::rootQ)
        .def("node_count", &MctsTree::nodeCount)
        .def("is_expanded", &MctsTree::isExpanded, py::arg("node"))
        .def("is_terminal", &MctsTree::isTerminal, py::arg("node"))
        .def("visits", &MctsTree::visits, py::arg("node"))
        .def("value", &MctsTree::value, py::arg("node"))
        .def("child", &MctsTree::child, py::arg("node"), py::arg("action"))
        .def_property_readonly("n_actions", &MctsTree::nActions)
        .def_property_readonly("c_puct", &MctsTree::cPuct)
        .def_property_readonly("virtual_loss", &MctsTree::virtualLoss);
    bind_observer(m);
    bind_search(m);
    m.attr("Gen3Battle") = gen3;  // older name
}
