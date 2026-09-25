#pragma once

// Decision-time search loop in C++ (the per-node work of rl/search.py SearchBattler).
//
// K determinized roots are prepared by the caller (Python: determinization specs, redraws,
// observer rebase, set_rng). The Searcher then runs, per tree, the same batched PUCT loop as
// SearchBattler.act_on: select `batch / K` leaves per tree and round (virtual loss), expand a
// leaf by cloning its parent's game and observer, sim_step(action), auto-play decisions without a
// choice (FactoryEnv plays move 0), observe, encode; evaluate all new leaves of the round with one
// call to the evaluator; back up in the same order as the Python code. Terminal leaves are worth
// 1 (won) / 0; a leaf at max_decisions keeps the network value and becomes terminal.
//
// The observer of a node is abstract (NodeObs): the C++ ObsMemory (gen3_observer.hpp, when built)
// or a Python BattleObserver + rl.encode.battle (src/python/search_bindings.cpp).

#include "gen3_game.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#if __has_include("gen3_observer.hpp")
#include "gen3_observer.hpp"
#define PKMN_HAVE_OBSERVER 1
#else
#define PKMN_HAVE_OBSERVER 0
#endif

namespace pkmn {

#if !PKMN_HAVE_OBSERVER
// Same layout as gen3_observer.hpp (rl/encode.py battle(), packed numeric arrays sized for the largest version).
struct EncodeCtx { int streak, battle, challenge, rents; };
constexpr int MON_IDS = 19, CTX_IDS = 2, N_DEFEATED = 6;
constexpr int MON_NUM_V3 = 106, MON_NUM_V4 = MON_NUM_V3 + N_DEFEATED, MOVE_NUM = 17, CTX_NUM = 99;
constexpr int MON_NUM_MAX = MON_NUM_V4, MOVE_NUM_MAX = MOVE_NUM, CTX_NUM_MAX = CTX_NUM;
struct EncodedObs {
    int64_t mon_ids[6][MON_IDS];
    float mon_num[6 * MON_NUM_MAX];
    float move_num[6 * 4 * MOVE_NUM_MAX];
    int64_t ctx_ids[CTX_IDS];
    float ctx_num[CTX_NUM_MAX];
    bool mask[7];
    int64_t active;
    int32_t mon_w, move_w, ctx_w;       // this observation's MON_NUM, MOVE_NUM, CTX_NUM
};
#endif

// What a search node knows about the battle so far (the player's memory), and its encoder.
class NodeObs {
public:
    virtual ~NodeObs() = default;
    virtual std::unique_ptr<NodeObs> clone() const = 0;
    // Called at every player decision along a path, in order (BattleObserver.observe).
    virtual void observe(Gen3Game& game, bool forced, uint8_t unusable, bool canSwitch) = 0;
    // The observation at the last observed decision (rl.encode.battle(view, ctx)).
    virtual void encode(Gen3Game& game, const EncodeCtx& ctx, EncodedObs& out) = 0;
};

#if PKMN_HAVE_OBSERVER
class CppNodeObs : public NodeObs {
public:
    explicit CppNodeObs(const ObsMemory& m) : m_mem(m) {}
    std::unique_ptr<NodeObs> clone() const override { return std::make_unique<CppNodeObs>(m_mem); }
    void observe(Gen3Game& g, bool forced, uint8_t unusable, bool canSwitch) override {
        obs_observe(m_mem, g, forced, unusable, canSwitch);
    }
    void encode(Gen3Game& g, const EncodeCtx& ctx, EncodedObs& out) override { encode_battle(m_mem, g, ctx, out); }
    ObsMemory& memory() { return m_mem; }

private:
    ObsMemory m_mem;
};
#endif

struct SearchRoot {
    Gen3Game* game;                // at the decision, determinized, RNG reseeded (caller-owned, only cloned)
    std::unique_ptr<NodeObs> obs;  // rebased on `game`
    int decisions;                 // decisions already taken in this battle
};

struct SearchConfig {
    int nSims = 256;
    int batch = 32;
    float cPuct = 1.5f;
    float virtualLoss = 1.0f;
    int maxDecisions = 300;
};

struct SearchStats {
    double visits[7] = {};         // root visits summed over the trees
    double q[7] = {};              // visit-weighted mean root Q
    int leaves = 0;                // leaves evaluated by the network
    int nodes = 0;                 // tree nodes (all trees)
    int netCalls = 0;              // evaluator calls
    int errors = 0;                // actions the game refused (counted as losses)
    double msTotal = 0.0;          // whole search
    double msEval = 0.0;           // inside the evaluator
};

// batch of B observations -> priors [B*7] (softmax, any mask), values [B] in [0, 1]
using BatchEvaluator = std::function<void(const EncodedObs* obs, int n, float* priors, float* values)>;

// Whether the player has a choice at an ACTION decision: a usable move or a switch target
// (BattleView.usable_moves / switch_targets). FactoryEnv plays move 0 otherwise.
bool hasChoice(Gen3Game& game);

SearchStats runSearch(std::vector<SearchRoot>& roots, const EncodeCtx& ctx, const float rootPriors[7],
                      const bool rootLegal[7], float rootValue, const SearchConfig& cfg,
                      const BatchEvaluator& evaluator);

}  // namespace pkmn
