// Decision-time search loop (include/gen3_search.hpp): a line-by-line port of the loop in
// rl/search.py SearchBattler.act_on / _advance, so both give the same trees for the same roots
// and the same evaluator.

#include "gen3_search.hpp"
#include "gen3_mcts.hpp"

extern "C" {
#include "gen3/search_host.h"
}

#include <algorithm>
#include <chrono>
#include <exception>
#include <map>
#include <stdexcept>
#include <utility>

namespace pkmn {

namespace {

constexpr int N_ACTIONS = 7;

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// A node's state: the game at a player decision and the observer that saw the path to it.
struct NodeState {
    std::unique_ptr<Gen3Game> game;
    std::unique_ptr<NodeObs> obs;
    int decisions = 0;
    int depth = 0;                                // decisions of ours from the root
};

struct TreeData {
    std::unique_ptr<MctsTree> tree;
    std::vector<NodeState> states;                // by node id (game == nullptr: no state)
    std::vector<uint8_t> isTerm;                   // by node id
    std::vector<float> termValue;

    void grow(int n) {
        if (static_cast<int>(states.size()) < n) {
            states.resize(n);
            isTerm.resize(n, 0);
            termValue.resize(n, 0.0f);
        }
    }
    bool hasState(int node) const { return node < static_cast<int>(states.size()) && states[node].game != nullptr; }
    bool terminal(int node) const { return node < static_cast<int>(isTerm.size()) && isTerm[node]; }
    void setTerm(int node, float v) {
        grow(node + 1);
        isTerm[node] = 1;
        termValue[node] = v;
    }
};

// rl/search.py _advance: play `action` from the node's state (already copied) and the decisions
// without a choice: over (value) or a new node after `decisions` decisions.
struct AdvanceResult {
    bool over;
    float value;
    int decisions;
};

AdvanceResult advance(Gen3Game& game, NodeObs& obs, int action, int ndec, int maxDecisions, int& errors) {
    int kind = action < 4 ? 0 : 1;
    int idx = action < 4 ? action : action - 4;
    while (true) {
        std::pair<int, int> r;
        try {
            r = game.simStep(kind, idx);
        } catch (const std::exception&) {
            errors++;                                   // the game refused the action: a loss
            return {true, 0.0f, ndec};
        }
        ndec++;
        int d = r.first;
        if (d == Gen3Game::BATTLE_OVER) return {true, r.second == 1 ? 1.0f : 0.0f, ndec};
        if (d != Gen3Game::ACTION && d != Gen3Game::SWITCH) return {true, 0.0f, ndec};   // TIMEOUT
        uint8_t unusable = game.unusableMoves(0);
        bool canSwitch = game.canSwitch(0);
        obs.observe(game, d == Gen3Game::SWITCH, unusable, canSwitch);
        if (d == Gen3Game::ACTION && ndec < maxDecisions) {
            game.unusableMoves(0);                      // (makes `game` the live state)
            if (!Gen3Search_HasChoice(unusable, canSwitch ? 1 : 0)) {
                kind = 0;
                idx = 0;                                // nothing to choose: FactoryEnv plays move 0
                continue;
            }
        }
        return {false, 0.0f, ndec};
    }
}

struct Pending {
    int k, leaf, parent, action;
};

struct ToEval {
    int k, leaf;
    bool trunc;
};

}  // namespace

bool hasChoice(Gen3Game& game) {
    uint8_t unusable = game.unusableMoves(0);
    bool canSwitch = game.canSwitch(0);
    game.unusableMoves(0);
    return Gen3Search_HasChoice(unusable, canSwitch ? 1 : 0) != 0;
}

SearchStats runSearch(std::vector<SearchRoot>& roots, const EncodeCtx& ctx, const float rootPriors[7],
                      const bool rootLegal[7], float rootValue, const SearchConfig& cfg,
                      const BatchEvaluator& evaluator) {
    auto t0 = Clock::now();
    SearchStats st;
    const int K = static_cast<int>(roots.size());
    if (K == 0) throw std::invalid_argument("no roots");
    if (cfg.batch <= 0 || cfg.nSims <= 0) throw std::invalid_argument("n_sims and batch must be positive");

    std::vector<float> rp(rootPriors, rootPriors + N_ACTIONS);
    std::vector<bool> rl(rootLegal, rootLegal + N_ACTIONS);
    std::vector<TreeData> trees(K);
    for (int k = 0; k < K; k++) {
        TreeData& t = trees[k];
        t.tree = std::make_unique<MctsTree>(N_ACTIONS, cfg.cPuct, cfg.virtualLoss);
        t.tree->select(1);
        t.tree->expand(0, rp, rl);
        t.tree->backup(0, rootValue);
        t.grow(1);
        // the root's state is the caller's (not copied): it is only cloned from
        t.states[0].game.reset();
        t.states[0].decisions = roots[k].decisions;
    }
    auto rootGame = [&](int k) -> Gen3Game& { return *roots[k].game; };
    auto rootObs = [&](int k) -> NodeObs& { return *roots[k].obs; };

    const int budget = std::max(1, cfg.nSims / K);
    const int per = std::max(1, cfg.batch / K);
    std::vector<int> done(K, 1);
    std::vector<Pending> pending;
    std::vector<ToEval> toEval;
    std::vector<std::pair<int, int>> dups;
    std::vector<EncodedObs> enc;
    std::vector<float> pri, val;
    std::map<std::pair<int, int>, float> valueOf;

    while (true) {
        pending.clear();
        for (int k = 0; k < K; k++) {
            int n = std::min(per, budget - done[k]);
            if (n > 0) {
                auto sel = trees[k].tree->select(n);
                done[k] = sel.empty() ? budget : done[k] + static_cast<int>(sel.size());
                for (auto& [leaf, parent, act] : sel) pending.push_back({k, leaf, parent, act});
            }
        }
        if (pending.empty()) break;

        toEval.clear();
        dups.clear();
        for (const Pending& p : pending) {
            TreeData& t = trees[p.k];
            MctsTree& tree = *t.tree;
            t.grow(tree.nodeCount());
            if (t.terminal(p.leaf)) {
                tree.backup(p.leaf, t.termValue[p.leaf]);
                continue;
            }
            if (p.leaf == 0 || t.hasState(p.leaf)) {     // (the root has a state, as in the Python code)
                dups.emplace_back(p.k, p.leaf);          // selected again: evaluated once
                continue;
            }
            if (p.parent < 0) {                          // (root: already expanded)
                tree.backup(p.leaf, rootValue);
                continue;
            }
            const bool parentIsRoot = p.parent == 0;
            Gen3Game& g0 = parentIsRoot ? rootGame(p.k) : *t.states[p.parent].game;
            NodeObs& o0 = parentIsRoot ? rootObs(p.k) : *t.states[p.parent].obs;
            int d0 = t.states[p.parent].decisions;
            auto g = std::make_unique<Gen3Game>(g0);
            auto obs = o0.clone();
            AdvanceResult res = advance(*g, *obs, p.action, d0, cfg.maxDecisions, st.errors);
            const int depth = (parentIsRoot ? 0 : t.states[p.parent].depth) + 1;
            st.maxDepth = std::max(st.maxDepth, depth);
            st.depthSum += depth;
            st.depthCount++;
            if (res.over) {
                tree.setTerminal(p.leaf, res.value);
                t.setTerm(p.leaf, res.value);
                tree.backup(p.leaf, res.value);
                continue;
            }
            NodeState& ns = t.states[p.leaf];
            ns.game = std::move(g);
            ns.obs = std::move(obs);
            ns.decisions = res.decisions;
            ns.depth = depth;
            if (static_cast<int>(enc.size()) <= static_cast<int>(toEval.size())) enc.resize(toEval.size() + 1);
            ns.obs->encode(*ns.game, ctx, enc[toEval.size()]);
            toEval.push_back({p.k, p.leaf, res.decisions >= cfg.maxDecisions});
        }

        if (!toEval.empty()) {
            const int B = static_cast<int>(toEval.size());
            pri.assign(static_cast<size_t>(B) * N_ACTIONS, 0.0f);
            val.assign(B, 0.0f);
            auto te = Clock::now();
            evaluator(enc.data(), B, pri.data(), val.data());
            st.msEval += msSince(te);
            st.netCalls++;
            st.leaves += B;
            valueOf.clear();
            for (int i = 0; i < B; i++) {
                const ToEval& e = toEval[i];
                TreeData& t = trees[e.k];
                float v = val[i];
                if (e.trunc) {
                    t.tree->setTerminal(e.leaf, v);
                    t.setTerm(e.leaf, v);
                } else {
                    std::vector<float> p(pri.begin() + static_cast<size_t>(i) * N_ACTIONS,
                                         pri.begin() + static_cast<size_t>(i + 1) * N_ACTIONS);
                    std::vector<bool> legal(enc[i].mask, enc[i].mask + N_ACTIONS);
                    t.tree->expand(e.leaf, p, legal);
                }
                t.tree->backup(e.leaf, v);
                valueOf[{e.k, e.leaf}] = v;
            }
            for (auto& [k, leaf] : dups) {
                auto it = valueOf.find({k, leaf});
                float v = it != valueOf.end() ? it->second
                                              : (trees[k].terminal(leaf) ? trees[k].termValue[leaf] : rootValue);
                trees[k].tree->backup(leaf, v);
            }
        } else {
            for (auto& [k, leaf] : dups)
                trees[k].tree->backup(leaf, trees[k].terminal(leaf) ? trees[k].termValue[leaf] : rootValue);
        }
    }

    double qsum[N_ACTIONS] = {};
    for (int k = 0; k < K; k++) {
        MctsTree& tree = *trees[k].tree;
        auto vis = tree.rootVisits();
        auto q = tree.rootQ();
        for (int a = 0; a < N_ACTIONS; a++) {
            st.visits[a] += static_cast<double>(vis[a]);
            qsum[a] += static_cast<double>(q[a]) * static_cast<double>(vis[a]);
        }
        st.nodes += tree.nodeCount();
    }
    for (int a = 0; a < N_ACTIONS; a++) st.q[a] = st.visits[a] > 0 ? qsum[a] / std::max(st.visits[a], 1e-9) : 0.0;
    st.msTotal = msSince(t0);
    return st;
}

}  // namespace pkmn
