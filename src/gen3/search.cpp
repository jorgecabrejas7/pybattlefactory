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
#include <cmath>
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

namespace {

// The per-tree state of one search and the batch step shared by the PUCT and the Gumbel loops.
struct Engine {
    std::vector<SearchRoot>& roots;
    const EncodeCtx& ctx;
    float rootValue;
    const SearchConfig& cfg;
    const BatchEvaluator& evaluator;
    SearchStats& st;
    int K;
    std::vector<TreeData> trees;
    std::vector<ToEval> toEval;
    std::vector<std::pair<int, int>> dups;
    std::vector<EncodedObs> enc;
    std::vector<float> pri, val;
    std::map<std::pair<int, int>, float> valueOf;

    Engine(std::vector<SearchRoot>& r, const EncodeCtx& c, float rv, const SearchConfig& cf,
           const BatchEvaluator& ev, SearchStats& s)
        : roots(r), ctx(c), rootValue(rv), cfg(cf), evaluator(ev), st(s), K(static_cast<int>(r.size())) {}

    void init(const float rootPriors[7], const bool rootLegal[7]) {
        std::vector<float> rp(rootPriors, rootPriors + N_ACTIONS);
        std::vector<bool> rl(rootLegal, rootLegal + N_ACTIONS);
        trees.resize(K);
        for (int k = 0; k < K; k++) {
            TreeData& t = trees[k];
            t.tree = std::make_unique<MctsTree>(N_ACTIONS, cfg.cPuct, cfg.virtualLoss);
            if (cfg.gumbel) t.tree->setGumbelRule(cfg.gumbelNonRoot, cfg.cVisit, cfg.cScale, cfg.rescale);
            t.tree->select(1);
            t.tree->expand(0, rp, rl);
            t.tree->backup(0, rootValue);
            t.grow(1);
            // the root's state is the caller's (not copied): it is only cloned from
            t.states[0].game.reset();
            t.states[0].decisions = roots[k].decisions;
        }
    }

    // Expand / evaluate / back up the selected leaves (one network call for the new ones).
    void process(const std::vector<Pending>& pending) {
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
            Gen3Game& g0 = parentIsRoot ? *roots[p.k].game : *t.states[p.parent].game;
            NodeObs& o0 = parentIsRoot ? *roots[p.k].obs : *t.states[p.parent].obs;
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

    // Root visits summed over the trees and the visit-weighted mean Q (the value sums over the visits).
    void aggregate(double visits[7], double q[7]) const {
        double qsum[N_ACTIONS] = {};
        for (int a = 0; a < N_ACTIONS; a++) visits[a] = 0.0;
        for (int k = 0; k < K; k++) {
            MctsTree& tree = *trees[k].tree;
            auto vis = tree.rootVisits();
            auto qq = tree.rootQ();
            for (int a = 0; a < N_ACTIONS; a++) {
                visits[a] += static_cast<double>(vis[a]);
                qsum[a] += static_cast<double>(qq[a]) * static_cast<double>(vis[a]);
            }
        }
        for (int a = 0; a < N_ACTIONS; a++) q[a] = visits[a] > 0 ? qsum[a] / std::max(visits[a], 1e-9) : 0.0;
    }

    void finish(Clock::time_point t0) {
        aggregate(st.visits, st.q);
        for (int k = 0; k < K; k++) st.nodes += trees[k].tree->nodeCount();
        st.msTotal = msSince(t0);
    }
};

// sigma(completed Q) at the root from the aggregated statistics (the transform of MctsTree::pickGumbel)
void rootSigma(const double visits[7], const double q[7], const double prior[7], const bool legal[7], double vhat,
               const SearchConfig& cfg, double out[7]) {
    double sumN = 0, maxN = 0, sumPi = 0, sumPiQ = 0;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (!legal[a]) continue;
        sumN += visits[a];
        maxN = std::max(maxN, visits[a]);
        if (visits[a] > 0) {
            sumPi += prior[a];
            sumPiQ += prior[a] * q[a];
        }
    }
    double vmix = sumPi > 0 ? (vhat + sumN * sumPiQ / sumPi) / (1.0 + sumN) : vhat;
    double cq[N_ACTIONS] = {}, lo = INFINITY, hi = -INFINITY;
    for (int a = 0; a < N_ACTIONS; a++) {
        if (!legal[a]) continue;
        cq[a] = visits[a] > 0 ? q[a] : vmix;
        lo = std::min(lo, cq[a]);
        hi = std::max(hi, cq[a]);
    }
    for (int a = 0; a < N_ACTIONS; a++) {
        double v = cfg.rescale ? (cq[a] - lo) / std::max(hi - lo, 1e-8) : cq[a];
        out[a] = legal[a] ? (cfg.cVisit + maxN) * cfg.cScale * v : 0.0;
    }
}

}  // namespace

SearchStats runSearch(std::vector<SearchRoot>& roots, const EncodeCtx& ctx, const float rootPriors[7],
                      const bool rootLegal[7], float rootValue, const SearchConfig& cfg,
                      const BatchEvaluator& evaluator) {
    auto t0 = Clock::now();
    SearchStats st;
    const int K = static_cast<int>(roots.size());
    if (K == 0) throw std::invalid_argument("no roots");
    if (cfg.batch <= 0 || cfg.nSims <= 0) throw std::invalid_argument("n_sims and batch must be positive");
    if (cfg.gumbel) throw std::invalid_argument("runSearch is the PUCT loop: use runGumbelSearch");

    Engine eng(roots, ctx, rootValue, cfg, evaluator, st);
    eng.init(rootPriors, rootLegal);

    const int budget = std::max(1, cfg.nSims / K);
    const int per = std::max(1, cfg.batch / K);
    std::vector<int> done(K, 1);
    std::vector<Pending> pending;
    while (true) {
        pending.clear();
        for (int k = 0; k < K; k++) {
            int n = std::min(per, budget - done[k]);
            if (n > 0) {
                auto sel = eng.trees[k].tree->select(n);
                done[k] = sel.empty() ? budget : done[k] + static_cast<int>(sel.size());
                for (auto& [leaf, parent, act] : sel) pending.push_back({k, leaf, parent, act});
            }
        }
        if (pending.empty()) break;
        eng.process(pending);
    }
    eng.finish(t0);
    return st;
}

std::vector<int> halvingPlan(int m, int nSims) {
    std::vector<int> out;
    if (m <= 0) return out;
    int phases = 1;
    while ((1 << phases) < m) phases++;
    int used = 0, r = m;
    for (int p = 0; p < phases; p++) {
        int b = p == phases - 1 ? nSims - used : nSims / phases;
        int per = std::max(1, b / r);
        out.push_back(per);
        used += per * r;
        r = (r + 1) / 2;
    }
    return out;
}

SearchStats runGumbelSearch(std::vector<SearchRoot>& roots, const EncodeCtx& ctx, const float rootPriors[7],
                            const bool rootLegal[7], float rootValue, const float gumbel[7],
                            const SearchConfig& cfg, const BatchEvaluator& evaluator) {
    auto t0 = Clock::now();
    SearchStats st;
    const int K = static_cast<int>(roots.size());
    if (K == 0) throw std::invalid_argument("no roots");
    if (cfg.batch <= 0 || cfg.nSims <= 0) throw std::invalid_argument("n_sims and batch must be positive");
    if (!cfg.gumbel) throw std::invalid_argument("runGumbelSearch needs cfg.gumbel");

    // the root prior normalized over the legal actions (as MctsTree::expand); logits = log prior
    double prior[N_ACTIONS] = {}, logit[N_ACTIONS] = {}, psum = 0;
    int nLegal = 0;
    for (int a = 0; a < N_ACTIONS; a++)
        if (rootLegal[a]) {
            nLegal++;
            if (rootPriors[a] > 0 && std::isfinite(rootPriors[a])) psum += rootPriors[a];
        }
    if (nLegal == 0) throw std::invalid_argument("no legal root action");
    for (int a = 0; a < N_ACTIONS; a++) {
        if (!rootLegal[a]) continue;
        prior[a] = psum > 0 ? ((rootPriors[a] > 0 && std::isfinite(rootPriors[a])) ? rootPriors[a] / psum : 0.0)
                            : 1.0 / nLegal;
        logit[a] = std::log(std::max(prior[a], 1e-30));
    }

    Engine eng(roots, ctx, rootValue, cfg, evaluator, st);
    eng.init(rootPriors, rootLegal);

    // Gumbel top-m
    std::vector<int> remaining;
    for (int a = 0; a < N_ACTIONS; a++)
        if (rootLegal[a]) remaining.push_back(a);
    std::stable_sort(remaining.begin(), remaining.end(),
                     [&](int x, int y) { return gumbel[x] + logit[x] > gumbel[y] + logit[y]; });
    const int m = std::min(static_cast<int>(remaining.size()), std::max(1, cfg.maxConsidered));
    remaining.resize(m);
    for (int a : remaining) st.considered[a] = 1;

    std::vector<int> plan = halvingPlan(m, cfg.nSims);
    const int per = std::max(1, cfg.batch / K);
    std::vector<std::vector<int>> queue(K);
    std::vector<Pending> pending;
    std::vector<int> retry;
    long rr = 0;                                     // round-robin over the trees
    double visits[N_ACTIONS], q[N_ACTIONS], sigma[N_ACTIONS];
    int used = 0;
    for (size_t p = 0; p < plan.size(); p++) {
        const int r = static_cast<int>(remaining.size());
        const bool last = p + 1 == plan.size();
        const int extra = last ? std::max(0, cfg.nSims - used - plan[p] * r) : 0;
        // this phase: plan[p] simulations per survivor (one more for the first `extra` in rank order), dealt to
        // the trees in turn
        int phaseN = 0;
        for (int i = 0; i < r; i++) {
            int n = plan[p] + (i < extra ? 1 : 0);
            for (int j = 0; j < n; j++) queue[(rr++) % K].push_back(remaining[i]);
            phaseN += n;
        }
        used += phaseN;
        st.phaseSims.push_back(phaseN);
        // the queues in rounds of `per` selections per tree (one network call per round); a selection that
        // collides with a leaf in flight waits for the next round
        std::vector<size_t> head(K, 0);
        while (true) {
            pending.clear();
            bool left = false;
            for (int k = 0; k < K; k++) {
                auto& qk = queue[k];
                retry.clear();
                int taken = 0;
                while (head[k] < qk.size() && taken < per) {
                    int a = qk[head[k]++];
                    auto sel = eng.trees[k].tree->selectForced(a);
                    if (sel.empty()) {
                        retry.push_back(a);
                        continue;
                    }
                    taken++;
                    auto& [leaf, parent, act] = sel[0];
                    pending.push_back({k, leaf, parent, act});
                }
                if (!retry.empty()) qk.insert(qk.begin() + static_cast<long>(head[k]), retry.begin(), retry.end());
                if (head[k] < qk.size()) left = true;
            }
            if (pending.empty()) {
                // (cannot happen: the first selection of a tree in a round never collides)
                st.errors++;
                break;
            }
            eng.process(pending);
            if (!left) break;
        }
        for (int k = 0; k < K; k++) queue[k].clear();
        // rank the survivors by g + logit + sigma(completed Q); the better half goes on
        eng.aggregate(visits, q);
        rootSigma(visits, q, prior, rootLegal, rootValue, cfg, sigma);
        std::stable_sort(remaining.begin(), remaining.end(), [&](int x, int y) {
            return gumbel[x] + logit[x] + sigma[x] > gumbel[y] + logit[y] + sigma[y];
        });
        if (!last) remaining.resize((r + 1) / 2);
    }
    st.winner = remaining.empty() ? -1 : remaining[0];
    eng.finish(t0);
    return st;
}

}  // namespace pkmn
