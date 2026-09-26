#pragma once

// PUCT search tree (AlphaZero-style) for single-agent search over Gen3Game decisions.
// Pure bookkeeping: the caller owns the game states and evaluates leaves.
//
//   tree.reset()
//   loop:
//     leaves = tree.select(k)              // (leaf, parent, action); virtual loss spreads the batch
//     evaluate leaves (clone parent's game, sim_step(action), network)
//     tree.expand(leaf, priors, legal)  or  tree.set_terminal(leaf, value)
//     tree.backup(leaf, value)             // value in [0,1], root player's perspective
//   tree.root_visits()
//
// Node 0 is the root. Children are created lazily when select() first reaches them.
// PUCT: Q + c_puct * P * sqrt(N_parent) / (1 + N). An unvisited child's Q is its parent's mean
// value (first-play urgency = parent value). Virtual loss counts as `virtual_loss` visits of
// value 0 per in-flight selection.
//
// Gumbel mode (alphazero_v2, Danihelka et al. 2022; setGumbelRule): select_forced(a) walks from the
// root through the root action `a` (the caller's sequential halving decides the root actions), and
// below the root picks deterministically argmax_a pi'(a) - N(a) / (1 + sum_b N(b)) with
// pi' = softmax(log prior + sigma(completed Q)): completed Q = the child's mean value if visited,
// else the node's v_mix (network value mixed with the prior-weighted Q of the visited children);
// optionally min-max rescaled over the node's legal actions; sigma(q) = (c_visit + max_b N(b)) *
// c_scale * q. N counts in-flight selections as virtual visits (of value 0), as PUCT does. The
// node's network value is the first value backed up at it (its leaf evaluation).

#include <cstdint>
#include <tuple>
#include <vector>

namespace pkmn {

class MctsTree {
public:
    explicit MctsTree(int nActions = 7, float cPuct = 1.5f, float virtualLoss = 1.0f);

    void reset();
    void expand(int node, const std::vector<float>& priors, const std::vector<bool>& legal);
    void setTerminal(int node, float value);
    std::vector<std::tuple<int, int, int>> select(int k);
    // Gumbel mode: one simulation through root action `action` (empty on a collision with an
    // in-flight leaf: nothing is left in flight then).
    std::vector<std::tuple<int, int, int>> selectForced(int action);
    // Non-root selection rule: gumbel=true (the deterministic rule above) or false (PUCT).
    void setGumbelRule(bool gumbel, float cVisit = 50.0f, float cScale = 0.1f, bool rescale = true);
    void backup(int node, float value);

    std::vector<int> rootVisits() const;
    std::vector<float> rootQ() const;
    int nodeCount() const { return static_cast<int>(m_parent.size()); }
    bool isExpanded(int node) const;
    bool isTerminal(int node) const;
    int visits(int node) const;
    float value(int node) const;     // mean backed-up value (0 if unvisited)
    float netValue(int node) const;  // the first value backed up at the node (its evaluation)
    std::vector<double> rootValueSums() const;
    int child(int node, int action) const;

    int nActions() const { return m_n; }
    float cPuct() const { return m_c; }
    float virtualLoss() const { return m_vl; }

private:
    int newNode(int parent, int action);
    void check(int node) const;
    void addVirtualLoss(int node, int delta);
    int pickPuct(int node) const;
    int pickGumbel(int node) const;
    // walk down from `node` (root action forced when forced >= 0) -> leaf; collision: in flight
    int walk(int forced, bool& collision);

    int m_n;
    float m_c, m_vl;
    // per node
    std::vector<int> m_parent, m_action, m_visits, m_inflight;
    std::vector<double> m_valueSum;
    std::vector<uint8_t> m_expanded, m_terminal;
    std::vector<float> m_terminalValue;
    std::vector<float> m_netValue;
    bool m_gumbel = false, m_rescale = true;
    float m_cVisit = 50.0f, m_cScale = 0.1f;
    // per node * action
    std::vector<int> m_children;
    std::vector<float> m_priors;
    std::vector<uint8_t> m_legal;
};

}  // namespace pkmn
