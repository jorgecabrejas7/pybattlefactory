#include "gen3_mcts.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pkmn {

MctsTree::MctsTree(int nActions, float cPuct, float virtualLoss)
    : m_n(nActions), m_c(cPuct), m_vl(virtualLoss) {
    if (nActions <= 0) throw std::invalid_argument("n_actions must be positive");
    reset();
}

void MctsTree::reset() {
    m_parent.clear();
    m_action.clear();
    m_visits.clear();
    m_inflight.clear();
    m_valueSum.clear();
    m_expanded.clear();
    m_terminal.clear();
    m_terminalValue.clear();
    m_children.clear();
    m_priors.clear();
    m_legal.clear();
    newNode(-1, -1);
}

int MctsTree::newNode(int parent, int action) {
    int id = static_cast<int>(m_parent.size());
    m_parent.push_back(parent);
    m_action.push_back(action);
    m_visits.push_back(0);
    m_inflight.push_back(0);
    m_valueSum.push_back(0.0);
    m_expanded.push_back(0);
    m_terminal.push_back(0);
    m_terminalValue.push_back(0.0f);
    m_children.insert(m_children.end(), m_n, -1);
    m_priors.insert(m_priors.end(), m_n, 0.0f);
    m_legal.insert(m_legal.end(), m_n, 0);
    if (parent >= 0) m_children[static_cast<size_t>(parent) * m_n + action] = id;
    return id;
}

void MctsTree::check(int node) const {
    if (node < 0 || node >= nodeCount()) throw std::out_of_range("no such node");
}

void MctsTree::expand(int node, const std::vector<float>& priors, const std::vector<bool>& legal) {
    check(node);
    if (static_cast<int>(priors.size()) != m_n || static_cast<int>(legal.size()) != m_n)
        throw std::invalid_argument("priors and legal must have n_actions entries");
    size_t base = static_cast<size_t>(node) * m_n;
    double sum = 0.0;
    int nLegal = 0;
    for (int a = 0; a < m_n; a++) {
        if (!legal[a]) continue;
        nLegal++;
        if (priors[a] > 0 && std::isfinite(priors[a])) sum += priors[a];
    }
    for (int a = 0; a < m_n; a++) {
        m_legal[base + a] = legal[a] ? 1 : 0;
        float p = 0.0f;
        if (legal[a]) {
            if (sum > 0) p = (priors[a] > 0 && std::isfinite(priors[a])) ? static_cast<float>(priors[a] / sum) : 0.0f;
            else p = 1.0f / nLegal;
        }
        m_priors[base + a] = p;
    }
    m_expanded[node] = 1;
}

void MctsTree::setTerminal(int node, float value) {
    check(node);
    m_terminal[node] = 1;
    m_terminalValue[node] = value;
}

void MctsTree::addVirtualLoss(int node, int delta) {
    for (int n = node; n >= 0; n = m_parent[n]) m_inflight[n] += delta;
}

std::vector<std::tuple<int, int, int>> MctsTree::select(int k) {
    std::vector<std::tuple<int, int, int>> out;
    if (k <= 0) return out;
    if (!m_expanded[0] && !m_terminal[0]) {
        out.emplace_back(0, -1, -1);
        return out;
    }
    std::vector<int> collisions;
    int attempts = 0;
    while (static_cast<int>(out.size()) < k && attempts < 4 * k) {
        attempts++;
        int node = 0, leaf = -1;
        bool collision = false;
        while (true) {
            if (m_terminal[node]) { leaf = node; break; }
            if (!m_expanded[node]) {
                // Created by an earlier select and not evaluated yet
                leaf = node;
                collision = m_inflight[node] > 0;
                break;
            }
            size_t base = static_cast<size_t>(node) * m_n;
            float nParent = m_visits[node] + m_inflight[node] * m_vl;
            float sqrtN = std::sqrt(std::max(nParent, 1.0f));
            float fpu = m_visits[node] > 0 ? static_cast<float>(m_valueSum[node] / m_visits[node]) : 0.5f;
            int best = -1;
            float bestScore = -INFINITY;
            for (int a = 0; a < m_n; a++) {
                if (!m_legal[base + a]) continue;
                int c = m_children[base + a];
                float nc = 0.0f, q = fpu;
                if (c >= 0) {
                    nc = m_visits[c] + m_inflight[c] * m_vl;
                    if (nc > 0) q = static_cast<float>(m_valueSum[c] / nc);
                }
                float score = q + m_c * m_priors[base + a] * sqrtN / (1.0f + nc);
                if (score > bestScore) { bestScore = score; best = a; }
            }
            if (best < 0) { leaf = node; break; }   // expanded without legal actions
            int c = m_children[base + best];
            if (c < 0) { leaf = newNode(node, best); break; }
            node = c;
        }
        addVirtualLoss(leaf, 1);
        if (collision) {
            collisions.push_back(leaf);
            continue;
        }
        out.emplace_back(leaf, m_parent[leaf], m_action[leaf]);
    }
    for (int leaf : collisions) addVirtualLoss(leaf, -1);
    return out;
}

void MctsTree::backup(int node, float value) {
    check(node);
    for (int n = node; n >= 0; n = m_parent[n]) {
        m_visits[n]++;
        m_valueSum[n] += value;
        if (m_inflight[n] > 0) m_inflight[n]--;
    }
}

std::vector<int> MctsTree::rootVisits() const {
    std::vector<int> v(m_n, 0);
    for (int a = 0; a < m_n; a++) {
        int c = m_children[a];
        if (c >= 0) v[a] = m_visits[c];
    }
    return v;
}

std::vector<float> MctsTree::rootQ() const {
    std::vector<float> q(m_n, 0.0f);
    for (int a = 0; a < m_n; a++) {
        int c = m_children[a];
        if (c >= 0 && m_visits[c] > 0) q[a] = static_cast<float>(m_valueSum[c] / m_visits[c]);
    }
    return q;
}

bool MctsTree::isExpanded(int node) const {
    check(node);
    return m_expanded[node] != 0;
}

bool MctsTree::isTerminal(int node) const {
    check(node);
    return m_terminal[node] != 0;
}

int MctsTree::visits(int node) const {
    check(node);
    return m_visits[node];
}

float MctsTree::value(int node) const {
    check(node);
    return m_visits[node] > 0 ? static_cast<float>(m_valueSum[node] / m_visits[node]) : 0.0f;
}

int MctsTree::child(int node, int action) const {
    check(node);
    if (action < 0 || action >= m_n) throw std::out_of_range("no such action");
    return m_children[static_cast<size_t>(node) * m_n + action];
}

}  // namespace pkmn
