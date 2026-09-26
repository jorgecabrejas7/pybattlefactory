"""Gumbel AlphaZero for the battler (alphazero_v2, docs/RL_DECISIONS.md §19.1; Danihelka et al. 2022, "Policy
improvement by planning with Gumbel").

    gb = GumbelBattler(evaluator, n_sims=256, n_determinizations=8, seed=0, opponent_prior="factory_sets")
    r = gb.search_root(backend, decisions, root_x)      # at a BATTLE / FORCED_SWITCH decision of a SimBackend
    r["action"]    the sequential-halving winner (what self-play plays)
    r["target"]    softmax(logits + sigma(completed Q)) over the legal actions (the policy target)
    r["root_q"]    visit-weighted mean root Q (for the value target)

The search is the C++ ensemble of §15 (rl/search.py: K legal-mode determinizations, each root's turn redrawn with
redraw_turn, the opponent drawn with opponent_prior) with a Gumbel root (pybattle_native.Searcher.search_gumbel):
    1. g(a) ~ Gumbel(0, 1) for every legal action (noise=False: g = 0);
    2. the m = min(legal, max_considered) actions with the largest g + logits are considered (logits = log prior);
    3. sequential halving: ceil(log2 m) phases, n_sims / phases simulations per phase split evenly over the
       survivors (halving_plan; the last phase takes what is left), each survivor's simulations dealt to the K trees
       in turn; after a phase the survivors are ranked by g + logits + sigma(q_hat), q_hat the action's Q aggregated
       over the trees (value sums over visits), and the better half goes on; the winner is the last survivor;
    4. below the root, the paper's deterministic rule (non_root="gumbel": argmax pi'(a) - N(a) / (1 + sum N), pi' =
       softmax(logits + sigma(completed Q)) at that node) or PUCT (non_root="puct").
completed Q = the search Q for visited actions and, for the unvisited ones, v_mix = (v_hat + sum_b N(b) *
sum_{visited} pi(a) q(a) / sum_{visited} pi(a)) / (1 + sum_b N(b)) with v_hat the network value (paper eq. 33);
min-max rescaled to [0, 1] over the legal actions (rescale=True, mctx's default); sigma(q) = (c_visit + max_b N(b))
c_scale q (c_visit 50, c_scale 0.1). The battler's values are win probabilities in [0, 1].

Perfect information cannot exist here: GumbelBattler is a legal-mode SearchBattler (no mode argument), and in
training the C++ Searcher refuses roots that are not full determinizations.
"""

import time

import numpy as np

from pybattle.backend import Phase
from . import encode
from .search import SearchBattler, run_ctx

try:
    from pybattle.pybattle_native import halving_plan
except ImportError:                     # pragma: no cover - depends on the build
    halving_plan = None

N_ACTIONS = 7


def normalized_prior(prior, legal):
    p = np.where(legal, np.maximum(np.nan_to_num(np.asarray(prior, np.float64)), 0.0), 0.0)
    s = p.sum()
    return p / s if s > 0 else legal / max(legal.sum(), 1)


def completed_q(visits, q, prior, legal, v_hat):
    """Search Q where visited, v_mix elsewhere (paper eq. 33). prior: normalized over the legal actions."""
    visits, q = np.asarray(visits, np.float64), np.asarray(q, np.float64)
    vis = legal & (visits > 0)
    sum_n = visits[legal].sum()
    sum_pi = prior[vis].sum()
    v_mix = (v_hat + sum_n * (prior[vis] * q[vis]).sum() / sum_pi) / (1.0 + sum_n) if sum_pi > 0 else v_hat
    return np.where(vis, q, v_mix)


def sigma(cq, visits, legal, c_visit=50.0, c_scale=0.1, rescale=True):
    cq = np.asarray(cq, np.float64)
    if rescale:
        lo, hi = cq[legal].min(), cq[legal].max()
        cq = (cq - lo) / max(hi - lo, 1e-8)
    max_n = float(np.asarray(visits, np.float64)[legal].max()) if legal.any() else 0.0
    return np.where(legal, (c_visit + max_n) * c_scale * cq, 0.0)


def improved_policy(prior, visits, q, legal, v_hat, c_visit=50.0, c_scale=0.1, rescale=True):
    """The policy target softmax(logits + sigma(completed Q)) over the legal actions (0 elsewhere)."""
    legal = np.asarray(legal, bool)
    p = normalized_prior(prior, legal)
    cq = completed_q(visits, q, p, legal, v_hat)
    z = np.where(legal, np.log(np.maximum(p, 1e-30)) + sigma(cq, visits, legal, c_visit, c_scale, rescale), -np.inf)
    z = np.exp(z - z[legal].max())
    return z / z.sum()


def halving_winner_score(prior, visits, q, legal, v_hat, g, c_visit=50.0, c_scale=0.1, rescale=True):
    """g + logits + sigma(completed Q): the score the sequential halving ranks by."""
    legal = np.asarray(legal, bool)
    p = normalized_prior(prior, legal)
    cq = completed_q(visits, q, p, legal, v_hat)
    return np.where(legal, g + np.log(np.maximum(p, 1e-30)) + sigma(cq, visits, legal, c_visit, c_scale, rescale),
                    -np.inf)


def plan(m, n_sims):
    """Simulations per survivor in each halving phase (the C++ halving_plan) and the phase totals."""
    per = list(halving_plan(m, n_sims))
    tot, r, used = [], m, 0
    for i, x in enumerate(per):
        n = x * r
        if i == len(per) - 1:
            n = max(n, n_sims - used)
        tot.append(n)
        used += n
        r = (r + 1) // 2
    return per, tot


class GumbelBattler(SearchBattler):
    """The legal-mode ensemble search with a Gumbel root (always legal: there is no mode argument)."""

    def __init__(self, evaluator, n_sims=256, n_determinizations=8, batch=32, seed=0, max_considered=16,
                 c_visit=50.0, c_scale=0.1, rescale=True, non_root="gumbel", c_puct=1.5, observer="auto",
                 opponent_prior="strict"):
        if observer == "auto" and encode.VERSION not in (3, 4):
            observer = "python"
        if non_root not in ("gumbel", "puct"):
            raise ValueError(f"non_root must be 'gumbel' or 'puct', not {non_root!r}")
        super().__init__(None, n_sims=n_sims, n_determinizations=n_determinizations, c_puct=c_puct, batch=batch,
                         mode="legal", seed=seed, impl="cpp", observer=observer, evaluator=evaluator,
                         opponent_prior=opponent_prior)
        if not hasattr(self.searcher, "search_gumbel"):
            raise RuntimeError("GumbelBattler needs pybattle_native.Searcher.search_gumbel (rebuild the extension)")
        self.max_considered, self.c_visit, self.c_scale, self.rescale = max_considered, c_visit, c_scale, rescale
        self.non_root = non_root
        self.np_rng = np.random.default_rng(seed)

    def search_root(self, backend, decisions=0, root_x=None, noise=True):
        t0 = time.perf_counter()
        self._net_calls, self._eval_ms = 0, 0.0
        if backend.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
            raise RuntimeError(f"not at a battle decision: {backend.phase}")
        forced = backend.phase == Phase.FORCED_SWITCH
        view = backend.view()
        ctx = run_ctx(backend)
        if root_x is None:
            root_x = encode.battle(view, ctx)
        legal = root_x["mask"].astype(bool)
        pri, val = self.evaluate([root_x])
        prior, value = pri[0].astype(np.float64), float(val[0])
        out = {"prior": prior, "value": value, "legal": legal}
        if legal.sum() <= 1:
            a = int(np.flatnonzero(legal)[0]) if legal.any() else 0
            out.update(action=a, target=np.eye(N_ACTIONS)[a], visits=np.eye(N_ACTIONS)[a], q=np.zeros(N_ACTIONS),
                       root_q=float("nan"), searched=False, leaves=0, max_depth=0, mean_depth=0.0,
                       g=np.zeros(N_ACTIONS), ms=(time.perf_counter() - t0) * 1000)
            return out
        g = np.where(legal, self.np_rng.gumbel(size=N_ACTIONS), 0.0) if noise else np.zeros(N_ACTIONS)
        roots = self._roots(backend, view, None, forced, native_obs=self.observer == "cpp")
        r = self.searcher.search_gumbel([(gm, obs, decisions) for gm, obs in roots], ctx, prior.tolist(),
                                        legal.tolist(), value, self.evaluate_batch, g.tolist(),
                                        self.max_considered, self.c_visit, self.c_scale, self.rescale,
                                        self.non_root == "gumbel")
        self.errors += r["errors"]
        visits, q = np.asarray(r["visits"], float), np.asarray(r["q"], float)
        visits = np.where(legal, visits, 0.0)
        a = int(r["winner"])
        if not legal[a]:                                   # (cannot happen: the winner is a considered action)
            raise RuntimeError(f"Gumbel search chose an illegal action {a}")
        target = improved_policy(prior, visits, q, legal, value, self.c_visit, self.c_scale, self.rescale)
        out.update(action=a, target=target, visits=visits, q=q,
                   root_q=float((visits * q).sum() / max(visits.sum(), 1e-9)), searched=True, leaves=r["leaves"],
                   max_depth=r.get("max_depth", 0), mean_depth=r.get("mean_depth", 0.0), g=g,
                   considered=np.asarray(r["considered"], bool), phase_sims=list(r["phase_sims"]),
                   ms=(time.perf_counter() - t0) * 1000)
        return out

    def act_on(self, backend, known=None, decisions=0):
        """Evaluation / watching: the winner without Gumbel noise."""
        from .search import to_backend_action
        r = self.search_root(backend, decisions, noise=False)
        self._record(time.perf_counter() - r["ms"] / 1000, r["action"], r["visits"], r["q"], r["prior"],
                     r["value"], r["leaves"], 0)
        return to_backend_action(r["action"])
