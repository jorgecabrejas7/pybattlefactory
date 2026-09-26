"""Stand-in for rl/team_eval.py (the team evaluator of docs/RL_DECISIONS.md §19.2) with the same API, for tests and
for running alphazero_v2 before the real evaluator exists.

    ev = TeamEvaluator.load(path, device)          # path "stub" (constant W) or "stub:varied" (W depends on the team)
    ev.w(teams, round_idx) -> [n, 3]               # W(normal), W(last), W(Noland)
    team_of_option(view, kind, option) -> Team
    option_features(ev, view, kind, ctx, options, v_next) -> [n_opt, 4]   # [E_round, W(normal), W(last|Noland), Delta]
    phi(ev, view, kind, ctx, v_next) -> float      # E_round + P(complete) * v_next of the current state

Conventions shared with the real module (what rl/alphazero_v2.py relies on):
    options   as rl.tactician_search.options_of gives them: rental (lead, pair index into rl.encode.PAIRS), swap
              0..9 (0 keep, 1 + 3 i + j: own slot i for the defeated team's slot j);
    ctx       rl.search.run_ctx / FactoryEnv._ctx: "battle" = battles already won in this round (0..6), so the next
              battle is k = battle + 1; "challenge" = rounds completed (round = challenge + 1);
    E_round   p_k + p_k p_{k+1} + ... + p_k...p_7, P(complete) = p_k...p_7, p_i = W(normal) for i < 7 and
              W(last or Noland) for i = 7;
    Delta     E_round of the option minus that of keeping the team (swap) or the mean over the options (rental).
"""

import numpy as np

W_CONST = 0.5


class Team(tuple):
    """(kind, option) of the stub: the real Team holds the 3 Pokemon as the player knows them, lead first."""


class TeamEvaluator:
    def __init__(self, varied=False, const=W_CONST):
        self.varied, self.const = varied, const

    @classmethod
    def load(cls, path, device="cpu"):
        return cls(varied=str(path).endswith(":varied"))

    def w(self, teams, round_idx):
        n = len(teams)
        if not self.varied:
            return np.full((n, 3), self.const, np.float64)
        out = np.empty((n, 3))
        for i, t in enumerate(teams):
            h = sum((j + 1) * int(np.sum(v)) for j, v in enumerate(np.ravel(t[1]))) if len(t) > 1 else 0
            base = 0.3 + 0.1 * ((h * 7 + int(round_idx)) % 6)
            out[i] = (base, base - 0.05, base - 0.1)
        return out


def team_of_option(view, kind, option):
    return Team((kind, option))


def _round_and_k(ctx):
    return int(ctx["challenge"]) + 1, int(ctx["battle"]) + 1


def _noland(ctx):
    return int(ctx["challenge"]) % 3 == 2        # rounds 3 and 6 (the stub's guess; the real module knows)


def _e_and_p(w_normal, w_last, k):
    e, p = 0.0, 1.0
    for i in range(k, 8):
        p *= w_last if i == 7 else w_normal
        e += p
    return e, p


def option_features(ev, view, kind, ctx, options, v_next):
    rnd, k = _round_and_k(ctx)
    if not options:
        return np.zeros((0, 4), np.float32)
    w = ev.w([team_of_option(view, kind, o) for o in options], rnd)
    last = w[:, 2] if _noland(ctx) else w[:, 1]
    e = np.array([_e_and_p(a, b, k)[0] for a, b in zip(w[:, 0], last)])
    ref = e[0] if kind == "swap" and options[0] == 0 else e.mean()
    return np.stack([e, w[:, 0], last, e - ref], 1).astype(np.float32)


def phi(ev, view, kind, ctx, v_next):
    rnd, k = _round_and_k(ctx)
    opts = [0] if kind == "swap" else [(0, 5)]
    w = ev.w([team_of_option(view, kind, o) for o in opts], rnd)
    last = w[0, 2] if _noland(ctx) else w[0, 1]
    e, p = _e_and_p(w[0, 0], last, k)
    return float(e + p * v_next)
