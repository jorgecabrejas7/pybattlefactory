"""Stand-in for the alphazero_v3 team evaluator interface (docs/RL_DECISIONS.md §20), for tests and for running
rl/alphazero_v3.py before the real module (planner + hint-conditioned W) is merged.

    N_OPT_FEATURES = 6
    TeamEvaluator.load(path, device)                                     # rl.team_eval_stub's: "stub" / "stub:varied"
    option_features_v3(ev, view, kind, ctx, options, v_next_table, hint, planner_cfg) -> [n_opt, N_OPT_FEATURES]
        [E_round, W(normal | hint), W(last or Noland), Delta, Q_plan, Q_plan - mean over the options]
    phi_v3(ev, view, kind, ctx, v_next_table, hint, planner_cfg) -> float
    q_eval_v3(features) -> [n_opt]                                      # the planner's Q_plan (column 4)

hint = (hint_type 0..18, 18 = none; hint_style 0..7). The stub has no planner: Q_plan = Phi of the state the option
leads to (E_round + P(complete) * v_next), and the hint only shifts W(normal) a little (so tests can see it matter).
Conventions (options, ctx, v_next_table[r] = mean wins from the start of round r) as rl/team_eval_stub.py and
rl/tactician_features.py.
"""

import numpy as np

from .team_eval_stub import TeamEvaluator, _e_and_p, _noland, _round_and_k, team_of_option  # noqa: F401

N_OPT_FEATURES = 6
NO_HINT = (18, 0)


def _v_next(table, ctx):
    if table is None or len(table) == 0:
        return 0.0
    r = int(ctx["challenge"]) + 2
    return float(table[min(r, len(table) - 1)])


def _hint_shift(hint):
    t, s = hint if hint is not None else NO_HINT
    return 0.0 if int(t) >= 18 else 0.01 * ((int(t) + 3 * int(s)) % 5 - 2)


def _w(ev, view, kind, ctx, options, hint):
    rnd, _ = _round_and_k(ctx)
    w = ev.w([team_of_option(view, kind, o) for o in options], rnd)
    wn = np.clip(w[:, 0] + _hint_shift(hint), 0.0, 1.0)
    last = w[:, 2] if _noland(ctx) else w[:, 1]
    return wn, last


def option_features_v3(ev, view, kind, ctx, options, v_next_table, hint, planner_cfg=None):
    if not options:
        return np.zeros((0, N_OPT_FEATURES), np.float32)
    _, k = _round_and_k(ctx)
    wn, last = _w(ev, view, kind, ctx, options, hint)
    ep = [_e_and_p(a, b, k) for a, b in zip(wn, last)]
    e = np.array([x[0] for x in ep])
    q = e + np.array([x[1] for x in ep]) * _v_next(v_next_table, ctx)
    ref = e[0] if kind == "swap" and options[0] == 0 else e.mean()
    return np.stack([e, wn, last, e - ref, q, q - q.mean()], 1).astype(np.float32)


def phi_v3(ev, view, kind, ctx, v_next_table, hint, planner_cfg=None):
    opts = [0] if kind == "swap" else [(0, 5)]
    _, k = _round_and_k(ctx)
    wn, last = _w(ev, view, kind, ctx, opts, hint)
    e, p = _e_and_p(wn[0], last[0], k)
    return float(e + p * _v_next(v_next_table, ctx))


def q_eval_v3(features):
    f = np.asarray(features, np.float64)
    return f[:, 4] if len(f) else np.zeros(0)
