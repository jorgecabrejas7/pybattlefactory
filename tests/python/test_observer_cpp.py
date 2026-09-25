"""The C++ observer + encoder (ObsMemory, include/gen3_observer.hpp) against pybattle.view.BattleObserver +
rl.encode.battle (encodings v3 and v4, one run each), side by side on real simulator runs.

Every encoded array must agree: ids exactly, floats within 1e-6. Paths covered: whole random-play Factory runs
(rounds 1-6 through the starting win streak, Noland, forced switches), ObsMemory.from_python at arbitrary points
followed to the end of the battle, and search-like paths (clone -> determinize(..., hidden_seed) -> rebase ->
sim_step -> observe), including the view at the rebased root itself (the observer's cached view).
"""

import json
import os
import random
import time

import numpy as np
import pytest

from pybattle.backend import Phase
from pybattle import pybattle_native as N
from rl import determinize as D
from rl import encode
from rl import search

from test_search import _decisions, _random_action

pytestmark = pytest.mark.skipif(not hasattr(N, "ObsMemory"), reason="C++ ObsMemory not built")

KEYS = ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask", "active")
ACTION, SWITCH = search.ACTION, search.SWITCH


def _use_version(v):
    old = encode.VERSION
    encode.set_version(v)
    assert N.encode_version() == v                  # rl.encode.set_version switches the C++ encoder too
    yield v
    encode.set_version(old)


@pytest.fixture(autouse=True)
def _v3():
    yield from _use_version(3)


@pytest.fixture(params=[3, 4], ids=["v3", "v4"])
def version(request):
    yield from _use_version(request.param)


def _diff(py, cpp):
    """Names of the arrays that differ (empty: equal)."""
    out = []
    for k in KEYS:
        a, b = np.asarray(py[k]), np.asarray(cpp[k])
        if a.shape != b.shape or a.dtype != b.dtype:
            out.append(f"{k}: shape/dtype {a.shape} {a.dtype} vs {b.shape} {b.dtype}")
        elif a.dtype.kind == "f":
            ints = np.uint32 if a.itemsize == 4 else np.uint64
            if not np.array_equal(a.view(ints), b.view(ints)):                  # bit for bit
                bad = np.argwhere(a.view(ints) != b.view(ints)) if a.ndim else [()]
                out.append(f"{k}: {len(bad)} values, first {tuple(bad[0])}: {a[tuple(bad[0])]!r} vs {b[tuple(bad[0])]!r}")
        elif not np.array_equal(a, b):
            bad = np.argwhere(a != b) if a.ndim else [()]
            out.append(f"{k}: first {tuple(bad[0])}: {a[tuple(bad[0])]} vs {b[tuple(bad[0])]}")
    return out


class Stats:
    def __init__(self):
        self.n = self.exact = 0
        self.fails = []

    def check(self, where, view, ctx, mem, game):
        py = encode.battle(view, ctx)
        cpp = mem.encode(game, ctx)
        d = _diff(py, cpp)
        self.n += 1
        self.exact += all(np.array_equal(np.asarray(py[k]), np.asarray(cpp[k])) for k in KEYS)
        if d:
            self.fails.append((where, d))
        assert not mem.overflow, where


def test_decomp_tables_match_game_data():
    d = json.load(open(os.path.join(os.path.dirname(encode.__file__), "..", "pybattle", "data", "game_data.json")))
    t = N.obs_tables()
    for i, (base, types, abil) in enumerate(t["species"]):
        s = d["species"][i]
        assert (base, types, abil) == (s["base"], s["types"], s["abilities"]), i
    for i, m in enumerate(t["moves"]):
        assert all(d["moves"][i][k] == v for k, v in m.items()), i
    assert [tuple(x) for x in t["items"]] == [(it["hold_effect"], it["hold_effect_param"]) for it in d["items"]]
    assert [list(r) for r in t["stat_stage_ratios"]] == d["stat_stage_ratios"]
    chart = t["type_chart"]
    assert [chart[i:i + 3] for i in range(0, len(chart), 3)] == d["type_effectiveness"]
    assert (t["abilities_count"], t["effects_count"]) == (d["counts"]["ABILITIES_COUNT"],
                                                          d["counts"]["NUM_BATTLE_MOVE_EFFECTS"])


def test_generated_tables_are_current():
    import subprocess
    import sys
    root = os.path.join(os.path.dirname(encode.__file__), "..")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "gen_observer_tables.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _args(b):
    g = b.game
    return g, b.phase == Phase.FORCED_SWITCH, g.unusable_moves(0), g.can_switch(0)


def test_real_runs_and_from_python(version):
    """Whole runs: a C++ memory per battle from its start, plus from_python conversions at random points, each
    followed to the end of its battle."""
    rng = random.Random(0)
    st = Stats()
    forced_n = noland = conv = 0
    rounds = set()
    cur_py, mem, chains = None, None, []
    for b, v, _ in _decisions(80, seed=21, streaks=(0, 7, 14, 20, 21, 28, 35, 41, 49)):
        o = b._observer
        g, forced, unusable, can_switch = _args(b)
        ctx = search.run_ctx(b)
        if o is not cur_py:                      # a new battle: a fresh memory with the attendant's hints
            cur_py, chains = o, []
            mem = N.ObsMemory(o.hint_type, o.hint_style)
        mem.observe(g, forced, unusable, can_switch)
        mem.observe(g, forced, unusable, can_switch)          # the same decision again: cached, no change
        st.check("run", v, ctx, mem, g)
        for c in chains:
            c.observe(g, forced, unusable, can_switch)
            st.check("from_python chain", v, ctx, c, g)
        if rng.random() < 0.08:
            c = N.ObsMemory.from_python(o)
            st.check("from_python", v, ctx, c, g)
            chains.append(c)
            conv += 1
        forced_n += forced
        noland += bool(b.game.factory_info.brain_status)
        rounds.add(ctx["challenge"])
    print(f"\nv{version} real runs: {st.n} encodings compared ({forced_n} forced switches, {noland} Noland decisions, "
          f"rounds {sorted(rounds)}, {conv} from_python conversions); bit-exact {st.exact}, mismatches {len(st.fails)}")
    for where, d in st.fails[:10]:
        print("  ", where, d)
    assert st.n > 3000 and forced_n > 50 and noland > 0 and set(range(7)) <= rounds
    assert not st.fails


def test_search_like_paths(version):
    """clone -> determinize(specs, hidden_seed) (or the true state) -> rebase -> [observe at the root] ->
    sim_step ... observe, the Python and C++ observers side by side; from_python on the rebased memory too."""
    rng = random.Random(1)
    st = Stats()
    roots = rebased_forced = steps = 0
    for b, v, tracker in _decisions(24, seed=22, streaks=(0, 14, 20, 28, 35)):
        if rng.random() < 0.5:
            continue
        forced = b.phase == Phase.FORCED_SWITCH
        ctx = search.run_ctx(b)
        ri = b.run_info()
        dctx = {"challenge": ri.challenge_num, "battle": ri.battle_in_challenge,
                "brain": b.game.factory_info.brain_status}
        base_mem = N.ObsMemory.from_python(b._observer)
        for k in range(2):
            g = b.game.clone()
            o = b._observer.fast_copy()
            m = base_mem.copy()
            if k == 0:
                specs = D.sample_determinization(v, dctx, rng)
                g.determinize(specs, D.hidden_counters(v), rng.getrandbits(32) if rng.random() < 0.7 else -1)
                o.rebase(g, forced)
                m.rebase(g, forced)
                rebased_forced += forced
                if rng.random() < 0.5:
                    m = N.ObsMemory.from_python(o)
            search.redraw_turn(g, rng.getrandbits(32))
            roots += 1
            if rng.random() < 0.5:                  # observe at the root itself: the cached (true) view
                args = (g, forced, g.unusable_moves(0), g.can_switch(0))
                vr = o.observe(*args)
                m.observe(*args)
                st.check("root", vr, ctx, m, g)
            view = v
            for depth in range(6):
                a = _random_action(b, view, rng) if depth == 0 else _leaf_action(view, rng)
                kind, idx = (0, a[1]) if a[0] == "move" else (1, a[1])
                if a[0] == "forfeit":
                    break
                d, _ = search.sim_step(g, kind, idx)
                if d not in (ACTION, SWITCH):
                    break
                args = (g, d == SWITCH, g.unusable_moves(0), g.can_switch(0))
                view = o.observe(*args)
                m.observe(*args)
                st.check(f"path depth {depth}", view, ctx, m, g)
                steps += 1
                if rng.random() < 0.15:
                    m = N.ObsMemory.from_python(o)
                    st.check("path from_python", view, ctx, m, g)
    print(f"\nv{version} search-like paths: {roots} roots ({rebased_forced} rebased at a forced switch), {steps} steps, "
          f"{st.n} encodings compared; bit-exact {st.exact}, mismatches {len(st.fails)}")
    for where, d in st.fails[:10]:
        print("  ", where, d)
    assert st.n > 1000 and rebased_forced > 0
    assert not st.fails


def _leaf_action(view, rng):
    if view.forced_switch:
        return ("switch", rng.choice(view.switch_targets))
    moves = [i for i, ok in enumerate(view.usable_moves) if ok]
    if view.switch_targets and (not moves or rng.random() < 0.15):
        return ("switch", rng.choice(view.switch_targets))
    return ("move", rng.choice(moves)) if moves else ("move", 0)


def test_benchmark():
    """µs per leaf: fast_copy + observe and encode, Python vs C++ (the search's per-leaf work)."""
    rng = random.Random(2)
    items = []
    for b, v, _ in _decisions(6, seed=23):
        if b.phase != Phase.BATTLE or rng.random() < 0.5:
            continue
        a = _random_action(b, v, rng)
        if a[0] == "forfeit":
            continue
        g = b.game.clone()
        d, _ = search.sim_step(g, 0 if a[0] == "move" else 1, a[1])
        if d in (ACTION, SWITCH):
            items.append((b._observer.fast_copy(), N.ObsMemory.from_python(b._observer), g, d == SWITCH,
                          search.run_ctx(b)))
    assert len(items) > 100
    t_po = t_pe = t_co = t_ce = 0.0
    for o, m, g, forced, ctx in items:
        args = (g, forced, g.unusable_moves(0), g.can_switch(0))
        t = time.perf_counter(); o2 = o.fast_copy(); view = o2.observe(*args); t_po += time.perf_counter() - t
        t = time.perf_counter(); encode.battle(view, ctx); t_pe += time.perf_counter() - t
        t = time.perf_counter(); m2 = m.copy(); m2.observe(*args); t_co += time.perf_counter() - t
        t = time.perf_counter(); m2.encode(g, ctx); t_ce += time.perf_counter() - t
    n = len(items)
    us = lambda x: x * 1e6 / n
    native = np.array([m.bench(g, forced, g.unusable_moves(0), g.can_switch(0), ctx, 20)
                       for o, m, g, forced, ctx in items])
    print(f"\nC++ timed natively: copy+observe {native[:, 0].mean():.2f} us, encode {native[:, 1].mean():.2f} us")
    print(f"\nper leaf ({n}): python observe {us(t_po):.0f} us, encode {us(t_pe):.0f} us; "
          f"C++ (through pybind) copy+observe {us(t_co):.1f} us, encode {us(t_ce):.1f} us")
    assert t_co + t_ce < (t_po + t_pe) / 5


def test_encode_version_switch():
    """Layout sizes per version, and the C++ switch refuses what it cannot encode."""
    sizes = {}
    for v in (2, 3, 4):
        encode.set_version(v)
        sizes[v] = (encode.MON_NUM, encode.MOVE_NUM, encode.CTX_NUM)
    encode.set_version(3)
    assert sizes[3] == (106, 17, 99) and sizes[4] == (106 + encode.N_DEFEATED, 17, 99)
    assert sizes[2][0] == 102 and sizes[2][1] == 14
    assert N.encode_version() == 3
    with pytest.raises(ValueError):
        N.set_encode_version(2)
    with pytest.raises(ValueError):
        encode.set_version(5)
