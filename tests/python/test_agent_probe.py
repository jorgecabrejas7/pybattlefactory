"""rl/agent_probe.py: per-agent probes (docs/RL_DECISIONS.md §25)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from rl import agent_probe as P

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V3 = os.path.join("runs", "alphazero_v3", "latest.pt")
V4 = os.path.join("runs", "alphazero_v4", "ckpt_00036012600.pt")
needs_ckpt = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, V3)), reason="runs/alphazero_v3/latest.pt absent")


def _args(**kw):
    a = P.build_parser().parse_args(["generate", "--procs", "1", "--device", "cpu", "--runs", "4",
                                     "--b-quota", "r3:6,r5:10,r6:10,noland:3", "--b-seeds", "2", "--c1-rental", "3",
                                     "--c1-swap", "6", "--c2-rental", "1", "--c2-swap", "2", "--c2-seeds", "4"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    if not os.path.exists(os.path.join(ROOT, V3)):
        pytest.skip("runs/alphazero_v3/latest.pt absent")
    cwd = os.getcwd()
    os.chdir(ROOT)
    old = P.PROBE_DIR
    P.PROBE_DIR = str(tmp_path_factory.mktemp("probe"))
    try:
        P.generate(_args())
        yield P.Bench()
    finally:
        P.PROBE_DIR = old
        os.chdir(cwd)


@pytest.fixture
def probe_dir(bench):
    old = P.PROBE_DIR
    P.PROBE_DIR = os.path.dirname(bench.dir)
    cwd = os.getcwd()
    os.chdir(ROOT)
    yield P.PROBE_DIR
    P.PROBE_DIR = old
    os.chdir(cwd)


# ---- pure functions ------------------------------------------------------------------------------------------------

def test_c2_regret_raw_and_cross_validated():
    table = {"s": {0: np.array([1, 1, 1, 1.]), 1: np.array([2, 2, 2, 2.]), 2: np.array([0, 4, 0, 4.])}}
    raw, cv, ids = P.c2_regret(table, {"s": 0})
    assert ids == ["s"] and raw == [1.0]
    # halves [0, 4] / [0, 4] mean 2 each: option 1 (2) ties option 2 (2) on each half -> the lower index wins: 1
    assert cv == [1.0]
    raw, cv, _ = P.c2_regret(table, {"s": 1})
    assert raw == [0.0] and cv == [0.0]
    # a state whose chosen option was not played is skipped
    assert P.c2_regret(table, {"s": 7})[2] == []


def test_cross_validation_removes_the_max_bias():
    rng = np.random.default_rng(0)
    # 10 equal options (true value 3), noisy: the raw regret of any choice is > 0, the validated one ~ 0
    raws, cvs = [], []
    for t in range(300):
        table = {"s": {o: rng.normal(3, 1, 32) for o in range(10)}}
        r, c, _ = P.c2_regret(table, {"s": 0})
        raws += r
        cvs += c
    assert np.mean(raws) > 0.2
    assert abs(np.mean(cvs)) < 0.05


def test_spec_parsing():
    assert P.parse_spec("net:a/b.pt") == ("net", "a/b.pt", None)
    assert P.parse_spec("search:a/b.pt") == ("search", "a/b.pt", 512)
    assert P.parse_spec("perfect:a/b.pt:4096") == ("perfect", "a/b.pt", 4096)
    with pytest.raises(ValueError):
        P.parse_spec("magic:x")


# ---- the benchmark -------------------------------------------------------------------------------------------------

@needs_ckpt
def test_benchmark_replays_exactly(bench, probe_dir):
    """Every state replays (state + observation hashes checked inside) and the option gives the same opponent."""
    states = bench.tactician + bench.battles
    n = 0
    for s, pre in P.replay_states(bench, states):
        P.check_opponent(s, pre, P._dec(s["action"]))
        n += 1
    assert n == len(states)


@needs_ckpt
def test_battle_start_with_the_games_rng_replays_the_generated_battle(bench, probe_dir):
    """With the game's own RNG (no reseed) and v3's network, the battle start plays the generation's battle again:
    same result."""
    agents = P.Agents("cpu")
    battler = P.Battler(agents, f"net:{V3}")
    res = {}
    games = []
    for s, pre in P.replay_states(bench, bench.battles):
        from rl.autopsy import start_battle
        games.append(P.Game(s["id"], start_battle(pre, P._dec(s["action"]), None, None), "battle", s))
    P.play_lockstep(iter(games), battler, lambda g: res.__setitem__(g.key, g.wins), width=7)
    assert len(res) == len(bench.battles)
    for s in bench.battles:
        assert res[s["id"]] == int(s["won"]), s["id"]


@needs_ckpt
def test_battler_results_are_paired_reproducible_and_resumable(bench, probe_dir):
    spec = f"net:{V3}"
    s1 = P.run_battler(spec, procs=1, device="cpu", width=3)
    prefix = P.battler_prefix(spec)
    rows = P.read_shards(prefix)
    assert len(rows) == len(bench.battles) * 2
    # the same games with another batch width give the same results (seeded, deterministic)
    path = prefix + ".p0.jsonl"
    keep = rows[: len(rows) // 3]
    P.write_jsonl(path, keep, mode="w")
    s2 = P.run_battler(spec, procs=1, device="cpu", width=11)
    assert s2["per_state"] == s1["per_state"]
    t = P.read_jsonl(os.path.join(P.PROBE_DIR, "timing.jsonl"))[-1]
    assert 0 < t["decisions"]        # only the missing games were played again


@needs_ckpt
def test_tactician_c1_and_planner(bench, probe_dir):
    P.run_tactician(V3, procs=1)
    net = P.load_choices(f"net:{V3}")
    plan = P.load_choices(f"planner:{V3}")
    assert set(net) == {s["id"] for s in bench.tactician}
    for s in bench.tactician:
        # v3's choice at C1 is the generation's option (same features and planner seed)
        assert net[s["id"]]["choice"] == s["option"]
        q = np.asarray(plan[s["id"]]["q"])
        assert plan[s["id"]]["choice"] == int(np.argmax(q))
        assert net[s["id"]]["q"] == plan[s["id"]]["q"]
    c1 = P.summarize_c1(f"planner:{V3}", f"net:{V3}")
    assert c1["swap"]["regret"][0] == 0.0 and c1["swap"]["agree"][0] == 1.0


@needs_ckpt
def test_outcomes_table_and_c2(bench, probe_dir):
    P.run_tactician(V3, procs=1)
    P.run_outcomes([f"net:{V3}", f"planner:{V3}"], procs=1, seeds=4, device="cpu", width=8)
    tab = P.outcome_table()
    assert set(tab) == {s["id"] for s in bench.outcomes}
    for s in bench.outcomes:
        if s["kind"] == "swap":
            assert len(tab[s["id"]]) == P.load_choices(f"net:{V3}")[s["id"]]["n"]
        for o, v in tab[s["id"]].items():
            assert len(v) == 4 and v.min() >= 0 and v.max() <= 8 - s["battle"]
    c2 = P.summarize_c2(f"net:{V3}")
    assert c2["all"]["raw"][0] >= 0


@needs_ckpt
def test_crossplay_routes_each_decision_to_its_network(probe_dir, monkeypatch):
    if not os.path.exists(os.path.join(ROOT, V4)):
        pytest.skip("v4 checkpoint absent")
    used = {"tactician": set(), "battler": set()}
    orig_decide, orig_act = P.tactician_decide, P.Battler.act_many

    def decide(agents, spec, be, seed, cache=None):
        used["tactician"].add(spec)
        return orig_decide(agents, spec, be, seed, cache)

    def act_many(self, games):
        if games:
            used["battler"].add(self.ckpt)
        return orig_act(self, games)

    monkeypatch.setattr(P, "tactician_decide", decide)
    monkeypatch.setattr(P.Battler, "act_many", act_many)
    s = P.run_crossplay(f"net:{V4}", f"net:{V3}", runs=2, rounds=(5,), procs=1, device="cpu")
    assert used == {"tactician": {f"net:{V4}"}, "battler": {V3}}
    assert s["rounds"][5]["n"] == 2
    # the planner as tactician: argmax of the checkpoint's own Q_plan
    s = P.run_crossplay(f"planner:{V3}", f"net:{V3}", runs=2, rounds=(5,), procs=1, device="cpu")
    assert f"planner:{V3}" in used["tactician"]


def test_refuses_to_run_in_a_training_process():
    code = ("from rl.search import mark_training; mark_training(); from rl import agent_probe as P\n"
            "try:\n    P.main(['report'])\nexcept RuntimeError as e:\n    print('refused', e)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert "refused" in out.stdout, out.stderr
