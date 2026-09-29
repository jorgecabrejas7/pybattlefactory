"""rl/autopsy.py: the loss autopsy (docs/RL_DECISIONS.md §21)."""

import json
import os
import re
import subprocess
import sys

import pytest

from pybattle.backend import Phase, SimBackend
from rl import autopsy as A
from rl import search

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(ROOT, "runs", "alphazero_v3", "latest.pt")
needs_ckpt = pytest.mark.skipif(not os.path.exists(CKPT), reason="runs/alphazero_v3/latest.pt not present")

TINY = ["--losses", "2", "--workers", "1", "--inference", "cpu", "--strong-sims", "16", "--strong-dets", "2",
        "--luck-seeds", "3", "--p0-net-seeds", "2", "--mistake-seeds", "2", "--max-rewinds", "1",
        "--team-screen-seeds", "2", "--team-top", "1", "--team-seeds", "2", "--min-pattern-losses", "1"]


# ---- messages and events ----------------------------------------------------------------------------------------

def test_string_ids_match_the_decomp():
    path = os.path.join(ROOT, "third_party", "pokeemerald", "include", "constants", "battle_string_ids.h")
    ids = {m.group(1): int(m.group(2)) for m in re.finditer(r"#define STRINGID_(\w+)\s+(\d+)", open(path).read())}
    for name, sid in A.STRINGIDS.items():
        assert ids[name] == sid, name


def test_parse_messages_sides_and_crit_ko():
    U, CRIT, FAINT, MISS, FRZ, PARA = 4, 217, 29, 23, 49, 57
    msgs = [(U, 1, 1, 0, 0, 33), (CRIT, 1, 1, 0, 0, 33), (FAINT, 1, 1, 0, 0, 33),      # opp crit KOs us
            (U, 0, 0, 1, 0, 85), (MISS, 0, 0, 1, 0, 85),                                  # our move misses
            (U, 1, 1, 0, 0, 58), (FRZ, 1, 1, 0, 0, 58),                                   # we get frozen (effect 0)
            (PARA, 1, 1, 0, 0, 0),                                                         # the opponent is fully paralyzed
            (U, 1, 1, 0, 0, A._MOVE_ID["MOVE_EXPLOSION"])]
    keys = A.event_keys(A.parse_messages(msgs))
    assert keys == {"opp_crit", "opp_crit_ko", "own_miss", "own_freeze", "opp_full_para", "opp_explosion"}
    c = {k: 1 for k in keys}
    # bad for us: opp crit, opp crit KO, our miss, our freeze; good: opp full paralysis
    assert A.luck_balance(c) == 4 - 1


def test_message_log_records_a_battle_and_clones_do_not_log():
    import random
    be = SimBackend(max_turns=10 ** 9)
    be.reset(seed=11, win_streak=28)
    be.game.log_messages(True)
    rng = random.Random(0)
    seen = []
    while be.phase != Phase.RUN_OVER and len(seen) < 400:
        v = be.view()
        if be.phase == Phase.RENTAL:
            sp = [m.species for m in v.candidates]
            be.act(next(t for t in [(a, b, c) for a in range(6) for b in range(6) for c in range(6)]
                        if len({sp[t[0]], sp[t[1]], sp[t[2]]}) == 3 and len(set(t)) == 3))
        elif be.phase == Phase.SWAP:
            be.act(None)
        else:
            cl = be.game.clone()
            be.act(rng.choice([a for a in v.legal_actions if a[0] != "forfeit"]))
            assert cl.take_messages() == []                 # a clone starts with an empty, disabled log
        seen += be.game.take_messages()
    sids = {m[0] for m in seen}
    assert A.STRINGIDS["USEDMOVE"] in sids and all(len(m) == 6 for m in seen)
    assert be.game.take_messages() == []                    # taken: cleared


# ---- labels and statistics -------------------------------------------------------------------------------------

def _cfg(**kw):
    cfg = A.config_of(A.build_parser().parse_args([]))
    cfg.update(kw)
    return cfg


def test_label_precedence():
    cfg = _cfg()
    mk = lambda found, delta=0.5: {"legal": {"found": found, "delta": delta}}
    assert A.label({"p0": 0.3, "mistake": mk(True), "team": {"gain": 0.6}}, cfg)[0] == "battler_error"
    assert A.label({"p0": 0.3, "mistake": mk(False), "team": {"gain": 0.6}}, cfg)[0] == "team"
    assert A.label({"p0": 0.9, "mistake": mk(False), "team": {"gain": 0.05}}, cfg)[0] == "luck"
    assert A.label({"p0": 0.6, "mistake": mk(False), "team": {"gain": 0.1}}, cfg)[0] == "mixed"
    assert A.label({"p0": 0.2, "mistake": mk(False), "team": {"gain": 0.1}}, cfg)[0] == "bad_matchup"
    # the mistake size threshold and the mode used for the label
    assert A.label({"p0": 0.9, "mistake": mk(True, 0.1)}, _cfg(mistake_delta=0.2))[0] == "luck"
    perf = {"perfect": {"found": True, "delta": 0.5}, "legal": {"found": False}}
    assert A.label({"p0": 0.9, "mistake": perf}, cfg)[0] == "luck"
    assert A.label({"p0": 0.9, "mistake": perf}, _cfg(label_mistake_mode="either"))[0] == "battler_error"


def test_bootstrap_fractions():
    labs = ["luck"] * 30 + ["team"] * 10
    out = A.bootstrap_fractions(labs, A.LABELS, B=500)
    assert out["luck"]["frac"] == 0.75 and out["luck"]["lo"] < 0.75 < out["luck"]["hi"]
    assert out["mixed"]["frac"] == 0.0 and out["team"]["n"] == 10


def test_estimate_scales_with_budget():
    a = A.estimate(_cfg(strong_sims=1024), 500)
    b = A.estimate(_cfg(strong_sims=2048), 500)
    assert b["hours"] > 1.8 * a["hours"] > 0


# ---- perfect information: only in this diagnostic entry point ---------------------------------------------------

def test_autopsy_is_not_a_training_entry_point():
    assert "rl.autopsy" not in search._TRAINING_MAINS
    src = open(os.path.join(ROOT, "rl", "autopsy.py")).read()
    assert "mark_training()" not in src
    # no training module imports the autopsy (its perfect-information battler never reaches training)
    for f in os.listdir(os.path.join(ROOT, "rl")):
        if f.endswith(".py") and f != "autopsy.py":
            assert "autopsy" not in open(os.path.join(ROOT, "rl", f)).read(), f


def test_autopsy_refuses_to_run_in_a_training_process(monkeypatch):
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    with pytest.raises(SystemExit):
        A.main(["--estimate"])


@needs_ckpt
def test_perfect_battler_allowed_here_refused_in_training(monkeypatch):
    W = A.Worker(_cfg(ckpt=CKPT, features=False, strong_sims=8, strong_dets=1))
    assert W.search("perfect", 8, 1).mode == "perfect"              # diagnostic process: allowed
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    W._search.clear()
    with pytest.raises(PermissionError):
        W.search("perfect", 8, 1)                                   # a training process: refused


# ---- end to end, replay determinism -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    if not os.path.exists(CKPT):
        pytest.skip("runs/alphazero_v3/latest.pt not present")
    out = tmp_path_factory.mktemp("autopsy")
    cwd = os.getcwd()
    os.chdir(ROOT)                          # the checkpoint's W path is relative to the repository
    try:
        summary = A.main(TINY + ["--ckpt", CKPT, "--out-dir", str(out)])
    finally:
        os.chdir(cwd)
    return out, summary


def test_tiny_end_to_end(tiny):
    out, s = tiny
    for f in ("collect.jsonl", "analysis.jsonl", "summary.json", "losses.csv", "report.md"):
        assert os.path.exists(os.path.join(out, f)), f
    assert s["n_losses"] == 2 and s["n_analyzed"] == 2 and not s["errors"]
    assert abs(sum(v["frac"] for v in s["labels"].values()) - 1.0) < 1e-9
    ana = A._load_jsonl(os.path.join(out, "analysis.jsonl"))
    for a in ana:
        assert a["label"] in A.LABELS and 0.0 <= a["p0"] <= 1.0 and len(a["p0_outcomes"]) == 3
        assert set(a["mistake"]) == {"legal", "perfect"}
        t = a["team"]
        assert t["n_options"] >= 2 and 0 <= t["our_option"] < t["n_options"] and len(t["alternatives"]) == 1
    recs = A._load_jsonl(os.path.join(out, "collect.jsonl"))
    loss = next(r for r in recs if not r["won"])
    assert loss["log"] and loss["turns"] and loss["round"] in (5, 6) and len(loss["opp"]) == 3
    assert all(m["sets"] for m in loss["opp"])                        # the true team: real Factory sets
    assert "## Etiquetas" in open(os.path.join(out, "report.md")).read()
    # rebuilt from the saved results
    s2 = A.main(["--out-dir", str(out), "--report-only", "--ckpt", CKPT, "--luck-seeds", "3",
                 "--min-pattern-losses", "1"])
    assert s2["labels"] == s["labels"]


def test_replay_is_exact_in_a_fresh_process(tiny, tmp_path):
    out, _ = tiny
    loss = next(r for r in A._load_jsonl(os.path.join(out, "collect.jsonl")) if not r["won"])
    p = tmp_path / "loss.json"
    p.write_text(json.dumps(loss))
    code = ("import json, sys; from rl import autopsy as A; rec = json.load(open(sys.argv[1])); "
            "pre, st = A.replay_battle(rec); print(len(st))")
    r = subprocess.run([sys.executable, "-c", code, str(p)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert int(r.stdout.strip()) == len(loss["turns"])


def test_battle_start_reproduces_the_real_battle(tiny):
    """start_battle with the actual option, the true team written back and the game's own RNG (seed None), then
    our logged actions: the very same battle (same end state)."""
    out, _ = tiny
    for loss in (r for r in A._load_jsonl(os.path.join(out, "collect.jsonl")) if not r["won"]):
        pre, states = A.replay_battle(loss)
        act = A.dec_action(loss["pre_action"])
        be = A.start_battle(pre, act, A.true_enemy_bytes(pre, act), None)
        for t in loss["turns"]:
            be.act(A.dec_action(t["action"]))
        assert be.phase == Phase.RUN_OVER and not be.last_battle_won
        assert A.state_hash(be.game) == loss["end_hash"]


def test_same_seed_same_outcome(tiny):
    out, _ = tiny
    cfg = _cfg(ckpt=CKPT, features=False, strong_sims=16, strong_dets=2)
    W = A.Worker(cfg)
    loss = next(r for r in A._load_jsonl(os.path.join(out, "collect.jsonl")) if not r["won"])
    pre, states = A.replay_battle(loss)
    act = A.dec_action(loss["pre_action"])
    enemy = A.true_enemy_bytes(pre, act)
    seeds = [A.stable_seed("t", i) for i in range(4)]
    for mode in ("net", "perfect", "legal"):
        c1, c2 = A.Counters(), A.Counters()
        p1, z1 = A.p_start(W, pre, act, enemy, seeds, mode, c1, "x")
        p2, z2 = A.p_start(W, pre, act, enemy, seeds, mode, c2, "x")
        assert z1 == z2 and c1.decisions == c2.decisions, mode
    # rewinds: the same seed replays the same continuation
    t, b = [(t, b) for t, b in states if t["n_choices"] >= 2][-1]
    q1 = A.p_from(W, b, 0, seeds[:2], "perfect")
    q2 = A.p_from(W, b, 0, seeds[:2], "perfect")
    assert q1 == q2
    # the true team is written back: every start faces the recorded opponent
    be = A.start_battle(pre, act, enemy, seeds[0])
    from pybattle.emu.decode import SYMBOLS as S, decode_party
    foe = decode_party(be.game.read(S.addr("gEnemyParty"), 300))
    assert [m.species for m in foe[:3]] == [m["species"] for m in loss["opp"]]


def test_team_alternatives_use_the_true_opponent(tiny):
    """Every option of the decision before the battle starts against the same (true) opponent team."""
    out, _ = tiny
    from rl import encode
    from rl.search import run_ctx
    from rl.tactician_search import options_of
    from pybattle.emu.decode import SYMBOLS as S, decode_party
    A.Worker(_cfg(ckpt=CKPT, features=False))           # (sets the checkpoint's encoding version)
    loss = next(r for r in A._load_jsonl(os.path.join(out, "collect.jsonl")) if not r["won"])
    pre, _ = A.replay_battle(loss)
    act = A.dec_action(loss["pre_action"])
    enemy = A.true_enemy_bytes(pre, act)
    kind = loss["pre_kind"]
    view, ctx = pre.view(), run_ctx(pre)
    obs = encode.rental(view, ctx) if kind == "rental" else encode.swap(view, ctx)
    opts = options_of(kind, obs)[:6]
    want = [m["species"] for m in loss["opp"]]
    for o in opts:
        be = A.start_battle(pre, A.option_action(kind, o), enemy, 123)
        assert be.phase in (Phase.BATTLE, Phase.FORCED_SWITCH, Phase.SWAP, Phase.RUN_OVER)
        if be.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
            assert [m.species for m in decode_party(be.game.read(S.addr("gEnemyParty"), 300))[:3]] == want


def test_find_mistake_takes_the_latest_clear_mistake(monkeypatch):
    """Backwards scan: skip agreements, skip differing decisions whose position was already lost (P < 0.5) or where
    our move did as well (delta < mistake_delta), stop at the latest clear mistake."""
    class B:
        def clone(self):
            return self

    class Strong:
        def __init__(self):
            self.search = type("S", (), {"last_stats": {"q": [0.0] * 7}})()

        def __call__(self, be, ndec):
            return {5: ("move", 0), 4: ("move", 1), 3: ("move", 1), 2: ("move", 1), 1: ("move", 1)}[be.li]

    class W:
        cfg = _cfg(max_rewinds=5, mistake_seeds=4, mistake_delta=0.25)

        def strong(self, mode, seed):
            return Strong()

    states = []
    for li in range(1, 6):
        b = B()
        b.li = li
        states.append(({"li": li, "turn": li, "n_choices": 3, "action": ["move", 0], "forced": False,
                        "act": [{"species": 1}, {"species": 2}]}, b))
    # li 5: agrees; li 4: differs, lost position; li 3: differs, P 0.75 but ours as good; li 2: the mistake
    p_d = {4: 0.25, 3: 0.75, 2: 1.0, 1: 1.0}
    p_our = {3: 0.75, 2: 0.25, 1: 0.0}

    def fake_p_from(W_, b, ndec, seeds, mode, first_action=None, cnt=None, phase="", stop_below=None):
        return (p_our[b.li] if first_action is not None else p_d[b.li]), len(seeds)

    monkeypatch.setattr(A, "p_from", fake_p_from)
    rec = {"uid": "u", "own": [{"species": 1, "moves": [1, 2, 3, 4]}]}
    res = A.find_mistake(W(), rec, states, "legal", A.Counters())
    assert res["found"] and res["turn"] == 2 and res["delta"] == 0.75
    assert res["p_found"] and res["p_found_turn"] == 3            # the bare definition stops at li 3
    assert res["compared"] == 4 and res["agree"] == 1 and res["scanned"] == 3
    assert res["move_kind"] == "move:" + A.move_name(1) and res["from_end"] == 3
