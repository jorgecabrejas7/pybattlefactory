"""rl/playlog.py (the raw logger), rl/playknow.py + rl/playstats.py (the metrics) and rl/play_sim.py (the simulator
replica): docs/RL_DECISIONS.md §23."""

import gzip
import json
import os

import numpy as np
import pytest

from pybattle.view import OwnMon, RentalView, RunInfo, SeenMon, SPECIES, NAMES
from rl import playknow as K
from rl import playstats as PS
from rl.playlog import PlayLogger, iter_runs, read_log, strip_truth, to_jsonable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(ROOT, "runs", "alphazero_v3", "latest.pt")
OPP = os.path.join(ROOT, "runs", "opp_model_v3", "model.pt")
needs_ckpt = pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(OPP)),
                                reason="runs/alphazero_v3/latest.pt or runs/opp_model_v3/model.pt not present")

SP = {v.replace("SPECIES_", ""): int(k) for k, v in NAMES["species"].items()}
MV = {v.replace("MOVE_", ""): int(k) for k, v in NAMES["moves"].items()}


def own(species, moves, item=0, ability=0, stats=(100, 100, 100, 100, 100), max_hp=300):
    sp = SP[species]
    return OwnMon(sp, 100, max_hp, max_hp, list(stats), [MV[m] for m in moves] + [0] * (4 - len(moves)),
                  [10, 10, 10, 10], item, ability, 0, [31] * 6, [0] * 6, 0, list(SPECIES[sp]["types"]))


def mon(species, moves, ability=0):
    return K.own_mon(to_jsonable(own(species, moves, ability=ability)))


# ---- logger ----------------------------------------------------------------------------------------------------------

def _rental_view():
    cands = [own("SNORLAX", ["BODY_SLAM", "CURSE", "REST", "SHADOW_BALL"]),
             own("STARMIE", ["SURF", "THUNDERBOLT", "ICE_BEAM", "RECOVER"]),
             own("SKARMORY", ["SPIKES", "DRILL_PECK", "TOXIC", "PROTECT"]),
             own("TYRANITAR", ["ROCK_SLIDE", "EARTHQUAKE", "DRAGON_DANCE", "CRUNCH"]),
             own("VENUSAUR", ["SLEEP_POWDER", "GIGA_DRAIN", "SLUDGE_BOMB", "SYNTHESIS"]),
             own("ARCANINE", ["FLAMETHROWER", "EXTREME_SPEED", "CRUNCH", "IRON_TAIL"])]
    return RentalView(cands, [1, 2, 3, 4, 5, 6], 11, 4)


def test_round_trip_and_truth_separated(tmp_path):
    path = str(tmp_path / "x.jsonl.gz")
    log = PlayLogger(path, {"ckpt": "c", "battler": "net", "arr": np.arange(3)})
    v = _rental_view()
    ri = RunInfo(0, 0, 0, True, 0, 0, False)
    pol = {"probs": np.array([0.25, 0.75], np.float32), "value": np.float32(1.5), "legal": [(0, 1), (1, 2)]}
    extras = {"features": np.ones((2, 7)), "hint": (11, 4), "set": {3, 1}, "nan": float("nan"),
              "truth": {"opp_team": [{"species": 143}]}, "obj": object()}
    log.decision("rental", v, ri, pol, (0, 1, 2), extras)
    log.battle_end(True, {"turns": 5, "events": {"own_crit": 1}, "truth": {"opp_team": [1, 2, 3]}})
    log.decision("swap", v, ri, pol, None, {})
    log.battle_end(False, {})
    log.run_end(1, {"start_round": 1})
    log.close()
    recs = list(read_log(path))
    assert [r["t"] for r in recs] == ["meta", "decision", "battle_end", "decision", "battle_end", "run_end"]
    meta = recs[0]
    assert meta["meta"]["arr"] == [0, 1, 2] and meta["truth_fields"] == ["truth"]
    assert "view" in meta["player_view_fields"]
    d = recs[1]
    assert d["kind"] == "rental" and d["action"] == [0, 1, 2] and d["battle"] == 0 and d["run"] == 0
    assert d["view"]["candidates"][0]["species"] == SP["SNORLAX"] and d["view"]["hint_type"] == 11
    assert d["policy"]["probs"] == pytest.approx([0.25, 0.75]) and d["policy"]["legal"] == [[0, 1], [1, 2]]
    assert d["extras"]["hint"] == [11, 4] and d["extras"]["set"] == [1, 3] and d["extras"]["nan"] is None
    assert "__repr__" in d["extras"]["obj"]
    # the truth is only in the record's own "truth" field, never inside the player's fields
    assert d["truth"] == {"opp_team": [{"species": 143}]} and "truth" not in d["extras"]
    assert recs[2]["truth"] == {"opp_team": [1, 2, 3]} and "truth" not in recs[2]["info"]
    for r in recs[1:]:
        assert "truth" not in json.dumps(strip_truth(r))
    assert recs[3]["action"] is None and recs[3]["battle"] == 1 and "truth" not in recs[3]
    assert recs[5]["streak"] == 1 and recs[5]["battles"] == 2
    runs = list(iter_runs([path]))
    assert len(runs) == 1 and len(runs[0]["battles"]) == 2
    assert runs[0]["battles"][0]["pre"][0]["kind"] == "rental" and runs[0]["battles"][1]["end"]["won"] is False
    with pytest.raises(ValueError):
        PlayLogger(str(tmp_path / "y.jsonl"), {}).decision("bogus", v, ri, pol, None, {})


def test_truncated_gzip_is_read_up_to_the_break(tmp_path):
    path = str(tmp_path / "t.jsonl.gz")
    log = PlayLogger(path, {})
    for i in range(50):
        log.battle_end(True, {"i": i})
    log.run_end(50, {})
    log.close()
    raw = open(path, "rb").read()
    open(path, "wb").write(raw[: len(raw) // 2])
    recs = list(read_log(path))
    assert 0 < len(recs) < 52
    assert list(iter_runs([path])) == []                 # an incomplete run is skipped


# ---- game knowledge / metrics on hand-made teams ------------------------------------------------------------------------

def test_coverage_hand_made():
    t = [mon("CHARIZARD", ["FLAMETHROWER"]), mon("BLASTOISE", ["SURF"]), mon("VENUSAUR", ["GIGA_DRAIN"])]
    d = K.def_coverage(t)
    # fire / water / grass: water resisted (Charizard? no: fire x0.5 vs water... Blastoise and Venusaur resist water)
    fire, water, grass, ground, rock, electric = 10, 11, 12, 4, 5, 13
    assert K.resist_count(t, water) == 2 and K.resist_count(t, ground) == 1 and K.resist_count(t, fire) == 2
    assert d["resisted"] + d["unresisted"] == 17 and d["resisted_phys"] + d["resisted_spec"] == d["resisted"]
    # rock: Charizard x4 and nobody else weak -> not shared; electric: Charizard (flying) and Blastoise weak
    assert K.effectiveness(electric, t[0]["types"]) == 2 and K.effectiveness(electric, t[1]["types"]) == 2
    assert d["shared_weak_spec"] >= 1
    o = K.off_coverage(t)
    # fire hits grass/ice/bug/steel; water hits fire/ground/rock; grass hits water/ground/rock: 4 + 3 + 1
    assert o["se"] == 8 and o["se_phys"] == 0 and o["se_spec"] == 8
    assert K.synergies(t)["core_fwg"]
    assert K.se_count(t, fire) == 1 and K.se_count(t, rock) == 2


def test_levitate_counts_as_resist():
    g = mon("GENGAR", ["SHADOW_BALL"], ability=K.ability_id("LEVITATE"))
    assert K.effectiveness(4, g["types"], g["ability"]) == 0
    assert K.resist_count([g], 4) == 1


def test_roles_fit_and_archetype():
    assert K.role_of(mon("TYRANITAR", ["ROCK_SLIDE", "EARTHQUAKE", "DRAGON_DANCE", "CRUNCH"])) == "setup"
    assert K.role_of(mon("SKARMORY", ["SPIKES", "DRILL_PECK", "TOXIC", "PROTECT"])) == "support"
    assert K.role_of(mon("STARMIE", ["SURF", "THUNDERBOLT", "ICE_BEAM", "RECOVER"])) == "fast"
    assert K.role_of(mon("STEELIX", ["EARTHQUAKE", "IRON_TAIL", "ROCK_SLIDE", "EXPLOSION"])) == "wall"
    assert K.role_of(mon("RHYDON", ["EARTHQUAKE", "ROCK_SLIDE", "MEGAHORN", "SURF"])) == "slow_attacker"
    # Alakazam (SpA 135 > Atk 50): special moves fit, a physical one does not; in Gen 3 Fire Punch is special (its
    # type decides), Brick Break physical
    assert K.stat_move_fit(mon("ALAKAZAM", ["PSYCHIC", "FIRE_PUNCH"])) == 1.0
    assert K.stat_move_fit(mon("ALAKAZAM", ["PSYCHIC", "BRICK_BREAK"])) == 0.5
    assert K.stat_move_fit(mon("ALAKAZAM", ["RECOVER"])) is None
    fast = [mon("JOLTEON", ["THUNDERBOLT"]), mon("ALAKAZAM", ["PSYCHIC"]), mon("CROBAT", ["SLUDGE_BOMB"])]
    slow = [mon("SNORLAX", ["BODY_SLAM"]), mon("STEELIX", ["EARTHQUAKE"]), mon("SLOWBRO", ["SURF"])]
    assert K.archetype(fast) == "fast_frail" and K.archetype(slow) == "bulky_slow"


def test_synergies_hand_made():
    rain = [mon("POLITOED", ["RAIN_DANCE", "HYPNOSIS", "ICE_BEAM", "TOXIC"]), mon("KINGDRA", ["SURF"]),
            mon("SNORLAX", ["BODY_SLAM"])]
    s = K.synergies(rain)
    assert s["rain"] and not s["sun"]
    assert not K.synergies([mon("POLITOED", ["RAIN_DANCE"]), mon("SNORLAX", ["BODY_SLAM"]),
                            mon("ARCANINE", ["FLAMETHROWER"])])["rain"]
    bp = [mon("JOLTEON", ["BATON_PASS", "AGILITY", "THUNDERBOLT"]), mon("SNORLAX", ["BODY_SLAM"]),
          mon("STEELIX", ["EARTHQUAKE"])]
    assert K.synergies(bp)["baton_pass"]
    sand = [mon("TYRANITAR", ["CRUNCH"], ability=K.ability_id("SAND_STREAM")), mon("STEELIX", ["EARTHQUAKE"]),
            mon("SNORLAX", ["BODY_SLAM"])]
    assert K.synergies(sand)["sand"]


def test_status_failures():
    tw, tox, wow, seed, cr = MV["THUNDER_WAVE"], MV["TOXIC"], MV["WILL_O_WISP"], MV["LEECH_SEED"], MV["CONFUSE_RAY"]
    ground, steel, fire, grass, normal = [4], [8], [10], [12], [0]
    f = K.status_failure
    assert f(tw, ground, 0, 0, False, False, False) == "immune_type"
    assert f(tw, normal, 0, 0, False, False, False) is None
    assert f(tw, normal, 5, 0, False, False, False) == "already_status"
    assert f(tw, normal, 0, 0, False, False, True) == "substitute"
    assert f(tw, normal, 0, K.ability_id("LIMBER"), False, False, False) == "immune_ability"
    assert f(tox, steel, 0, 0, False, False, False) == "immune_type"
    assert f(wow, fire, 0, 0, False, False, False) == "immune_type"
    assert f(seed, grass, 0, 0, False, False, False) == "immune_type"
    assert f(seed, normal, 0, 0, False, True, False) == "already_seeded"
    assert f(cr, normal, 0, 0, True, False, False) == "already_confused"
    assert f(tw, normal, 0, 0, False, False, False, safeguard=True) == "safeguard"
    assert f(MV["SURF"], normal, 0, 0, False, False, False) is None


def test_hidden_power_type():
    assert K.hidden_power_type([31] * 6) == 17           # all odd: dark
    assert K.hidden_power_type([30] * 6) == 1            # all even: fighting


def test_bootstrap_and_controlled_difference():
    b = PS.Boot(8, B=500, seed=1)
    x = [1, 1, 0, 0, 1, 0, 1, 1]
    cl = [0, 0, 1, 1, 2, 2, 3, 3]
    m = b.mean(x, cl)
    assert m["m"] == pytest.approx(5 / 8) and m["lo"] <= m["m"] <= m["hi"] and m["n"] == 8
    assert b.mean([None, 1.0], [0, 1])["n"] == 1
    # within each stratum group A wins 1.0 and group B 0.0: the controlled difference is exactly 1
    x = [1, 0, 1, 0]
    g = [True, False, True, False]
    st = ["a", "a", "b", "b"]
    d = b.diff(x, [0, 1, 2, 3], g, st)
    assert d["m"] == pytest.approx(1.0) and d["n1"] == 2 and d["n0"] == 2
    # Simpson: a raw difference that the strata remove
    x = [1, 1, 1, 0, 0, 0, 0, 1]
    g = [True, True, False, False, True, False, False, False]
    st = ["hi", "hi", "hi", "hi", "lo", "lo", "lo", "lo"]
    d = b.diff(x, list(range(8)), g, st)
    raw = np.mean([v for v, gg in zip(x, g) if gg]) - np.mean([v for v, gg in zip(x, g) if not gg])
    assert d["m"] == pytest.approx(0.5 * (1 - 0.5) + 0.5 * (0 - 1 / 3))
    assert d["m"] != pytest.approx(raw)


def test_percentile_of_a_choice():
    assert PS._percentile(3, [1, 2, 3], False) == pytest.approx((1 + 1 + 0.5) / 3)
    assert PS._percentile(1, [1, 2, 3], True) == pytest.approx((0.5 + 1 + 1) / 3)


# ---- end to end: the simulator replica, logged and analyzed ------------------------------------------------------------

@needs_ckpt
def test_tiny_sim_run_and_report(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from rl import play_sim
    out = str(tmp_path / "study")
    tot = play_sim.main(["--mode", "net", "--battles", "4", "--procs", "1", "--max-battles", "4", "--out", out,
                         "--nice", "0", "--late-runs", "1"])
    assert tot["battles"] >= 4 and tot.get("errors", 0) == 0
    files = tot["files"]
    recs = [r for f in files for r in read_log(f)]
    kinds = {r.get("kind") for r in recs if r["t"] == "decision"}
    assert {"rental", "battle"} <= kinds
    dec = [r for r in recs if r["t"] == "decision"]
    # tactician decisions carry the per-option features and the probes; battle decisions the estimates
    t = next(r for r in dec if r["kind"] in ("rental", "swap"))
    assert len(t["extras"]["features"]) == len(t["policy"]["legal"]) and "feat_zero" in t["extras"]["probes"]
    b = next(r for r in dec if r["kind"] == "battle")
    assert "estimates" in b["extras"] and "truth" in b and "truth" not in b["extras"]
    assert abs(sum(b["policy"]["probs"]) - 1) < 1e-3
    ends = [r for r in recs if r["t"] == "battle_end"]
    assert all("truth" in r and "opp_team" in r["truth"] for r in ends)
    # the player's side never holds the true opponent team
    for r in dec:
        assert "opp_team" not in json.dumps(strip_truth(r))
    R = PS.main(["--logs", os.path.dirname(files[0]), "--out", str(tmp_path / "rep"), "--boot", "50", "--min-n", "1"])
    md = open(tmp_path / "rep" / "report.md").read()
    for h in ("## 1. Resultados", "## 2. Alquiler", "## 3. Cambios", "## 4. Sinergias", "## 5. Influencia",
              "## 6. Estados", "## 7. Elección", "## 8. Cambios dentro", "## 9. Confianza"):
        assert h in md
    assert R["counts"]["battles"] == tot["battles"]
    assert os.path.exists(tmp_path / "rep" / "stats.json")


@needs_ckpt
def test_search_agent_logs_visits(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from rl.play_agent import PlayAgent
    from rl.play_sim import play_run
    agent = PlayAgent(CKPT, battler="search", sims=16, dets=2, probes=False)
    path = str(tmp_path / "s.jsonl")
    log = PlayLogger(path, agent.meta)
    play_run(agent, log, 4242, 1, max_battles=1)
    log.close()
    b = [r for r in read_log(path) if r["t"] == "decision" and r["kind"] == "battle"]
    assert b and all("search" in r["extras"] and len(r["extras"]["search"]["visits"]) == 7 for r in b)
    assert any(r["extras"]["search"]["searched"] for r in b)
