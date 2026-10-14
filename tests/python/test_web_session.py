"""The browser build's FactorySession (src/web/factory_session.cpp) plays exactly like SimBackend: same phases,
same views (pybattle/web_schema.py JSON), same run info and the same battle encoding, on random play."""

import json
import random

import numpy as np
import pytest

from pybattle.backend import Phase, SimBackend
from pybattle.pybattle_native import FactorySession
from pybattle.web_schema import view_json
from rl import damage, encode
from rl.real_agents import ctx_of

PHASES = [Phase.RENTAL, Phase.SWAP, Phase.BATTLE, Phase.FORCED_SWITCH, Phase.RUN_OVER]


def _random_action(rng, phase, view):
    if phase == Phase.RENTAL:
        while True:
            a = rng.sample(range(6), 3)
            if len({view.candidates[i].species for i in a}) == 3:
                return tuple(a)
    if phase == Phase.SWAP:
        if rng.random() < 0.5:
            return None
        own = {m.species for m in view.own_party}
        opts = [(i, j) for i in range(3) for j in range(3)
                if view.enemy_party[j].species not in own - {view.own_party[i].species}]
        return rng.choice(opts) if opts else None
    legal = [a for a in view.legal_actions if a[0] != "forfeit"]
    if rng.random() < 0.002 and phase == Phase.BATTLE:
        return ("forfeit",)
    return rng.choice(legal)


def _act(fs, action, phase):
    if phase == Phase.RENTAL:
        fs.act_rental(*action)
    elif phase == Phase.SWAP:
        fs.act_swap(-1, 0) if action is None else fs.act_swap(*action)
    elif action[0] == "move":
        fs.act_move(action[1])
    elif action[0] == "switch":
        fs.act_switch(action[1])
    else:
        fs.act_forfeit()


def _same_encoding(py, cpp):
    for k in ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask"):
        a, b = np.asarray(py[k]), np.asarray(cpp[k])
        assert a.shape == b.shape, k
        assert np.array_equal(a, b) if a.dtype.kind in "biu" else np.allclose(a, b, atol=1e-6, rtol=0), k
    assert int(py["active"]) == int(cpp["active"])


@pytest.mark.parametrize("seed", range(200))
def test_session_matches_simbackend(seed):
    old = encode.VERSION
    encode.set_version(5)
    try:
        rng = random.Random(seed)
        streak = rng.choice([0, 3, 7, 13, 20, 27, 41, 48])
        sim, fs = SimBackend(max_turns=150), FactorySession(True, 150)
        sim.reset(seed, streak)
        fs.reset(seed, streak, 0)
        for _ in range(3000):
            assert PHASES[fs.phase] == sim.phase
            view = sim.view()
            assert json.loads(fs.view_json()) == view_json(view)
            if sim.phase == Phase.RUN_OVER:
                break
            info = sim.run_info()
            web_info = json.loads(fs.run_info_json())
            assert isinstance(web_info.pop("trainer_id"), int)
            assert web_info == view_json(info)
            if sim.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
                _same_encoding(encode.battle(view, ctx_of(info)), fs.encode_battle())
                own, foe, threat, speed = damage.battle_estimates(view, ctx_of(info), 4)
                est = json.loads(fs.estimates_json())
                assert np.allclose(est["own"], own, atol=1e-5)
                assert np.allclose(est["threat"], threat, atol=1e-5) and np.allclose(est["speed"], speed, atol=1e-5)
                assert np.allclose(np.reshape([[f["min"], f["max"], f["ko"]] for f in est["foe"]], (-1, 3)), np.reshape(foe, (-1, 3)), atol=1e-5)
            action = _random_action(rng, sim.phase, view)
            phase = sim.phase
            sim.act(action)
            _act(fs, action, phase)
            assert fs.wins == sim.wins
    finally:
        encode.set_version(old)


def test_save_load_resumes_identically():
    rng = random.Random(7)
    fs = FactorySession(True, 150)
    fs.reset(11, 0, 0)
    for _ in range(25):
        view = json.loads(fs.view_json())
        phase = PHASES[fs.phase]
        if phase == Phase.RUN_OVER:
            break
        if phase == Phase.RENTAL:
            _act(fs, (0, 1, 2), phase)
        elif phase == Phase.SWAP:
            _act(fs, None, phase)
        else:
            _act(fs, tuple(rng.choice([a for a in view["legal_actions"] if a[0] != "forfeit"])), phase)
    state = fs.save_state()
    a, b = FactorySession(True, 150), FactorySession(True, 150)
    a.load_state(state)
    b.load_state(state)
    for _ in range(200):
        assert a.view_json() == b.view_json() and a.phase == b.phase
        if PHASES[a.phase] == Phase.RUN_OVER:
            break
        view = json.loads(a.view_json())
        phase = PHASES[a.phase]
        act = (0, 1, 2) if phase == Phase.RENTAL else None if phase == Phase.SWAP else \
            tuple(rng.choice([x for x in view["legal_actions"] if x[0] != "forfeit"]))
        _act(a, act, phase)
        _act(b, act, phase)
