import pytest

import pybattle


def test_native_module_exposes_engine():
    assert hasattr(pybattle, "BattleEngine")
    assert hasattr(pybattle, "FactoryGenerator")


@pytest.mark.xfail(strict=True, reason="fainted mons are never replaced (engine onFaint TODO)")
def test_battle_runs_and_terminates():
    gen = pybattle.FactoryGenerator(1)
    engine = pybattle.BattleEngine()
    engine.reset(1)
    pool = gen.generate_player_team(0, True)
    engine.set_player_team(pool[:3])
    engine.set_opponent_team(gen.generate_opponent_team(0, 0, True))
    for _ in range(500):
        legal = engine.get_legal_actions()
        result = engine.step(legal[0])
        if result.done:
            break
    assert result.done, "battle must end within 500 turns"
