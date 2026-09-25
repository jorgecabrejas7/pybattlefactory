"""The BattleView must come out identical on the real game (mGBA) and in the simulator.

A few battles are played on the ROM with a random policy; at every decision the emulator's
view is built by BattleObserver from the emulator's RAM. The battle is then replayed in the
simulator from the same start, and the simulator's views must be equal field for field,
including the observer's memory (turn events, counters, reveals).

Needs the ROM and save (PYBATTLE_ROM / PYBATTLE_SAVE, or the default location); skipped otherwise.
"""

import dataclasses
import os
import random

import pytest

_ROMS = "/home/apollo/Desktop/Dev/pokemon-battle-factory/roms"
ROM = os.environ.get("PYBATTLE_ROM", os.path.join(_ROMS, "Pokemon - Emerald Version (USA, Europe).gba"))
SAVE = os.environ.get("PYBATTLE_SAVE", os.path.join(_ROMS, "Pokemon - Emerald Version (USA, Europe).sav"))

pytestmark = pytest.mark.skipif(not (os.path.exists(ROM) and os.path.exists(SAVE)), reason="ROM/save not available")


def _record(n_battles: int, seed: int):
    from pybattle.emu.driver import Decision, FactoryDriver
    from pybattle.emu.trace import random_policy, record_battle
    from pybattle.emu_backend import _FieldOrderRam, can_switch, unusable_moves
    from pybattle.view import BattleObserver

    d = FactoryDriver(ROM, SAVE)
    d.boot_to_overworld()
    lobby = d.save_state()
    rng = random.Random(seed)
    out = []
    for _ in range(n_battles):
        d.load_state(lobby)
        d.emu.run_frames(rng.randrange(1, 2000))
        d.set_factory_streak(rng.choice([0, 21, 42]), rents=rng.randrange(0, 50))
        d.start_challenge(open_level=True)
        hints = d.hints()
        d.pick_rentals(rng.sample(range(6), 3))
        obs = BattleObserver(*hints)
        views = []
        policy = random_policy(rng)

        def observing_policy(drv, dec):
            v = obs.observe(_FieldOrderRam(drv), dec == Decision.PARTY_MENU, unusable_moves(drv.emu), can_switch(drv.emu))
            views.append(dataclasses.asdict(v))
            return policy(drv, dec)

        trace = record_battle(d, observing_policy)
        late = obs.copy()
        obs.finish(_FieldOrderRam(d))                  # at the frame the battle was decided
        # EmuBackend notices the outcome in advance_factory, up to 8 frames later (pressing A): the same records
        d.mash_until(lambda: False, max_frames=16)
        late.finish(_FieldOrderRam(d))
        assert late.foe_records() == obs.foe_records()
        out.append((trace, hints, views, [dataclasses.asdict(r) for r in obs.foe_records()]))
    return out


def _replay_views(trace, hints):
    from pybattle.diff import setup_battle
    from pybattle.pybattle_native import Gen3Game
    from pybattle.view import BattleObserver

    g = setup_battle(trace["start"])
    obs = BattleObserver(*hints)
    views = []
    for dec in trace["decisions"]:
        kind = g.run()
        forced = kind == Gen3Game.Decision.SWITCH
        views.append(dataclasses.asdict(obs.observe(g, forced, g.unusable_moves(0), g.can_switch(0))))
        action = dec["action"]
        if "move" in action:
            g.choose_move(action["move"])
        else:
            g.choose_switch(action["switch"])
    assert g.run() == Gen3Game.Decision.BATTLE_OVER
    obs.finish(g)
    return views, [dataclasses.asdict(r) for r in obs.foe_records()]


def test_views_identical_on_emulator_and_simulator():
    compared = 0
    records = 0
    for trace, hints, emu_views, emu_records in _record(n_battles=3, seed=7):
        sim_views, sim_records = _replay_views(trace, hints)
        assert len(sim_views) == len(emu_views)
        for i, (e, s) in enumerate(zip(emu_views, sim_views)):
            diff = {k: (e[k], s[k]) for k in e if e[k] != s[k]}
            assert not diff, f"decision {i}: emulator != simulator: {diff}"
            compared += 1
        # the defeated-opponent records of the swap view (BattleObserver.finish at the end of the battle)
        assert emu_records == sim_records, f"records: emulator {emu_records} != simulator {sim_records}"
        records += sum(r["turns"] > 0 for r in emu_records)
    assert compared > 10 and records > 0
