"""Search core (C++): Gen3Game.sim_step / set_rng / determinize and MctsTree."""

import itertools
import random
import struct

import pytest

from pybattle.backend import Phase, SimBackend
from pybattle.diff import _mask_battle_mons
from pybattle.emu.decode import BATTLE_MON_SIZE, SYMBOLS as S, decode_battle_mon, decode_party
from pybattle.pybattle_native import Gen3Game, MctsTree

D = Gen3Game.Decision


# --- helpers ---------------------------------------------------------------------------

def start_battle(seed, win_streak=0):
    b = SimBackend()
    b.reset(seed=seed, win_streak=win_streak)
    species = [m.species for m in b.view().candidates]
    b.act(next(t for t in itertools.combinations(range(6), 3) if len({species[i] for i in t}) == 3))
    assert b.phase == Phase.BATTLE
    return b


def enemy_party(g):
    return decode_party(g.read(S.addr("gEnemyParty"), 300))


def battle_mon(g, battler):
    base = S.addr("gBattleMons") + battler * BATTLE_MON_SIZE
    return decode_battle_mon(g.read(base, BATTLE_MON_SIZE))


def enemy_active_slot(g):
    return struct.unpack("<4H", g.read(S.addr("gBattlerPartyIndexes"), 8))[1]


def true_set_ids(g):
    return list(struct.unpack("<3H", g.read(S.addr("gFrontierTempParty"), 6)))


def random_step(g, decision, rng):
    """(kind, index) for a random legal decision, from the game itself."""
    party = decode_party(g.read(S.addr("gPlayerParty"), 300))
    active = struct.unpack("<4H", g.read(S.addr("gBattlerPartyIndexes"), 8))[0]
    bench = [i for i, m in enumerate(party) if i != active and m.species and m.hp > 0]
    if decision == D.SWITCH:
        return 1, rng.choice(bench)
    unusable = g.unusable_moves(0)
    moves = [i for i in range(4) if not unusable & (1 << i)]
    if bench and g.can_switch(0) and (not moves or rng.random() < 0.1):
        return 1, rng.choice(bench)
    return 0, (rng.choice(moves) if moves else 0)


def play_out(g, decision, rng, max_steps=2000):
    steps = 0
    while decision in (D.ACTION, D.SWITCH):
        decision, outcome = g.sim_step(*random_step(g, decision, rng))
        steps += 1
        assert steps < max_steps
    assert decision == D.BATTLE_OVER
    return outcome, steps


def advance(g, n, rng):
    """n random decisions; returns the decision reached (ACTION/SWITCH) or None if the battle ended."""
    decision = D.ACTION
    for _ in range(n):
        decision, _ = g.sim_step(*random_step(g, decision, rng))
        if decision == D.BATTLE_OVER:
            return None
    return decision


# --- sim_step ----------------------------------------------------------------------------

BATTLE_RAM = ("gBattleMons", "gEnemyParty", "gPlayerParty", "gDisableStructs", "gBattlerPartyIndexes")


def ram(g, name):
    raw = g.read(S.addr(name), S.size(name))
    # bytes the game copies from uninitialised stack memory (see pybattle/diff.py)
    return _mask_battle_mons(raw) if name == "gBattleMons" else raw


@pytest.mark.parametrize("seed", [1, 5, 11, 23])
def test_sim_step_reproduces_backend(seed):
    b = start_battle(seed, win_streak=seed % 3 * 14)
    g = b.game.clone()
    rng = random.Random(seed)
    steps = 0
    while True:
        view = b.view()
        if b.phase == Phase.FORCED_SWITCH:
            action, step = ("switch", rng.choice(view.switch_targets)), None
        else:
            moves = [i for i, ok in enumerate(view.usable_moves) if ok]
            if view.switch_targets and rng.random() < 0.1:
                action = ("switch", rng.choice(view.switch_targets))
            else:
                action = ("move", rng.choice(moves) if moves else 0)
        step = (0 if action[0] == "move" else 1, action[1])
        decision, outcome = g.sim_step(*step)
        b.act(action)
        steps += 1
        if decision == D.BATTLE_OVER:
            assert outcome in (1, 2)
            assert b.last_battle_won == (outcome == 1)
            break
        assert decision in (D.ACTION, D.SWITCH) and outcome == 0
        assert (decision == D.SWITCH) == (b.phase == Phase.FORCED_SWITCH)
        assert g.rng == b.game.rng
        for name in BATTLE_RAM:
            assert ram(g, name) == ram(b.game, name), name
        assert steps < 2000
    # the Factory run did not advance in the search copy
    assert g.factory_phase == Gen3Game.FactoryPhase.BATTLE
    assert g.factory_info.wins == 0
    assert g.battle_outcome == outcome
    with pytest.raises(RuntimeError):
        g.sim_step(0, 0)


def test_sim_step_rejects_bad_input():
    g = start_battle(2).game.clone()
    with pytest.raises((ValueError, IndexError)):
        g.sim_step(0, 4)
    with pytest.raises(ValueError):
        g.sim_step(2, 0)


# --- set_rng -----------------------------------------------------------------------------

def test_set_rng_changes_outcomes_and_is_repeatable():
    base = start_battle(7).game

    def run(seed):
        g = base.clone()
        g.set_rng(seed)
        assert g.rng == seed
        return play_out(g, D.ACTION, random.Random(0)) + (g.rng,)

    results = {run(s) for s in range(12)}
    assert len(results) > 1
    assert run(3) == run(3)


# --- determinize ---------------------------------------------------------------------------

def true_specs(g, hp_fraction=-1.0):
    ids = true_set_ids(g)
    specs = []
    for slot, mon in enumerate(enemy_party(g)[:3]):
        specs.append((slot, ids[slot], mon.ivs[0], mon.ability_num, hp_fraction))
    return specs


PARTY_FIELDS = ("species", "held_item", "moves", "pp", "evs", "ivs", "ability_num", "status", "level", "hp",
                "max_hp", "stats", "friendship", "experience", "pp_bonuses")
BATTLE_FIELDS = ("species", "stats", "moves", "ivs", "ability_num", "stat_stages", "ability", "types", "pp", "hp",
                 "level", "max_hp", "item", "status1", "status2")


@pytest.mark.parametrize("seed,turns", [(s, s % 8) for s in range(16)])
def test_determinize_with_true_sets_reproduces_the_game(seed, turns):
    b = start_battle(seed, win_streak=seed)
    g = b.game.clone()
    rng = random.Random(seed)
    decision = advance(g, turns, rng)
    if decision is None:
        pytest.skip("battle ended early")
    before_party = enemy_party(g)[:3]
    before_mon = battle_mon(g, 1)
    ids = true_set_ids(g)
    assert all(ids[s] < 882 for s in range(3))

    d = g.clone()
    d.determinize(true_specs(g), -1)
    after_party = enemy_party(d)[:3]
    for a, o in zip(after_party, before_party):
        assert a.checksum_ok
        for f in PARTY_FIELDS:
            assert getattr(a, f) == getattr(o, f), f
    after_mon = battle_mon(d, 1)
    for f in BATTLE_FIELDS:
        assert getattr(after_mon, f) == getattr(before_mon, f), f
    assert d.rng == g.rng

    # The personality is the only difference, so the rest of the battle plays out identically.
    r1, r2 = random.Random(1), random.Random(1)
    dec1 = dec2 = decision
    while dec1 != D.BATTLE_OVER:
        s1, s2 = random_step(g, dec1, r1), random_step(d, dec2, r2)
        assert s1 == s2
        dec1, o1 = g.sim_step(*s1)
        dec2, o2 = d.sim_step(*s2)
        assert (dec1, o1) == (dec2, o2)
        assert g.rng == d.rng


def test_determinize_keep_slot_is_a_no_op():
    g = start_battle(8).game.clone()
    before = g.read(S.addr("gEnemyParty"), 300), g.read(S.addr("gBattleMons"), 352)
    g.determinize([(0, -1, 0, 0, -1.0), (1, -1, 0, 0, -1.0), (2, -1, 0, 0, -1.0)], -1)
    assert (g.read(S.addr("gEnemyParty"), 300), g.read(S.addr("gBattleMons"), 352)) == before


@pytest.mark.parametrize("seed", range(16))
def test_determinize_with_other_sets_changes_the_mons_and_battles_finish(seed):
    rng = random.Random(seed)
    b = start_battle(seed + 30, win_streak=35)
    g = b.game.clone()
    decision = advance(g, rng.randrange(4), rng)
    if decision is None:
        pytest.skip("battle ended early")
    before = enemy_party(g)[:3]
    active = enemy_active_slot(g)
    used = set()
    specs = []
    for slot in range(3):
        while True:
            set_id = rng.randrange(372, 882)
            if set_id not in used:
                break
        used.add(set_id)
        frac = rng.choice([-1.0, 0.5, 1.0])
        specs.append((slot, set_id, rng.choice([3, 6, 31]), rng.randrange(2), frac))
    d = g.clone()
    d.determinize(specs, seed)
    after = enemy_party(d)[:3]
    for (slot, set_id, iv, _, frac), a, o in zip(specs, after, before):
        assert a.checksum_ok and a.ivs == [iv] * 6 and a.friendship == 0
        assert sum(a.evs) in range(500, 511)
        assert a.status == o.status or (o.status & 7)   # sleep counters may be resampled
        if o.hp == 0 and frac < 0:
            assert a.hp == 0
        elif frac > 0:
            assert a.hp == max(1, int(a.max_hp * frac + 0.5))
    mon = battle_mon(d, 1)
    p = after[active]
    assert (mon.species, mon.stats, mon.max_hp, mon.hp) == (p.species, p.stats, p.max_hp, p.hp)
    assert mon.moves == p.moves and mon.ivs == p.ivs
    assert mon.stat_stages == battle_mon(g, 1).stat_stages
    assert mon.status1 & ~7 == battle_mon(g, 1).status1 & ~7
    assert any(a.species != o.species or a.stats != o.stats for a, o in zip(after, before))
    if d.battle_outcome == 0:
        play_out(d, decision, random.Random(seed))


def test_determinize_hp_fraction_and_bad_specs():
    g = start_battle(12).game.clone()
    specs = true_specs(g, hp_fraction=0.25)
    g.determinize(specs, -1)
    for m in enemy_party(g)[:3]:
        assert m.hp == max(1, int(m.max_hp * 0.25 + 0.5))
    assert battle_mon(g, 1).hp == enemy_party(g)[enemy_active_slot(g)].hp
    with pytest.raises(ValueError):
        g.determinize([(3, 0, 3, 0, 1.0)], -1)
    with pytest.raises(ValueError):
        g.determinize([(0, 900, 3, 0, 1.0)], -1)


def test_hidden_counters_are_resampled():
    g = start_battle(13).game.clone()
    status_addr = S.addr("gBattleMons") + BATTLE_MON_SIZE + 0x4C
    g.write(status_addr, struct.pack("<I", 5))                    # asleep, 5 turns left
    g.write(status_addr + 4, struct.pack("<I", 4 | (1 << 4)))     # confused (4) + uproar bits kept
    seen_sleep, seen_conf = set(), set()
    for seed in range(40):
        d = g.clone()
        d.determinize([], seed)
        s1, s2 = struct.unpack("<2I", d.read(status_addr, 8))
        assert 1 <= s1 <= 5 and s1 & ~7 == 0
        assert 1 <= s2 & 7 <= 5 and s2 & ~7 == 1 << 4
        seen_sleep.add(s1)
        seen_conf.add(s2 & 7)
    assert seen_sleep == set(range(1, 6)) and seen_conf == set(range(1, 6))
    d = g.clone()
    d.determinize([], -1)
    assert struct.unpack("<2I", d.read(status_addr, 8)) == (5, 4 | (1 << 4))
    # same seed, same result
    a, c = g.clone(), g.clone()
    a.determinize([], 7)
    c.determinize([], 7)
    assert a.read(status_addr, 8) == c.read(status_addr, 8)


# --- MctsTree ------------------------------------------------------------------------------

def run_bandit(tree, rewards, sims, batch, legal=None, noise=None):
    legal = legal or [True] * len(rewards)
    rng = random.Random(0)
    done = 0
    while done < sims:
        leaves = tree.select(batch)
        assert leaves
        for leaf, parent, action in leaves:
            if leaf == 0:
                tree.expand(0, [1.0] * len(rewards), legal)
                tree.backup(0, 0.5)
            else:
                assert parent == 0 and legal[action]
                tree.set_terminal(leaf, rewards[action])
                r = rewards[action] if noise is None else float(rng.random() < rewards[action])
                tree.backup(leaf, r)
            done += 1


def test_mcts_concentrates_on_the_best_arm():
    rewards = [0.1, 0.3, 0.8, 0.5, 0.2, 0.75, 0.0]
    legal = [True] * 6 + [False]
    tree = MctsTree()
    run_bandit(tree, rewards, 2000, 8, legal)
    v = tree.root_visits()
    assert v[6] == 0
    assert max(range(7), key=v.__getitem__) == 2
    assert v[2] > 0.5 * sum(v)
    q = tree.root_q()
    assert q[2] == pytest.approx(0.8)
    assert tree.node_count() == 7        # root + 6 legal children


def test_mcts_noisy_bandit():
    rewards = [0.2, 0.4, 0.7, 0.3, 0.5, 0.1, 0.6]
    tree = MctsTree(c_puct=1.0)
    run_bandit(tree, rewards, 5000, 4, noise=True)
    v = tree.root_visits()
    assert max(range(7), key=v.__getitem__) == 2


def test_mcts_virtual_loss_spreads_a_batch():
    tree = MctsTree()
    assert tree.select(4) == [(0, -1, -1)]
    tree.expand(0, [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05], [True] * 7)
    tree.backup(0, 0.5)
    leaves = tree.select(7)
    assert sorted(a for _, _, a in leaves) == list(range(7))
    assert len({l for l, _, _ in leaves}) == 7 and all(p == 0 for _, p, _ in leaves)
    assert tree.node_count() == 8
    for leaf, _, _ in leaves:
        assert not tree.is_expanded(leaf)
        tree.expand(leaf, [1.0] * 7, [True] * 7)
        tree.backup(leaf, 0.5)
    assert tree.root_visits() == [1] * 7
    # the next batch goes one level deeper, still spread out
    leaves = tree.select(5)
    assert len({(p, a) for _, p, a in leaves}) == 5 and all(p != 0 for _, p, _ in leaves)

    # without virtual loss, a batch collapses onto one pending leaf
    t2 = MctsTree(virtual_loss=0.0)
    t2.select(1)
    t2.expand(0, [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05], [True] * 7)
    t2.backup(0, 0.5)
    assert len(t2.select(7)) == 1


def test_mcts_priors_renormalized_and_reset():
    tree = MctsTree(n_actions=3, c_puct=2.0)
    tree.select(1)
    tree.expand(0, [0.0, 5.0, 5.0], [True, False, True])
    tree.backup(0, 0.5)
    (leaf, parent, action), = tree.select(1)
    assert (parent, action) == (0, 2)               # the only legal action with prior mass
    tree.set_terminal(leaf, 1.0)
    tree.backup(leaf, 1.0)
    assert tree.select(1) == [(leaf, 0, 2)]        # terminal leaves are returned themselves
    assert tree.root_visits()[1] == 0
    tree.reset()
    assert tree.node_count() == 1 and not tree.is_expanded(0)
    with pytest.raises(ValueError):
        tree.expand(0, [1.0], [True])
