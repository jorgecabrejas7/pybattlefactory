"""BattleView contents: nothing the player could not know, counters that behave, turn events
that match what happened (checked against a frame-by-frame replay of the turn)."""

import dataclasses
import os
import random
import struct

import pytest

from pybattle.backend import Phase, SimBackend
from pybattle.emu.decode import SYMBOLS as S, decode_party
from pybattle.view import (ACTION_MOVE, ACTION_NONE, HP_BAR_PIXELS, MOVE, MOVES,
                           SPECIES, UNSEEN_HP, BattleView, SeenMon, ability_of, hp_bar_pixels)


def _rent(b, v):
    picks = [0]
    for i in range(1, 6):
        if len(picks) < 3 and v.candidates[i].species not in {v.candidates[p].species for p in picks}:
            picks.append(i)
    b.act(tuple(picks))


def _random_battle_views(n_runs=20, seed0=0, max_steps=500):
    """(backend, view) at every battle decision of random runs."""
    for seed in range(seed0, seed0 + n_runs):
        rng = random.Random(seed)
        b = SimBackend()
        b.reset(seed=seed, win_streak=[0, 21, 42][seed % 3])
        steps = 0
        while b.phase != Phase.RUN_OVER and steps < max_steps:
            steps += 1
            v = b.view()
            if b.phase == Phase.RENTAL:
                _rent(b, v)
                continue
            if b.phase == Phase.SWAP:
                b.act(None)
                continue
            yield b, v
            if b.phase == Phase.FORCED_SWITCH:
                b.act(("switch", rng.choice(v.switch_targets)))
                continue
            moves = [i for i, ok in enumerate(v.usable_moves) if ok]
            if v.switch_targets and (not moves or rng.random() < 0.15):
                b.act(("switch", rng.choice(v.switch_targets)))
            else:
                b.act(("move", rng.choice(moves) if moves else 0))


def _enemy_party(b):
    return decode_party(b.game.read(S.addr("gEnemyParty"), 300))


# ---------------------------------------------------------------------------------------------
# no leakage
# ---------------------------------------------------------------------------------------------

def test_unseen_enemy_slots_reveal_nothing():
    unseen = 0
    for b, v in _random_battle_views(15):
        enemy = _enemy_party(b)
        for i, m in enumerate(v.enemy_party):
            if not m.seen:
                unseen += 1
                assert m == SeenMon(), f"unseen slot {i} leaks {m}"
                assert m.hp_pixels == UNSEEN_HP and m.species == 0 and m.level == 0 and m.status == 0
                assert not m.fainted and not m.revealed_moves and m.revealed_item is None
                continue
            true = enemy[i]
            assert m.species == true.species and m.level == true.level
            assert m.types == SPECIES[true.species]["types"] and m.base_stats == SPECIES[true.species]["base"]
            assert ability_of(true.species, true.ability_num) in m.possible_abilities
            if m.revealed_ability:      # announced abilities are the real one
                assert m.revealed_ability == ability_of(true.species, true.ability_num)
            assert set(m.revealed_moves) <= set(true.moves)
            assert 0 <= m.hp_pixels <= HP_BAR_PIXELS
        # the opponent's substitute HP is not shown
        assert v.enemy_active.substitute_hp == -1
    assert unseen > 50


def test_enemy_view_has_no_exact_hp_or_hidden_fields():
    names = {f.name for f in dataclasses.fields(SeenMon)}
    for hidden in ("hp", "max_hp", "stats", "pp", "item", "ability", "ivs", "evs", "nature", "moves"):
        assert hidden not in names


def test_hints_reach_the_battle_view():
    b = SimBackend()
    b.reset(seed=5, win_streak=7)
    rental = b.view()
    _rent(b, rental)
    v = b.view()
    assert (v.hint_type, v.hint_style) == (rental.hint_type, rental.hint_style)
    rng = random.Random(0)
    while b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
        v = b.view()
        b.act(rng.choice(v.legal_actions[:-1]))        # never forfeit
    if b.phase == Phase.SWAP:
        swap = b.view()
        b.act(None)
        assert (b.view().hint_type, b.view().hint_style) == (swap.hint_type, swap.hint_style)


# ---------------------------------------------------------------------------------------------
# counters
# ---------------------------------------------------------------------------------------------

def _battle(seed=3):
    b = SimBackend()
    b.reset(seed=seed)
    _rent(b, b.view())
    assert b.phase == Phase.BATTLE
    return b


def _safe_move(v: BattleView):
    """A usable move that is not a switching / healing / field move (keeps the scenario simple)."""
    me = v.own_party[v.own_active.party_index]
    for i, ok in enumerate(v.usable_moves):
        if ok and MOVES[me.moves[i]]["power"] > 1 and MOVES[me.moves[i]]["priority"] == 0:
            return ("move", i)
    return ("move", v.usable_moves.index(True))


def test_reflect_turns_left_count_down():
    b = _battle()
    g = b.game
    side = S.addr("gSideStatuses")
    g.write(side, struct.pack("<H", struct.unpack("<H", g.read(side, 2))[0] | 1))    # SIDE_STATUS_REFLECT
    g.write(S.addr("gSideTimers"), bytes([5]))                                     # reflectTimer
    seen = []
    for _ in range(6):
        if b.phase != Phase.BATTLE:
            break
        v = b.view()
        seen.append(v.own_side.reflect_turns)
        b.act(_safe_move(v))
    assert seen[:5] == [5, 4, 3, 2, 1][:len(seen[:5])]
    if len(seen) == 6:
        assert seen[5] == 0


def test_sleep_turns_count_attempts_not_turns_left():
    b = _battle(seed=11)
    g = b.game
    mons = S.addr("gBattleMons")
    # both active Pokemon fall asleep for 4 "attempts" (hidden counter 4)
    for battler in (0, 1):
        addr = mons + battler * 0x58 + 0x4C
        g.write(addr, struct.pack("<I", 4))
    counts = []
    for _ in range(3):
        v = b.view()
        me = v.own_party[v.own_active.party_index]
        foe = v.enemy_party[v.enemy_active.party_index]
        counts.append((me.sleep_turns, foe.sleep_turns, me.status, foe.status))
        if b.phase != Phase.BATTLE:
            break
        b.act(_safe_move(v))
    assert counts[0][:2] == (0, 0)
    for prev, cur in zip(counts, counts[1:]):
        for k in (0, 1):
            if cur[2 + k] == 1:                  # still asleep: one more "fast asleep"
                assert cur[k] == prev[k] + 1
    assert all(c[0] <= 4 and c[1] <= 4 for c in counts)
    # nothing in the view says how long it will still sleep
    for f in dataclasses.fields(SeenMon):
        assert "left" not in f.name


def test_elapsed_and_remaining_counters_stay_in_range():
    maxima = {}
    for b, v in _random_battle_views(25):
        for a in (v.own_active, v.enemy_active):
            for name in ("confusion_turns", "encore_turns", "disable_turns", "wrapped_turns", "uproar_turns",
                         "rampage_turns"):
                maxima[name] = max(maxima.get(name, 0), getattr(a, name))
            assert 0 <= a.taunt_turns_left <= 2 and 0 <= a.yawn_turns_left <= 2
            assert -1 <= a.perish_count <= 3 and 0 <= a.bide_turns_left <= 2 and 0 <= a.stockpile <= 3
        for sd in (v.own_side, v.enemy_side):
            for name in ("reflect_turns", "light_screen_turns", "safeguard_turns", "mist_turns"):
                assert 0 <= getattr(sd, name) <= 5
            assert 0 <= sd.future_sight_turns <= 3 and 0 <= sd.wish_turns <= 2 and 0 <= sd.spikes <= 3
        assert 0 <= v.weather_turns_left <= 5
        assert not (v.weather_permanent and v.weather_turns_left)
        for m in v.own_party + [m for m in v.enemy_party if m.seen]:
            assert 0 <= m.sleep_turns <= 4
    # random durations: confusion 2-5 (<= 4 failed-or-not attempts while still confused), etc.
    assert maxima["confusion_turns"] <= 4 and maxima["wrapped_turns"] <= 6 and maxima["encore_turns"] <= 6


# ---------------------------------------------------------------------------------------------
# turn events against a frame-by-frame replay of each turn
# ---------------------------------------------------------------------------------------------

def _frame(g):
    mons = g.read(S.addr("gBattleMons"), 0xB0)
    return dict(tan=g.read(S.addr("gCurrentTurnActionNumber"), 1)[0], order=g.read(S.addr("gBattlerByTurnOrder"), 2),
                acts=g.read(S.addr("gActionsByTurnOrder"), 2),
                hp=[struct.unpack_from("<H", mons, i * 0x58 + 0x28)[0] for i in range(2)],
                mx=[struct.unpack_from("<H", mons, i * 0x58 + 0x2C)[0] for i in range(2)],
                idx=g.read(S.addr("gBattlerPartyIndexes"), 4),
                turn=g.read(S.addr("gBattleResults") + 0x13, 1)[0])


def _step_turn(g, max_frames=4000):
    """Run the turn one frame at a time (validation only) and split it into the two actions."""
    frames = [_frame(g)]
    devnull, saved = os.open(os.devnull, os.O_WRONLY), os.dup(2)
    os.dup2(devnull, 2)                         # run(1) reports a timeout on stderr each frame
    try:
        for _ in range(max_frames):
            g.run(1)
            frames.append(_frame(g))
            if frames[-1]["turn"] != frames[0]["turn"]:
                break
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)
    wins, started, prev = {}, frames[0]["tan"] >= 2, None
    for fs in frames:
        if not started:
            started = fs["tan"] >= 2
        elif fs["tan"] in (0, 1):
            w = wins.setdefault(fs["tan"], dict(drops=[0, 0], start=[None, None], mx=[0, 0]))
            w["actor"], w["act"] = fs["order"][fs["tan"]], fs["acts"][fs["tan"]]
            for bt in (0, 1):
                if prev["idx"][2 * bt:2 * bt + 2] == fs["idx"][2 * bt:2 * bt + 2] and prev["hp"][bt] > fs["hp"][bt]:
                    if w["start"][bt] is None:
                        w["start"][bt], w["mx"][bt] = prev["hp"][bt], prev["mx"][bt]
                    w["drops"][bt] += prev["hp"][bt] - fs["hp"][bt]
        prev = fs
    return wins


def test_turn_events_match_the_turn_as_played():
    stats = dict(first=0, first_bad=0, dmg=0, dmg_bad=0, px=0, px_bad=0)
    for seed in range(8):
        rng = random.Random(seed)
        b = SimBackend()
        b.reset(seed=seed, win_streak=[0, 21, 42][seed % 3])
        truth, steps, first_decision = None, 0, True
        while b.phase != Phase.RUN_OVER and steps < 300:
            steps += 1
            v = b.view()
            if b.phase == Phase.RENTAL:
                _rent(b, v)
                first_decision = True
                continue
            if b.phase == Phase.SWAP:
                b.act(None)
                first_decision = True
                continue
            ev = v.last_turn
            if first_decision:
                assert ev.turn == -1 and ev.own_action == ACTION_NONE and ev.enemy_action == ACTION_NONE
                first_decision = False
            # always-true properties
            assert ev.first in (-1, 0, 1) and ev.own_crit in (-1, 0, 1) and ev.enemy_crit in (-1, 0, 1)
            assert 0 <= ev.damage_dealt_pixels <= HP_BAR_PIXELS and ev.damage_taken >= 0
            if ev.own_action != ACTION_MOVE:
                assert ev.own_move == 0
            if ev.enemy_action != ACTION_MOVE:
                assert ev.enemy_move == 0
            elif ev.enemy_move != MOVE["STRUGGLE"] and not v.enemy_active.transformed:
                assert any(ev.enemy_move in m.revealed_moves for m in v.enemy_party)     # a used move is revealed
            if truth and ev.turns == 1 and not ev.in_progress:
                both = [w for w in truth.values() if w["act"] in (0, 2)]
                if len(both) == 2 and ev.first != -1:
                    stats["first"] += 1
                    stats["first_bad"] += ev.first != truth[0]["actor"]
                taken = sum(w["drops"][0] for w in truth.values() if w["actor"] == 1)
                stats["dmg"] += 1
                stats["dmg_bad"] += ev.damage_taken != taken
                px = sum(hp_bar_pixels(w["start"][1], w["mx"][1]) - hp_bar_pixels(w["start"][1] - w["drops"][1], w["mx"][1])
                         for w in truth.values() if w["actor"] == 0 and w["drops"][1])
                stats["px"] += 1
                stats["px_bad"] += ev.damage_dealt_pixels != px
            if b.phase == Phase.FORCED_SWITCH:
                b.act(("switch", rng.choice(v.switch_targets)))
                truth = None
                continue
            moves = [i for i, ok in enumerate(v.usable_moves) if ok]
            b._turns += 1
            if v.switch_targets and (not moves or rng.random() < 0.15):
                b.game.choose_switch(rng.choice(v.switch_targets))
            else:
                b.game.choose_move(rng.choice(moves) if moves else 0)
            truth = _step_turn(b.game)
            b._sync()
    assert stats["first"] > 100 and stats["dmg"] > 150
    # the rare misses are confusion self-hits in the same turn as a hit (see GUIDE)
    assert stats["first_bad"] <= 0.02 * stats["first"], stats
    assert stats["dmg_bad"] <= 0.03 * stats["dmg"] and stats["px_bad"] <= 0.03 * stats["px"], stats


def test_forced_switch_after_a_faint_describes_the_whole_turn():
    n = 0
    for b, v in _random_battle_views(20):
        if b.phase == Phase.FORCED_SWITCH and v.own_party[v.own_active.party_index].hp == 0:
            ev = v.last_turn
            assert not ev.in_progress and ev.own_fainted
            assert ev.turn == v.turn
            n += 1
    assert n > 10


# ---------------------------------------------------------------------------------------------
# Struggle, clone
# ---------------------------------------------------------------------------------------------

def test_struggle_is_explicit_in_the_view():
    b = _battle()
    b.game.write(S.addr("gBattleMons") + 0x24, bytes(4))          # no PP left on any move
    v = b.view()
    assert v.must_struggle and not any(v.usable_moves)
    assert ("move", 0) in v.legal_actions and ("forfeit",) in v.legal_actions
    assert all(a[0] != "move" or a == ("move", 0) for a in v.legal_actions)
    b.act(("move", 0))
    if b.phase == Phase.BATTLE:
        assert b.view().last_turn.own_move == MOVE["STRUGGLE"]


def test_legal_actions_match_usable_moves_and_switches():
    for b, v in list(_random_battle_views(5))[:200]:
        if v.forced_switch:
            assert v.legal_actions == [("switch", i) for i in v.switch_targets]
        else:
            moves = [a for a in v.legal_actions if a[0] == "move"]
            assert moves == ([("move", i) for i, ok in enumerate(v.usable_moves) if ok] or [("move", 0)])
            assert v.must_struggle == (not any(v.usable_moves))


def test_clone_copies_observer_memory():
    b = _battle(seed=4)
    rng = random.Random(1)
    for _ in range(6):
        if b.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
            break
        v = b.view()
        b.act(rng.choice(v.legal_actions[:-1]) if not v.forced_switch else rng.choice(v.legal_actions))
    if b.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
        pytest.skip("battle ended early")
    c = b.clone()
    assert c._observer is not b._observer
    # the memory observe() edits in place is the clone's own (the turn-start snapshot is immutable and shared)
    o, oc = b._observer, c._observer
    assert oc._start is not None
    for name in ("revealed_moves", "revealed_items", "revealed_abilities", "seen", "_last_enemy_item",
                 "_bench_status", "_sleep", "_vol"):
        assert getattr(oc, name) is not getattr(o, name) and getattr(oc, name) == getattr(o, name), name
    assert all(oc.revealed_moves[k] is not o.revealed_moves[k] for k in o.revealed_moves)
    assert all(oc._sleep[k] is not o._sleep[k] for k in o._sleep)
    assert all(x is not y for x, y in zip(oc._vol, o._vol))
    assert c.view() == b.view()
    # the same actions from here give the same views: the memory went along with the game
    rng_b, rng_c = random.Random(9), random.Random(9)
    for _ in range(6):
        if b.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
            break
        vb, vc = b.view(), c.view()
        assert vb == vc
        b.act(rng_b.choice(vb.legal_actions[:-1]) if not vb.forced_switch else rng_b.choice(vb.legal_actions))
        c.act(rng_c.choice(vc.legal_actions[:-1]) if not vc.forced_switch else rng_c.choice(vc.legal_actions))
    # and acting on the original does not touch the clone's memory
    before = dict(c._observer.revealed_moves), c._observer._turns_done
    if b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
        b.act(b.view().legal_actions[0])
    assert (dict(c._observer.revealed_moves), c._observer._turns_done) == before


# ---------------------------------------------------------------------------------------------
# the defeated opponents' records (SwapView.defeated)
# ---------------------------------------------------------------------------------------------

def test_defeated_records_add_up_to_the_battle():
    """At each swap, the per-opponent records account for the whole battle as the player saw it: the HP our team
    lost adds up (exactly, when none of ours was seen healing), the knockouts are our fainted Pokemon, every
    battle turn is credited once, the boosts are at least those seen, a status on us came from some opponent."""
    from rl.baselines import MaxDamageBattler
    from pybattle.view import FoeRecord, SwapView
    swaps = exact = statuses = 0
    for seed in range(40):
        rng = random.Random(seed)
        battler = MaxDamageBattler(seed)
        b = SimBackend()
        b.reset(seed=seed, win_streak=[0, 7, 14][seed % 3])
        views, steps = [], 0
        while b.phase != Phase.RUN_OVER and steps < 3000:
            steps += 1
            v = b.view()
            if b.phase == Phase.RENTAL:
                _rent(b, v)
                views = []
                continue
            if b.phase == Phase.SWAP:
                assert isinstance(v, SwapView) and len(v.defeated) == 3
                rec = v.defeated
                own = v.own_party                         # as the battle left them (healed when the next starts)
                assert rec[0].team_max_hp == sum(m.max_hp for m in own)
                lost = sum(m.max_hp - m.hp for m in own)
                hps = [[m.hp for m in w.own_party] for w in views] + [[m.hp for m in own]]
                healed = any(w[i] > u[i] for u, w in zip(hps, hps[1:]) for i in range(3))
                total = sum(r.damage for r in rec)
                assert total >= lost and (healed or total == lost), (total, lost, healed)
                exact += not healed
                assert sum(r.knockouts for r in rec) == sum(m.hp == 0 for m in own)
                n_battle = sum(not w.forced_switch for w in views)
                assert sum(r.turns for r in rec) >= n_battle
                for j in range(3):
                    seen = [sum(max(0, s) for s in w.enemy_active.stat_stages) for w in views
                            if w.enemy_active.party_index == j]
                    assert rec[j].max_boosts >= max(seen, default=0)
                    assert 0 <= rec[j].damage_frac <= 1 and rec[j].hits_taken <= rec[j].turns
                rested = any(w.last_turn.own_action == ACTION_MOVE and MOVES[w.last_turn.own_move]["effect"] ==
                             MOVES[MOVE["REST"]]["effect"] for w in views)
                if any(m.status and m.hp for m in own) and not rested:
                    assert any(r.inflicted_status for r in rec)
                    statuses += 1
                assert all(isinstance(r, FoeRecord) for r in rec)
                swaps += 1
                b.act(None)
                views = []
                continue
            views.append(v)
            if not v.forced_switch and v.switch_targets and rng.random() < 0.1:
                b.act(("switch", rng.choice(v.switch_targets)))
            else:
                b.act(battler(v))
    print(f"\n{swaps} swaps ({exact} with the HP sum checked exactly, {statuses} with a status on us)")
    assert swaps > 30 and exact > 10 and statuses > 0, (swaps, exact, statuses)
