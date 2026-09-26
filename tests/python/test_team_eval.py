"""Team evaluator (rl/team_eval.py, docs/RL_DECISIONS.md §19.2) and its data generator (rl/team_eval_data.py)."""

import random
import struct

import numpy as np
import pytest
import torch

from pybattle.backend import Phase, SimBackend
from pybattle.emu.decode import SYMBOLS as S, decode_party
from pybattle.view import OwnMon
from rl import determinize as DET
from rl import encode, search
from rl import team_eval as T
from rl import team_eval_data as TD
from rl.envs import NO_ACTION
from rl.tactician_search import options_of


@pytest.fixture(autouse=True)
def _keep_encode_version():
    v = encode.VERSION
    yield
    encode.set_version(v)


def _evaluator(seed=0):
    torch.manual_seed(seed)
    return T.TeamEvaluator(T.TeamNet(d=32, d_emb=8, layers=1))


def _battle_to_swap(b, rng):
    while b.phase != Phase.SWAP:
        if b.phase == Phase.RUN_OVER:
            return False
        v = b.view()
        if b.phase == Phase.RENTAL:
            b.act(T.rental_pick(rng.choice(T.legal_rentals(v))))
        elif b.phase == Phase.FORCED_SWITCH:
            b.act(("switch", v.switch_targets[0]))
        else:
            moves = [i for i, ok in enumerate(v.usable_moves) if ok]
            b.act(("move", moves[0]) if moves else ("switch", v.switch_targets[0]) if v.switch_targets else ("move", 0))
    return True


def _decision(kind, seed, streak=7):
    rng = random.Random(seed)
    for s in range(seed, seed + 50):
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=s, win_streak=streak, rents_count=2)
        if kind == "swap" and not _battle_to_swap(b, rng):
            continue
        ctx = search.run_ctx(b)
        v = b.view()
        return b, v, ctx, (encode.rental(v, ctx) if kind == "rental" else encode.swap(v, ctx))
    raise RuntimeError("no swap decision reached")


def _party_after(b, kind, option):
    g = b.game.clone()
    if kind == "rental":
        assert g.factory_rent(*T.rental_pick(option))
    else:
        t = T.swap_trade(option)
        assert g.factory_swap(-1) if t is None else g.factory_swap(*t)
    return decode_party(g.read(S.addr("gPlayerParty"), 300))[:3]


# ---- team_of_option is the team the game really has -----------------------------------------------------------------

def test_team_of_option_rental_matches_the_game():
    b, v, ctx, obs = _decision("rental", 1)
    opts = options_of("rental", obs)
    assert opts == T.legal_rentals(v) and len(opts) > 20
    for o in opts:
        team = T.team_of_option(v, "rental", o)
        real = [T.TeamMon.from_own(OwnMon.from_party(m)) for m in _party_after(b, "rental", o)]
        assert list(team.mons) == real                       # species, moves, item, ability, stats; lead first
        assert team.lead == real[0]


def test_team_of_option_swap_matches_the_game():
    b, v, ctx, obs = _decision("swap", 2)
    opts = options_of("swap", obs)
    assert 0 in opts and len(opts) > 1
    for o in opts + [None, NO_ACTION]:
        team = T.team_of_option(v, "swap", o)
        party = _party_after(b, "swap", o)
        assert team.species == tuple(m.species for m in party)
        t = T.swap_trade(o)
        for i, (tm, pm) in enumerate(zip(team.mons, party)):
            if t is not None and i == t[0]:
                # the traded-in Pokemon: only what was seen in battle; nothing contradicts the real one
                assert tm.stats is None
                assert all(m == T.UNKNOWN or m in pm.moves for m in tm.moves)
                assert tm.item in (T.UNKNOWN, pm.held_item)
            else:
                assert tm == T.TeamMon.from_own(OwnMon.from_party(pm))
    # (own_slot, foe_slot) pairs and action indices agree
    assert T.team_of_option(v, "swap", (1, 2)) == T.team_of_option(v, "swap", 1 + 3 * 1 + 2)


# ---- the formulas ---------------------------------------------------------------------------------------------------------

def test_battle_types():
    assert T.battle_types(1) == [T.NORMAL] * 6 + [T.LAST]
    assert T.battle_types(3)[6] == T.NOLAND and T.battle_types(6)[6] == T.NOLAND
    assert [T.battle_types(r)[6] for r in (2, 4, 5, 7, 8, 9, 12)] == [T.LAST] * 5 + [T.NOLAND] * 2
    assert T.round_of({"streak": 0}) == (1, 1) and T.round_of({"streak": 20}) == (3, 7)
    assert T.round_of({"streak": 13}) == (2, 7) and T.round_of({"streak": 14}) == (3, 1)


def test_round_values_on_hand_made_w():
    W = np.array([[0.5, 1.0, 0.0],        # normal 0.5, last 1, Noland 0
                  [1.0, 0.2, 0.3],
                  [0.9, 0.8, 0.7]])
    e, p = T.round_values(W, 6, 1)        # battles 6 (normal) and 7 (last)
    assert np.allclose(e, [0.5 + 0.5 * 1.0, 1.0 + 1.0 * 0.2, 0.9 + 0.9 * 0.8])
    assert np.allclose(p, [0.5, 0.2, 0.72])
    e, p = T.round_values(W, 6, 3)        # battle 7 of round 3 is Noland
    assert np.allclose(e, [0.5, 1.0 + 0.3, 0.9 + 0.9 * 0.7]) and np.allclose(p, [0.0, 0.3, 0.63])
    e, p = T.round_values(W, 1, 2)        # a whole round: sum_{j=1..7} prod_{i<=j} p_i
    w, l = 0.9, 0.8
    ref = sum(w ** j for j in range(1, 7)) + w ** 6 * l
    assert np.isclose(e[2], ref) and np.isclose(p[2], w ** 6 * l)
    e, p = T.round_values(W, 7, 4)
    assert np.allclose(e, W[:, T.LAST]) and np.allclose(p, W[:, T.LAST])


class _FixedW(T.TeamEvaluator):
    """W = a fixed function of the team (for formula tests)."""

    def __init__(self):
        pass

    def w(self, teams, round_idx):
        return np.array([[0.5 + 0.01 * (t.lead.species % 40), 0.4, 0.3] for t in teams])


def test_option_features_and_phi_values():
    ev = _FixedW()
    b, v, ctx, obs = _decision("swap", 2, streak=14)
    r, k = T.round_of(ctx)
    opts = options_of("swap", obs)
    f = T.option_features(ev, v, "swap", ctx, opts, 5.0)
    W = ev.w([T.team_of_option(v, "swap", o) for o in opts], r)
    e, p = T.round_values(W, k, r)
    assert np.allclose(f[:, 0], e, atol=1e-6) and np.allclose(f[:, 1], W[:, 0], atol=1e-6)
    assert np.allclose(f[:, 2], W[:, T.last_type(r)], atol=1e-6)
    assert np.allclose(f[:, 3], e - e[opts.index(0)], atol=1e-6) and f[opts.index(0), 3] == 0
    q = T.phi_options(ev, v, "swap", ctx, opts, 5.0)
    assert np.allclose(q, e + p * 5.0)
    assert np.isclose(T.phi(ev, v, "swap", ctx, 5.0), q[opts.index(0)])          # current team = keep
    assert np.isclose(T.phi(ev, v, ("swap", opts[-1]), ctx, 5.0), q[-1])
    assert np.isclose(T.phi(ev, v, T.team_of_option(v, "swap", opts[-1]), ctx, 5.0), q[-1])
    b, v, ctx, obs = _decision("rental", 3)
    opts = options_of("rental", obs)
    f = T.option_features(ev, v, "rental", ctx, opts, 5.0)
    assert f.shape == (len(opts), 4) and abs(f[:, 3].mean()) < 1e-5
    assert np.isclose(T.phi(ev, v, "rental", ctx, 5.0), T.phi_options(ev, v, "rental", ctx, opts, 5.0).mean())


@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_option_features_shapes_with_a_network(kind):
    ev = _evaluator()
    b, v, ctx, obs = _decision(kind, 4, streak=14)
    opts = options_of(kind, obs)
    f = T.option_features(ev, v, kind, ctx, opts, 3.0)
    assert f.shape == (len(opts), 4) and f.dtype == np.float32 and np.isfinite(f).all()
    k = T.round_of(ctx)[1]
    assert (f[:, 1:3] >= 0).all() and (f[:, 1:3] <= 1).all()
    assert (f[:, 0] >= 0).all() and (f[:, 0] <= 8 - k).all()
    W = ev.w([T.team_of_option(v, kind, o) for o in opts], 3)
    assert W.shape == (len(opts), 3) and ((W >= 0) & (W <= 1)).all()
    # rounds past 6 are round 6 (the game's pools and IVs; see rl/team_eval.py)
    assert np.allclose(ev.w([T.team_of_option(v, kind, opts[0])], 9), ev.w([T.team_of_option(v, kind, opts[0])], 6))


# ---- information: the team comes from the view only ---------------------------------------------------------------------

def _rebuild_defeated_team(b, rng):
    """A clone of `b` (swap screen) whose defeated team keeps its species but gets other moves, items, IVs, EVs
    and natures in RAM (the hidden part), and whose real next opponent is other sets."""
    c = b.clone()
    party = decode_party(c.game.read(S.addr("gEnemyParty"), 300))[:3]
    specs = []
    for i, m in enumerate(party):
        mv = rng.sample([x for x in DET.learnable(m.species) if x not in m.moves], 4)
        specs.append((i, m.species, mv, 0, [rng.randrange(32)] * 6, [0] * 6, rng.randrange(25), 0, 1.0))
    c.game.determinize(specs)
    addr = S.addr("gFrontierTempParty")
    ids = struct.unpack("<3H", c.game.read(addr, 6))
    c.game.write(addr, struct.pack("<3H", *[(x + 41) % 800 + 1 for x in ids]))
    after = decode_party(c.game.read(S.addr("gEnemyParty"), 300))[:3]
    assert [m.species for m in after] == [m.species for m in party]
    assert any(a.moves != p.moves for a, p in zip(after, party))
    return c


def test_features_never_read_the_opponents_hidden_state():
    ev = _evaluator(1)
    b, v, ctx, obs = _decision("swap", 5, streak=14)
    opts = options_of("swap", obs)
    f1 = T.option_features(ev, v, "swap", ctx, opts, 2.0)
    c = _rebuild_defeated_team(b, random.Random(0))
    v2 = c.view()
    f2 = T.option_features(ev, v2, "swap", search.run_ctx(c), opts, 2.0)
    assert np.array_equal(f1, f2)
    for o in opts:
        assert T.featurize([T.team_of_option(v, "swap", o)])[0].tolist() == \
            T.featurize([T.team_of_option(v2, "swap", o)])[0].tolist()


def test_from_seen_uses_only_what_was_seen():
    from pybattle.view import SeenMon
    m = SeenMon(species=6, seen=True, revealed_moves=[53], revealed_item=None, revealed_ability=0)
    tm = T.TeamMon.from_seen(m)
    assert tm.moves == (53, T.UNKNOWN, T.UNKNOWN, T.UNKNOWN) and tm.item == T.UNKNOWN and tm.stats is None
    ids, num = T.featurize([T.Team((tm, tm, tm))])
    assert ids[0, 0, 1] == encode.ITEM_UNK and ids[0, 0, 4] == encode.MOVE_UNK and num[0, 0, 0] == 0


# ---- save / load ------------------------------------------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    ev = _evaluator(2)
    b, v, ctx, obs = _decision("rental", 6)
    teams = [T.team_of_option(v, "rental", o) for o in options_of("rental", obs)[:9]]
    path = str(tmp_path / "m.pt")
    ev.save(path, {"battler_ckpt": "x.pt", "data_battles": 12})
    ev2 = T.TeamEvaluator.load(path)
    assert ev2.meta["battler_ckpt"] == "x.pt" and ev2.meta["data_battles"] == 12
    for r in (1, 3, 6):
        assert np.allclose(ev.w(teams, r), ev2.w(teams, r), atol=1e-6)
    assert ev.w([], 1).shape == (0, 3)


# ---- the data generator ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("round_type", [(1, 0), (2, 1), (3, 2), (6, 2), (5, 0)])
def test_make_case_builds_the_recorded_team(round_type):
    rng = random.Random(sum(round_type))
    be, own, know, rec = TD.make_case(rng, p_swap=1.0, round_type=round_type)
    r, t = round_type
    assert rec["round"] == r and rec["btype"] == t and know.challenge == r - 1 and know.noland == (t == 2)
    ctx = search.run_ctx(be)
    assert ctx["streak"] == 7 * (r - 1) + rec["battle"] - 1 and ctx["battle"] == rec["battle"] - 1
    party = decode_party(be.game.read(S.addr("gPlayerParty"), 300))[:3]
    assert [m.species for m in party] == rec["species"] and len(set(rec["species"])) == 3
    assert any(rec["swapped"])
    for m, sw, sid in zip(party, rec["swapped"], rec["set_id"]):
        assert DET.SET_SPECIES[sid] == m.species
        if sw:                                  # a swapped-in set: built as the game built the opponent
            spec = DET.set_spec(0, sid, DET.fixed_iv(0, False))
            assert list(m.ivs) == spec[4] and list(m.evs) == spec[5] and m.nature == spec[6]
            assert list(m.moves) == spec[2] and m.held_item == spec[3] and m.friendship == 0
            assert sid in DET.pool(r - 1)
    # a battle can start and be played
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(search.TRAINING_ENV, "1")
        c = TD.start_battle(be, own, know, rng)
    foe = decode_party(c.game.read(S.addr("gEnemyParty"), 300))[:3]
    assert all(m.species for m in foe)
    iv = DET.fixed_iv(r + 1, False) if t == 2 else DET.fixed_iv(0, t == 1)
    assert all(set(m.ivs) == {iv} for m in foe)
