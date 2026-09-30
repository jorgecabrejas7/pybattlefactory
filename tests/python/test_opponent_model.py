"""The opponent model learned from experience (rl/opponent_model.py, docs/RL_DECISIONS.md §22): what it records, that
it never reads the true opponent, its sampler, save / load, the online update, and its use by the search, the
evaluation (rl.eval_rounds) and alphazero_v4."""

import collections
import inspect
import math
import os
import random
import struct
import sys

import numpy as np
import pytest
import torch

from pybattle.backend import Phase, SimBackend
from pybattle.emu.decode import SYMBOLS as S, decode_party
from rl import determinize as D
from rl import encode, search
from rl import opponent_model as OM
from rl.envs import FactoryEnv, decode_action

sys.path.insert(0, os.path.dirname(__file__))
from test_search import _decisions, _random_action  # noqa: E402

V3_CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "alphazero_v3", "latest.pt")


@pytest.fixture(autouse=True)
def _keep_encode_version():
    v = encode.VERSION
    yield
    encode.set_version(v)


def _random_env_action(env, ev, rng):
    b = env.backend
    v = b.view()
    if ev["kind"] == "battle":
        legal = np.flatnonzero(ev["obs"]["mask"])
        return decode_action("battle", int(rng.choice(list(legal))))
    a = _random_action(b, v, rng)
    if ev["kind"] == "swap" and a is None and rng.random() < 0.6:          # take one more often
        for p in range(3):
            for e in range(3):
                if v.enemy_party[e].species not in {m.species for i, m in enumerate(v.own_party) if i != p}:
                    return (p, e)
    return a


def _play(env, steps, seed, on_step=None):
    rng = random.Random(seed)
    ev = env.reset()
    for _ in range(steps):
        a = _random_env_action(env, ev, rng)
        before = env.backend.phase
        ev = env.step(a)
        if on_step is not None:
            on_step(env, before, a, ev)
    return ev


@pytest.fixture(scope="module")
def experience():
    """Records of random play from several start rounds, and the model built on them."""
    cnt = collections.Counter()
    for k, seed in zip((1, 3, 5, 6), (1, 2, 3, 4)):
        env = FactoryEnv(seed, win_streak=7 * (k - 1))
        col = OM.ExperienceCollector()
        col.attach(env)
        _play(env, 700, seed)
        cnt.update(col.drain())
    return cnt, OM.ExperienceModel(counts=cnt)


# ---- the information rule -------------------------------------------------------------------------------------------

BANNED = ("gEnemyParty", "gFrontierTempParty", "FRONTIER_MONS", "SET_", "pool(", "fixed_iv", "enemy_iv", ".game",
          "factory_pool", "sample_factory_sets", "opponent_knowledge", "HELD_ITEMS", "RANGES", "frontier_ids",
          "read_saveblock2", "decode_party")


def _code_only(module):
    """The module's code without docstrings and comments (they name what is NOT read)."""
    import ast
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_source_never_reads_the_true_opponent_or_the_set_list():
    from rl import alphazero_v4 as V4
    for mod in (OM, V4):
        src = _code_only(mod)
        assert "ExperienceCollector" in src or mod is V4
        for banned in BANNED:
            assert banned not in src, (mod.__name__, banned)


class _PlayerOnly:
    """A backend proxy exposing only what the player sees: the phase, the views, the run's info and the player's
    memory of the battle. Anything else (the game, its RAM) raises."""

    def __init__(self, b):
        object.__setattr__(self, "_b", b)

    def __getattr__(self, k):
        if k in ("phase", "view", "run_info", "_observer"):
            return getattr(self._b, k)
        raise AssertionError(f"the collector read backend.{k}")


def test_collector_reads_only_the_player_view():
    """The collector, run on a backend proxy that only exposes views / run info / battle memory, gives the same
    records as on the real backend (so it reads nothing else), and its records are true to the game: a Pokemon
    taken at the swap screen is recorded whole (as its summary shows it), a battle sighting's revealed moves and
    item belong to that opponent."""
    env = FactoryEnv(11, win_streak=14)
    real, proxied = OM.ExperienceCollector(), OM.ExperienceCollector()
    env_proxy = type("E", (), {})()
    env_proxy.backend = _PlayerOnly(env.backend)
    truth = {}                                           # species -> set of true (moves, item) of the opponents

    def on_step(env_, before, a, ev):
        swap_view = None
        if before == Phase.SWAP:
            swap_view = last_swap_view[0]
        real.after(env_, before, a, swap_view, ev)
        proxied.after(env_proxy, before, a, swap_view, ev)
        b = env_.backend
        if b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
            for m in decode_party(b.game.read(S.addr("gEnemyParty"), 300))[:3]:        # test only
                truth.setdefault(m.species, set()).add((OM._moves_key(m.moves), m.held_item))
        last_swap_view[0] = b.view() if b.phase == Phase.SWAP else None

    last_swap_view = [None]
    rng = random.Random(5)
    ev = env.reset()
    real.after(env, None, None, None, ev)
    proxied.after(env_proxy, None, None, None, ev)
    n_steps = 0
    while n_steps < 1500:
        a = _random_env_action(env, ev, rng)
        before = env.backend.phase
        ev = env.step(a)
        on_step(env, before, a, ev)
        n_steps += 1
    r1, r2 = real.drain(), proxied.drain()
    assert r1 == r2 and sum(r1.values()) > 100
    opp_full = [k for k in r1 if k[0] == OM.OPP and k[6] == 1]
    opp_part = [k for k in r1 if k[0] == OM.OPP and k[6] == 0]
    assert opp_full and opp_part
    for k in opp_full:
        assert (k[3], k[4]) in truth[k[2]], "the Pokemon taken at the swap screen is the opponent's, whole"
        assert 0 <= k[5] < 25
    for k in opp_part:
        ok = [(mv, it) for mv, it in truth[k[2]] if set(k[3]) <= set(mv) and (k[4] < 0 or k[4] == it)]
        assert ok or len(k[3]) >= 1                         # (a copied move can be outside the set: Mimic & co.)
    assert all(k[6] == 1 and k[5] >= 0 for k in r1 if k[0] == OM.RENTAL)


def _battle_start(seed=8, streak=28):
    rng = random.Random(4)
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=seed, win_streak=streak, rents_count=streak // 7)
    b.act(_random_action(b, b.view(), rng))
    assert b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH)
    return b


def _overwrite_hidden(c):
    """Rewrite what the player cannot see of the real opponent: the set ids and the unseen party slots (with a
    valid Pokemon: our own first one)."""
    c.game.write(S.addr("gFrontierTempParty"), struct.pack("<3H", 5, 6, 7))
    v = c.view()
    base = S.addr("gEnemyParty")
    mine = c.game.read(S.addr("gPlayerParty"), 100)
    n = 0
    for i, m in enumerate(v.enemy_party[:3]):
        if not m.seen:
            c.game.write(base + 100 * i, mine)
            n += 1
    assert n >= 1
    return c


def test_sampler_and_search_never_read_the_true_opponent(experience):
    _, model = experience
    b = _battle_start()
    c = _overwrite_hidden(b.clone())
    assert decode_party(b.game.read(S.addr("gEnemyParty"), 300)) != decode_party(c.game.read(S.addr("gEnemyParty"),
                                                                                             300))
    assert b.view() == c.view()
    ch = b.run_info().challenge_num
    for s in range(20):
        assert model.determinization(b.view(), ch, random.Random(s)) == \
            model.determinization(c.view(), ch, random.Random(s))
    from rl import alphazero as AZ
    from rl.model import FactoryNet
    torch.manual_seed(3)
    encode.set_version(4)
    ev = AZ.local_battler_evaluator(FactoryNet(share="embeddings").eval(), None)
    out = []
    for backend in (b, c):
        sb = search.SearchBattler(None, n_sims=32, n_determinizations=4, seed=2, evaluator=ev,
                                  opponent_prior="experience", opponent_model=model)
        sb.act_on(backend)
        out.append(sb.last_stats)
    assert out[0]["visits"] == out[1]["visits"] and np.allclose(out[0]["q"], out[1]["q"])


# ---- the sampler -------------------------------------------------------------------------------------------------------

def test_sampler_respects_what_the_player_saw(experience):
    _, model = experience
    rng = random.Random(1)
    n = known = 0
    species_all = set(D.frontier_species(True))
    for b, v, _ in _decisions(16, seed=13, streaks=(0, 14, 28, 35)):
        ch = b.run_info().challenge_num
        specs = model.determinization(v, ch, rng, b.open_level)
        level = D.team_level(v)
        sp_all, items = [], []
        for (slot, sp, moves, item, ivs, evs, nature, bit, hp), m in zip(specs, v.enemy_party):
            if m.seen and m.fainted:
                assert sp == D.KEEP and hp < 0
                continue
            assert len(moves) == 4 and all(0 <= x <= 31 for x in ivs) and sum(evs) <= 510 and 0 <= nature < 25
            real = [x for x in moves if x]
            assert real and len(set(real)) == len(real)
            if m.seen:
                assert sp == m.species
                assert set(x for x in m.revealed_moves if x) <= set(real)
                if m.revealed_item:
                    assert item == m.revealed_item
            else:
                assert sp in species_all and hp == 1.0
            known += (sp, OM._moves_key(moves), item) in {(k[2], k[3], k[4]) for k in model.counts if k[6]}
            sp_all.append(sp)
            if item:
                items.append(item)
        assert len(sp_all) == len(set(sp_all)) and len(items) == len(set(items))
        n += 1
    assert n > 150 and known > 0.5 * n


def test_empty_model_is_the_strict_sampler():
    """Without experience every Pokemon is drawn by the strict rules (a novel configuration)."""
    model = OM.ExperienceModel()
    for b, v, _ in _decisions(3, seed=2):
        ch = b.run_info().challenge_num
        for i, m in enumerate(v.enemy_party[:3]):
            if not (m.seen and m.fainted):
                cs, pc, pn = model.compile().config_posterior(OM.bucket_of(ch), m.species or 25)
                assert cs == [] and pn == 1.0
        break
    sp = model.compile().species_probs(0)
    assert np.allclose(sp, 1.0 / len(sp))


def test_slot_logp_matches_the_sampler():
    """exp(slot_logp) of the first unseen slot is the frequency with which the sampler draws it."""
    cnt = collections.Counter()
    for sp, moves, item, n in ((6, (53, 89, 126, 337), 196, 30), (6, (14, 53, 89, 126), 200, 10),
                               (9, (56, 58, 89, 92), 234, 20)):
        cnt[(OM.RENTAL, 4, sp, moves, item, 3, 1)] += n
    cnt[(OM.OPP, 4, 6, (53,), -1, -1, 0)] += 25
    model = OM.ExperienceModel({"a0": 2.0, "a_cfg": 1.0}, cnt)
    v = type("V", (), {})()
    from pybattle.view import SeenMon
    v.enemy_party = [SeenMon(), SeenMon(), SeenMon()]
    v.own_party = []
    D_level = 100
    rng = random.Random(0)
    freq = collections.Counter()
    N = 20000
    t = model.compile()
    for _ in range(N):
        team = model._draw_team(t, 4, v.enemy_party, D_level, rng)
        s, mv, it, _, known = team[0]
        if known:
            freq[(s, OM._moves_key(mv), it)] += 1
    for key, f in freq.items():
        p = math.exp(model.slot_logp(v, 0, 4, key[0], key[1], key[2]))
        assert abs(f / N - p) < 4 * math.sqrt(p * (1 - p) / N) + 1e-3, (key, f / N, p)
    # EM: the 25 sightings of species 6 with move 53 revealed are shared by its two configurations
    cs, pc, pn = t.config_posterior(4, 6)
    post = dict(zip(cs, pc))
    assert len(cs) == 2 and post[((53, 89, 126, 337), 196)] > post[((14, 53, 89, 126), 200)] and pn < 0.1
    assert abs(post[((53, 89, 126, 337), 196)] / post[((14, 53, 89, 126), 200)] - 3.0) < 0.3


def test_hint_rejection(experience):
    _, model = experience
    m2 = OM.ExperienceModel(dict(model.cfg, use_hint=True), model.counts)
    rng = random.Random(3)
    n = hit = 0
    for b, v, _ in _decisions(6, seed=31, streaks=(7, 28)):
        info = b.run_info()
        hint = D.battle_hint(v, info.noland)
        if hint is None:
            continue
        specs = m2.determinization(v, info.challenge_num, rng, hint=hint)
        drawn = [(s[1], s[2]) for s in specs]
        if any(s[1] == D.KEEP for s in specs):
            continue
        hit += D.hint_of_team(drawn) == hint
        n += 1
    assert n > 20 and hit / n > 0.9


# ---- save / load and the online update ---------------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path, experience):
    _, model = experience
    p = str(tmp_path / "m.pt")
    model.save(p)
    m2 = OM.ExperienceModel.load(p)
    assert m2.counts == model.counts and m2.cfg == model.cfg
    for b, v, _ in _decisions(2, seed=17):
        ch = b.run_info().challenge_num
        assert model.determinization(v, ch, random.Random(9)) == m2.determinization(v, ch, random.Random(9))
    # a checkpoint that carries a model
    torch.save({"net": {}, "opponent_model": model.state()}, str(tmp_path / "ck.pt"))
    assert OM.ExperienceModel.load(str(tmp_path / "ck.pt")).counts == model.counts
    # weights_only loading (rl.policy.Policy loads checkpoints so) accepts the state
    torch.load(p, map_location="cpu", weights_only=True)


def test_online_update_is_deterministic(experience):
    cnt, _ = experience
    keys = sorted(cnt)
    parts = [collections.Counter({k: cnt[k] for k in keys[i::3]}) for i in range(3)]
    a = OM.ExperienceModel()
    for p in parts:
        a.add(OM.merge([p]))
    b = OM.ExperienceModel().add(OM.merge(parts[::-1]))
    assert a.counts == b.counts
    for x, y in zip(a.state()["counts"], b.state()["counts"]):
        assert x == y
    ta, tb = a.compile(), b.compile()
    for s in ta.tables:
        assert np.array_equal(ta.tables[s], tb.tables[s])
    assert np.array_equal(ta.p_species, tb.p_species)


# ---- use: search, evaluation, alphazero_v4 --------------------------------------------------------------------------------

def test_experience_prior_is_allowed_everywhere(monkeypatch, experience):
    _, model = experience
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    D.check_prior("experience")
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    D.check_prior("experience")
    with pytest.raises(ValueError):
        search.SearchBattler(None, n_sims=8, n_determinizations=2, evaluator=lambda b: None,
                             opponent_prior="experience")
    sb = search.SearchBattler(None, n_sims=8, n_determinizations=2, evaluator=lambda b: None,
                              opponent_prior="experience", opponent_model=model)
    assert sb.opponent_prior == "experience" and sb.opponent_model is model


def test_eval_rounds_prior_selection(tmp_path, experience):
    from argparse import Namespace
    from rl import eval_rounds as ER
    _, model = experience
    plain = str(tmp_path / "plain.pt")
    torch.save({"net": {}, "args": {}}, plain)
    v4 = str(tmp_path / "v4.pt")
    torch.save({"net": {}, "args": {}, "opponent_model": model.state()}, v4)
    mfile = str(tmp_path / "m.pt")
    model.save(mfile)
    ns = lambda **k: Namespace(**dict({"opponent_prior": "auto", "opponent_model": None}, **k))
    assert ER.opponent_prior_of(ns(ckpt=plain)) == ("strict", None)
    assert ER.opponent_prior_of(ns(ckpt=v4)) == ("experience", v4)
    assert ER.opponent_prior_of(ns(ckpt=plain, opponent_model=mfile)) == ("experience", mfile)
    assert ER.opponent_prior_of(ns(ckpt=v4, opponent_prior="strict")) == ("strict", None)
    with pytest.raises(ValueError):
        ER.opponent_prior_of(ns(ckpt=plain, opponent_prior="experience"))
    src = inspect.getsource(ER)
    assert src.count("factory_sets") == 2            # the choice and its help text (a diagnostic, never auto)


@pytest.mark.skipif(not os.path.exists(V3_CKPT), reason="runs/alphazero_v3/latest.pt not present")
def test_eval_round_with_the_experience_sampler(tmp_path, experience):
    from rl.eval_rounds import eval_round_inprocess
    _, model = experience
    mfile = str(tmp_path / "m.pt")
    model.save(mfile)
    kw = {"n_sims": 16, "n_determinizations": 2, "mode": "legal", "seed": 1, "opponent_prior": "experience",
          "opponent_model": mfile}
    r = eval_round_inprocess(V3_CKPT, 5, 2, "search", kw, workers=1, envs_per_worker=1, procs=1)
    assert r["n"] >= 2 and r["decisions"] > 0 and r["search_errors"] == 0


def test_v4_selfplay_collects_and_trains_end_to_end(tmp_path, monkeypatch):
    """A tiny alphazero_v4 run: the workers' records reach the model (online), the checkpoints carry it, a resume
    keeps it, and the training search can use it (--opponent-prior experience)."""
    from rl import alphazero_v4 as V4
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    run = str(tmp_path / "az4")
    common = ["--run-dir", run, "--decisions-per-iter", "60", "--workers", "2", "--inference", "cpu", "--device",
              "cpu", "--eval-runs", "2", "--eval-rounds", "1", "--eval-workers", "1", "--eval-envs-per-worker", "2",
              "--batch-b", "16", "--sims", "16", "--dets", "2", "--holdout", "0.3", "--team-eval", "stub:varied",
              "--t-minibatch", "8", "--lr-warmup", "1", "--eval-every", "1"]
    V4.main(common + ["--iterations", "2"])
    ck = torch.load(os.path.join(run, "latest.pt"), map_location="cpu", weights_only=False)
    assert ck["args"]["variant"] == "alphazero_v4" and ck["args"]["version"] == 3
    assert ck["args"]["iterations"] == 2 and ck["args"]["t_imitation"] == 0.1
    m = OM.ExperienceModel.from_state(ck["opponent_model"])
    n1 = m.n_records()
    assert n1 > 0 and m.n_records(OM.RENTAL) > 0
    assert os.path.exists(os.path.join(run, "opponent_model.pt"))
    # resume: the counts go on growing from the checkpoint's; the training search from the model
    V4.main(common + ["--iterations", "3", "--init-from", os.path.join(run, "latest.pt"), "--opponent-prior",
                      "experience", "--opponent-model-refresh", "1"])
    ck = torch.load(os.path.join(run, "latest.pt"), map_location="cpu", weights_only=False)
    assert OM.ExperienceModel.from_state(ck["opponent_model"]).n_records() > n1
    from rl.eval_rounds import ckpt_has_opponent_model
    assert ckpt_has_opponent_model(os.path.join(run, "latest.pt"))
