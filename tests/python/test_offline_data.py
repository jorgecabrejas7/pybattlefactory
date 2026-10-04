"""The v5 offline dataset (rl/offline_data.py, docs/RL_DECISIONS.md §27): recorded positions and their replay keys.
Replaying a key must reproduce the recorded encoded input exactly, deterministically, from public state."""

import argparse
import os

import numpy as np
import pytest
import torch

from rl import encode, gumbel as G, search
from rl import offline_data as OD

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CKPT = os.path.join(ROOT, "runs", "alphazero_v4", "stopped_it1111.pt")

pytestmark = [pytest.mark.skipif(not os.path.exists(CKPT), reason="needs runs/alphazero_v4/stopped_it1111.pt"),
              pytest.mark.skipif(G.halving_plan is None or not hasattr(search.NativeSearcher, "search_gumbel"),
                                 reason="needs the extension's Gumbel search")]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A small recording with the v4 checkpoint (CPU, 16 simulations), consolidated."""
    from rl import alphazero as AZ
    from rl import alphazero_v3 as V3
    out = str(tmp_path_factory.mktemp("v5_offline"))
    v = encode.VERSION
    torch.set_num_threads(2)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(search.TRAINING_ENV, "1")                     # factory_sets: the training-only sampler
        ck = torch.load(os.path.join(ROOT, CKPT), map_location="cpu", weights_only=False)
        args = argparse.Namespace(**ck["args"])
        args.sims, args.dets = 16, 2
        encode.set_version(5)
        cfg = V3.worker_cfg(args, ck["args"]["opt_feat"])
        cfg.update(opponent_model=CKPT, rec_out=os.path.join(out, "shards"))
        sp = OD._player_class()(cfg, seed=11)
        sp.load(CKPT)
        sp.set_v_next(ck["v_next"]["table"])
        sp.set_team_eval(ck["team_eval"])
        sp.set_weights(AZ.start_weights_from_eval({int(k): x for k, x in ck["round_eval"].items()}, 0.3).tolist())
        n = 0
        for _ in range(6):
            _, st = sp.play(60)
            n += st["positions"]
            if n >= 80:
                break
        OD.consolidate(argparse.Namespace(out=out))
    encode.set_version(v)
    return out


def test_fields(recorded):
    ds = OD.Dataset(recorded)
    n = len(ds)
    assert n >= 40
    t = np.asarray(ds["target"])
    mask = ds.obs_rows(np.arange(n))["mask"]
    assert np.allclose(t.sum(1), 1, atol=1e-5) and (t[~mask] == 0).all()        # the target is over legal actions
    s = np.asarray(ds["searched"])
    assert np.isfinite(np.asarray(ds["root_q"])[s]).all()
    z = np.asarray(ds["z"])
    assert ((z >= 0) & (z <= 1)).all()
    assert set(np.unique(ds["round"])) <= set(range(1, 13))
    run = np.asarray(ds["run"])
    for u in np.unique(run):                                        # held out by run
        assert len(set(ds.held[run == u])) == 1
    # every row's played action is legal; one-legal decisions are one-hot
    a = np.asarray(ds["action"])
    assert mask[np.arange(n), a].all()
    assert (t[~s].max(1) == 1).all()


def test_replay_reproduces_inputs_exactly(recorded):
    ds = OD.Dataset(recorded)
    rows = np.arange(len(ds))
    got = {rp.key: rp.obs for rp in OD.replay_positions(rows, recorded, check=True)}
    assert sorted(got) == list(rows)
    ref = ds.obs_rows(rows)
    for i in rows:
        for k in OD.OBS_KEYS:
            assert np.array_equal(np.asarray(got[i][k]), ref[k][i]), (i, k)
            assert np.asarray(got[i][k]).dtype == ref[k].dtype


def test_replay_is_deterministic_and_keys_agree(recorded):
    from rl.autopsy import state_hash
    ds = OD.Dataset(recorded)
    rows = np.arange(len(ds))
    keys = ds.keys(rows)
    h1 = {rp.key: state_hash(rp.backend.game) for rp in OD.replay_positions(keys, recorded)}
    h2 = {}
    for rp in OD.replay_positions(rows, recorded):
        h2[keys[rp.key]] = state_hash(rp.backend.game)
    assert h1 == h2
    assert (ds.rows_of(keys) == rows).all()


def test_replayed_backends_are_private(recorded):
    """Acting on a yielded backend (a clone) does not disturb the next positions of the same run."""
    ds = OD.Dataset(recorded)
    rows = np.arange(len(ds))
    for rp in OD.replay_positions(rows, recorded, check=True):
        legal = np.flatnonzero(rp.obs["mask"])
        rp.backend.act(search.to_backend_action(int(legal[0])))


def test_view_and_observer_are_the_players(recorded):
    ds = OD.Dataset(recorded)
    rp = next(iter(OD.replay_positions([0], recorded)))
    assert rp.observer is rp.backend._observer
    assert rp.ctx == search.run_ctx(rp.backend)
    again = encode.battle(rp.view, rp.ctx)
    for k in OD.OBS_KEYS:
        assert np.array_equal(again[k], rp.obs[k])
    assert ds.keys([0])[0][0] == int(ds["run"][0])


def test_research_target_on_replayed_position(recorded):
    """A replayed position can be searched again (experiment 4): a legal target over the recorded mask."""
    from rl.inference import make_evaluator
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(search.TRAINING_ENV, "1")
        gb = G.GumbelBattler(make_evaluator(CKPT), n_sims=16, n_determinizations=2, seed=0,
                             opponent_prior="factory_sets")
        ds = OD.Dataset(recorded)
        i = int(np.flatnonzero(np.asarray(ds["searched"]))[0])
        rp = next(iter(OD.replay_positions([i], recorded, check=True)))
        r = gb.search_root(rp.backend, int(ds["ndec"][i]), rp.obs, noise=True)
    assert np.isclose(r["target"].sum(), 1) and (r["target"][~rp.obs["mask"].astype(bool)] == 0).all()
