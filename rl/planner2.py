"""The improved tactician planner (docs/RL_DECISIONS.md §27.x "Planificador mejorado"), measured as a tactician by
rl/agent_probe.py ("planner2:<ckpt>[:options]"). No training uses it yet.

    cfg = Planner2Config(n_opp=32, hint_next=True)                    # or parse_options("n=32,h=1")
    sampler = OpponentSampler("experience", model)                     # rl.opponent_model.ExperienceModel
    out = plan_values2(ev, view, kind, ctx, options, v_next, hint, cfg, sampler, backend=be, noland_mc=mc)
    out["q_plan"]                                                       # [n_opt], in wins (as rl.team_eval)

It is the planner of rl/team_eval.py (§20.2.4: Q_plan = expectimax over the next opponent and our best swap answer,
W as the leaf, common random numbers), with switchable changes, each off by default except the sampler:

    prior       "experience" (default): the opponents of battle k (and k+1, k+2 at depth 2) are drawn from the opponent
                model learned from the player's own games (rl/opponent_model.py, §22), the round's bucket, no
                repeated species / held item, none of the species on screen, with the heard hint (rejection on
                hint_of_arrays, as the other samplers). "factory_sets": the Factory's set list exactly as
                rl.team_eval.plan_values draws it (the same random calls: same values for the same seed); it is the
                game's hidden list, so it is refused outside training unless PYBATTLE_DIAGNOSTIC_PRIOR=factory_sets
                (a diagnostic ceiling). NOTE: rl.team_eval.plan_values (the planner of v3 / v4) always draws from that
                list, at evaluation too.
    n_opp       opponents of battle k (8 in v3 / v4).
    depth       1: one swap ahead (v3 / v4); 2: two swaps ahead (n_opp2 opponents of battle k + 1 per level-1
                opponent, the `beam` best level-1 answers by V_0 expanded).
    hint_next   the future swap decision HEARS the hint of the battle after it (the swap screen shows it): for each
                level-1 opponent n, battle k + 1's team is drawn (none of the visible species nor opponent n's), its
                hint h'_n computed as the game does, and the level-1 answers are valued with p_{k+1} = W(. | h'_n).
                Without it (v3 / v4) the answer is chosen with W(. | unknown): the planner ignores that it will know
                the next opponent's type / style when it swaps (max of an average instead of average of maxes).
                Noland's battle has no hint (unknown, as the game). At depth 2 the opponents of battle k + 1 are drawn
                with that hint.
    noland      "w" (default): Noland's battle is valued by W's Noland head (trained on battles against teams drawn
                with his rules: rl/team_eval_data.py). "mc": at the swap right before Noland's battle (battle 7 of
                rounds 3 and 6: GetFrontierBrainStatus with the symbols of that streak), p_7 of each option is the
                win rate of `noland_sims` simulated battles of the checkpoint's battler network (greedy) against
                Noland teams drawn as the game draws his (FillFactoryBrainParty): the round's opponent distribution
                (here the experience model's), no repeated species / item, NOT the sets of the 6 Pokemon on the
                swap screen (our team after the swap and the defeated team; the one we give away becomes possible),
                species of ours allowed, every IV = his fixed IV (GetFactoryMonFixedIV(challenge + 2): 15 in round
                3, 31 from round 6), shrunk toward W: p = (noland_w_n W + wins) / (noland_w_n + sims). The same
                teams and battle seeds for every option (common random numbers).

Information: the planner reads the view (our team, the candidates / the defeated team as seen, the hint), the round,
the opponent model (what the player saw) and, for "mc", clones of the game where the hidden next opponent is
overwritten before the first frame (rl.tactician_search.start_simulated_battle's rule). Never the true opponents.
"""

import random
from dataclasses import dataclass, fields, asdict
from typing import Optional

import numpy as np

from pybattle.view import possible_abilities
from . import determinize as DET
from . import team_eval as T

PRIORS = ("experience", "factory_sets")
NOLAND_MODES = ("w", "mc")
NO_HINT_CODE = -1


@dataclass
class Planner2Config:
    depth: int = 1
    n_opp: int = 8
    n_opp2: int = 4
    beam: int = 3
    hint_next: bool = False
    prior: str = "experience"
    noland: str = "w"
    noland_sims: int = 32
    noland_w_n: float = 8.0
    seed: Optional[int] = None
    chunk: int = 16384
    max_tries: int = 4096          # hint rejection: draws per wanted team before the fallback

    def __post_init__(self):
        if self.prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, not {self.prior!r}")
        if self.noland not in NOLAND_MODES:
            raise ValueError(f"noland must be one of {NOLAND_MODES}, not {self.noland!r}")
        if self.depth not in (0, 1, 2):
            raise ValueError("depth must be 0, 1 or 2")


# short option names of the probe spec "planner2:<ckpt>:n=32,h=1,d=2,nol=mc"
SHORT = {"n": "n_opp", "d": "depth", "n2": "n_opp2", "b": "beam", "h": "hint_next", "p": "prior", "nol": "noland",
         "ns": "noland_sims", "nw": "noland_w_n"}
_PRIOR_SHORT = {"exp": "experience", "fs": "factory_sets"}


def parse_options(s) -> Planner2Config:
    """'n=32,h=1,d=2,p=fs,nol=mc,ns=32,nw=8' (any subset, any order; '' = defaults) -> Planner2Config."""
    kw = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        name = SHORT.get(k, k)
        typ = {f.name: f.type for f in fields(Planner2Config)}.get(name)
        if typ is None:
            raise ValueError(f"unknown planner2 option {k!r}")
        if name == "prior":
            kw[name] = _PRIOR_SHORT.get(v, v)
        elif name == "noland":
            kw[name] = v
        elif name == "hint_next":
            kw[name] = v.lower() in ("1", "true", "yes")
        elif name == "noland_w_n":
            kw[name] = float(v)
        else:
            kw[name] = int(v)
    return Planner2Config(**kw)


def options_tag(cfg: Planner2Config) -> str:
    """Canonical option string of the non-default options ('n=32,h=1'; '' for the defaults): parse_options reads it
    back; file_tag(...) makes it a file-name part."""
    d = Planner2Config()
    inv = {v: k for k, v in SHORT.items()}
    out = []
    for name in ("prior", "n_opp", "depth", "n_opp2", "beam", "hint_next", "noland", "noland_sims", "noland_w_n"):
        v = getattr(cfg, name)
        if v == getattr(d, name):
            continue
        if name == "prior":
            v = {b: a for a, b in _PRIOR_SHORT.items()}[v]
        elif isinstance(v, bool):
            v = int(v)
        elif isinstance(v, float) and v.is_integer():
            v = int(v)
        out.append(f"{inv[name]}={v}")
    return ",".join(out)


def file_tag(opts: str) -> str:
    """'n=32,h=1' -> 'n32-h1' (file names)."""
    return opts.replace("=", "").replace(",", "-")


# ---- opponent teams ---------------------------------------------------------------------------------------------

@dataclass
class Draw:
    """n opponent teams: species [n, 3], moves [n, 3, 4], item [n, 3], set_ids [n, 3] (factory_sets only)."""
    species: np.ndarray
    moves: np.ndarray
    item: np.ndarray
    set_ids: Optional[np.ndarray] = None
    cfg: Optional[list] = None          # experience: per slot the model's configuration (or None: novel)

    def __len__(self):
        return len(self.species)


def hint_codes(species, moves):
    """The game's hint of each team -> (hint_type, hint_style) arrays."""
    return DET.hint_of_arrays(np.asarray(species, np.int64), np.asarray(moves, np.int64))


class OpponentSampler:
    """Draws whole opponent teams for the planner. prior "experience" (model: rl.opponent_model.ExperienceModel) or
    "factory_sets" (the Factory's set list: training / explicit diagnostic only)."""

    def __init__(self, prior="experience", model=None, open_level=True):
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, not {prior!r}")
        if prior == "factory_sets":
            DET.check_prior("factory_sets")
        elif model is None:
            raise ValueError("the experience sampler needs its rl.opponent_model.ExperienceModel")
        self.prior, self.model, self.open_level = prior, model, open_level
        self.stats = {"calls": 0, "fallback": 0}

    # -- normal trainers --
    def normal(self, challenge, n, g, excluded_species=(), hint=None, level=100, max_tries=4096) -> Draw:
        """n teams of a normal trainer of the round (no repeated species / held item, none of excluded_species),
        each with the hint `hint` if given (rejection; after max_tries draws per team, a non-matching one)."""
        self.stats["calls"] += 1
        if self.prior == "factory_sets":
            ids = DET.sample_pool_teams(DET.pool(challenge, self.open_level), n, g, excluded_species, hint,
                                        max_tries)
            return self._of_ids(ids)
        return self._exp_draw(challenge, n, g, excluded_species, hint, level, None, max_tries)

    def noland(self, challenge, n, g, excluded=None, level=100) -> Draw:
        """n teams of Noland: the round's distribution, no repeated species / held item, none of the excluded
        configurations (excluded(table) -> bool mask over the table's entries for experience; a set of set ids for
        factory_sets). No hint, no species exclusion (FillFactoryBrainParty)."""
        self.stats["calls"] += 1
        if self.prior == "factory_sets":
            ids = DET.pool(challenge, self.open_level)
            if excluded:
                ids = ids[~np.isin(ids, np.array(sorted(excluded), np.int64))]
            return self._of_ids(DET.sample_pool_teams(ids, n, g))
        return self._exp_draw(challenge, n, g, (), None, level, excluded, 4096)

    @staticmethod
    def _of_ids(ids):
        ids = np.asarray(ids, np.int64)
        mv = np.array([[list(DET.SET_MOVES_AS_BUILT[int(x)]) for x in row] for row in ids], np.int64).reshape(-1, 3, 4)
        return Draw(DET.SET_SPECIES[ids], mv, DET.SET_ITEM[ids], ids)

    def _exp_draw(self, challenge, n, g, excluded_species, hint, level, excluded_cfg, max_tries):
        from .opponent_model import bucket_of
        t = self.model.compile()
        b = bucket_of(challenge)
        tab = t.joint_table(b, frozenset(), level, set(int(x) for x in excluded_species))
        bad = None
        if excluded_cfg is not None:
            bad = np.asarray(excluded_cfg(tab), bool)
            if bad.any():
                tab = dict(tab, cum=np.cumsum(np.where(bad, 0.0, np.diff(np.concatenate([[0.0], tab["cum"]])))))
        got = []                              # (idx [3], sp [3], mv [3, 4], it [3])
        last = None
        tried, M = 0, max(64, 4 * n)
        while len(got) < n and tried < max_tries * n:
            rows = [t.draw_rows(tab, M, g, level) for _ in range(3)]
            idx = np.stack([r[0] for r in rows], 1)
            sp = np.stack([r[1] for r in rows], 1)
            mv = np.stack([r[2] for r in rows], 1)
            it = np.stack([r[3] for r in rows], 1)
            ok = (sp[:, 0] != sp[:, 1]) & (sp[:, 0] != sp[:, 2]) & (sp[:, 1] != sp[:, 2])
            for a, c in ((0, 1), (0, 2), (1, 2)):
                ok &= (it[:, a] <= 0) | (it[:, a] != it[:, c])
            tried += M
            valid = np.flatnonzero(ok)
            if len(valid):
                last = (idx[valid], sp[valid], mv[valid], it[valid])
                if hint is not None:
                    ht, hs = hint_codes(sp[valid], mv[valid])
                    valid = valid[(ht == int(hint[0])) & (hs == int(hint[1]))]
                for j in valid[:n - len(got)]:
                    got.append((idx[j], sp[j], mv[j], it[j]))
            M = min(M * 4, 8192)
        if len(got) < n:
            self.stats["fallback"] += n - len(got)
            if last is None:
                raise RuntimeError("the opponent model drew no valid team")
            for j in range(n - len(got)):
                k = j % len(last[0])
                got.append(tuple(a[k] for a in last))
        idx = np.stack([x[0] for x in got])
        sp = np.stack([x[1] for x in got]).astype(np.int64)
        mv = np.stack([x[2] for x in got]).astype(np.int64)
        it = np.stack([x[3] for x in got]).astype(np.int64)
        # a novel configuration's item is drawn now (useful items, not one a teammate holds), as the model does
        for i, s in zip(*np.nonzero(it < 0)):
            used = {int(x) for x in it[i] if x > 0}
            items = [x for x in DET.plausible_items(int(sp[i, s])) if x not in used]
            it[i, s] = items[int(g.integers(len(items)))] if items else 0
        cfg = [[tab["cfg"][int(k)] for k in row] for row in idx]
        return Draw(sp, mv, it, None, cfg)


def seen_mon(species, moves, item, g, reveal=T.REVEAL) -> T.TeamMon:
    """A drawn opponent Pokemon as the swap screen shows it after we beat it (rl.team_eval.seen_set_mon's masking:
    REVEAL's number of moves seen, the item with REVEAL's probability, the ability if unique or with REVEAL's
    probability, stats unknown); the same random calls as seen_set_mon."""
    sp = int(species)
    real = [int(x) for x in moves if x]
    pm = np.asarray(reveal["moves"], np.float64)
    r = int(g.choice(len(pm), p=pm / pm.sum()))
    keep = [real[i] for i in g.permutation(len(real))[:r]] if real else []
    mv = tuple((keep + [T.UNKNOWN] * 4)[:4])
    it = int(item) if g.random() < reveal["item"] else T.UNKNOWN
    pa = possible_abilities(sp)
    ab = pa[0] if len(pa) == 1 else (int(pa[int(g.integers(2))]) if g.random() < reveal["ability"] else 0)
    return T.TeamMon(sp, mv, it, int(ab), None)


def draw_seen_mons(draw: Draw, table, g):
    """Mon-table indices [n, 3] of the drawn teams as seen at the swap screen."""
    if draw.set_ids is not None:
        return np.array([[table.add(T.seen_set_mon(int(x), g)) for x in row] for row in draw.set_ids], np.int64)
    return np.array([[table.add(seen_mon(draw.species[i, j], draw.moves[i, j], draw.item[i, j], g))
                      for j in range(3)] for i in range(len(draw))], np.int64)


# ---- W with per-team hints ------------------------------------------------------------------------------------------

class WCache:
    """W(team | hint) of the distinct (team, hint) pairs the planner meets (mon-index triples of a _MonTable; hint
    code = 16 * type + style, NO_HINT_CODE = unknown), evaluated in batched network calls, each pair once."""

    def __init__(self, ev, table, round_idx, chunk=16384):
        self.ev, self.table, self.round_idx, self.chunk = ev, table, round_idx, chunk
        self.uses_hint = T.ev_uses_hint(ev)
        self.store = {}                     # hint code -> (sorted keys, W)
        self.pending = {}
        self._mon = None

    @property
    def n(self):
        return sum(len(v[0]) for v in self.store.values()) + sum(len(v) for v in self.pending.values())

    def _code(self, codes, shape):
        c = np.broadcast_to(np.asarray(codes, np.int64), shape)
        return c if self.uses_hint else np.full(shape, NO_HINT_CODE, np.int64)

    def add(self, teams, codes=NO_HINT_CODE):
        t = np.asarray(teams, np.int64)
        keys = T._team_keys(t).reshape(-1)
        c = self._code(codes, t.shape[:-1]).reshape(-1)
        for h in np.unique(c):
            u = np.unique(keys[c == h])
            have = self.store.get(int(h), (np.zeros(0, np.int64), None))[0]
            u = u[~np.isin(u, have)]
            if len(u):
                p = self.pending.get(int(h))
                self.pending[int(h)] = u if p is None else np.union1d(p, u)

    def evaluate(self):
        if not self.pending:
            return
        if self._mon is None or len(self._mon[0]) < len(self.table.mons):
            self._mon = T._mon_rows(self.table.mons)
        hs = sorted(self.pending)
        keys = np.concatenate([self.pending[h] for h in hs])
        codes = np.concatenate([np.full(len(self.pending[h]), h, np.int64) for h in hs])
        self.pending = {}
        triples = T._keys_to_teams(keys)
        W = np.zeros((len(triples), 3))
        unk = codes < 0
        for grp in (np.flatnonzero(unk), np.flatnonzero(~unk)):        # unknown: the network's own "no hint" call
            for i in range(0, len(grp), self.chunk):
                sel = grp[i:i + self.chunk]
                tr, cd = triples[sel], codes[sel]
                ids = self._mon[0][tr]
                num = self._mon[1][tr].copy()
                num[:, 0, 7] = 1.0
                if self.uses_hint and cd[0] >= 0:
                    W[sel] = self.ev.w_features(ids, num, self.round_idx, np.minimum(cd // 16, T.HINT_UNK_TYPE),
                                                np.minimum(cd % 16, T.HINT_UNK_STYLE))
                else:
                    W[sel] = self.ev.w_features(ids, num, self.round_idx)
        for h in hs:
            sel = codes == h
            k_old, w_old = self.store.get(int(h), (np.zeros(0, np.int64), np.zeros((0, 3))))
            k = np.concatenate([k_old, keys[sel]])
            w = np.concatenate([w_old, W[sel]])
            o = np.argsort(k, kind="stable")
            self.store[int(h)] = (k[o], w[o])

    def w(self, teams, codes=NO_HINT_CODE):
        t = np.asarray(teams, np.int64)
        keys = T._team_keys(t)
        c = self._code(codes, keys.shape)
        out = np.zeros(keys.shape + (3,))
        for h in np.unique(c):
            m = c == h
            k, W = self.store[int(h)]
            out[m] = W[np.searchsorted(k, keys[m])]
        return out


def code_of(hint):
    return NO_HINT_CODE if hint is None else 16 * int(hint[0]) + int(hint[1])


# ---- the planner ------------------------------------------------------------------------------------------------

def team_level(view, kind):
    return int(view.candidates[0].level if kind == "rental" else view.own_party[0].level) or 100


def plan_values2(ev, view, kind, ctx, options, v_next, hint=None, cfg=None, sampler=None, backend=None,
                 noland_mc=None, with_current=False):
    """Q_plan of the options with the improvements of Planner2Config (module docstring). -> dict as
    rl.team_eval.plan_values (q_plan, phi, e_round, w_hint, w_unknown, teams) plus "noland_mc" (the simulated
    p_7 per option, when used)."""
    cfg = cfg if isinstance(cfg, Planner2Config) else Planner2Config(**(cfg or {}))
    if sampler is None:
        sampler = OpponentSampler(cfg.prior)
    r, k = T.round_of(ctx)
    types = T.battle_types(r)
    noland_k = types[k - 1] == T.NOLAND
    if noland_k:
        hint = None
    g = np.random.default_rng(cfg.seed)
    table = T._MonTable()
    root_teams = [T.team_of_option(view, kind, o) for o in options]
    if with_current and kind == "swap":
        root_teams.append(T.current_team(view))
    R = len(root_teams)
    if R == 0:
        z = np.zeros(0)
        return {"q_plan": z, "phi": z, "e_round": z, "w_hint": np.zeros((0, 3)), "w_unknown": np.zeros((0, 3)),
                "teams": 0}
    root = np.array([[table.add(m) for m in t.mons] for t in root_teams], np.int64)          # [R, 3]
    depth = cfg.depth if k < 7 else 0
    visible = T._visible_species(view, kind)
    challenge = int(ctx.get("challenge", r - 1))
    level = team_level(view, kind)
    wc = WCache(ev, table, r, cfg.chunk)
    hc = code_of(hint) if hint is not None else NO_HINT_CODE
    wc.add(root)
    wc.add(root, hc)
    L1 = L2 = next_codes = None
    if depth >= 1:
        opp1 = sampler.normal(challenge, cfg.n_opp, g, visible, hint, level, cfg.max_tries)          # battle k
        m1 = draw_seen_mons(opp1, table, g)
        L1 = T._swap_children(root[:, None, :], m1[None])                                          # [R, N, 10, 3]
        wc.add(L1)
        next_noland = types[k] == T.NOLAND                                                         # battle k + 1
        if cfg.hint_next and not next_noland:
            nxt = [sampler.normal(challenge, 1, g, visible | set(int(x) for x in opp1.species[n]), None, level)
                   for n in range(len(opp1))]
            ht = [int(hint_codes(d.species, d.moves)[0][0]) for d in nxt]
            hs = [int(hint_codes(d.species, d.moves)[1][0]) for d in nxt]
            next_codes = np.array([16 * a + b for a, b in zip(ht, hs)], np.int64)                  # [N]
            wc.add(L1, next_codes[None, :, None])
        if depth >= 2 and k + 1 < 7:
            m2 = []
            for n in range(len(opp1)):
                h2 = None if next_codes is None else (int(next_codes[n]) // 16, int(next_codes[n]) % 16)
                d2 = sampler.normal(challenge, cfg.n_opp2, g, visible | set(int(x) for x in opp1.species[n]), h2,
                                    level, cfg.max_tries)
                m2.append(draw_seen_mons(d2, table, g))
            L2 = np.stack(m2)                                                                    # [N, N2, 3]
    wc.evaluate()
    W_u_root = wc.w(root)
    W_h_root = wc.w(root, hc)
    phi_root = T.leaf_values(W_h_root, W_u_root, k, r, v_next)
    e_root = T.leaf_values(W_h_root, W_u_root, k, r, 0.0)
    p_k = W_h_root[:, types[k - 1]]
    extra = {}
    if depth == 0:
        q = phi_root
    else:
        Wl1u = wc.w(L1)                                                                          # [R, N, 10, 3]
        Wl1h = Wl1u if next_codes is None else wc.w(L1, next_codes[None, :, None])
        v0_l1 = T.leaf_values(Wl1h, Wl1u, k + 1, r, v_next)                                      # [R, N, 10]
        if L2 is None:
            q = p_k * (1.0 + v0_l1.max(-1).mean(-1))
        else:
            B = min(max(1, cfg.beam), 10)
            top = np.argsort(-v0_l1, axis=-1, kind="stable")[..., :B]                            # [R, N, B]
            E = np.take_along_axis(L1, top[..., None], axis=2)                                   # [R, N, B, 3]
            L2t = T._swap_children(E[:, :, :, None, :], L2[None, :, None, :, :])                 # [R,N,B,N2,10,3]
            wc.add(L2t)
            wc.evaluate()
            Wl2 = wc.w(L2t)
            v2 = T.leaf_values(Wl2, Wl2, k + 2, r, v_next).max(-1).mean(-1)                      # [R, N, B]
            WE = np.take_along_axis(Wl1h, top[..., None], axis=2)                                # [R, N, B, 3]
            v1 = WE[..., types[k]] * (1.0 + v2)
            q = p_k * (1.0 + v1.max(-1).mean(-1))
    if cfg.noland == "mc" and noland_k and kind == "swap" and noland_mc is not None and backend is not None:
        opts = list(options) + ([0] if with_current else [])
        p_sim, n_sim = noland_mc.p_win(backend, opts, view, challenge, g, level)
        w = float(cfg.noland_w_n)
        p = (w * W_h_root[:, T.NOLAND] + p_sim * n_sim) / (w + n_sim)
        q = p * (1.0 + float(v_next))                                                             # k = 7: Phi
        extra["noland_mc"] = p_sim
    out = {"q_plan": q, "phi": phi_root, "e_round": e_root, "w_hint": W_h_root, "w_unknown": W_u_root,
           "teams": wc.n, **extra}
    if with_current and kind == "swap":
        out["current"] = {key: v[-1] for key, v in out.items() if key != "teams"}
        for key in ("q_plan", "phi", "e_round", "w_hint", "w_unknown", "noland_mc"):
            if key in out:
                out[key] = out[key][:-1]
    return out


# ---- Noland, simulated ------------------------------------------------------------------------------------------

def noland_iv(challenge):
    """Noland's fixed IV (FillFactoryBrainParty: GetFactoryMonFixedIV(challengeNum + 2, FALSE)): 15 in round 3
    (challenge 2), 31 from round 6 (challenge 5) on."""
    return DET.fixed_iv(int(challenge) + 2, False)


def _own_cfg(m):
    return int(m.species), frozenset(int(x) for x in m.moves if x), int(m.item)


def noland_exclusion(view, option):
    """The configurations Noland cannot have after the swap `option` (0..9) at a SwapView: our team after the swap
    (fully known: summary) and the defeated team (known as seen in battle). -> (own [(species, moves, item)],
    seen [(species, revealed moves, revealed item or None)])."""
    own = [_own_cfg(m) for m in view.own_party[:3]]
    seen = [(int(m.species), frozenset(int(x) for x in m.revealed_moves if x),
             None if m.revealed_item is None else int(m.revealed_item)) for m in view.enemy_party[:3]]
    a = int(option)
    if a:
        i, j = (a - 1) // 3, (a - 1) % 3
        own.pop(i)                       # given away: Noland may have its set again
    return own, seen


def exclusion_mask(tab, own, seen):
    """Experience table entries excluded: a known configuration equal to one of `own`, or consistent with a seen
    defeated Pokemon (same species, its revealed moves among the configuration's, its item if revealed)."""
    bad = np.zeros(len(tab["sp"]), bool)
    own_set = set(own)
    by_sp = {}
    for s, R, I in seen:
        by_sp.setdefault(s, []).append((R, I))
    for e, (s, c) in enumerate(zip(tab["sp"], tab["cfg"])):
        if c is None:
            continue
        s = int(s)
        mv = frozenset(int(x) for x in c[0] if x)
        if (s, mv, int(c[1])) in own_set:
            bad[e] = True
            continue
        for R, I in by_sp.get(s, ()):
            if R <= mv and (I is None or I == int(c[1])):
                bad[e] = True
                break
    return bad


def trained_evs(rng):
    """EVs of a trained Pokemon (the experience model does not learn them): the 510 points split evenly over two
    random stats (255 each, capped at 252), instead of the strict sampler's random partial spread."""
    a, b = rng.sample(range(6), 2)
    ev = [0] * 6
    ev[a] = ev[b] = 252
    return ev


class NolandMC:
    """p_7 of each swap option against Noland by simulated battles of the battler network (greedy, lockstep; the
    rl.tactician_search.BattleSimulator), against Noland teams drawn from `sampler` with his rules; common random
    numbers: the same team draws (filtered per option by its exclusions) and battle seeds for every option."""

    def __init__(self, evaluator, sampler, sims=32, max_decisions=300, ev_mode="trained"):
        from .tactician_search import BattleSimulator
        self.ev_mode = ev_mode
        self.sim = BattleSimulator(evaluator, max_decisions)
        self.sampler, self.sims = sampler, sims
        self.battles = 0

    def p_win(self, backend, options, view, challenge, g, level):
        from .tactician_search import _Sim
        n = self.sims
        iv = noland_iv(challenge)
        if self.sampler.prior == "experience":
            seen = noland_exclusion(view, 0)[1]
            # common draws: excluding only the defeated team's (excluded for every option), then per option its own
            pool = self.sampler.noland(challenge, 4 * n, g, lambda tab: exclusion_mask(tab, [], seen), level)
            team_key = [[(int(pool.species[i, j]), frozenset(int(x) for x in pool.moves[i, j] if x),
                          int(pool.item[i, j])) for j in range(3)] for i in range(len(pool))]
            excl = [set(noland_exclusion(view, o)[0]) for o in options]
        else:
            # factory_sets (diagnostic ceiling): the true exclusions (set ids) of each option, the sets' EVs / nature
            from .envs import decode_action
            excl = [set(backend.opponent_knowledge_for(decode_action("swap", o)).set_ids) for o in options]
            common = set.intersection(*excl) if excl else set()
            pool = self.sampler.noland(challenge, 4 * n, g, common, level)
            team_key = [[int(x) for x in row] for row in pool.set_ids]
        seeds = [int(x) for x in g.integers(0, 2 ** 32, n)]
        rng = random.Random(int(g.integers(2 ** 31)))
        specs_all = [self._specs(pool, i, rng, iv, self.sampler.model, self.ev_mode) for i in range(len(pool))]
        sims, tags = [], []
        for oi, o in enumerate(options):
            ok = [i for i in range(len(pool)) if not (set(team_key[i]) & excl[oi])][:n]
            for m, i in enumerate(ok):
                be = self._start(backend, o, specs_all[i], seeds[m])
                sims.append(_Sim(be, oi, self.sim.cpp))
                tags.append(oi)
        self.sim.run(sims)
        self.battles += len(sims)
        tot, cnt = np.zeros(len(options)), np.zeros(len(options))
        for s in sims:
            tot[s.tag] += s.value
            cnt[s.tag] += 1
        return tot / np.maximum(cnt, 1), cnt

    @staticmethod
    def _specs(pool, i, rng, iv, model=None, ev_mode="trained"):
        if pool.set_ids is not None:
            return [DET.set_spec(j, int(pool.set_ids[i, j]), iv, DET._ability_bit(int(pool.species[i, j]), None, rng),
                                 1.0) for j in range(3)]
        specs = []
        for j in range(3):
            s = int(pool.species[i, j])
            cfg = pool.cfg[i][j] if pool.cfg is not None else None
            nature = model.compile().nature(s, cfg, rng) if (model is not None and cfg is not None) \
                else rng.randrange(DET.NUM_NATURES)
            evs = trained_evs(rng) if ev_mode == "trained" else DET._evs(rng)
            specs.append((j, s, [int(x) for x in pool.moves[i, j]], int(pool.item[i, j]), [iv] * 6, evs,
                          nature, DET._ability_bit(s, None, rng), 1.0))
        return specs

    @staticmethod
    def _start(backend, option, specs, seed):
        from pybattle.backend import Phase
        from pybattle.emu.decode import decode_party
        from .tactician_search import _PLAYER_PARTY
        be = backend.clone()
        g = be.game
        a = int(option)
        ok = g.factory_swap(-1) if a == 0 else g.factory_swap((a - 1) // 3, (a - 1) % 3)
        if not ok:
            raise ValueError(f"swap option {option} refused by the game")
        own = g.read(_PLAYER_PARTY, 300)
        g.write_party(1, own)              # the real Noland team is overwritten before any frame
        if not g.determinize(specs):
            raise RuntimeError("the simulated Noland team was not fully rebuilt")
        g.set_rng(seed)
        be.phase = Phase.BATTLE
        return be


# ---- the probe's tactician -------------------------------------------------------------------------------------------

def opponent_model_for(ckpt, default="runs/opp_model_v3/model.pt"):
    """The experience model a checkpoint plays with: its own (v4) or the warm-start one (v3)."""
    from .opponent_model import ExperienceModel
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("opponent_model"):
        return ExperienceModel.from_state(ck["opponent_model"])
    return ExperienceModel.load(default)


class Planner2Tactician:
    """argmax of plan_values2's Q_plan with a checkpoint's W and v_next table (rl.alphazero_v3.features_hook);
    experience model from the checkpoint (or the warm-start model); Noland's simulation with the checkpoint's
    battler network (`evaluator`)."""

    def __init__(self, ckpt, cfg: Planner2Config, hook, evaluator=None, model=None):
        self.cfg, self.hook = cfg, hook
        if cfg.prior == "experience":
            model = model if model is not None else opponent_model_for(ckpt)
        self.sampler = OpponentSampler(cfg.prior, model)
        self.mc = NolandMC(evaluator, self.sampler, cfg.noland_sims) if cfg.noland == "mc" and evaluator else None
        self.ms = []

    def q(self, be, kind, options, ctx, seed):
        import time
        t0 = time.perf_counter()
        view = be.view()
        hint = (int(view.hint_type), int(view.hint_style))
        ev = self.hook._load()
        cfg = Planner2Config(**dict(asdict(self.cfg), seed=int(seed)))
        out = plan_values2(ev, view, kind, ctx, options, T.v_next_value(self.hook.table(), ctx), hint, cfg,
                           self.sampler, backend=be, noland_mc=self.mc)
        self.ms.append((kind, (time.perf_counter() - t0) * 1000))
        return np.asarray(out["q_plan"], np.float64), out
