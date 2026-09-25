"""Factory environments for training: one run of the Factory per environment, many in worker processes.

Each environment advances to the next decision of either agent and reports it as an event:

    {"kind": "battle" | "rental" | "swap", "obs": encoded observation,
     "close_b": (reward, done, trunc) | None,   closes the battler's previous transition
     "close_t": (reward, done, trunc) | None,   closes the tactician's previous transition
     "truncate": bool,                          the battle hit max_decisions: bootstrap only, then send "reset"
     "stats": {...}}                            battles / runs that just finished

Rewards (docs/RL_DECISIONS.md §3): battler +1 win, 0 loss, plus potential-based shaping beta*(gamma*Phi' - Phi)
with Phi = 0 at the end of the battle; tactician +1 per battle won since its previous decision.
"""

import multiprocessing as mp
import random
import struct

import numpy as np

from pybattle.backend import Phase, SimBackend, rents_offset
from pybattle.emu.decode import SYMBOLS as S, decode_battle_mon, decode_party
from . import encode

NO_ACTION = object()      # "just run to the next decision" (None is a real action: keep the team at a swap)


def potential(game, view, w_hp=0.4, w_alive=0.4, w_stages=0.2):
    """Phi(s) from the true state: HP fractions, Pokemon alive, and the 7 stat stages of both active Pokemon."""
    own = decode_party(game.read(S.addr("gPlayerParty"), 300))[:3]
    foe = decode_party(game.read(S.addr("gEnemyParty"), 300))[:3]
    mons = [decode_battle_mon(game.read(S.addr("gBattleMons") + b * 0x58, 0x58)) for b in range(2)]
    idx = struct.unpack("<2H", game.read(S.addr("gBattlerPartyIndexes"), 4))

    def side(party, active, battle_mon):
        hp = [battle_mon.hp if i == active else m.hp for i, m in enumerate(party)]
        frac = np.mean([h / m.max_hp if m.max_hp else 0.0 for h, m in zip(hp, party)])
        return frac, sum(h > 0 for h in hp)

    hp_o, alive_o = side(own, idx[0], mons[0])
    hp_e, alive_e = side(foe, idx[1], mons[1])
    stages = np.mean([(a - b) / 12.0 for a, b in zip(view.own_active.stat_stages, view.enemy_active.stat_stages)])
    return w_hp * (hp_o - hp_e) + w_alive * (alive_o - alive_e) / 3.0 + w_stages * stages


class FactoryEnv:
    def __init__(self, seed, gamma=0.99, beta=0.5, max_decisions=300, win_streak=0, start_p0=1.0,
                 start_max_round=5):
        """start_p0: probability that a run starts at round 1 (streak 0); otherwise it starts at the beginning of
        a round drawn uniformly from 2..start_max_round+1 (streak 7k), with a random rental counter in the range a
        player could have there: [k, 7k] (each completed round adds 1 for the rental plus up to 6 swaps; the
        counter only moves on wins). Defaults: always from streak `win_streak`."""
        self.rng = random.Random(seed)
        self.gamma, self.beta, self.max_decisions, self.win_streak = gamma, beta, max_decisions, win_streak
        self.start_p0, self.start_max_round = start_p0, start_max_round
        self.start_rng = random.Random(seed ^ 0x5EED)     # separate stream: run seeds stay the same as before
        self.backend = SimBackend(max_turns=10 ** 9)          # truncation is handled here, not by forfeiting
        self._new_run()

    def _new_run(self):
        seed = self.rng.getrandbits(32)
        if self.start_p0 < 1.0 and self.start_rng.random() >= self.start_p0:
            k = self.start_rng.randint(1, self.start_max_round)
            self.start_streak, rents = 7 * k, self.start_rng.randint(k, 7 * k)
        else:
            k = self.win_streak // 7          # a fixed later start gets a random feasible rental counter too
            self.start_streak, rents = self.win_streak, (self.start_rng.randint(k, 7 * k) if k else 0)
        # (SimBackend.reset gives the run the symbols a player with this streak holds: Noland's silver from 21,
        # the gold from 42; without them the game would not schedule his later battles, GetFrontierBrainStatus)
        self.backend.reset(seed=seed, win_streak=self.start_streak, rents_count=rents)
        self.b_pending, self.b_phi = False, 0.0
        self.t_pending, self.t_acc = False, 0
        self.decisions = 0
        self.battle_rewards = 0.0

    def _ctx(self):
        info = self.backend.run_info()
        rents = int.from_bytes(self.backend.game.read_saveblock2(rents_offset(self.backend.open_level), 2),
                               "little")
        return {"streak": info.win_streak, "battle": info.battle_in_challenge, "challenge": info.challenge_num,
                "rents": rents}

    def reset(self):
        self._new_run()
        return self._advance(NO_ACTION)

    def step(self, action):
        if isinstance(action, str) and action == "reset":
            return self.reset()
        return self._advance(action)

    def _advance(self, action):
        """Apply `action` (NO_ACTION right after a reset) and run to the next decision."""
        be = self.backend
        close_b = close_t = None
        stats = {}
        if action is not NO_ACTION:
            battle_phase = be.phase in (Phase.BATTLE, Phase.FORCED_SWITCH)
            be.act(action)
            if battle_phase and be.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
                won = bool(be.last_battle_won)
                r = float(won) - self.beta * self.b_phi                   # gamma * Phi(terminal) = 0
                close_b, self.b_pending = (r, True, False), False
                stats["battle"] = {"won": won, "decisions": self.decisions,
                                   "shaped_return": self.battle_rewards + r}
                self.decisions, self.battle_rewards = 0, 0.0
                self.t_acc += int(won)
        while True:
            # skip decisions where nothing can be chosen (the game picks the move itself)
            if be.phase == Phase.BATTLE:
                view = be.view()
                if not any(view.usable_moves) and not view.switch_targets and self.decisions < self.max_decisions:
                    be.act(("move", 0))
                    self.decisions += 1                                 # counts towards truncation too
                    if be.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):   # the battle ended on its own
                        won = bool(be.last_battle_won)
                        r = float(won) - self.beta * self.b_phi
                        close_b, self.b_pending = (r, True, False), False
                        stats["battle"] = {"won": won, "decisions": self.decisions,
                                           "shaped_return": self.battle_rewards + r}
                        self.decisions, self.battle_rewards = 0, 0.0
                        self.t_acc += int(won)
                    continue
            if be.phase == Phase.RUN_OVER:
                if self.t_pending:
                    close_t = (float(self.t_acc), True, False)
                stats["run"] = {"streak": be.wins, "start": self.start_streak}
                self._new_run()
                continue
            break
        if be.phase in (Phase.RENTAL, Phase.SWAP):
            if self.t_pending:
                close_t = (float(self.t_acc), False, False)
            self.t_pending, self.t_acc = True, 0
            view = be.view()
            kind = "rental" if be.phase == Phase.RENTAL else "swap"
            # the swap mask mirrors Swap_AlreadyHasSameSpecies (host_factory_screen.c)
            obs = encode.rental(view, self._ctx()) if kind == "rental" else encode.swap(view, self._ctx())
            return {"kind": kind, "obs": obs, "close_b": close_b, "close_t": close_t, "truncate": False,
                    "stats": stats}
        view = be.view()
        phi = potential(be.game, view) if self.beta else 0.0       # (every shaping term is beta * ...)
        truncate = self.decisions >= self.max_decisions
        if self.b_pending:
            r = self.beta * (self.gamma * phi - self.b_phi)
            self.battle_rewards += r
            close_b = (r, False, truncate)
        obs = encode.battle(view, self._ctx())
        if truncate:
            stats["battle"] = {"won": None, "decisions": self.decisions, "shaped_return": self.battle_rewards}
            stats["truncated"] = 1
            stats["wins"] = be.wins
            close_t = (float(self.t_acc), False, True) if self.t_pending else None
            self.b_pending = self.t_pending = False
            return {"kind": "battle", "obs": obs, "close_b": close_b, "close_t": close_t, "truncate": True,
                    "stats": stats}
        self.b_pending, self.b_phi = True, phi
        self.decisions += 1
        return {"kind": "battle", "obs": obs, "close_b": close_b, "close_t": close_t, "truncate": False,
                "stats": stats}


def decode_action(kind, a):
    """Network action index -> backend action."""
    if kind == "battle":
        a = int(a)
        return ("move", a) if a < 4 else ("switch", a - 4)
    if kind == "swap":
        a = int(a)
        return None if a == 0 else ((a - 1) // 3, (a - 1) % 3)
    lead, pair = int(a[0]), int(a[1])
    j, k = encode.PAIRS[pair]
    return (lead, j, k)


# ---- worker processes ---------------------------------------------------------------------------------------

def _worker(conn, seeds, env_kwargs):
    envs = [FactoryEnv(s, **env_kwargs) for s in seeds]
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send([e.reset() for e in envs])
            elif cmd == "start":                    # the run started at construction (keeps run seeds fixed)
                conn.send([e._advance(NO_ACTION) for e in envs])
            elif cmd == "step":
                conn.send([e.step(a) for e, a in zip(envs, data)])
            elif cmd == "set":                      # change an attribute (e.g. the shaping weight beta)
                for e in envs:
                    setattr(e, data[0], data[1])
                conn.send(True)
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass


class VecEnv:
    """n_workers processes x envs_per_worker environments, stepped in lockstep (one decision each)."""

    def __init__(self, n_workers, envs_per_worker, seed=0, **env_kwargs):
        ctx = mp.get_context("fork")
        self.n = n_workers * envs_per_worker
        self.k = envs_per_worker
        self.conns, self.procs = [], []
        for w in range(n_workers):
            a, b = ctx.Pipe()
            seeds = [seed * 100_003 + w * envs_per_worker + i for i in range(envs_per_worker)]
            p = ctx.Process(target=_worker, args=(b, seeds, env_kwargs), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return [ev for c in self.conns for ev in c.recv()]

    def start(self):
        for c in self.conns:
            c.send(("start", None))
        return [ev for c in self.conns for ev in c.recv()]

    def step(self, actions):
        for w, c in enumerate(self.conns):
            c.send(("step", actions[w * self.k:(w + 1) * self.k]))
        return [ev for c in self.conns for ev in c.recv()]

    def set(self, attr, value):
        for c in self.conns:
            c.send(("set", (attr, value)))
        for c in self.conns:
            c.recv()

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except BrokenPipeError:
                pass
        for p in self.procs:
            p.join(timeout=5)
