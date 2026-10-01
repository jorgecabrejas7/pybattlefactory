"""The real game's current battle, mirrored in the simulator (so a search can run on it).

Decision-time search (rl.search.SearchBattler) clones and steps a native `Gen3Game`; the emulator has no such
state. But the simulator *is* the game's battle code, and pybattle.diff checks that, started from the battle's
starting point (both parties, battle flags, Frontier save data and gRngValue at BeginBattleIntro) and given the
player's actions, it reproduces the emulator's battle RAM and RNG byte for byte at every decision. So:

  1. `WatchedMachine` wraps the emulator and looks at every frame for the battle's start (gBattleMainFunc ==
     BeginBattleIntro) and takes the start snapshot there (pybattle.emu.trace.battle_start), as the trace
     recorder does;
  2. `SimMirror.at_decision(driver)` starts the simulated battle from it on the battle's first decision, runs the
     simulator to its decision and compares it with the emulator (gRngValue, the battle regions of the traces,
     both parties); `apply(action)` gives the simulator the action the emulator got;
  3. while every comparison holds, `MirroredBackend` is a SimBackend-like view of the emulator for the search:
     the emulator's views / run info / battle memory with the simulator's game (`backend.game`).

If a comparison ever fails (it never has in the differential tests) the mirror stops for the rest of that battle
(`synced` False, `error` says why) and the caller falls back to the network.
"""

from typing import Optional

from .decode import SB2_RENTAL_MONS, SYMBOLS as S


class WatchedMachine:
    """An emulator (or a ViewedMachine) with a per-frame look for the start of a battle. Same interface."""

    def __init__(self, emu):
        self.emu = emu
        self._main_func = S.addr("gBattleMainFunc")
        self._intro = S.functions["BeginBattleIntro"]
        self._cb2 = S.addr("gMain") + 4
        self._cb2_ok = {S.functions["CB2_HandleStartBattle"], S.functions["BattleMainCB2"]}
        self.pending_start = None          # battle_start() dict of the battle that just began
        self._armed = True
        self.starts = 0

    def run_frames(self, n: int = 1):
        emu = self.emu
        for _ in range(n):
            emu.run_frames(1)
            if emu.read32(self._main_func) == self._intro and emu.read32(self._cb2) in self._cb2_ok:
                if self._armed:
                    self._armed = False
                    self.pending_start = _battle_start(emu)
                    self.starts += 1
            elif not self._armed and emu.read32(self._cb2) not in self._cb2_ok:
                self._armed = True         # out of the battle: the next BeginBattleIntro is a new battle

    def take_start(self):
        s, self.pending_start = self.pending_start, None
        return s

    def __getattr__(self, name):
        return getattr(self.emu, name)


def _battle_start(emu) -> dict:
    """pybattle.emu.trace.battle_start, from the machine alone, plus the rental save data (the search's own set
    ids are read from the simulator's save block)."""
    import struct
    sb2 = emu.read32(S.addr("gSaveBlock2Ptr"))
    return {
        "rng": emu.read32(S.addr("gRngValue")),
        "battle_type_flags": emu.read32(S.addr("gBattleTypeFlags")),
        "trainer_id": emu.read16(S.addr("gTrainerBattleOpponent_A")),
        "lvl_mode": emu.read8(sb2 + 0xCA9) & 3,
        "battle_num": emu.read16(sb2 + 0xCB2),
        "factory_win_streaks": list(struct.unpack("<4H", emu.read(sb2 + 0xDE2, 8))),
        "tower_win_streaks": list(struct.unpack("<8H", emu.read(sb2 + 0xCE0, 16))),
        "player_party": emu.read(S.addr("gPlayerParty"), 300).hex(),
        "enemy_party": emu.read(S.addr("gEnemyParty"), 300).hex(),
        "rental_mons": emu.read(sb2 + SB2_RENTAL_MONS, 72).hex(),
    }


class SimMirror:
    def __init__(self, check: bool = True):
        self.game = None
        self.synced = False
        self.error: Optional[str] = None
        self.check = check
        self.decisions = 0
        self.battles = 0
        self.desyncs = 0

    def begin(self, start: dict):
        from ..diff import setup_battle
        self.error = None
        self.decisions = 0
        try:
            g = setup_battle(start)
            if start.get("rental_mons"):
                g.write_saveblock2(SB2_RENTAL_MONS, bytes.fromhex(start["rental_mons"]))
            self.game, self.synced = g, True
            self.battles += 1
        except Exception as e:                                  # pragma: no cover (never seen)
            self._fail(f"setup: {type(e).__name__}: {e}")

    def _fail(self, why: str):
        self.synced, self.error = False, why
        self.desyncs += 1

    def at_decision(self, driver, forced: bool) -> bool:
        """The emulator is at a battle decision: bring the simulator to its decision and compare. -> synced."""
        if not self.synced:
            return False
        from ..pybattle_native import Gen3Game
        kind = self.game.run()
        want = Gen3Game.Decision.SWITCH if forced else Gen3Game.Decision.ACTION
        if kind != want:
            self._fail(f"decision {self.decisions}: simulator at {kind}, emulator at {'switch' if forced else 'action'}")
            return False
        if self.check:
            from ..diff import compare, sim_snapshot
            from .trace import snapshot
            emu_snap = snapshot(driver)
            mism = compare(self.decisions, emu_snap, sim_snapshot(self.game, emu_snap["regions"]))
            if mism:
                m = mism[0]
                self._fail(f"decision {self.decisions}: {m.what} {m.expected} {m.got}"[:300])
                return False
        return True

    def apply(self, action):
        """The action the emulator was given at this decision."""
        if not self.synced:
            return
        if action[0] == "move":
            self.game.choose_move(int(action[1]))
        elif action[0] == "switch":
            self.game.choose_switch(int(action[1]))
        else:
            self.synced = False                                 # forfeit: the battle is over
        self.decisions += 1


class MirroredBackend:
    """What rl.search.SearchBattler reads from a backend, for the emulator's current battle decision: the
    emulator backend's phase, views, run info and battle memory, and the mirrored simulator as `game`."""

    def __init__(self, emu_backend, mirror: SimMirror):
        self._be, self.game = emu_backend, mirror.game
        self.phase = emu_backend.phase
        self.open_level = emu_backend.open_level
        self._observer = emu_backend._observer

    def view(self):
        return self._be.view()

    def run_info(self):
        return self._be.run_info()
