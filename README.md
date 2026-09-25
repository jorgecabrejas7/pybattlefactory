# pybattlefactory

A headless, game-exact **Battle Factory** (Pokémon Emerald, singles) for training RL agents, and
the same interface on the real game in mGBA, so agents trained on the simulator can be tested on
the real thing.

A run goes through three kinds of decisions (`pybattle.backend.Phase`):

| Phase | Decision | Action |
|---|---|---|
| `RENTAL` | take 3 of the 6 rental Pokémon (order = party order, first one leads) | `(a, b, c)` select-screen slots |
| `SWAP` | after a win: keep the team or trade one of ours for one of the defeated team's | `None` or `(own_slot, enemy_slot)` |
| `BATTLE` / `FORCED_SWITCH` | pick a move or a switch / pick a replacement after a faint | `("move", slot)`, `("switch", party_index)`, `("forfeit",)` |

**New here? Read [docs/GUIDE.md](docs/GUIDE.md)**: a plain-English walkthrough of how to run everything.

## How exactness works

The simulator *is* the game's code. `third_party/pokeemerald` holds pokeemerald's battle engine,
battle AI, Battle Factory and Pokémon code, vendored verbatim from the decompilation and compiled
for the host (`src/gen3`). Battle and AI scripts run from bytes extracted out of the ROM. Graphics,
sound, text windows and the link cable are stubbed. Battle controllers are headless: the player's
decisions come from Python, the opponent's from the game's own AI.

- All battle RAM lives in one linker section, so a battle is saved, restored or cloned with a
  memcpy. Many independent instances can exist; see `Gen3Game.clone()`.
- Struct layouts follow the GBA compiler's rules, checked at compile time against `pokeemerald.elf`
  (`src/gen3/generated/rom_layout_check.c`).
- **Differential testing:** battles recorded on the real ROM in mGBA (`scripts/record_traces.py`) are
  replayed in the simulator. The simulator must reproduce `gRngValue`, all battle RAM and both
  parties at every decision, byte for byte (`tests/python/test_gen3_exact.py`). This covers streaks
  0–56, every AI level and the Factory Head.
- **Factory generation:** rentals, opponents and trainers match the game given the same RNG state.
  The only thing not modelled is how many frames the player spends walking between rooms, since the
  overworld RNG advances every frame.

Speed: ~2,400 battles/s single-threaded (bare battles), ~1,000 Factory battles/s through `SimBackend`.

## Build

```bash
pip install numpy gymnasium pyelftools pytest pillow
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python -m pytest tests/python          # exactness + backend tests
```

The build fetches pybind11 and a minimal static libmgba 0.10.2 (headless emulator, no system
dependencies). The Python modules land in `pybattle/`. The gen3 core builds on Linux only for now.

Regenerating the vendored game code requires a *built* pokeemerald checkout with
`pokeemerald.elf` and `pokeemerald.map`:

```bash
scripts/regen_gen3.sh ~/Dev/pokeemerald   # vendor sources, extract ROM scripts, regenerate stubs
python scripts/extract_game_data.py       # data tables + RAM symbol map (pybattle/data, pybattle/emu)
```

## Use

```python
from pybattle.backend import SimBackend, Phase

b = SimBackend()
b.reset(seed=1, win_streak=0)
while b.phase != Phase.RUN_OVER:
    view = b.view()          # RentalView | SwapView | BattleView — only what the player can see
    b.act(...)               # see the Phase table above
```

- Views (`pybattle/view.py`) show the player's own Pokémon in full and the opponent the way the
  game does: species, level, HP bar in pixels (0–48), status, stat changes, and the moves and item
  revealed so far. The swap screen shows only the defeated team's species. `BattleView.usable_moves`
  and `switch_targets` are the game's own legality checks.
- `b.wins`, `b.run_info()` and `b.last_battle_won` report progress; `SimBackend.clone()` copies a
  whole game state (e.g. for search).
- Gen 3 battles have no turn limit. `SimBackend(max_turns=500)` forfeits a battle after that many
  player turns so a stalled battle cannot run forever.

## The real game

`EmuBackend` has the same interface on the unmodified ROM and your save. The save must be standing
in the Battle Factory lobby, in front of the singles attendant. Views are built by the same code
from the same RAM addresses as in the simulator.

```python
from pybattle.emu_backend import EmuBackend
b = EmuBackend("Emerald.gba", "Emerald.sav")      # headless mGBA, ~1.5 s per battle
```

### Watch agents play (or play yourself)

```bash
python scripts/watch.py --rom Emerald.gba --save Emerald.sav [--speed 2] [--step] [--human]
```

This loads your save, walks the player to the singles attendant and plays runs one after another
until you close the window. The window shows the game, what the agent sees at each decision, its
choice with a reason, and a decision log (also written to `watch_log.jsonl`).

The window can be resized, maximized or made fullscreen, and everything scales with it. It has a header
(round, battle n/7, streak, wins, rentals) and the game screen with a decision log. Next to them are
panels showing what the agent sees (team cards with HP bars, moves in type colours and damage
estimates; the opponent as far as it has been seen) and its decision (P(win) or expected wins, and
every option as a probability bar). The controls are buttons along the bottom: **Pause/Resume**,
**Next**, **Step on/off**, speed **−** / **+**, **Human on/off**, text size **A−/A+**, **Light/Dark**,
**Fullscreen** and **Quit**. In human mode the keyboard plays the game (arrows, Z = A, X = B,
Enter = Start, Backspace = Select, A/S = L/R) and the panels show what the agents would do. See
[docs/GUIDE.md §5](docs/GUIDE.md).

The default agents are the random dummies in `examples/random_agents.py`. Plug in your own with
`--agents mymodule:MyTactician,mymodule:MyBattler`: classes with
`act(phase, view, run_info) -> pybattle.agent.Choice`. The interface is in `pybattle/agent.py`.

To watch it live in mgba-qt:
1. Open the ROM and save.
2. Go to *Tools → Scripting → Load* and load `emu/mgba/bridge.lua`.
3. Use `EmuBackend(machine=LuaMachine())` (`from pybattle.emu import LuaMachine`).

## Layout

| Path | What |
|---|---|
| `third_party/pokeemerald/` | vendored decomp headers and sources (generated; do not edit) |
| `src/gen3/` | host layer (`host.c`), Factory run (`factory_run.c`), C++ wrapper (`game.cpp`), generated ROM scripts/stubs |
| `pybattle/backend.py`, `emu_backend.py` | `SimBackend` / `EmuBackend` |
| `pybattle/view.py` | observations (information set shown to the player) |
| `pybattle/emu/` | headless mGBA driver, RAM decoders, trace recorder, Lua bridge client |
| `pybattle/diff.py` | trace replayer (exactness checks) |
| `pybattle/agent.py`, `examples/random_agents.py` | agent interface, random dummy tactician/battler |
| `pybattle/emu/viewer.py`, `pybattle/emu/ui.py`, `scripts/watch.py` | live window: watch agents / play yourself |
| `scripts/` | data extraction, vendoring, stub generation, trace recording |
| `src/battle_engine.cpp`, `pybattle/pkmn_env.py`, `factory_hrl_env.py` | legacy hand-written engine and envs (superseded) |
