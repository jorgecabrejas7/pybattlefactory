"""The game's OPTIONS menu, read and written in RAM (SaveBlock2, include/global.h):

    0x13  u8  optionsButtonMode
    0x14  u16 optionsTextSpeed:3 (0 slow, 1 mid, 2 fast) | optionsWindowFrameType:5 | optionsSound:1 |
              optionsBattleStyle:1 (0 shift, 1 set) | optionsBattleSceneOff:1

The game reads these bits live (text printers take the speed when a message starts; the battle code checks
optionsBattleSceneOff before every animation), so writing them right after the save loads is the same as choosing
them in the OPTIONS menu, without the menu. The .sav file on disk is never touched (the emulator keeps save
writes in memory), but the game itself saves them into its in-memory save when the Factory saves before a
challenge, like any option a player changed.

Effect on the game (docs/RL_DECISIONS.md §24): text speed and battle scene only change how many frames things take.
Inside a battle the RNG is advanced only by the battle engine (animations never call Random(); the simulator,
which has no animations at all, reproduces the emulator's gRngValue at every decision), so a battle with given
actions and starting RNG plays out the same with the scene on or off. Outside battles the overworld advances the
RNG every frame, so changing the number of frames changes the next opponents: a recorded run must be replayed with
the options it was played with (play_real stores them in the run log and the replay re-applies them).
"""

TEXT_SPEEDS = {"slow": 0, "mid": 1, "fast": 2}
_OFF_BUTTON = 0x13
_OFF_BITS = 0x14


def _sb2(emu) -> int:
    from .decode import SYMBOLS as S
    return emu.read32(S.addr("gSaveBlock2Ptr"))


def read_options(emu) -> dict:
    sb2 = _sb2(emu)
    v = emu.read16(sb2 + _OFF_BITS)
    speed = v & 7
    return {"text_speed": next((k for k, x in TEXT_SPEEDS.items() if x == speed), speed),
            "window_frame": (v >> 3) & 31, "sound": "stereo" if (v >> 8) & 1 else "mono",
            "battle_style": "set" if (v >> 9) & 1 else "shift", "battle_scene": not ((v >> 10) & 1),
            "button_mode": emu.read8(sb2 + _OFF_BUTTON)}


def write_options(emu, text_speed=None, battle_scene=None, battle_style=None) -> dict:
    """Set the given options (None: leave as is). Returns the options after the write."""
    sb2 = _sb2(emu)
    v = emu.read16(sb2 + _OFF_BITS)
    if text_speed is not None:
        v = (v & ~7) | TEXT_SPEEDS[text_speed]
    if battle_style is not None:
        v = (v & ~(1 << 9)) | ((battle_style == "set") << 9)
    if battle_scene is not None:
        v = (v & ~(1 << 10)) | ((not battle_scene) << 10)
    emu.write(sb2 + _OFF_BITS, int(v & 0xFFFF).to_bytes(2, "little"))
    return read_options(emu)
