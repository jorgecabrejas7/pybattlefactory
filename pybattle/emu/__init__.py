"""Emulator side: headless mGBA host, RAM symbols and decoders."""

from .. import pybattle_emu as native
from .lua_machine import LuaMachine

Emulator = native.Emulator
