"""Talk to a live mgba-qt through emu/mgba/bridge.lua, with the same interface as the headless
emulator (pybattle.emu.Emulator), so FactoryDriver and EmuBackend can run on either.

The game runs at the speed mgba-qt runs (use its fast-forward to speed things up).
"""

import socket
import struct


class LuaMachine:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.sock = socket.create_connection((host, port))
        self.file = self.sock.makefile("rb")

    def _cmd(self, line: str) -> str:
        self.sock.sendall((line + "\n").encode())
        reply = self.file.readline().decode().strip()
        if reply.startswith("ERR"):
            raise RuntimeError(reply)
        return reply

    # --- the pybattle.emu.Emulator interface ------------------------------------------------

    def read(self, addr: int, size: int) -> bytes:
        return bytes.fromhex(self._cmd(f"R {addr} {size}"))

    def read8(self, addr):
        return self.read(addr, 1)[0]

    def read16(self, addr):
        return struct.unpack("<H", self.read(addr, 2))[0]

    def read32(self, addr):
        return struct.unpack("<I", self.read(addr, 4))[0]

    def write(self, addr: int, data: bytes):
        self._cmd(f"W {addr} {bytes(data).hex()}")

    def write32(self, addr: int, value: int):
        self.write(addr, struct.pack("<I", value))

    def set_keys(self, mask: int):
        self._cmd(f"K {mask}")

    def run_frames(self, n: int = 1):
        self._cmd(f"F {n}")

    @property
    def frame(self) -> int:
        return int(self._cmd("N"))

    def save_state(self) -> bytes:
        return bytes.fromhex(self._cmd("S"))

    def load_state(self, state: bytes):
        self._cmd(f"L {state.hex()}")

    def screenshot(self):
        raise NotImplementedError("watch the mgba-qt window instead")
