#!/usr/bin/env python3
"""Assemble the browser "game pack" (what the web app loads to play the Factory and run the AI client-side):

    dist/packs/emerald-factory/<version>/
        manifest.json          what is in the pack (see below)
        engine/gen3_web.mjs    the game (emscripten build of src/web/web_api.cpp: scripts/build_wasm.sh gen3_web)
        engine/gen3_web.wasm
        models/*.onnx          the networks (scripts/export_onnx.py), models.json
        data/game.json         names and tables (scripts/export_web_data.py), data/planner.json (tactician tables)
        fixtures/              parity fixtures for the web repo's tests (scripts/web_fixtures.py)
        sprites/{front,back,icon}/<species>.png   (scripts/build_sprites.py)
        engine/gba_web.{mjs,wasm}  mGBA (emu/web/gba_web.cpp: scripts/build_wasm.sh gba_web)
        game/emerald.gba, game/emerald.sav, game/symbols.json   the only ROM and save the app runs (pinned)
        battle/                the game's battle screen graphics and font (scripts/build_battle_assets.py)
    dist/packs/emerald-factory-<version>.tar.gz (+ .sha256)

The pack holds ROM-derived bytes (the battle scripts inside the wasm) and Nintendo sprites: keep it private.

    build_pack.py [--version 0.1.0] [--skip-build] [--publish contabo:~/packs/]
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(REPO, "dist")
PACK_ID = "emerald-factory"
ROM_DIR = os.path.expanduser("~/Desktop/Dev/pokemon-battle-factory/roms")
ROM_NAME = "Pokemon - Emerald Version (USA, Europe)"
# the only ROM / save the web app runs (pinned: the app refuses anything else)
ROM_SHA1 = None     # filled from the file; pokeemerald's build hash is checked below
ABI = 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--skip-build", action="store_true", help="reuse build-wasm/ and dist/{models,data,sprites}")
    ap.add_argument("--ckpt", default="runs/alphazero_v3/snapshot.pt")
    ap.add_argument("--rom", default=os.path.join(ROM_DIR, ROM_NAME + ".gba"))
    ap.add_argument("--save", default=os.path.join(ROM_DIR, ROM_NAME + ".sav"))
    ap.add_argument("--publish", default=None, help="rsync destination for the tarball (e.g. contabo:~/packs/)")
    args = ap.parse_args()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip())
    version = f"{args.version}+{git}{'.dirty' if dirty else ''}"

    if not args.skip_build:
        run("bash", "scripts/build_wasm.sh", "gen3_web")
        run("bash", "scripts/build_wasm.sh", "gba_web")
        run("python3", "scripts/export_onnx.py", "--ckpt", args.ckpt, "--out", os.path.join(DIST, "models"))
        run("python3", "scripts/export_web_data.py", "--out", os.path.join(DIST, "data", "game.json"))
        run("python3", "scripts/export_opponent_model.py", "--out", os.path.join(DIST, "data", "opponent_model.json"))
        run("python3", "scripts/build_sprites.py", "--out", os.path.join(DIST, "sprites"))
        run("python3", "scripts/build_battle_assets.py", "--out", os.path.join(DIST, "battle"))
        run("python3", "scripts/web_fixtures.py", "--ckpt", args.ckpt, "--out", os.path.join(DIST, "fixtures", "tactician.json"))

    out = os.path.join(DIST, "packs", PACK_ID, version)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "engine"))
    for f in ("gen3_web.mjs", "gen3_web.wasm", "gba_web.mjs", "gba_web.wasm"):
        shutil.copy(os.path.join(REPO, "build-wasm", f), os.path.join(out, "engine", f))
    shutil.copytree(os.path.join(DIST, "models"), os.path.join(out, "models"))
    shutil.copytree(os.path.join(DIST, "data"), os.path.join(out, "data"))
    shutil.copytree(os.path.join(DIST, "sprites"), os.path.join(out, "sprites"))
    shutil.copytree(os.path.join(DIST, "battle"), os.path.join(out, "battle"))
    if os.path.isdir(os.path.join(DIST, "fixtures")):          # parity fixtures for the web repo's tests
        shutil.copytree(os.path.join(DIST, "fixtures"), os.path.join(out, "fixtures"))

    # the game itself: the ROM (must be the pokeemerald build this project's symbols come from) and the save
    os.makedirs(os.path.join(out, "game"))
    import hashlib as _h
    rom_sha1 = _h.sha1(open(args.rom, "rb").read()).hexdigest()
    elf_rom = os.path.expanduser("~/Dev/pokeemerald/pokeemerald.gba")
    if os.path.exists(elf_rom) and _h.sha1(open(elf_rom, "rb").read()).hexdigest() != rom_sha1:
        raise SystemExit("the ROM is not the pokeemerald build the symbols come from")
    shutil.copy(args.rom, os.path.join(out, "game", "emerald.gba"))
    shutil.copy(args.save, os.path.join(out, "game", "emerald.sav"))
    # the game's RAM / function addresses (the AI reads the emulator by symbol, like pybattle/emu/driver.py)
    shutil.copy(os.path.join(REPO, "pybattle", "emu", "symbols.json"), os.path.join(out, "game", "symbols.json"))

    if dirty:
        # a dirty tree has no commit naming its content: the version names it instead, so a browser never mixes the
        # files of two builds under the same URLs (packs are cached as immutable)
        h = hashlib.sha256()
        for root, _, files in sorted(os.walk(out)):
            for f in sorted(files):
                h.update(os.path.relpath(os.path.join(root, f), out).encode())
                h.update(open(os.path.join(root, f), "rb").read())
        version = f"{version}.{h.hexdigest()[:8]}"
        final = os.path.join(DIST, "packs", PACK_ID, version)
        shutil.rmtree(final, ignore_errors=True)
        os.rename(out, final)
        out = final
    models = json.load(open(os.path.join(out, "models", "models.json")))
    manifest = {
        "schema": 1,
        "id": PACK_ID,
        "version": version,
        "title": "Emerald Battle Factory",
        "engine": {"js": "engine/gen3_web.mjs", "wasm": "engine/gen3_web.wasm", "abi": ABI,
                   "sha256": sha256(os.path.join(out, "engine", "gen3_web.wasm"))},
        "emulator": {"js": "engine/gba_web.mjs", "wasm": "engine/gba_web.wasm",
                     "rom": "game/emerald.gba", "rom_sha256": sha256(os.path.join(out, "game", "emerald.gba")),
                     "save": "game/emerald.sav", "save_sha256": sha256(os.path.join(out, "game", "emerald.sav")),
                     "symbols": "game/symbols.json"},
        "models": {k: dict(v, file=f"models/{v['file']}", sha256=sha256(os.path.join(out, "models", v["file"])))
                   for k, v in models["models"].items()},
        "agents": [{"id": "v3-net", "title": "AlphaZero v3 (network)", "battler": "net", "tactician": "net+planner",
                    "encode_version": models["encode_version"], "opt_feat": models["opt_feat"],
                    "v_next": models.get("v_next"), "planner": {"depth": 1, "n_opp": 8}},
                   {"id": "v3-search", "title": "AlphaZero v3 (search)", "battler": "search", "tactician": "net+planner",
                    "search": {"prior": "experience", "mode": "legal", "c_puct": 1.5, "batch": 32, "redraws": 4,
                               "budgets": {"fast": [64, 4], "normal": [128, 4], "strong": [512, 8]}}}],
        "data": {"game": "data/game.json", "planner": "data/planner.json", "opponent_model": "data/opponent_model.json"},
        "sprites": {"front": "sprites/front/{id}.png", "back": "sprites/back/{id}.png", "icon": "sprites/icon/{id}.png",
                    "front_frames": 2, "icon_frames": 2, "size": 64},
        "build": {"git": git, "dirty": dirty, "ckpt": models["ckpt"], "built": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "license": "private: ROM-derived data and Nintendo assets; do not redistribute",
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    tar = os.path.join(DIST, "packs", f"{PACK_ID}-{version}.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname=f"{PACK_ID}/{version}")
    digest = sha256(tar)
    with open(tar + ".sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(tar)}\n")
    print(f"{tar}: {os.path.getsize(tar) / 1e6:.1f} MB sha256 {digest}")
    if args.publish:
        run("rsync", "-a", tar, tar + ".sha256", args.publish)


if __name__ == "__main__":
    main()
