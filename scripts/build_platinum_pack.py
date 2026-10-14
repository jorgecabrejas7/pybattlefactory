#!/usr/bin/env python3
"""The Platinum game pack for the web app (battle-arcade, games/platinum-frontier): the DS emulator built for the
browser, the game and a save with the Battle Frontier open. No AI yet (the singles and doubles models are in
training); the pack grows models/ and data/ like the Emerald one (scripts/build_pack.py) when they exist.

    dist/packs/platinum-frontier/<version>/
        manifest.json
        engine/nds_web.{mjs,wasm}      DeSmuME (emu/web/nds_web.cpp, scripts/build_wasm.sh nds_web)
        game/platinum.nds              the pokeplatinum build (US Rev 1); the web repo keeps it out of the public site
        game/platinum.sav              a save at the Battle Frontier
    dist/packs/platinum-frontier-<version>.tar.gz (+ .sha256)

    build_platinum_pack.py [--version 0.1.0] [--skip-build] [--publish contabo:~/packs/]
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
PACK_ID = "platinum-frontier"
ROM_DIR = os.path.expanduser("~/Desktop/Dev/pokemon-battle-factory/roms")
ROM_NAME = "Pokemon - Platinum Version (USA) (Rev 1)"
ROM_SHA1 = "0862ec35b24de5c7e2dcb88c9eea0873110d755c"      # pret/pokeplatinum, US Rev 1


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
    ap.add_argument("--skip-build", action="store_true", help="reuse build-wasm/nds_web.*")
    ap.add_argument("--rom", default=os.path.join(ROM_DIR, ROM_NAME + ".nds"))
    ap.add_argument("--save", default=os.path.join(ROM_DIR, ROM_NAME + ".sav"))
    ap.add_argument("--publish", default=None, help="rsync destination for the tarball (e.g. contabo:~/packs/)")
    args = ap.parse_args()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip())

    if not args.skip_build:
        run("bash", "scripts/build_wasm.sh", "nds_web")
    if hashlib.sha1(open(args.rom, "rb").read()).hexdigest() != ROM_SHA1:
        raise SystemExit(f"{args.rom}: not Pokémon Platinum (USA, Rev 1) as pret/pokeplatinum builds it")

    out = os.path.join(DIST, "packs", PACK_ID, "_staging")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "engine"))
    os.makedirs(os.path.join(out, "game"))
    for f in ("nds_web.mjs", "nds_web.wasm"):
        shutil.copy(os.path.join(REPO, "build-wasm", f), os.path.join(out, "engine", f))
    shutil.copy(args.rom, os.path.join(out, "game", "platinum.nds"))
    shutil.copy(args.save, os.path.join(out, "game", "platinum.sav"))

    # the version names the content (packs are cached as immutable): the commit, plus a content hash when dirty
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(out)):
        for f in sorted(files):
            h.update(os.path.relpath(os.path.join(root, f), out).encode())
            h.update(open(os.path.join(root, f), "rb").read())
    version = f"{args.version}+{git}" + (f".dirty.{h.hexdigest()[:8]}" if dirty else "")
    final = os.path.join(DIST, "packs", PACK_ID, version)
    shutil.rmtree(final, ignore_errors=True)
    os.rename(out, final)
    out = final

    manifest = {
        "schema": 1,
        "id": PACK_ID,
        "version": version,
        "title": "Platinum Battle Factory",
        "emulator": {"js": "engine/nds_web.mjs", "wasm": "engine/nds_web.wasm",
                     "rom": "game/platinum.nds", "rom_sha256": sha256(os.path.join(out, "game", "platinum.nds")),
                     "save": "game/platinum.sav", "save_sha256": sha256(os.path.join(out, "game", "platinum.sav"))},
        "modes": ["singles", "doubles"],
        "models": {},
        "agents": [],
        "build": {"git": git, "dirty": dirty, "built": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "license": "private: ROM and Nintendo assets; do not redistribute. Emulator: DeSmuME (GPL-2.0).",
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
