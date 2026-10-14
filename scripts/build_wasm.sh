#!/bin/bash
# Build the gen3 core for the browser (emscripten, in Docker; no local emsdk needed).
#   scripts/build_wasm.sh [target ...]      (default: all web targets) -> build-wasm/
# Targets: gen3_web, gen3_smoke, gba_web (mGBA), nds_web (DeSmuME; needs a desmume-wasm clone, see below).
# nds_web compiles github.com/44670/desmume-wasm from a local clone, DESMUME_WASM_DIR (default
# /home/apollo/Dev/desmume-wasm), mounted read-only at /desmume inside the container.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EMSDK_IMAGE="${EMSDK_IMAGE:-emscripten/emsdk:4.0.10}"
DESMUME_WASM_DIR="${DESMUME_WASM_DIR:-/home/apollo/Dev/desmume-wasm}"
TARGETS=("${@:-all}")
MOUNTS=(-v "$REPO:/src")
CMAKE_ARGS=()
if [ -d "$DESMUME_WASM_DIR/desmume/src" ]; then
    MOUNTS+=(-v "$DESMUME_WASM_DIR:/desmume:ro")
    CMAKE_ARGS+=(-DDESMUME_WASM_DIR=/desmume)
fi
run() { docker run --rm "${MOUNTS[@]}" -w /src -u "$(id -u):$(id -g)" -e EM_CACHE=/src/build-wasm/.emcache "$EMSDK_IMAGE" "$@"; }
run emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release "${CMAKE_ARGS[@]}" >/dev/null
for t in "${TARGETS[@]}"; do run cmake --build build-wasm -j"$(nproc)" --target "$t"; done
