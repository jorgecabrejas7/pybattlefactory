#!/bin/bash
# Build the gen3 core for the browser (emscripten, in Docker; no local emsdk needed).
#   scripts/build_wasm.sh [target ...]      (default: all web targets) -> build-wasm/
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EMSDK_IMAGE="${EMSDK_IMAGE:-emscripten/emsdk:4.0.10}"
TARGETS=("${@:-all}")
run() { docker run --rm -v "$REPO:/src" -w /src -u "$(id -u):$(id -g)" -e EM_CACHE=/src/build-wasm/.emcache "$EMSDK_IMAGE" "$@"; }
run emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release >/dev/null
for t in "${TARGETS[@]}"; do run cmake --build build-wasm -j"$(nproc)" --target "$t"; done
