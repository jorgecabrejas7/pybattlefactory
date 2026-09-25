#!/bin/bash
# Regenerate everything the gen3 battle core is built from:
#   third_party/pokeemerald (vendored decomp), src/gen3/generated/* (ROM scripts, RAM map,
#   layout check, stubs). Needs a built pokeemerald checkout (pokeemerald.elf + .map).
# Usage: scripts/regen_gen3.sh [path/to/pokeemerald]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PE="${1:-$HOME/Dev/pokeemerald}"
python3 "$REPO/scripts/sync_decomp.py" --pokeemerald "$PE"
python3 "$REPO/scripts/extract_rom_scripts.py" --pokeemerald "$PE"

# Stubs: compile everything, collect unresolved symbols, generate definitions for them.
WORK="$(mktemp -d "$REPO/.scratch/regen.XXXX" 2>/dev/null || mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
CF=(-std=gnu17 -O0 -w -DGEN3_HOST -funsigned-char -fPIC
    -iquote "$REPO/third_party/pokeemerald/include" -iquote "$REPO/third_party/pokeemerald/src" -iquote "$REPO/src")
: > "$REPO/src/gen3/generated/stubs.c"
for f in "$REPO"/third_party/pokeemerald/src/*.c "$REPO/src/gen3/host.c" "$REPO/src/gen3/factory_run.c" "$REPO/src/gen3/generated/rom_script_tables.c"; do
    gcc "${CF[@]}" -c "$f" -o "$WORK/$(basename "$f" .c).o"
done
(cd "$REPO/src/gen3/generated" && gcc -c rom_scripts.S -o "$WORK/rom_scripts.o" && gcc -c rom_ram_map.S -o "$WORK/rom_ram_map.o")
ld -r "$WORK"/*.o -o "$WORK/all.o"
nm -u "$WORK/all.o" | awk '{print $2}' | sort > "$WORK/undefined.txt"
python3 "$REPO/scripts/gen_stubs.py" "$WORK/undefined.txt"
gcc "${CF[@]}" -c "$REPO/src/gen3/generated/stubs.c" -o "$WORK/stubs.o"
gcc "${CF[@]}" -c "$REPO/src/gen3/generated/rom_layout_check.c" -o /dev/null
rm "$WORK/all.o"; gcc -shared -o "$WORK/check.so" "$WORK"/*.o -Wl,--no-undefined
echo "gen3 regenerated and links cleanly"
