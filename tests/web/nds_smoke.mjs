// Smoke test of the DS emulator for the browser (build-wasm/nds_web.mjs, emu/web/nds_web.cpp) in node:
//   node tests/web/nds_smoke.mjs <rom.nds> [save.sav] [outdir]
// Boots the ROM (fresh, or with the raw save), runs frames pressing A/START, reports fps, checks the picture,
// writes PNGs of both screens, checks that a savestate replays deterministically, reads main RAM and the save.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { deflateSync, crc32 } from "node:zlib";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const [romPath, savePath, outDirArg] = process.argv.slice(2);
if (!romPath) {
  console.error("usage: node tests/web/nds_smoke.mjs <rom.nds> [save.sav|-] [outdir]");
  process.exit(2);
}
const save = savePath && savePath !== "-" ? readFileSync(savePath) : null;
const outDir = outDirArg ?? path.resolve(here, "../../build-wasm/nds_smoke");
mkdirSync(outDir, { recursive: true });

const KEY = { A: 1, B: 2, SELECT: 4, START: 8, RIGHT: 16, LEFT: 32, UP: 64, DOWN: 128, R: 256, L: 512, X: 1024, Y: 2048 };
let failures = 0;
const check = (ok, what) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${what}`);
  if (!ok) failures++;
};

// ---- PNG (RGBA, 8-bit) ----
function png(file, rgba, w, h) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 4 + 1)] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + y * w * 4, w * 4).copy(raw, y * (w * 4 + 1) + 1);
  }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, "ascii"), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(td) >>> 0);
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  writeFileSync(file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]));
}

// ---- module ----
const createNds = (await import(process.env.NDS_WEB_MJS ? path.resolve(process.env.NDS_WEB_MJS) : path.resolve(here, "../../build-wasm/nds_web.mjs"))).default;
const M = await createNds();
const W = 256, H = 384;
const video = () => M.HEAPU8.slice(M._nds_video(), M._nds_video() + W * H * 4);
const sha = (b) => createHash("sha1").update(b).digest("hex").slice(0, 12);
const ramHash = () => {
  const p = M._malloc(0x400000);
  M._nds_read(0x02000000, p, 0x400000);
  const h = sha(M.HEAPU8.subarray(p, p + 0x400000));
  M._free(p);
  return h;
};

// Init failure path first: a non-ROM must be rejected (and must not leave a half-built emulator usable... it does,
// so this runs in its own module instance).
{
  const M2 = await createNds();
  const p = M2._malloc(4096);
  M2.HEAPU8.fill(0, p, p + 4096);
  check(M2._nds_init(p, 4096, 0, 0) !== 0, "init rejects a zero-filled 4 KiB 'ROM'");
}

const rom = readFileSync(romPath);
const romPtr = M._nds_rom_alloc(rom.length);              // zero-copy upload
M.HEAPU8.set(rom, romPtr);
let savePtr = 0;
if (save) {
  savePtr = M._malloc(save.length);
  M.HEAPU8.set(save, savePtr);
}
let t0 = performance.now();
const rc = M._nds_init(romPtr, rom.length, savePtr, save ? save.length : 0);
check(rc === 0, `nds_init (${save ? "with " + save.length + "-byte save" : "fresh, no save"}) -> ${rc} in ${(performance.now() - t0).toFixed(0)} ms`);
if (rc !== 0) process.exit(1);
if (savePtr) M._free(savePtr);
M._nds_set_rtc(Date.UTC(2026, 0, 1, 12, 0, 0));   // fixed clock -> deterministic runs

// Input script. Fresh: A pulses (START pulses early on) through the intro until frame 1500 (Prof. Rowan's speech).
// With a save: A through the logos, title and main menu (CONTINUE) until frame 1000, then B pulses to close the
// journal until 1700; the game is then standing in the overworld.
const keysAt = save
  ? (f) => (f % 40 >= 4 ? 0 : f < 1000 ? KEY.A : f < 1700 ? KEY.B : 0)
  : (f) => (f >= 1500 ? 0 : f % 40 < 4 ? KEY.A : f % 120 >= 60 && f % 120 < 64 && f < 900 ? KEY.START : 0);

const FRAMES = save ? 2500 : 1500;
const LAST = 500;   // the last frames are timed separately (overworld with a save)
let audioFrames = 0, tLast = 0, minAudio = 1e9, maxAudio = 0;
t0 = performance.now();
for (let f = 0; f < FRAMES; f++) {
  if (f === FRAMES - LAST) tLast = performance.now();
  M._nds_set_keys(keysAt(f));
  M._nds_run_frame(1);
  const a = M._nds_audio_read();
  audioFrames += a;
  minAudio = Math.min(minAudio, a);
  maxAudio = Math.max(maxAudio, a);
}
const tEnd = performance.now();
const dt = (tEnd - t0) / 1000;
console.log(`ran ${FRAMES} frames (drawing) in ${dt.toFixed(2)} s = ${(FRAMES / dt).toFixed(1)} fps; nds_frame()=${M._nds_frame()}`);
console.log(`last ${LAST} frames (${save ? "overworld" : "intro speech"}): ${(LAST / ((tEnd - tLast) / 1000)).toFixed(1)} fps`);
console.log(`audio: ${audioFrames} stereo frames = ${(audioFrames / FRAMES).toFixed(1)}/frame [${minAudio}..${maxAudio}] (48 kHz at 59.83 fps = 802.3)`);
check(M._nds_frame() === FRAMES, "frame counter");
check(Math.abs(audioFrames / FRAMES - 802.3) < 2, "audio rate ~48 kHz");
{
  const p = M._nds_audio_ptr() >> 1;
  let peak = 0;
  for (let i = 0; i < 2 * minAudio; i++) peak = Math.max(peak, Math.abs(M.HEAP16[p + i]));
  console.log(`audio peak in the last frame's samples: ${peak}`);
}

console.log(`wasm heap: ${(M.HEAPU8.length / 2 ** 20).toFixed(0)} MiB; save written by the game during the run: ${M._nds_save_changed()}`);

// Picture
const v = video();
const colours = new Set();
for (let i = 0; i < v.length; i += 4 * 97) colours.add(v[i] | (v[i + 1] << 8) | (v[i + 2] << 16));
let alphaOk = true;
for (let i = 3; i < v.length; i += 4) if (v[i] !== 255) { alphaOk = false; break; }
check(colours.size > 4, `picture is not flat (${colours.size} sampled colours)`);
check(alphaOk, "alpha is 255");
const tag = save ? "continue" : "fresh";
png(path.join(outDir, `nds_${tag}_${FRAMES}.png`), v, W, H);
console.log(`wrote ${path.join(outDir, `nds_${tag}_${FRAMES}.png`)}`);

// Main RAM
const b = [];
for (let i = 0; i < 16; i++) b.push(M._nds_read8(0x02000000 + i).toString(16).padStart(2, "0"));
console.log(`main RAM 0x02000000: ${b.join(" ")}  read32=0x${(M._nds_read32(0x02000000) >>> 0).toString(16)}`);
check((M._nds_read32(0x02000000) >>> 0) === ((M._nds_read16(0x02000000) | (M._nds_read16(0x02000002) << 16)) >>> 0),
  "read32 == two read16");
{
  const p = M._malloc(4);
  M.HEAPU8.set([0xde, 0xad, 0xbe, 0xef], p);
  const addr = 0x023ff000;   // keep the old bytes, write, read back, restore
  const old = M._malloc(4);
  M._nds_read(addr, old, 4);
  M._nds_write(addr, p, 4);
  check((M._nds_read32(addr) >>> 0) === 0xefbeadde, "nds_write/nds_read32 roundtrip");
  M._nds_write(addr, old, 4);
  M._free(p);
  M._free(old);
}

// Speed without drawing (frame skip), same idle input
t0 = performance.now();
for (let f = 0; f < 300; f++) { M._nds_set_keys(0); M._nds_run_frame(0); M._nds_audio_read(); }
const dt0 = (performance.now() - t0) / 1000;
console.log(`300 frames without drawing: ${(300 / dt0).toFixed(1)} fps`);
M._nds_run_frame(1);

// Savestate determinism: save, run N frames with scripted keys, hash; load, rerun, hash.
t0 = performance.now();
const n = M._nds_state_save();
const stateMs = performance.now() - t0;
check(n > 0, `nds_state_save -> ${n} bytes (${stateMs.toFixed(1)} ms)`);
const state = M.HEAPU8.slice(M._nds_state_ptr(), M._nds_state_ptr() + n);
const replay = () => {
  for (let f = 0; f < 180; f++) {
    M._nds_set_keys(f % 30 < 3 ? KEY.A : f % 50 < 5 ? KEY.DOWN : 0);
    if (f >= 60 && f < 70) M._nds_touch(1, 128, 96); else M._nds_touch(0, 0, 0);
    M._nds_run_frame(1);
    M._nds_audio_read();
  }
  return { video: sha(video()), ram: ramHash(), frame: M._nds_frame() };
};
const r1 = replay();
const sp = M._malloc(n);
M.HEAPU8.set(state, sp);
t0 = performance.now();
const lrc = M._nds_state_load(sp, n);
const loadMs = performance.now() - t0;
M._free(sp);
check(lrc === 0, `nds_state_load -> ${lrc} (${loadMs.toFixed(1)} ms)`);
const r2 = replay();
console.log(`replay 1: ${JSON.stringify(r1)}\nreplay 2: ${JSON.stringify(r2)}`);
check(r1.video === r2.video && r1.ram === r2.ram && r1.frame === r2.frame, "savestate replay is deterministic (video + 4 MiB RAM)");

// Cartridge save
const ss = M._nds_save_size();
const changed = M._nds_save_changed();
console.log(`cartridge save: ${ss} bytes, changed since init=${changed}, sha1 ${sha(M.HEAPU8.subarray(M._nds_save_ptr(), M._nds_save_ptr() + ss))}`);
if (save) {
  check(ss === save.length, "save size equals the given save");
  const now = M.HEAPU8.subarray(M._nds_save_ptr(), M._nds_save_ptr() + ss);
  console.log(`save bytes identical to the input: ${Buffer.compare(Buffer.from(now), save) === 0}`);
  // restore roundtrip
  const p = M._malloc(save.length);
  M.HEAPU8.set(save, p);
  check(M._nds_save_restore(p, save.length) === 0, "nds_save_restore");
  M._free(p);
  const back = M.HEAPU8.subarray(M._nds_save_ptr(), M._nds_save_ptr() + M._nds_save_size());
  check(Buffer.compare(Buffer.from(back), save) === 0, "save after restore == input");
}
// Reset (the cartridge save is kept)
M._nds_reset();
for (let f = 0; f < 120; f++) { M._nds_set_keys(0); M._nds_run_frame(1); M._nds_audio_read(); }
check(M._nds_frame() === 120, `nds_reset restarts the console (frame ${M._nds_frame()})`);
check(M._nds_save_size() === (save ? save.length : M._nds_save_size()), "save survives reset");

console.log(failures ? `${failures} FAILED` : "all ok");
process.exit(failures ? 1 : 0);
