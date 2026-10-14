// Runs the ROM + save in the wasm mGBA for N frames (with a key script) and prints a hash of the frame and RAM.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import createGba from "../../build-wasm/gba_web.mjs";
const [rom, sav, frames] = [process.argv[2], process.argv[3], Number(process.argv[4] ?? 600)];
const M = await createGba();
const put = (b) => { const p = M._malloc(b.length); M.HEAPU8.set(b, p); return p; };
const r = readFileSync(rom), s = readFileSync(sav);
const rc = M._gba_init(put(r), r.length, put(s), s.length);
if (rc) throw new Error("init " + rc);
M._gba_rtc_fake_epoch(1767268800000);
const t0 = performance.now();
let audio = 0;
for (let f = 0; f < frames; f++) {
  M._gba_set_keys(f % 120 < 4 ? 1 : 0);   // tap A every 2 s
  M._gba_run_frames(1);
  audio += M._gba_audio_read();
}
const ms = performance.now() - t0;
const v = M._gba_video();
const img = M.HEAPU8.slice(v, v + 240 * 160 * 4);
const ram = new Uint8Array(0x40000); const p = M._malloc(0x40000); M._gba_read(0x02000000, p, 0x40000);
ram.set(M.HEAPU8.subarray(p, p + 0x40000));
const h = (b) => createHash("sha1").update(b).digest("hex").slice(0, 16);
console.log(JSON.stringify({ frames, fps: Math.round(frames / (ms / 1000)), audio, video: h(img), ram: h(ram) }));
