// Plays Factory runs through the wasm C API (build-wasm/gen3_web.mjs) with a fixed pseudo-random policy and prints
// one JSON line per decision (view, run info, log, estimates, encoding). tests/python/test_web_wasm.py runs the
// same policy on the native FactorySession and compares the transcripts.
//   node tests/web/transcript.mjs <runs> <seed>
import createGen3 from "../../build-wasm/gen3_web.mjs";

const [runs, seed0] = [Number(process.argv[2] ?? 3), Number(process.argv[3] ?? 1)];
const M = await createGen3();
const str = (p) => (p ? M.UTF8ToString(p) : null);
const call = (name, ...args) => M[`_${name}`](...args);

let rngState = 0n;
const rnd = (n) => {   // same LCG as the Python side
  rngState = (rngState * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
  return Number((rngState >> 33n) % BigInt(n));
};

const h = call("fs_new", 1, 150);
const layout = JSON.parse(str(call("fs_encoded_layout", h)));
const encoded = () => {
  const p = call("fs_encode", h);
  if (!p) return null;
  const out = {};
  for (const [k, v] of Object.entries(layout)) {
    if (k === "size") continue;
    const n = v.shape.reduce((a, b) => a * b, 1);
    const at = p + v.offset;
    const arr = v.type === "float32" ? M.HEAPF32.subarray(at >> 2, (at >> 2) + n)
      : v.type === "int32" ? M.HEAP32.subarray(at >> 2, (at >> 2) + n) : M.HEAPU8.subarray(at, at + n);
    out[k] = v.shape.length ? Array.from(arr) : arr[0];
  }
  return out;
};

for (let r = 0; r < runs; r++) {
  rngState = BigInt(seed0 * 7919 + r);
  const streak = (r * 9) % 49;
  if (call("fs_reset", h, seed0 * 1000003 + r, streak, 0) !== 0) throw new Error(str(call("fs_last_error")));
  for (let d = 0; d < 3000; d++) {
    const phase = call("fs_phase", h);
    const view = JSON.parse(str(call("fs_view_json", h)));
    const line = { r, d, phase, view, log: JSON.parse(str(call("fs_take_log_json", h))) };
    if (phase === 4) { console.log(JSON.stringify(line)); break; }
    line.info = JSON.parse(str(call("fs_run_info_json", h)));
    if (phase === 2 || phase === 3) {
      line.est = JSON.parse(str(call("fs_estimates_json", h)));
      line.enc = encoded();
    }
    console.log(JSON.stringify(line));
    let rc;
    if (phase === 0) {
      for (;;) {
        const a = rnd(6), b = rnd(6), c = rnd(6);
        if ((rc = call("fs_act", h, 0, a, b, c)) === 0) break;
      }
    } else if (phase === 1) {
      const keep = rnd(2), own = rnd(3), foe = rnd(3);
      if (keep === 0 || call("fs_act", h, 1, own, foe, 0) !== 0) call("fs_act", h, 1, -1, 0, 0);
    } else {
      const legal = view.legal_actions.filter((a) => a[0] !== "forfeit");
      const a = legal[rnd(legal.length)];
      rc = call("fs_act", h, a[0] === "move" ? 2 : 3, a[1], 0, 0);
      if (rc !== 0) throw new Error(str(call("fs_last_error")));
    }
  }
}
