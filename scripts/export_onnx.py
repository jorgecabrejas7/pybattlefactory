#!/usr/bin/env python3
"""Export the alphazero networks to ONNX for the browser (onnxruntime-web, fp32).

    battler.onnx   FactoryNet.battler: probs [B,7] (masked softmax), p_win [B] (value denormalized, clipped to
                   [0, 1] like rl/inference.py battler_outputs), logits [B,7]
    rental.onnx    FactoryNet.rental_joint: lead_logp [B,6], pair_logp [B,6,15] (log-softmax per lead row; the
                   greedy rental is argmax lead, then argmax pair of that row), critic [B] (denormalized)
    swap.onnx      FactoryNet.swap: probs [B,10], critic [B] (denormalized)
    teamnet.onnx   TeamNet: W [B,3] = sigmoid(logits): W(normal), W(last battle), W(Noland)

Inputs use int32 ids (cast to int64 inside the graph) and bool masks, so JavaScript needs no BigInt arrays. The
battle / tactician encodings are rl/encode.py's at the checkpoint's version (5 for alphazero_v3/v4).

    export_onnx.py [--ckpt runs/alphazero_v3/snapshot.pt] [--team-eval PATH] [--out dist/models] [--check 300]

--check N compares onnxruntime against torch on N real observations of each kind (SimBackend random play).
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl import encode  # noqa: E402
from rl.model import NEG  # noqa: E402
from rl.policy import Policy  # noqa: E402

OPSET = 17


def _std(norm):
    return max(norm["var"], 1e-4) ** 0.5 if norm else 1.0


class _Base(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    @staticmethod
    def x(mon_ids, mon_num, move_num, ctx_ids, ctx_num, **rest):
        return dict(mon_ids=mon_ids.long(), mon_num=mon_num, move_num=move_num, ctx_ids=ctx_ids.long(),
                    ctx_num=ctx_num, **rest)


class Battler(_Base):
    def __init__(self, net, norm):
        super().__init__(net)
        self.mean, self.std = float(norm["mean"]) if norm else 0.0, _std(norm)

    def forward(self, mon_ids, mon_num, move_num, ctx_ids, ctx_num, mask, active):
        # FactoryNet.battler with the active Pokemon picked by gather (its arange(len(active)) would freeze the
        # batch size in the traced graph); check() compares against FactoryNet.battler itself
        n = self.net
        mons, ctx, moves = n.trunk(self.x(mon_ids, mon_num, move_num, ctx_ids, ctx_num), 0)
        a = active.long()
        am = torch.gather(moves, 1, a[:, None, None, None].expand(-1, 1, 4, moves.shape[-1]))[:, 0]
        ah = torch.gather(mons, 1, a[:, None, None].expand(-1, 1, mons.shape[-1]))[:, 0]
        mv = n.b_move(torch.cat([am, ah[:, None].expand(-1, 4, -1), ctx[:, None].expand(-1, 4, -1)], -1))[..., 0]
        sw = n.b_switch(torch.cat([mons[:, :3], ctx[:, None].expand(-1, 3, -1)], -1))[..., 0]
        logits = torch.cat([mv, sw], 1).masked_fill(~mask, NEG)
        v = n.b_value(n._pool(mons, ctx))[:, 0]
        return torch.softmax(logits, -1), (v * self.std + self.mean).clamp(0.0, 1.0), logits

    def reference(self, mon_ids, mon_num, move_num, ctx_ids, ctx_num, mask, active):
        logits, v = self.net.battler(self.x(mon_ids, mon_num, move_num, ctx_ids, ctx_num, mask=mask,
                                            active=active.long()))
        return torch.softmax(logits, -1), (v * self.std + self.mean).clamp(0.0, 1.0), logits


class Rental(_Base):
    def __init__(self, net, norm):
        super().__init__(net)
        self.mean, self.std = float(norm["mean"]) if norm else 0.0, _std(norm)

    def forward(self, mon_ids, mon_num, move_num, ctx_ids, ctx_num, lead_mask, pair_mask, opt_feat):
        lead, pair, v = self.net.rental_joint(self.x(mon_ids, mon_num, move_num, ctx_ids, ctx_num, lead_mask=lead_mask,
                                                     pair_mask=pair_mask, opt_feat=opt_feat))
        return torch.log_softmax(lead, -1), torch.log_softmax(pair, -1), v * self.std + self.mean


class Swap(_Base):
    def __init__(self, net, norm):
        super().__init__(net)
        self.mean, self.std = float(norm["mean"]) if norm else 0.0, _std(norm)

    def forward(self, mon_ids, mon_num, move_num, ctx_ids, ctx_num, mask, opt_feat):
        logits, v = self.net.swap(self.x(mon_ids, mon_num, move_num, ctx_ids, ctx_num, mask=mask, opt_feat=opt_feat))
        return torch.softmax(logits, -1), v * self.std + self.mean


class Team(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, ids, num, rnd, hint_type, hint_style):
        if self.net.hint:
            return torch.sigmoid(self.net(ids.long(), num, rnd.long(), hint_type.long(), hint_style.long()))
        return torch.sigmoid(self.net(ids.long(), num, rnd.long())) + 0 * (hint_type + hint_style)[:, None].float()


# ---- sample observations ------------------------------------------------------------------------------------------

def sample_observations(n, seed=0, opt_feat=7):
    """Real battle / rental / swap encodings from random SimBackend play (opt_feat: random option features)."""
    from pybattle.backend import Phase, SimBackend
    from rl.real_agents import ctx_of
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    out = {"battle": [], "rental": [], "swap": []}
    s = 0
    while min(len(out["battle"]), len(out["rental"]) * 4, len(out["swap"]) * 4) < n and s < 20000:
        sim = SimBackend(max_turns=100)
        sim.reset(s, rng.choice([0, 7, 14, 21, 35, 48]))
        s += 1
        while sim.phase != Phase.RUN_OVER:
            v, info = sim.view(), sim.run_info()
            ctx = ctx_of(info)
            if sim.phase == Phase.RENTAL:
                o = dict(encode.rental(v, ctx))
                o["opt_feat"] = nrng.normal(size=(6, 15, opt_feat)).astype(np.float32)
                out["rental"].append(o)
                while True:
                    a = tuple(rng.sample(range(6), 3))
                    if len({v.candidates[i].species for i in a}) == 3:
                        break
            elif sim.phase == Phase.SWAP:
                o = dict(encode.swap(v, ctx))
                o["opt_feat"] = nrng.normal(size=(10, opt_feat)).astype(np.float32)
                out["swap"].append(o)
                a = None
            else:
                out["battle"].append(encode.battle(v, ctx))
                a = rng.choice([x for x in v.legal_actions if x[0] != "forfeit"])
            sim.act(a)
    return {k: v[:n] for k, v in out.items()}


INPUTS = {
    "battler": ["mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask", "active"],
    "rental": ["mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "lead_mask", "pair_mask", "opt_feat"],
    "swap": ["mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask", "opt_feat"],
}
OUTPUTS = {"battler": ["probs", "p_win", "logits"], "rental": ["lead_logp", "pair_logp", "critic"],
           "swap": ["probs", "critic"], "teamnet": ["w"]}


def to_inputs(obs_list, names):
    """Batch encodings into the ONNX input dtypes (int32 ids, bool masks, float32 numbers)."""
    out = {}
    for k in names:
        a = np.stack([np.asarray(o[k]) for o in obs_list])
        if a.dtype.kind in "iu":
            a = a.astype(np.int32)
        elif a.dtype.kind == "b":
            a = a.astype(np.bool_)
        else:
            a = a.astype(np.float32)
        out[k] = a
    return out


def export(module, inputs, names, out_names, path):
    args = tuple(torch.from_numpy(inputs[k]) for k in names)
    dyn = {k: {0: "batch"} for k in names + out_names}
    torch.onnx.export(module, args, path, input_names=names, output_names=out_names, dynamic_axes=dyn,
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)


def check(path, module, inputs, names, atol=1e-4):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, inputs)
    with torch.no_grad():
        ref = getattr(module, "reference", module)
        want = ref(*(torch.from_numpy(inputs[k]) for k in names))
    want = want if isinstance(want, tuple) else (want,)
    worst = 0.0
    for g, w in zip(got, want):
        w = w.numpy()
        # masked log-probs are ~-1e9: compare the finite ones
        ok = np.abs(w) < 1e6
        worst = max(worst, float(np.max(np.abs(g[ok] - w[ok]))) if ok.any() else 0.0)
    if worst > atol:
        raise SystemExit(f"{os.path.basename(path)}: onnxruntime differs from torch by {worst:.2e}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "runs", "alphazero_v3", "snapshot.pt"))
    ap.add_argument("--team-eval", default=None, help="TeamNet checkpoint (default: the one the checkpoint names)")
    ap.add_argument("--out", default=os.path.join(REPO, "dist", "models"))
    ap.add_argument("--check", type=int, default=300)
    args = ap.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    pol = Policy(args.ckpt, "cpu")
    net = pol.net.eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    norms = pol.value_norm
    obs = sample_observations(max(args.check, 4), opt_feat=net.opt_feat or 7)
    meta = {"ckpt": os.path.relpath(args.ckpt, REPO), "encode_version": pol.encode_version, "opset": OPSET,
            "opt_feat": net.opt_feat, "value_norm": norms, "models": {}}

    mods = {"battler": Battler(net, norms.get("battler")), "rental": Rental(net, norms.get("tactician")),
            "swap": Swap(net, norms.get("tactician"))}
    kinds = {"battler": "battle", "rental": "rental", "swap": "swap"}
    for name, mod in mods.items():
        mod.eval()
        names = INPUTS[name] if net.opt_feat else [k for k in INPUTS[name] if k != "opt_feat"]
        inputs = to_inputs(obs[kinds[name]], names)
        path = os.path.join(args.out, f"{name}.onnx")
        export(mod, {k: v[:2] for k, v in inputs.items()}, names, OUTPUTS[name], path)
        err = check(path, mod, inputs, names) if args.check else None
        meta["models"][name] = {"file": f"{name}.onnx", "inputs": {k: [list(v.shape[1:]), str(v.dtype)]
                                                                   for k, v in inputs.items()},
                                "outputs": OUTPUTS[name], "max_abs_err": err}
        print(f"{name}: {os.path.getsize(path) / 1e6:.1f} MB, max |ort - torch| = {err}")

    # TeamNet (W)
    from rl import team_eval as TE
    te_path = args.team_eval or ck.get("team_eval")
    if te_path:
        te_path = te_path if os.path.isabs(te_path) else os.path.join(REPO, te_path)
        ev = TE.TeamEvaluator.load(te_path)
        tmod = Team(ev.net).eval()
        rng = np.random.default_rng(0)
        # teams of real rental candidates
        teams = []
        from pybattle.backend import SimBackend
        sim = SimBackend()
        for s in range(max(4, args.check // 10)):
            sim.reset(1000 + s, 0)
            cands = sim.view().candidates
            for _ in range(10):
                pick = rng.choice(6, 3, replace=False)
                teams.append(TE.Team(tuple(TE.TeamMon.from_own(cands[i]) for i in pick)))
        ids, num = TE.featurize(teams)
        n = len(teams)
        tin = {"ids": ids.astype(np.int32), "num": num.astype(np.float32),
               "rnd": rng.integers(0, TE.N_ROUNDS, n).astype(np.int32),
               "hint_type": rng.integers(0, TE.N_HINT_TYPES + 1, n).astype(np.int32),
               "hint_style": rng.integers(0, TE.N_HINT_STYLES + 1, n).astype(np.int32)}
        tnames = ["ids", "num", "rnd", "hint_type", "hint_style"]
        path = os.path.join(args.out, "teamnet.onnx")
        export(tmod, {k: v[:2] for k, v in tin.items()}, tnames, OUTPUTS["teamnet"], path)
        err = check(path, tmod, tin, tnames) if args.check else None
        meta["models"]["teamnet"] = {"file": "teamnet.onnx", "inputs": {k: [list(v.shape[1:]), str(v.dtype)]
                                                                         for k, v in tin.items()},
                                     "outputs": OUTPUTS["teamnet"], "max_abs_err": err, "hint": bool(ev.net.hint),
                                     "source": os.path.relpath(te_path, REPO)}
        print(f"teamnet: {os.path.getsize(path) / 1e6:.1f} MB, max |ort - torch| = {err}")
    meta["v_next"] = ck.get("v_next")
    with open(os.path.join(args.out, "models.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
