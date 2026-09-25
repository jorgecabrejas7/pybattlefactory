"""Shared network for the tactician and the battler (docs/RL_DECISIONS.md §6).

Every decision is a set of tokens: 6 Pokemon tokens + 1 context token (the battle field for the battler, the
streak context for the tactician). Pokemon tokens are built by one shared encoder from shared embedding tables;
a transformer without positional encoding mixes them; action heads score the entity each action refers to.

Batched inputs (see rl/encode.py for the layout):
    mon_ids   int64  [B, 6, MON_IDS]     species, 2 types, item, ability, 4 moves, 4 effects, 4 move types, 2 possible abilities
    mon_num   float  [B, 6, MON_NUM]
    move_num  float  [B, 6, 4, MOVE_NUM]
    ctx_ids   int64  [B, CTX_IDS]        last moves (battler) / zeros (tactician)
    ctx_num   float  [B, CTX_NUM]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encode as E

NEG = -1e9
PAIRS = E.PAIRS                    # the 15 unordered pairs of 6 rental slots


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class Trunk(nn.Module):
    """Embeddings + move / Pokemon / context encoders + transformer. `emb_from`: another Trunk whose embedding
    tables this one shares (the rest of the weights are its own)."""

    def __init__(self, d_emb=64, d=128, layers=2, heads=4, emb_from=None):
        super().__init__()
        if emb_from is None:
            self.species = nn.Embedding(E.N_SPECIES_TOK, d_emb)
            self.move = nn.Embedding(E.N_MOVE_TOK, d_emb)
            self.effect = nn.Embedding(E.N_EFFECT_TOK, d_emb)
            self.type_emb = nn.Embedding(E.N_TYPE_TOK, d_emb)
            self.item = nn.Embedding(E.N_ITEM_TOK, d_emb)
            self.ability = nn.Embedding(E.N_ABILITY_TOK, d_emb)
        else:
            for k in ("species", "move", "effect", "type_emb", "item", "ability"):
                setattr(self, k, getattr(emb_from, k))
        self.move_enc = mlp(3 * d_emb + E.MOVE_NUM, d, d)
        self.mon_enc = mlp(5 * d_emb + d + E.MON_NUM, d, d)
        self.ctx_enc = mlp(2 * d_emb + E.CTX_NUM, d, d)
        self.kind = nn.Embedding(3, d)            # 0 battle, 1 rental, 2 swap: tells the trunk which decision it is
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, x, kind: int):
        ids, num, mnum = x["mon_ids"], x["mon_num"], x["move_num"]
        moves = torch.cat([self.move(ids[..., 5:9]), self.effect(ids[..., 9:13]), self.type_emb(ids[..., 13:17]), mnum], -1)
        moves = self.move_enc(moves)                                            # [B, 6, 4, d]
        mon = torch.cat([self.species(ids[..., 0]), self.type_emb(ids[..., 1]) + self.type_emb(ids[..., 2]),
                         self.item(ids[..., 3]), self.ability(ids[..., 4]),
                         self.ability(ids[..., 17]) + self.ability(ids[..., 18]), moves.sum(2), num], -1)
        mon = self.mon_enc(mon)                                                 # [B, 6, d]
        cid = x["ctx_ids"]
        ctx = self.ctx_enc(torch.cat([self.move(cid[:, 0]), self.move(cid[:, 1]), x["ctx_num"]], -1))
        ctx = ctx + self.kind.weight[kind]
        h = self.norm(self.mix(torch.cat([mon, ctx[:, None]], 1)))              # [B, 7, d]
        return h[:, :6], h[:, 6], moves


class FactoryNet(nn.Module):
    """share="all": one trunk for both agents (ppo_joint_v1/v2). share="embeddings": the tactician has its own
    encoders and transformer (trunk_t) and shares only the embedding tables with the battler."""

    def __init__(self, d_emb=64, d=128, layers=2, heads=4, share="all"):
        super().__init__()
        self.trunk = Trunk(d_emb, d, layers, heads)
        if share != "all":
            self.trunk_t = Trunk(d_emb, d, layers, heads, emb_from=self.trunk)
        self.share = share
        # battler
        self.b_move = mlp(3 * d, d, 1)
        self.b_switch = mlp(2 * d, d, 1)
        self.b_value = mlp(2 * d, d, 1)
        # tactician
        self.t_lead = mlp(2 * d, d, 1)
        self.t_pair = mlp(3 * d, d, 1)
        self.t_keep = mlp(2 * d, d, 1)
        self.t_swap = mlp(3 * d, d, 1)
        self.t_value = mlp(2 * d, d, 1)
        self.register_buffer("pairs", torch.tensor(PAIRS, dtype=torch.long))

    def tactician_trunk(self):
        return self.trunk if self.share == "all" else self.trunk_t

    @staticmethod
    def _pool(mons, ctx):
        return torch.cat([mons.mean(1), ctx], -1)

    # ---- battler: 7 logits (moves 0-3 of the active Pokemon, switch to own party slot 0-2) --------------------
    def battler(self, x):
        mons, ctx, moves = self.trunk(x, 0)
        active = x["active"]                                                   # [B] own party index 0-2
        b = torch.arange(len(active), device=active.device)
        am = moves[b, active]                                                  # [B, 4, d]
        ah = mons[b, active]                                                   # [B, d]
        mv = self.b_move(torch.cat([am, ah[:, None].expand(-1, 4, -1), ctx[:, None].expand(-1, 4, -1)], -1))[..., 0]
        sw = self.b_switch(torch.cat([mons[:, :3], ctx[:, None].expand(-1, 3, -1)], -1))[..., 0]
        logits = torch.cat([mv, sw], 1).masked_fill(~x["mask"], NEG)
        return logits, self.b_value(self._pool(mons, ctx))[:, 0]

    # ---- tactician: rental (lead over 6, then pair over 15 pairs given the lead) --------------------------------
    def rental(self, x, lead=None):
        """Returns lead logits [B,6], pair logits [B,15] (conditioned on `lead`, sampled here if None), value."""
        mons, ctx, _ = self.tactician_trunk()(x, 1)
        c6 = ctx[:, None].expand(-1, 6, -1)
        lead_logits = self.t_lead(torch.cat([mons, c6], -1))[..., 0].masked_fill(~x["lead_mask"], NEG)
        if lead is None:
            lead = torch.distributions.Categorical(logits=lead_logits).sample()
        b = torch.arange(len(lead), device=lead.device)
        lh = mons[b, lead]
        pair_h = mons[:, self.pairs[:, 0]] + mons[:, self.pairs[:, 1]]           # [B, 15, d]
        pair_logits = self.t_pair(torch.cat([pair_h, lh[:, None].expand(-1, 15, -1),
                                             ctx[:, None].expand(-1, 15, -1)], -1))[..., 0]
        pair_logits = pair_logits.masked_fill(~x["pair_mask"][b, lead], NEG)
        return lead_logits, pair_logits, lead, self.t_value(self._pool(mons, ctx))[:, 0]

    def rental_joint(self, x):
        """Every lead at once: lead logits [B,6], pair logits [B,6,15] (row l: the pair given lead l, masked; a
        row of an impossible lead is all masked), value [B]. log pi(lead, pair) = log_softmax(lead)[l] +
        log_softmax(pair[l])[p], the same factorization as rental()."""
        mons, ctx, _ = self.tactician_trunk()(x, 1)
        c6 = ctx[:, None].expand(-1, 6, -1)
        lead_logits = self.t_lead(torch.cat([mons, c6], -1))[..., 0].masked_fill(~x["lead_mask"], NEG)
        pair_h = mons[:, self.pairs[:, 0]] + mons[:, self.pairs[:, 1]]           # [B, 15, d]
        pair_logits = self.t_pair(torch.cat([pair_h[:, None].expand(-1, 6, -1, -1),
                                             mons[:, :, None].expand(-1, -1, 15, -1),
                                             ctx[:, None, None].expand(-1, 6, 15, -1)], -1))[..., 0]
        pair_logits = pair_logits.masked_fill(~x["pair_mask"], NEG)            # [B, 6, 15]
        return lead_logits, pair_logits, self.t_value(self._pool(mons, ctx))[:, 0]

    # ---- tactician: swap (keep, or own slot i x enemy slot j) -> 10 logits ------------------------------------
    def swap(self, x):
        mons, ctx, _ = self.tactician_trunk()(x, 2)
        own, foe = mons[:, :3], mons[:, 3:]
        grid = torch.cat([own[:, :, None].expand(-1, 3, 3, -1), foe[:, None].expand(-1, 3, 3, -1),
                          ctx[:, None, None].expand(-1, 3, 3, -1)], -1)
        sw = self.t_swap(grid)[..., 0].flatten(1)                                # [B, 9]  index = 3*i + j
        keep = self.t_keep(self._pool(mons, ctx))
        logits = torch.cat([keep, sw], 1).masked_fill(~x["mask"], NEG)
        return logits, self.t_value(self._pool(mons, ctx))[:, 0]
