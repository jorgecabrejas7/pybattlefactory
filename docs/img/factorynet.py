"""Render docs/img/factorynet.{png,svg}: FactoryNet architecture diagram.

Hand-laid-out with matplotlib (no auto layout) so that no edges cross.
    python3 docs/img/factorynet.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- palette
BG, INK, MUTE, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985"
C_IN, C_EMB, C_ENC, C_TR = "#eef4fc", "#fdf0ea", "#e8f7f1", "#f0edfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FRAME = "#d9d8d3"

plt.rcParams.update({
    "font.family": "Inter",
    "svg.fonttype": "none",
    "text.color": INK,
})

W, H = 24.0, 18.0                 # figure units (inches) -> 2400 x 1800 px
fig = plt.figure(figsize=(W, H), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)                 # y grows downward
ax.set_facecolor(BG)
ax.axis("off")

FS = 13            # body text (13pt ~ 18px at 100 dpi)
FS_S = 11.5        # secondary lines
FS_T = 12          # group titles / edge labels


# ---------------------------------------------------------------- helpers
def box(x, y, w, h, name, detail=None, fill="#ffffff", edge=LINE, lw=1.2,
        detail2=None, rounding=0.16):
    """Rounded box centred at (x, y); returns dict of anchors."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       fc=fill, ec=edge, lw=lw, zorder=3)
    ax.add_patch(p)
    lines = [detail, detail2]
    n = 1 + sum(1 for l in lines if l)
    gap = 0.28
    y0 = y - (n - 1) * gap / 2
    ax.text(x, y0, name, ha="center", va="center", fontsize=FS, weight="bold",
            color=INK, zorder=4)
    k = 1
    for l in lines:
        if l:
            ax.text(x, y0 + k * gap, l, ha="center", va="center",
                    fontsize=FS_S, color=MUTE, zorder=4)
            k += 1
    return dict(x=x, y=y, l=x - w / 2, r=x + w / 2, t=y - h / 2, b=y + h / 2)


def frame(x0, y0, x1, y1, title=None, edge=FRAME, lw=1.4, fill="none",
          title_color=MUTE, title_size=FS_T, title_weight="bold"):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle="round,pad=0,rounding_size=0.25",
                       fc=fill, ec=edge, lw=lw, zorder=1)
    ax.add_patch(p)
    if title:
        ax.text(x0 + 0.35, y0 + 0.36, title, ha="left", va="center",
                fontsize=title_size, weight=title_weight, color=title_color,
                zorder=4)


def arrow(pts, color=LINE, lw=1.4, dashed=False, head=True, z=2):
    """Poly-line arrow through pts (list of (x,y)); arrowhead on last segment."""
    ls = (0, (5, 4)) if dashed else "-"
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round",
                zorder=z)
    a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-",
                        mutation_scale=16, color=color, lw=lw, ls=ls,
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def elabel(x, y, s, rot=0, ha="center", va="center", color=MUTE):
    ax.text(x, y, s, fontsize=FS_T, color=color, ha=ha, va=va, rotation=rot,
            rotation_mode="anchor", zorder=5,
            bbox=dict(fc=BG, ec="none", pad=2.5))


# ---------------------------------------------------------------- title
ax.text(0.7, 0.75, "FactoryNet — shared network (847k params)", fontsize=20,
        weight="bold", ha="left", va="center", color=INK)
ax.text(0.7, 1.28, "One network, two agents: the tactician (rental / swap "
        "decisions) and the battler (in-battle actions).",
        fontsize=FS_S, ha="left", va="center", color=MUTE)

# legend (top right)
lx, ly = 15.2, 0.75
for i, (c, ec, lab) in enumerate([
        (C_IN, LINE, "input"), (C_EMB, LINE, "embedding"),
        (C_ENC, LINE, "encoder"), (C_TR, LINE, "transformer"),
        ("#ffffff", BLUE, "battler head"), ("#ffffff", ORANGE, "tactician head")]):
    col = i % 3
    row = i // 3
    x = lx + col * 2.75
    y = ly + row * 0.5
    ax.add_patch(FancyBboxPatch((x, y - 0.14), 0.42, 0.28,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                fc=c, ec=ec, lw=1.4, zorder=3))
    ax.text(x + 0.6, y, lab, fontsize=FS_S, color=MUTE, va="center")

# ---------------------------------------------------------------- rows
Y_IDS, Y_EMB, Y_ENC, Y_SUM, Y_PENC, Y_TOK, Y_TR = 2.3, 4.2, 6.3, 7.75, 9.05, 10.75, 12.25
BH = 0.95   # standard box height

# --- id inputs
mon_ids = box(10.5, Y_IDS, 9.2, BH + 0.3, "mon_ids", "6 × 19 ids per Pokémon",
              fill=C_IN, detail2="species · 2 types · item · ability · 4 moves · "
              "4 effects · 4 move types · 2 possible abilities")
ctx_ids = box(20.4, Y_IDS, 3.3, BH + 0.3, "ctx_ids", "2 · last moves", fill=C_IN)

# --- embeddings band
EX0, EX1 = 3.6, 21.9
frame(EX0, Y_EMB - 0.78, EX1, Y_EMB + 0.78,
      title="Shared embeddings · 64-d · each table has NONE / UNKNOWN tokens",
      fill=C_EMB, edge="#f3d9cb")
chips = [("species", "413"), ("move", "356"), ("effect", "216"),
         ("type", "20"), ("item", "378"), ("ability", "79")]
cw, cg = 2.0, 0.22
cx0 = (EX0 + EX1) / 2 - (len(chips) * cw + (len(chips) - 1) * cg) / 2 + cw / 2
chip_x = {}
for i, (n, sz) in enumerate(chips):
    x = cx0 + i * (cw + cg)
    chip_x[n] = x
    box(x, Y_EMB + 0.2, cw, 0.62, n, None, fill="#ffffff", edge="#e9c9b6",
        rounding=0.1)
    ax.text(x + 0.62, Y_EMB + 0.2, sz, fontsize=FS_S, color=MUTE,
            va="center", ha="left")
EMB_T, EMB_B = Y_EMB - 0.78, Y_EMB + 0.78

arrow([(mon_ids["x"], mon_ids["b"]), (mon_ids["x"], EMB_T)])
arrow([(ctx_ids["x"], ctx_ids["b"]), (ctx_ids["x"], EMB_T)])

# --- encoders row
move_num = box(1.95, Y_ENC, 2.5, BH + 0.3, "move_num", "6 × 4 × 14", fill=C_IN)
menc = box(5.55, Y_ENC, 3.9, BH + 0.3, "Move encoder", "MLP 206 → 128 → 128",
           fill=C_ENC, detail2="shared by the 6 × 4 move slots")
ctx_num = box(15.35, Y_ENC, 2.5, BH + 0.3, "ctx_num", "93", fill=C_IN)
cenc = box(19.55, Y_ENC, 4.9, BH + 0.3, "Context encoder",
           "MLP 221 → 128 → 128", fill=C_ENC,
           detail2="+ decision-kind emb (battle / rental / swap)")

arrow([(move_num["r"], Y_ENC), (menc["l"], Y_ENC)])
arrow([(ctx_num["r"], Y_ENC), (cenc["l"], Y_ENC)])
# embeddings -> move encoder (move, effect, type)
arrow([(menc["x"], EMB_B), (menc["x"], menc["t"])])
elabel(menc["x"] + 0.15, (EMB_B + menc["t"]) / 2, "move · effect · type", ha="left")
# embeddings -> context encoder (last-move embeddings) ; ctx_ids joins band
arrow([(cenc["x"], EMB_B), (cenc["x"], cenc["t"])])
elabel(cenc["x"] + 0.15, (EMB_B + cenc["t"]) / 2, "2 last-move emb", ha="left")
# ctx_ids down to the band top-right corner (drawn above)
# embeddings -> Pokémon encoder (straight down through the free gap)
PX = 10.5

# --- Σ and Pokémon encoder
sm = box(5.55, Y_SUM, 2.7, 0.7, "Σ over 4 moves", None, fill="#ffffff",
         rounding=0.35)
arrow([(menc["x"], menc["b"]), (sm["x"], sm["t"])])
elabel(menc["x"] + 0.15, (menc["b"] + sm["t"]) / 2, "6 × 4 × 128",
       ha="left")

penc = box(PX, Y_PENC, 5.4, BH + 0.3, "Pokémon encoder", "MLP 550 → 128 → 128",
           fill=C_ENC, detail2="shared by the 6 Pokémon slots")
mon_num = box(15.35, Y_PENC, 2.5, BH + 0.3, "mon_num", "6 × 102", fill=C_IN)
arrow([(mon_num["l"], Y_PENC), (penc["r"], Y_PENC)])
arrow([(sm["x"], sm["b"]), (sm["x"], Y_PENC), (penc["l"], Y_PENC)])
elabel(sm["x"] + 0.15, (sm["b"] + Y_PENC) / 2, "6 × 128", ha="left")
arrow([(PX, EMB_B), (PX, penc["t"])])
elabel(PX + 0.15, 7.45, "species · types · item · ability · possible abilities",
       ha="left")

# --- 7 tokens
TX0, TX1 = 5.6, 18.4
frame(TX0, Y_TOK - 0.55, TX1, Y_TOK + 0.55, edge=LINE, fill="#ffffff", lw=1.2)
ax.text((TX0 + TX1) / 2, Y_TOK - 0.32, "7 tokens × 128 · no positional encoding",
        fontsize=FS_S, color=MUTE, ha="center", va="center", zorder=4)
tw, tg = 1.5, 0.16
tx0 = 7.35
for i in range(7):
    x = tx0 + i * (tw + tg)
    lab = "context" if i == 6 else f"Pokémon {i + 1}"
    box(x, Y_TOK + 0.2, tw, 0.42, lab, None,
        fill=C_ENC if i < 6 else "#ffffff", edge="#bcd9cc" if i < 6 else LINE,
        rounding=0.08)
ctx_tok_x = tx0 + 6 * (tw + tg)
arrow([(PX, penc["b"]), (PX, Y_TOK - 0.55)])
elabel(PX + 0.15, (penc["b"] + Y_TOK - 0.55) / 2, "6 × 128", ha="left")
arrow([(cenc["x"], cenc["b"]), (cenc["x"], Y_TOK - 1.0), (ctx_tok_x, Y_TOK - 1.0),
       (ctx_tok_x, Y_TOK - 0.55)])
elabel(cenc["x"] + 0.15, (cenc["b"] + Y_TOK - 1.0) / 2, "1 × 128", ha="left")

# --- Transformer
tr = box((TX0 + TX1) / 2, Y_TR, 12.8, 1.15, "Transformer encoder",
         "2 layers · 4 heads · d = 128 · FF 256 · pre-norm · final LayerNorm",
         fill=C_TR)
arrow([((TX0 + TX1) / 2, Y_TOK + 0.55), ((TX0 + TX1) / 2, tr["t"])])

# --- heads
Y_HT = 13.55         # top of the head frames
Y_HB = 17.55
BX0, BX1 = 1.15, 11.75
OX0, OX1 = 12.25, 23.3
frame(BX0, Y_HT, BX1, Y_HB, "Battler heads", edge=BLUE, lw=1.6,
      title_color=BLUE)
frame(OX0, Y_HT, OX1, Y_HB, "Tactician heads", edge=ORANGE, lw=1.6,
      title_color=ORANGE)
ax.text(BX1 - 0.35, Y_HT + 0.36, "MLP → 1 score per action · illegal actions masked before softmax",
        fontsize=10.5, color=MUTE, ha="right", va="center")
ax.text(OX1 - 0.35, Y_HT + 0.36, "MLP → 1 score per action · illegal actions masked before softmax",
        fontsize=10.5, color=MUTE, ha="right", va="center")

# bus from transformer to both groups
ybus = (tr["b"] + Y_HT) / 2
bxc, oxc = (BX0 + BX1) / 2, (OX0 + OX1) / 2
ax.plot([tr["x"], tr["x"]], [tr["b"], ybus], color=LINE, lw=1.4, zorder=2)
ax.plot([bxc, oxc], [ybus, ybus], color=LINE, lw=1.4, zorder=2)
arrow([(bxc, ybus), (bxc, Y_HT)])
arrow([(oxc, ybus), (oxc, Y_HT)])
elabel(tr["x"] + 0.15, (tr["b"] + ybus) / 2 - 0.02, "7 × 128", ha="left")

HW, HH = 3.2, 1.2
R1, R2 = Y_HT + 1.35, Y_HT + 3.05
# battler: 3 columns
bcols = [BX0 + 1.85, bxc, BX1 - 1.85]
bm = box(bcols[0], R1, HW, HH, "Move i  (×4)", "MLP[move_i ‖ active ‖ ctx] → 1",
         edge=BLUE)
bs = box(bcols[1], R1, HW, HH, "Switch j  (×3)", "MLP[mon_j ‖ ctx] → 1",
         edge=BLUE)
bv = box(bcols[2], R1, HW, HH, "Value V(s)", "MLP[mean(mons) ‖ ctx] → 1",
         edge=BLUE)
bo = box((bcols[0] + bcols[1]) / 2, R2, HW * 2 + 0.3, HH - 0.25, "π(a | s)",
         "7 logits → mask illegal → softmax", edge=BLUE, fill=C_IN)
arrow([(bm["x"], bm["b"]), (bm["x"], bo["t"])], color=BLUE)
arrow([(bs["x"], bs["b"]), (bs["x"], bo["t"])], color=BLUE)

# tactician: 3 columns
ocols = [OX0 + 1.85, oxc, OX1 - 1.85]
tl = box(ocols[0], R1, HW, HH, "Rental 1 · lead", "MLP[cand_i ‖ ctx] → 6 logits",
         edge=ORANGE)
tk = box(ocols[1], R1, HW + 0.4, HH, "Swap", "keep: MLP[pool] → 1",
         edge=ORANGE, detail2="trade: MLP[own_i ‖ foe_j ‖ ctx] → 9")
tv = box(ocols[2], R1, HW, HH, "Value V(s)", "MLP[mean(mons) ‖ ctx] → 1",
         edge=ORANGE)
tp = box(ocols[0] + 0.4, R2, HW + 0.8, HH, "Rental 2 · pair | lead",
         "MLP[cand_j + cand_k ‖ lead ‖ ctx] → 15 logits", edge=ORANGE)
arrow([(tl["x"], tl["b"]), (tl["x"], tp["t"])], color=ORANGE)
elabel(tl["x"] + 0.15, (tl["b"] + tp["t"]) / 2, "sampled lead", ha="left")
ax.text(tv["x"], tv["b"] + 0.3, "shared by rental and swap",
        fontsize=10.5, color=MUTE, ha="center", va="center")
ax.text(tk["x"], tk["b"] + 0.3, "1 + 9 logits → mask → softmax",
        fontsize=10.5, color=MUTE, ha="center", va="center")

# --- skip connection: move vectors -> battler move head, along the left edge
SX = 0.55
arrow([(menc["l"] + 0.6, menc["b"]), (menc["l"] + 0.6, menc["b"] + 0.45),
       (SX, menc["b"] + 0.45), (SX, R1), (bm["l"], R1)],
      dashed=True, color=BLUE)
elabel(SX - 0.02, (Y_ENC + 1.2 + R1) / 2, "skip: move vectors of the active Pokémon (4 × 128)",
       rot=90, ha="center", va="center", color=BLUE)

fig.savefig(OUT / "factorynet.png", dpi=100, facecolor=BG)
fig.savefig(OUT / "factorynet.svg", facecolor=BG)
print("wrote", OUT / "factorynet.png", OUT / "factorynet.svg")
