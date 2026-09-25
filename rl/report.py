"""Training plots from TensorBoard logs.

    python -m rl.report --run runs/ppo_joint_v2 --compare runs/ppo_joint_v1
Writes <run>/plots/training.png and, with --compare, <run>/plots/compare.png.
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "lines.linewidth": 2, "legend.frameon": False})


class Run:
    def __init__(self, path):
        self.path, self.name = path, os.path.basename(path.rstrip("/"))
        self.ea = EventAccumulator(path, size_guidance={"scalars": 0})
        self.ea.Reload()

    def has(self, tag):
        return tag in self.ea.Tags()["scalars"]

    def s(self, tag):
        v = self.ea.Scalars(tag) if self.has(tag) else []
        return np.array([e.step for e in v]) / 1e6, np.array([e.value for e in v])


def ema(y, a=0.8):
    if len(y) == 0:
        return y
    o, m = np.empty_like(y), y[0]
    for i, x in enumerate(y):
        m = a * m + (1 - a) * x
        o[i] = m
    return o


def title(a, t):
    a.set_title(t, loc="left", color=INK)
    a.set_xlabel("M decisiones del combatiente")


def training(run, out):
    fig, ax = plt.subplots(3, 3, figsize=(15, 11))
    ax = ax.ravel()
    _, hrs = run.s("perf/hours")
    x, g = run.s("eval_greedy/streak_mean")
    fig.suptitle(f"{run.name} — {x[-1] if len(x) else 0:.1f} M decisiones, {hrs[-1] if len(hrs) else 0:.1f} h",
                 color=INK, fontsize=13, x=0.01, ha="left")
    a = ax[0]
    _, lo = run.s("eval_greedy/streak_ci95_low"); _, hi = run.s("eval_greedy/streak_ci95_high")
    xs, sm = run.s("eval_sampled/streak_mean")
    a.fill_between(x, lo, hi, color=BLUE, alpha=0.15, lw=0)
    a.plot(x, g, color=BLUE, label="acción más probable (IC95)"); a.plot(xs, sm, color=ORANGE, label="muestreando")
    for t, nm in (("eval_baseline_maxdamage/streak_mean", "máx. potencia"), ("eval_baseline_random/streak_mean", "aleatorio")):
        _, v = run.s(t)
        if len(v):
            a.axhline(v[-1], color=INK2, ls="--", lw=1)
            a.annotate(f"{nm} {v[-1]:.2f}", (1, v[-1]), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                       color=INK2, fontsize=8)
    title(a, "Evaluación desde racha 0: victorias seguidas"); a.legend(loc="upper left", fontsize=8)
    a = ax[1]
    for k, c in zip(range(1, 7), (BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#4a3aa7")):
        xx, v = run.s(f"eval_greedy/reach_round_{k}")
        if len(v):
            a.plot(xx, v * 100, color=c, label=f"≥ {k} rondas ({7 * k})")
    title(a, "Evaluación: % de rachas que completan k rondas"); a.set_ylabel("%"); a.legend(fontsize=7, ncol=2)
    a = ax[2]
    for k, c in zip(range(1, 7), (BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#4a3aa7")):
        xx, v = run.s(f"train_by_start/rounds_reached_from_round_{k}")
        if len(v):
            a.plot(xx, ema(v), color=c, label=f"empieza en ronda {k}")
    title(a, "Entrenamiento: ronda alcanzada según la ronda de inicio"); a.legend(fontsize=7)
    a = ax[3]
    for t, c, l in (("battler/explained_variance", BLUE, "combatiente"), ("tactician/explained_variance", ORANGE, "táctico")):
        xx, v = run.s(t); a.plot(xx, ema(v, 0.9), color=c, label=l)
    title(a, "Varianza explicada del crítico"); a.legend(fontsize=8)
    a = ax[4]
    for t, c, l in (("battler/entropy", BLUE, "combatiente"), ("tactician/entropy", ORANGE, "táctico")):
        xx, v = run.s(t); a.plot(xx, ema(v, 0.9), color=c, label=l)
    title(a, "Entropía de la política (nats)"); a.legend(fontsize=8)
    a = ax[5]
    for t, c, l in (("battler/approx_kl", BLUE, "combatiente"), ("tactician/approx_kl", ORANGE, "táctico")):
        xx, v = run.s(t); a.plot(xx, ema(v, 0.9), color=c, label=l)
    a.axhline(0.02, color=INK2, ls="--", lw=1)
    title(a, "KL aproximada por actualización"); a.legend(fontsize=8)
    a = ax[6]
    for t, c, l in (("actions/battler_switch_frac", BLUE, "combatiente: % cambios"),
                    ("actions/tactician_swap_frac", ORANGE, "táctico: % intercambia")):
        xx, v = run.s(t); a.plot(xx, ema(v) * 100, color=c, label=l)
    title(a, "Acciones elegidas"); a.set_ylabel("%"); a.legend(fontsize=8)
    a = ax[7]
    xx, v = run.s("train/battle_win_rate"); a.plot(xx, ema(v) * 100, color=BLUE)
    title(a, "Entrenamiento: % combates ganados (todas las rondas)"); a.set_ylabel("%")
    a = ax[8]
    xx, v = run.s("train/battle_decisions"); a.plot(xx, ema(v), color=BLUE)
    title(a, "Decisiones por combate")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=110); plt.close(fig)


COLORS = (BLUE, ORANGE, AQUA, YELLOW)


def compare(runs, out):
    colors = COLORS
    panels = [("eval_greedy/streak_mean", "Desde la ronda 1: victorias seguidas (acción más probable)", 1),
              ("eval_sampled/streak_mean", "Desde la ronda 1: victorias seguidas (muestreando)", 1),
              ("eval_greedy/challenge_completed_rate", "Desde la ronda 1: % completa ≥ 1 ronda", 100),
              ("battler/explained_variance", "Varianza explicada (combatiente)", 1),
              ("tactician/explained_variance", "Varianza explicada (táctico)", 1),
              ("actions/tactician_swap_frac", "Táctico: % intercambia", 100)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    fig.suptitle(" vs ".join(r.name for r in runs) + " (mismas semillas de evaluación)", color=INK, fontsize=13,
                 x=0.01, ha="left")
    xmax = max(r.s("eval_greedy/streak_mean")[0][-1] for r in runs)
    for a, (tag, t, scale) in zip(ax, panels):
        for r, c in zip(runs, colors):
            x, v = r.s(tag)
            keep = x <= xmax * 1.02
            x, v = x[keep], v[keep]
            smooth = 0.9 if tag.startswith(("battler", "tactician", "actions")) else 0.6
            a.plot(x, ema(v, smooth) * scale, color=c, label=r.name)
            if tag.startswith("eval"):
                a.plot(x, v * scale, color=c, alpha=0.2, lw=1)
        if tag == "eval_greedy/streak_mean":
            _, v = runs[0].s("eval_baseline_maxdamage/streak_mean")
            if len(v):
                a.axhline(v[-1], color=INK2, ls="--", lw=1)
                a.annotate(f"máx. potencia {v[-1]:.2f}", (1, v[-1]), xycoords=("axes fraction", "data"), ha="right",
                           va="bottom", color=INK2, fontsize=8)
        a.set_xlim(0, xmax * 1.02)
        title(a, t); a.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=110); plt.close(fig)


def load_rounds(run):
    import glob, json
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(run.path, "eval_rounds", "*.json")))]
    return sorted(rows, key=lambda r: r["steps"])


def compare_rounds(runs, out):
    """Per-round evaluation (runs start at round k): comparable whatever each policy was trained on."""
    colors = COLORS
    data = [load_rounds(r) for r in runs]
    fig, ax = plt.subplots(2, 6, figsize=(20, 7.5), sharey="row")
    fig.suptitle("Evaluación por ronda — rachas que empiezan al inicio de la ronda k (mismas semillas; banda = ±1,96 EE). "
                 "Comparar a igual eje X.", color=INK, fontsize=13, x=0.01, ha="left")
    for k in range(1, 7):
        for row, (key, label) in enumerate((("complete", "P(completar la ronda)"),
                                            ("battle", "% combates ganados en la ronda"))):
            a = ax[row, k - 1]
            for r, rows, c in zip(runs, data, colors):
                if not rows:
                    continue
                x = np.array([d["steps"] for d in rows]) / 1e6
                y = np.array([d["rounds"][str(k)][key] for d in rows])
                n = np.array([d["rounds"][str(k)]["n"] for d in rows])
                se = np.sqrt(y * (1 - y) / (n if key == "complete" else n * 3))   # battle: ~3+ battles per run
                a.fill_between(x, (y - 1.96 * se) * 100, (y + 1.96 * se) * 100, color=c, alpha=0.15, lw=0)
                a.plot(x, y * 100, color=c, marker="o", ms=4, label=r.name)
            a.set_title(f"Ronda {k}" + (" (Noland)" if k in (3, 6) else ""), loc="left", color=INK)
            if k == 1:
                a.set_ylabel(label + " (%)")
            if row == 1:
                a.set_xlabel("M decisiones del combatiente")
            if k == 1 and row == 0:
                a.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--compare", default=None)
    args = p.parse_args()
    run = Run(args.run)
    os.makedirs(os.path.join(args.run, "plots"), exist_ok=True)
    training(run, os.path.join(args.run, "plots", "training.png"))
    if args.compare:
        compare([Run(args.compare), run], os.path.join(args.run, "plots", "compare.png"))
        compare_rounds([Run(args.compare), run], os.path.join(args.run, "plots", "compare_rounds.png"))
    print("ok")


if __name__ == "__main__":
    main()
