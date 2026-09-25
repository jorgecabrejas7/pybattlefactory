"""One command for the periodic report: every number is computed the same way for every run, so runs are
comparable at equal training steps.

    python -m rl.full_report --main runs/ppo_joint_v3 --others runs/ppo_joint_v1,runs/ppo_joint_v2

1. Per-round evaluation (runs start at round k; the only metric comparable across start distributions) for every
   run on a common grid of battler steps (every 5 M, up to what each run reached), 384 runs per round, cached.
2. Noland / per-position analysis (rounds 3 and 6) for the main run's latest checkpoint and the other runs at the
   same step (if they got there).
3. Strategy analysis of the main run (rentals, swaps, battler moves).
4. Plots: training curves, per-round comparison, from-round-1 comparison. Summary in <main>/report.md.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

GRID = 5e6


def ckpts(run):
    out = {}
    for c in glob.glob(os.path.join(run, "ckpt_*.pt")):
        out[int(os.path.basename(c)[5:-3])] = c
    return out


def nearest(run, step, tol=0.6e6):
    cs = ckpts(run)
    if not cs:
        return None
    s = min(cs, key=lambda k: abs(k - step))
    return cs[s] if abs(s - step) <= tol else None


def sh(*args):
    subprocess.run([sys.executable, "-m", *args], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True)
    p.add_argument("--others", default="")
    p.add_argument("--runs", type=int, default=384)
    args = p.parse_args()
    others = [o for o in args.others.split(",") if o]
    runs = others + [args.main]
    latest = max(ckpts(args.main))
    grid = [g for g in np.arange(GRID, latest + 1, GRID)] + [latest]

    # 1. per-round evaluation on the common grid (each run up to where it got)
    for r in runs:
        top = max(ckpts(r))
        steps = [g for g in grid if g <= top + 0.6e6]
        if steps:
            sh("rl.eval_rounds", "--run", r, "--steps", ",".join(str(float(g)) for g in steps), "--runs", str(args.runs))

    # 2. per-position / Noland analysis at the main run's latest step
    noland = {}
    for r in runs:
        c = nearest(r, latest)
        if c:
            sh("rl.analyze_rounds", "--ckpt", c, "--rounds", "3,6", "--runs", "400")
            noland[os.path.basename(r)] = (os.path.basename(c), json.load(open(os.path.join(r, "analysis_rounds.json"))))

    # 3. strategy of the main run
    sh("rl.analyze", "--ckpt", os.path.join(args.main, os.path.basename(ckpts(args.main)[latest])), "--runs", "300")

    # 4. plots (per-round comparison and from-round-1 comparison with all runs)
    from .report import Run, compare, compare_rounds, training
    main_run = Run(args.main)
    os.makedirs(os.path.join(args.main, "plots"), exist_ok=True)
    training(main_run, os.path.join(args.main, "plots", "training.png"))
    all_runs = [Run(r) for r in runs]
    compare(all_runs, os.path.join(args.main, "plots", "compare.png"))
    compare_rounds(all_runs, os.path.join(args.main, "plots", "compare_rounds.png"))

    # summary
    def rows(r):
        return sorted([json.load(open(f)) for f in glob.glob(os.path.join(r, "eval_rounds", "*.json"))],
                      key=lambda d: d["steps"])
    lines = [f"# Informe — {os.path.basename(args.main)} a {latest / 1e6:.1f} M decisiones", ""]
    lines.append("## Evaluación por ronda al mismo número de decisiones (P completar la ronda | empieza en ella)")
    lines.append("")
    common = min(latest, *[max(ckpts(r)) for r in runs])
    lines.append(f"Paso común más alto: {common / 1e6:.1f} M (media de los 2 últimos puntos de la rejilla ≤ ese paso)")
    lines.append("")
    lines.append("| Ronda | " + " | ".join(os.path.basename(r) for r in runs) + " |")
    lines.append("|---" * (len(runs) + 1) + "|")
    for k in range(1, 7):
        cells = []
        for r in runs:
            pts = [d for d in rows(r) if d["steps"] <= common + 0.6e6][-2:]
            if pts:
                c = np.mean([d["rounds"][str(k)]["complete"] for d in pts])
                b = np.mean([d["rounds"][str(k)]["battle"] for d in pts])
                cells.append(f"{c * 100:.1f} % ({b * 100:.0f} % comb.)")
            else:
                cells.append("—")
        lines.append(f"| {k}{' (Noland)' if k in (3, 6) else ''} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"## {os.path.basename(args.main)} por ronda a lo largo del entrenamiento (P completar, %)")
    lines.append("")
    rm = rows(args.main)
    lines.append("| M decisiones | " + " | ".join(f"R{k}" for k in range(1, 7)) + " |")
    lines.append("|---" * 7 + "|")
    for d in rm:
        lines.append(f"| {d['steps'] / 1e6:.1f} | " + " | ".join(f"{d['rounds'][str(k)]['complete'] * 100:.1f}"
                                                            for k in range(1, 7)) + " |")
    lines.append("")
    lines.append(f"## Noland y posición en la ronda (paso ≈ {latest / 1e6:.1f} M)")
    lines.append("")
    for name, (c, a) in noland.items():
        for k in ("3", "6"):
            if k in a:
                x = a[k]
                lines.append(f"- {name} ({c}), ronda {k}: victorias por posición {x['win_rate_by_position']}; "
                             f"Noland {x['noland_win_rate']} en {x['noland_battles']} combates; resto {x['other_win_rate']}")
    lines.append("")
    an = json.load(open(os.path.join(args.main, "analysis_greedy.json")))
    lines.append("## Estrategia (acción más probable)")
    lines.append("")
    for k in ("streak_mean", "rental_top_by_rate", "rental_bottom_by_rate", "lead_most", "items_chosen", "swap_rate",
              "swap_rate_by_battle", "swap_slot_given", "swapped_in_most", "battler_switch_rate",
              "battler_agrees_with_maxdamage", "battler_status_move_share", "moves_top", "turn1_effects_top"):
        v = an[k]
        if isinstance(v, list):
            v = "; ".join(" ".join(map(str, x)) for x in v[:10])
        lines.append(f"- **{k}**: {v}")
    open(os.path.join(args.main, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
