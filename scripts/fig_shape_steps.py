"""Shape shifts: success rate vs MPC step for each of the 7 shapes, Frozen / AdaJEPA / Ours, GD and CEM.

Counterpart of the visual-shift step curves. All three methods are taken on the same round (default round 5) of the
five-round stream, so they plan the same targets; Ours is the online continual learner (arm cls6a_free) after the
preceding rounds, AdaJEPA (w0) and Frozen reset every episode. The line is the mean over chains (one chain per seed
stream); the band is +-1 SD when there are at least two chains. Two layouts:

    shape_steps.pdf        one row, GD and CEM overlaid (thumbnail of each shape above its panel)
    shape_steps_split.pdf  GD row and CEM row

    python scripts/fig_shape_steps.py                                   # default chains, those present
    python scripts/fig_shape_steps.py --run cem:S7t5=eval_outputs/cont_S7t3_cem --round 5 --out reports/figs_shape
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fig_shape_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ["frozen", "w0", "cls6a_free"]


def collect(runs, rnd):
    """{(planner, method, shape): [per-chain curve (MAX_STEPS + 1,) in %]} and the chains used per planner."""
    curves, used = {}, {}
    for planner, stream, root in runs:
        if not root.exists():
            continue
        chunks = C.round_chunks(stream, rnd)
        got = False
        for m in METHODS:
            for shape, chunk in chunks.items():
                d = root / m / chunk
                if C.finished(d):
                    curves.setdefault((planner, m, shape), []).append(100 * C.success_by_step(d).mean(0))
                    got = True
        if got:
            used.setdefault(planner, []).append(stream)
    return curves, used


def draw(ax, curves, planner, m, shape, label=None):
    cs = curves.get((planner, m, shape))
    if not cs:
        return
    cs = np.stack(cs)
    x = np.arange(C.MAX_STEPS + 1)
    mu = cs.mean(0)
    ax.plot(x, mu, color=C.COLOR[planner], ls=C.LS[m], lw=1.6 if m == "cls6a_free" else 1.3, label=label)
    if len(cs) >= 2:
        sd = cs.std(0, ddof=1)
        ax.fill_between(x, mu - sd, mu + sd, color=C.COLOR[planner], alpha=0.12, lw=0)


def style_ax(ax, first, xlabel=True):
    ax.set_xlim(0, C.MAX_STEPS)
    ax.set_ylim(0, 100)
    ax.set_xticks([10, 20, 30])
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True)
    if xlabel:
        ax.set_xlabel("MPC Step")
    else:
        ax.tick_params(labelbottom=False)
    if not first:
        ax.tick_params(labelleft=False)


def thumbs_row(fig, axes_top, shapes, mark_unseen):
    for ax, s in zip(axes_top, shapes):
        ax.imshow(C.thumbnail(s))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d0d0d0")
        title = C.shape_name(s) + ("*" if mark_unseen and s not in C.SEEN else "")
        ax.set_title(title, fontsize=12, pad=6)


def caption(used, rnd, mark_unseen):
    n = {p: len(v) for p, v in used.items()}
    chains = ", ".join(f"{C.PLANNER[p]}: {k} chain{'s' if k > 1 else ''}" for p, k in n.items())
    sd = "Mean +/- 1 SD over chains" if all(k >= 2 for k in n.values()) else "Mean over chains (no band with one chain)"
    star = " *: shape not seen in training." if mark_unseen else ""
    return f"{sd}. All methods on round {rnd} targets ({chains}); Ours continues from rounds 1-{rnd - 1}.{star}"


def fig_overlaid(curves, used, planners, rnd, mark_unseen):
    fig = plt.figure(figsize=(14.2, 3.9))
    gs = fig.add_gridspec(2, 7, height_ratios=[0.85, 1.55], hspace=0.28, wspace=0.07)
    thumbs_row(fig, [fig.add_subplot(gs[0, i]) for i in range(7)], C.SHAPES, mark_unseen)
    for i, s in enumerate(C.SHAPES):
        ax = fig.add_subplot(gs[1, i])
        for p in planners:
            for m in METHODS:
                draw(ax, curves, p, m, s, label=f"{C.PLANNER[p]} ({C.ARMS[m]})" if i == 0 else None)
        style_ax(ax, i == 0)
        if i == 0:
            ax.set_ylabel("Success Rate (%)")
    fig.legend(loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.04), handlelength=3)
    fig.text(0.07, -0.04, caption(used, rnd, mark_unseen), fontsize=8.5, color="#555555")
    return fig


def fig_split(curves, used, planners, rnd, mark_unseen):
    fig = plt.figure(figsize=(14.2, 5.6))
    gs = fig.add_gridspec(1 + len(planners), 7, height_ratios=[0.85] + [1.4] * len(planners), hspace=0.18, wspace=0.07)
    thumbs_row(fig, [fig.add_subplot(gs[0, i]) for i in range(7)], C.SHAPES, mark_unseen)
    for r, p in enumerate(planners):
        for i, s in enumerate(C.SHAPES):
            ax = fig.add_subplot(gs[1 + r, i])
            for m in METHODS:
                draw(ax, curves, p, m, s, label=f"{C.PLANNER[p]} ({C.ARMS[m]})" if i == 0 else None)
            style_ax(ax, i == 0, xlabel=r == len(planners) - 1)
            if i == 0:
                ax.set_ylabel(f"{C.PLANNER[p]}\nSuccess Rate (%)")
    fig.legend(loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=3)
    fig.text(0.07, 0.0, caption(used, rnd, mark_unseen), fontsize=8.5, color="#555555")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", help="planner:stream=root (default: all standard chains that exist)")
    ap.add_argument("--round", type=int, default=5)
    ap.add_argument("--out", default="reports/figs_shape")
    ap.add_argument("--no-mark-unseen", action="store_true")
    a = ap.parse_args()
    C.setup_style()
    curves, used = collect(C.parse_runs(a.run), a.round)
    planners = [p for p in ("gd", "cem") if p in used]
    if not planners:
        sys.exit("no finished chunks for this round in the given runs")
    print("chains:", used)
    out = C.ROOT / a.out
    mark = not a.no_mark_unseen
    C.save(fig_overlaid(curves, used, planners, a.round, mark), out, "shape_steps")
    C.save(fig_split(curves, used, planners, a.round, mark), out, "shape_steps_split")


if __name__ == "__main__":
    main()
