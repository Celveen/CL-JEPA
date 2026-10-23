"""Paper-style figures: official AdaJEPA shape weights vs ours frozen at round 5, on the held-out stream H7t3.

Bars are the mean over experiments (official: 3 test seeds; ours: round-5 states x 3 seeds); the number above a
pair is the paired difference ours - official in points (blue when its 95% bootstrap interval excludes 0; the
per-shape figure shows only those). Unseen shapes are shaded and starred. Reads the statistics written by
scripts/cmp_official_stats.py, where ours is keyed "v6a" (internal name, never shown in a figure).

    python scripts/cmp_official_stats.py && python scripts/fig_cmp_official.py
    # -> reports/figs/cmp_overall.{pdf,png} (all / seen / unseen), reports/figs/cmp_shapes.{pdf,png} (per shape)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = {"official": "#c4c9d0", "v6a": fig_style.BLUE}
LBL = {"official": "AdaJEPA (official weights)", "v6a": "Ours (round 5)"}
TITLE = {"gd": "GD", "cem": "CEM"}
SHAPE = {"square": "Square", "small_tee": "Small T"}  # tick labels; the letter shapes keep their id


def bars(ax, P, keys, names, width=0.36, band=None, label="all"):
    """Grouped bars (official, ours); the gap ours - official above each pair (label="all") or only where its 95%
    interval excludes 0 (label="sig")."""
    x = np.arange(len(keys))
    if band:
        for i, k in enumerate(keys):
            if band(k):
                ax.axvspan(i - 0.5, i + 0.5, color="#f3f4f6", lw=0, zorder=0)
    for j, m in enumerate(("official", "v6a")):
        ax.bar(x + (j - 0.5) * width, [P["models"][m][k]["mean"] for k in keys], width * 0.94, color=C[m], label=LBL[m],
               zorder=2, edgecolor="none")
    for i, k in enumerate(keys):
        d = P["diffs"]["v6a-official"][k]
        sig = d["ci"][0] > 0 or d["ci"][1] < 0
        if label == "sig" and not sig:
            continue
        top = max(P["models"]["official"][k]["mean"], P["models"]["v6a"][k]["mean"])
        ax.text(i, top + 1.5, f"{d['diff']:+.1f}", ha="center", va="bottom", fontsize=7,
                color=fig_style.BLUE if sig else fig_style.GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="reports/cmp/report_data.json")
    ap.add_argument("--out", default="reports/figs")
    ap.add_argument("--font-dir", default="reports/figs/fonts", help="TrueType fonts to use (falls back to DejaVu Sans)")
    a = ap.parse_args()
    D = json.loads((ROOT / a.data).read_text())
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    fig_style.setup(ROOT / a.font_dir)
    shapes = [s["id"] for s in D["shapes"]]
    seen = {s["id"] for s in D["shapes"] if s["seen"]}

    keys, names = ["all", "seen", "unseen"], ["All", "Seen", "Unseen"]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2), sharey=True)
    for ax, p in zip(axes, ("gd", "cem")):
        bars(ax, D["planners"][p], keys, names)
        ax.set_title(TITLE[p], pad=3)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))
    axes[0].set_ylabel("Success rate (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.15), handlelength=1.2)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cmp_overall.{ext}", bbox_inches="tight")
    plt.close(fig)

    names = [SHAPE.get(s, s) + ("*" if s not in seen else "") for s in shapes]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    for ax, p in zip(axes, ("gd", "cem")):
        bars(ax, D["planners"][p], shapes, names, width=0.38, band=lambda k: k not in seen, label="sig")
        ax.set_title(TITLE[p], pad=3)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))
    axes[0].set_ylabel("Success rate (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.15), handlelength=1.2)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cmp_shapes.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("written:", sorted(p.name for p in out.glob("cmp_*.p*")))


if __name__ == "__main__":
    main()
