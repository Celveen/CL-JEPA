"""Shape shifts: success rate of Ours (online continual learning, arm cls6a_free) per round of the five-round
stream, GD and CEM. Counterpart of the visual-shift "Seven visual domains" round panel.

A round is the 7 shapes x 50 episodes; its success rate pools those 350 episodes. The line is the mean over chains
(seed streams) with a +-1 SD band when there are at least two. Only rounds every chain of a planner has finished
are drawn. --with-adajepa adds AdaJEPA (w0, resets every episode) as a thin reference line per planner.

    python scripts/fig_shape_rounds.py
    python scripts/fig_shape_rounds.py --run cem:S7t5=eval_outputs/cont_S7t3_cem --with-adajepa
    # -> reports/figs_shape/shape_rounds.{pdf,png}, shape_rounds.json (the plotted numbers)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_shape_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

N_ROUNDS = 5


def round_rate(root, arm, stream, rnd):
    """Pooled success rate (%) of one round, or None if any of its chunks is unfinished."""
    succ = []
    for chunk in C.round_chunks(stream, rnd).values():
        d = root / arm / chunk
        if not C.finished(d):
            return None
        succ.append(np.load(d / "final_eval_results.npz")["successes"].astype(float))
    return 100 * np.concatenate(succ).mean()


def collect(runs, arm):
    """{planner: (rounds, (n_chains, n_rounds) array, streams)} over rounds every present chain has finished."""
    per = {}
    for planner, stream, root in runs:
        if not root.exists():
            continue
        rates = []
        for r in range(1, N_ROUNDS + 1):
            v = round_rate(root, arm, stream, r)
            if v is None:
                break
            rates.append(v)
        if rates:
            per.setdefault(planner, []).append((stream, rates))
    out = {}
    for planner, chains in per.items():
        k = min(len(r) for _, r in chains)
        out[planner] = (np.arange(1, k + 1), np.array([r[:k] for _, r in chains]), [s for s, _ in chains])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", help="planner:stream=root (default: all standard chains that exist)")
    ap.add_argument("--out", default="reports/figs_shape")
    ap.add_argument("--with-adajepa", action="store_true")
    ap.add_argument("--ylim", type=float, nargs=2, default=[40, 100])
    a = ap.parse_args()
    C.setup_style()
    runs = C.parse_runs(a.run)
    ours = collect(runs, "cls6a_free")
    if not ours:
        sys.exit("no finished rounds of Ours in the given runs")
    base = collect(runs, "w0") if a.with_adajepa else {}

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    summary = {}
    for p in ("gd", "cem"):
        if p not in ours:
            continue
        x, R, streams = ours[p]
        mu = R.mean(0)
        ax.plot(x, mu, color=C.COLOR[p], lw=1.8, marker="o", ms=3, label=f"Ours ({C.PLANNER[p]})")
        if len(R) >= 2:
            sd = R.std(0, ddof=1)
            ax.fill_between(x, mu - sd, mu + sd, color=C.COLOR[p], alpha=0.15, lw=0)
        summary[p] = {"streams": streams, "rounds": x.tolist(), "ours": R.tolist()}
        if p in base:
            bx, B, _ = base[p]
            k = min(len(bx), len(x))
            ax.plot(bx[:k], B[:, :k].mean(0), color=C.COLOR[p], lw=1.1, ls=C.LS["w0"], label=f"AdaJEPA ({C.PLANNER[p]})")
            summary[p]["adajepa"] = B[:, :k].tolist()
    ax.set_title("Seven shape domains")
    ax.set_xlabel("Continual-adaptation round")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlim(0.8, N_ROUNDS + 0.2)
    ax.set_xticks(range(1, N_ROUNDS + 1))
    ax.set_ylim(*a.ylim)
    ax.grid(True, axis="y")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    n = {p: len(ours[p][2]) for p in ours}
    band = "Mean +/- 1 SD across chains" if all(k >= 2 for k in n.values()) else "Mean across chains"
    chains = ", ".join(f"{C.PLANNER[p]}: {k}" for p, k in n.items())
    fig.text(0.0, -0.06, f"{band} ({chains}). Each round: 7 shapes x 50 episodes, new targets per round.",
             fontsize=8, color="#555555")
    out = C.ROOT / a.out
    C.save(fig, out, "shape_rounds")
    (out / "shape_rounds.json").write_text(json.dumps(summary, indent=1))
    for p, s in summary.items():
        print(p, "ours by round:", np.round(np.mean(s["ours"], 0), 1).tolist())


if __name__ == "__main__":
    main()
