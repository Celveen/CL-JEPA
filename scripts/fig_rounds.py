"""Paper-style figures: round-by-round success of ours (online continual learning, arm cls6a_free) vs AdaJEPA (arm
w0), GD and CEM.

Each planner has three seed streams (S7t3, S7t3s400, S7t3s700) run for five rounds (5-round manifests S7t5,
S7t5s400, S7t5s700); a round is 7 shapes x 50 episodes. Lines are the mean over the three streams with a light
+-1 s.e.m. band (n = 3 streams); the arrow at round 5 marks ours - AdaJEPA in points when that gap is significant. Per-round paired McNemar tests
(pooled episodes) go to rounds_summary.{csv,md,json} rather than onto the figure. Only rounds that every stream
has finished for both methods are drawn. Writes {gd,cem}_rounds{,_split}.{pdf,png} and the summary to --out.

    python scripts/fig_rounds.py
    python scripts/fig_rounds.py --root gd:S7t3=/other/checkout/eval_outputs/cont_S7t3   # outputs kept elsewhere

Output roots default to the cont_run_stream.py runs: eval_outputs/cont_<stream> for GD and
eval_outputs/cont_<stream>_cem for CEM (arms cls6a_free and w0 inside).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEN = {"T", "L", "Z", "+"}
STREAMS = {  # planner -> [(label, 5-round manifest, default output root)]
    "gd": [("S7t3", "S7t5", "eval_outputs/cont_S7t3"), ("S7t3s400", "S7t5s400", "eval_outputs/cont_S7t3s400"),
           ("S7t3s700", "S7t5s700", "eval_outputs/cont_S7t3s700")],
    "cem": [("S7t3", "S7t5", "eval_outputs/cont_S7t3_cem"), ("S7t3s400", "S7t5s400", "eval_outputs/cont_S7t3s400_cem"),
            ("S7t3s700", "S7t5s700", "eval_outputs/cont_S7t3s700_cem")],
}
ARMS = {"cls6a_free": "Ours", "w0": "AdaJEPA"}  # figure labels: no internal arm names
TITLE = {"gd": "GD", "cem": "CEM"}


def mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def chunk(d):
    logs = d / "logs.json"
    if not ((d / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()):
        return None
    return np.load(d / "final_eval_results.npz")["successes"].astype(int)


def load(planner, roots):
    """Per stream: {round: {arm: {group: 0/1 array}}} for rounds finished by both arms; rounds every stream has."""
    per_stream = []
    for label, stream, _ in STREAMS[planner]:
        root = roots[(planner, label)]
        blocks = json.loads((ROOT / "data/cont_streams" / stream / "manifest.json").read_text())["blocks"]
        rounds = {}
        for r in range(5):
            rb = blocks[7 * r: 7 * r + 7]
            res = {}
            for arm in ARMS:
                xs = [chunk(root / arm / b["chunk"]) for b in rb]
                if any(x is None for x in xs):
                    res = None
                    break
                res[arm] = {"all": np.concatenate(xs),
                            "seen": np.concatenate([x for x, b in zip(xs, rb) if b["shape"] in SEEN]),
                            "unseen": np.concatenate([x for x, b in zip(xs, rb) if b["shape"] not in SEEN])}
            if res is not None:
                rounds[r + 1] = res
        per_stream.append((label, rounds))
    done = [r for r in range(1, 6) if all(r in rounds for _, rounds in per_stream)]
    return per_stream, done


def summarize(planner, roots):
    per_stream, done = load(planner, roots)
    out = {"rounds": done, "streams": [l for l, _ in per_stream], "groups": {}}
    for g in ("all", "seen", "unseen"):
        rows = []
        for r in done:
            rates = {arm: [100 * rounds[r][arm][g].mean() for _, rounds in per_stream] for arm in ARMS}
            x = np.concatenate([rounds[r]["cls6a_free"][g] for _, rounds in per_stream])
            y = np.concatenate([rounds[r]["w0"][g] for _, rounds in per_stream])
            b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
            rows.append({"round": r, "n": int(len(x)),
                         **{f"{arm}_mean": float(np.mean(v)) for arm, v in rates.items()},
                         **{f"{arm}_sem": float(np.std(v, ddof=1) / np.sqrt(len(v))) for arm, v in rates.items()},
                         **{f"{arm}_streams": [round(t, 1) for t in v] for arm, v in rates.items()},
                         "diff": float(np.mean(rates["cls6a_free"]) - np.mean(rates["w0"])), "only_ours": b, "only_ada": c,
                         "p": mcnemar_p(b, c)})
        out["groups"][g] = rows
    return out


def draw(ax, rows, title, legend=True, ylabel=True, xlabel=True):
    """Two lines with markers, a light +-1 s.e.m. band each, and the round-5 gap marked once at the right."""
    r = np.array([row["round"] for row in rows])
    series = [("w0", fig_style.GREY, "--", "s"), ("cls6a_free", fig_style.BLUE, "-", "o")]
    for arm, color, ls, mk in series:
        y = np.array([row[f"{arm}_mean"] for row in rows])
        e = np.array([row[f"{arm}_sem"] for row in rows])
        ax.fill_between(r, y - e, y + e, color=color, alpha=0.14, lw=0, zorder=1)
        ax.plot(r, y, color=color, ls=ls, lw=1.5, marker=mk, ms=3.8, mfc=color if mk == "o" else "white", mec=color,
                mew=1.0, label=ARMS[arm], zorder=3)
    v, a = rows[-1]["cls6a_free_mean"], rows[-1]["w0_mean"]
    if rows[-1]["p"] < 0.05:  # mark the round-5 gap only when it is significant
        x = r[-1] + 0.2
        ax.annotate("", xy=(x, v), xytext=(x, a), arrowprops=dict(arrowstyle="<->", lw=0.7, color=fig_style.BLUE,
                                                                   shrinkA=0, shrinkB=0, mutation_scale=6))
        ax.text(x + 0.08, (v + a) / 2, f"{v - a:+.1f}", va="center", ha="left", fontsize=7, color=fig_style.BLUE)
    lo = min(min(row[f"{k}_mean"] - row[f"{k}_sem"] for row in rows) for k in ARMS)
    hi = max(max(row[f"{k}_mean"] + row[f"{k}_sem"] for row in rows) for k in ARMS)
    pad = max(1.0, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad * 2.2)
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.75, 5.75)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True))
    if xlabel:
        ax.set_xlabel("Round")
    if ylabel:
        ax.set_ylabel("Success rate (%)")
    ax.set_title(title, pad=3)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)
    if legend:
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l[::-1], loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.2, labelspacing=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/figs")
    ap.add_argument("--font-dir", default="reports/figs/fonts", help="TrueType fonts to use (falls back to DejaVu Sans)")
    ap.add_argument("--root", action="append", default=[], help="override an output root: planner:stream=PATH, e.g. gd:S7t3=/x")
    a = ap.parse_args()
    roots = {(p, l): ROOT / r for p, ss in STREAMS.items() for l, _, r in ss}
    for kv in a.root:
        key, path = kv.split("=", 1)
        p, l = key.split(":")
        assert (p, l) in roots, f"unknown stream {key}; known: {sorted(f'{p}:{l}' for p, l in roots)}"
        roots[(p, l)] = Path(path)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    fig_style.setup(ROOT / a.font_dir)

    summ = {p: summarize(p, roots) for p in STREAMS}
    for p, S in summ.items():
        if not S["rounds"]:
            continue
        fig, ax = plt.subplots(figsize=(3.3, 2.35))
        draw(ax, S["groups"]["all"], TITLE[p])
        fig.tight_layout(pad=0.3)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{p}_rounds.{ext}", bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15))
        for k, (g, t) in enumerate((("all", "All shapes"), ("unseen", "Unseen shapes"), ("seen", "Seen shapes"))):
            draw(axes[k], S["groups"][g], f"{TITLE[p]}: {t}", legend=(k == 0), ylabel=(k == 0), xlabel=True)
        fig.tight_layout(pad=0.3, w_pad=1.0)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{p}_rounds_split.{ext}", bbox_inches="tight")
        plt.close(fig)

    # the numbers behind the figures
    lines = ["planner,group,round,n_episodes,ours_mean,ours_sem,adajepa_mean,adajepa_sem,diff_pp,only_ours,only_adajepa,"
             "p_mcnemar,ours_per_stream,adajepa_per_stream"]
    md = ["# Round-by-round success (mean over 3 seed streams ± s.e.m.)", "",
          "Numbers = Ours − AdaJEPA (points); p = McNemar on the round's pooled episodes.", ""]
    for p, S in summ.items():
        md += [f"## {TITLE[p]} (streams: {', '.join(S['streams'])}; rounds drawn: {S['rounds']})", "",
               "| group | round | Ours | AdaJEPA | Ours − AdaJEPA | McNemar p |", "|---|---:|---:|---:|---:|---:|"]
        for g in ("all", "unseen", "seen"):
            for row in S["groups"][g]:
                lines.append(",".join(str(x) for x in [
                    p, g, row["round"], row["n"], f"{row['cls6a_free_mean']:.2f}", f"{row['cls6a_free_sem']:.2f}",
                    f"{row['w0_mean']:.2f}", f"{row['w0_sem']:.2f}", f"{row['diff']:.2f}", row["only_ours"], row["only_ada"],
                    f"{row['p']:.3g}", "/".join(map(str, row["cls6a_free_streams"])), "/".join(map(str, row["w0_streams"]))]))
                md.append(f"| {g} | {row['round']} | {row['cls6a_free_mean']:.1f} ± {row['cls6a_free_sem']:.1f} | "
                          f"{row['w0_mean']:.1f} ± {row['w0_sem']:.1f} | {row['diff']:+.1f} | {row['p']:.2g} |")
        md.append("")
    (out / "rounds_summary.csv").write_text("\n".join(lines) + "\n")
    (out / "rounds_summary.md").write_text("\n".join(md) + "\n")
    (out / "rounds_summary.json").write_text(json.dumps(summ, indent=1))
    for p, S in summ.items():
        print(p, "rounds drawn:", S["rounds"])
        for row in S["groups"]["all"]:
            print(f"   round {row['round']}: ours {row['cls6a_free_mean']:.1f}±{row['cls6a_free_sem']:.1f}  "
                  f"AdaJEPA {row['w0_mean']:.1f}±{row['w0_sem']:.1f}  diff {row['diff']:+.1f}  p={row['p']:.2g}")


if __name__ == "__main__":
    main()
