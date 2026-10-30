"""Shape-shift figures in the style of the visual-shift figure set (offline; numpy, matplotlib, pillow only).

    python scripts/build_figures.py            # from the package root (the folder holding data/ and replay/)

Inputs : data/shape7_summary.json, data/ours_rounds.json, data/mpc_steps.json, data/trajectories.json,
         thumbnails/<shape>.png, replay/<planner>/<shape>/<method>/{step_XX,goal}.png
Outputs: figures/NN_*.pdf (vector curves/text; simulator frames are raster), preview/NN_*.png,
         形状偏移图表合集.pdf (all pages), figure_manifest.json, English_captions.txt, 图表索引.html
"""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for d in ["figures", "preview"]:
    (ROOT / d).mkdir(exist_ok=True)


def read(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


S = read("shape7_summary.json")
R = read("ours_rounds.json")
M = read("mpc_steps.json")
T = read("trajectories.json")

PLANNERS = ["GD", "CEM"]
METHODS = ["frozen", "adajepa", "ours"]
SHAPES = ["I", "L", "square", "Z", "small_tee", "plus", "T"]
NAME = {"I": "I", "L": "L", "square": "square", "Z": "Z", "small_tee": "smallT", "plus": "+", "T": "T"}
LABEL = {"frozen": "Frozen", "adajepa": "AdaJEPA", "ours": "Ours"}
# Same palette and line styles as the visual-shift figure set (data/style.json there).
PLANNER_COLOR = {"GD": "#708CC3", "CEM": "#86B99C"}
METHOD_COLOR = {"frozen": "#3268DD", "adajepa": "#EC4436", "ours": "#26955B"}
BAR_COLOR = {"frozen": "#CFCAC2", "adajepa": "#7BCDBF", "ours": "#7FB5E8"}
BAR_BACKGROUND = "#F7F7F7"
METHOD_STYLE = {"frozen": "--", "adajepa": "-.", "ours": "-"}
SEEN_TXT = "T, L, Z, +"
UNSEEN_TXT = "I, square, smallT"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12,
    "axes.linewidth": .65, "axes.edgecolor": "#555555", "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "text.color": "#151515", "axes.labelcolor": "#151515", "xtick.color": "#333333",
    "ytick.color": "#333333", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
MANIFEST, FIGS = [], []

COMMON = ("Held-out shape-shift targets H7t3: seven shapes x 50 episodes x test seeds 2000/2100/2200, identical "
          "initial/goal pairs for every method; none was used during continual learning. Frozen and AdaJEPA use the "
          "official AdaJEPA shape-shift checkpoint (Frozen: no adaptation; AdaJEPA: per-episode test-time adaptation, "
          "reset every episode). Ours is the round-5 state of each of three continual-learning chains per planner (GD: "
          "streams S7t3, S7t3s400, S7t3s700; CEM: S7t3, S7t3b, S7t3s700, where S7t3b is the S7t3 stream run again "
          "on a second machine) on its own base checkpoint, evaluated read-only: within-episode adaptation as usual, "
          "but no learner is written, trained or merged. Means are over the three test-seed blocks (Ours pools its "
          "three chains inside each block: 150 episodes per shape per block, baselines 50); +/- 1 sample SD across "
          "the three blocks, not a confidence interval. Seen shapes T, L, Z, + are in our base model's training "
          "data; I, square, small T are not. Planning budget: at most 30 MPC replans; GD 100 steps, CEM 200 samples, "
          "top 30, 10 steps. Differences combine checkpoint and adaptation effects.")


def tidy(ax, ylim=(0, 100), both=True):
    ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.grid(axis="both" if both else "y", color="#D0D0D0", ls=":", lw=.55)
    ax.tick_params(length=2.8, pad=3)


def bar_tidy(ax, ylim):
    tidy(ax, ylim, both=False)
    ax.set_facecolor(BAR_BACKGROUND)
    ax.grid(axis="y", color="#E3E3E3", ls="-", lw=.5)
    for sp in ax.spines.values():
        sp.set_color("#D0D0D0")


def method_legend(fig, y=.985):
    fig.legend(handles=[Patch(facecolor=BAR_COLOR[m], label=LABEL[m]) for m in METHODS], loc="upper center",
               bbox_to_anchor=(.53, y), ncol=3, frameon=False, columnspacing=3)


def save(fig, name, title, caption):
    idx = len(MANIFEST) + 1
    fn = f"{idx:02d}_{name}"
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    out = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()
           and (lambda b: b.x0 < -.5 or b.y0 < -.5 or b.x1 > w + .5 or b.y1 > h + .5)(t.get_window_extent(rend))]
    assert not out, (fn, out)
    for ax in fig.axes:
        for line in ax.lines:
            gid = line.get_gid()
            if gid == "ours":
                assert line.get_linestyle() == "-" and line.get_marker() in [None, "None", "", " "], fn
            if gid == "adajepa":
                assert line.get_linestyle() == "-.", fn
    meta = {"Title": title, "Author": "AdaJEPA experiment figures", "Subject": caption, "Creator": "Python / Matplotlib"}
    fig.savefig(ROOT / "figures" / f"{fn}.pdf", metadata=meta)
    fig.savefig(ROOT / "preview" / f"{fn}.png", dpi=180)
    FIGS.append(fig)
    MANIFEST.append({"number": idx, "name": fn, "title": title, "caption": caption})
    print("Saved", fn, flush=True)


# 1-2: cumulative success by MPC step, seven shapes (layout of the visual-shift Figures 1-2)
for split in [False, True]:
    fig = plt.figure(figsize=(13.5, 5.65 if split else 3.85))
    gs = fig.add_gridspec(3 if split else 2, 7, height_ratios=[1, 1.5, 1.5] if split else [1, 1.7],
                          left=.055, right=.993, top=.805 if split else .76, bottom=.15 if split else .155,
                          wspace=.08, hspace=.10)
    for j, shape in enumerate(SHAPES):
        th = fig.add_subplot(gs[0, j])
        th.imshow(Image.open(ROOT / "thumbnails" / f"{shape}.png"))
        th.set_xticks([])
        th.set_yticks([])
        for sp in th.spines.values():
            sp.set_color("#CCCCCC")
            sp.set_linewidth(.6)
        th.set_title(NAME[shape] + ("" if shape in ("T", "L", "Z", "plus") else "*"), fontsize=12, pad=4)
        for ri, opts in enumerate([["GD"], ["CEM"]] if split else [PLANNERS]):
            ax = fig.add_subplot(gs[ri + 1, j])
            tidy(ax)
            for planner in opts:
                for method in METHODS:
                    c = S["curves"][planner][shape][method]
                    mu, sd = np.r_[0, c["mean"]], np.r_[0, c["sd"]]
                    assert abs(mu[-1] - S["by_shape"][planner][shape][method]["mean"]) < 1e-8
                    assert np.all(np.diff(mu) >= -1e-8)
                    x = np.arange(31)
                    col = PLANNER_COLOR[planner]
                    ax.fill_between(x, np.clip(mu - sd, 0, 100), np.clip(mu + sd, 0, 100), color=col, alpha=.085, lw=0)
                    ax.plot(x, mu, color=col, ls=METHOD_STYLE[method], gid=method,
                            lw=1.65 if method == "ours" else 1.35, marker=None)
            ax.set(xlim=(0, 30), xticks=[10, 20, 30], yticks=[0, 20, 40, 60, 80, 100])
            ax.tick_params(labelleft=j == 0, labelbottom=(not split or ri == 1))
            if not split or ri == 1:
                ax.set_xlabel("MPC Step", labelpad=3)
            if j == 0:
                ax.set_ylabel((("GD\n" if ri == 0 else "CEM\n") + "Success Rate (%)") if split else "Success Rate (%)")
    handles = [Line2D([], [], color=PLANNER_COLOR[p], ls=METHOD_STYLE[m], lw=1.6, marker=None, label=f"{p} ({LABEL[m]})")
               for p in PLANNERS for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.525, .995), ncol=6, frameon=False,
               columnspacing=1.6, handlelength=2.8)
    fig.text(.055, .025, "Mean +/- 1 SD across test seeds 2000/2100/2200. * shape not in our base model's training data. "
             "Ours: round-5 states of three chains, read-only; baselines: official shape-shift checkpoint.",
             fontsize=8.3, color="#555555")
    save(fig, "shape7_by_planner" if split else "shape7_success",
         "Seven shape domains" + (" - separate planners" if split else ""),
         "Planning success across seven shape domains. Curves show the mean cumulative success rate by MPC step; "
         "shading denotes +/- 1 sample SD, clipped to [0, 100]. " + COMMON +
         " Encoding: blue GD, green CEM; dashed Frozen, dash-dot AdaJEPA, plain solid Ours without markers.")

# 3: grouped bars - all / seen / unseen shapes
fig, ax = plt.subplots(figsize=(10.2, 3.9))
fig.subplots_adjust(left=.075, right=.99, top=.83, bottom=.23)
bar_tidy(ax, (0, 109))
ax.set_yticks(range(0, 101, 20))
method_legend(fig)
centers = np.array([0, 1, 2.5, 3.5, 5, 6])
width = .23
for j, method in enumerate(METHODS):
    ss = [S["rates"][p][method][g] for g in ["all", "seen", "unseen"] for p in PLANNERS]
    vals, sd = [x["mean"] for x in ss], [x["sd"] for x in ss]
    bars = ax.bar(centers + (j - 1) * width, vals, width, color=BAR_COLOR[method], edgecolor="white", linewidth=.6,
                  yerr=sd, capsize=2, error_kw={"elinewidth": .65, "capthick": .65})
    for b, v, e in zip(bars, vals, sd):
        ax.text(b.get_x() + width / 2, v + e + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
for i, group in enumerate(["All seven shapes", f"Seen shapes ({SEEN_TXT})", f"Unseen shapes ({UNSEEN_TXT})"]):
    ax.text(float(centers[2 * i:2 * i + 2].mean()), -.19, group, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11)
ax.set(xlim=(-.6, 6.6), ylabel="Success Rate (%)", xticks=centers)
ax.set_xticklabels(["GD", "CEM"] * 3)
fig.text(.075, .025, "Error bars: +/- 1 SD across test seeds 2000/2100/2200. 1,050 episodes per method and planner "
         "(Ours: 3,150, three chains).", fontsize=8.5, color="#555555")
save(fig, "shape7_grouped", "Seven shape domains - seen and unseen shapes",
     "Success rates on all seven shapes, on the seen shapes (T, L, Z, +) and on the unseen shapes (I, square, "
     "small T). Bars show mean success rates; error bars denote +/- 1 sample SD across test-seed blocks. " + COMMON)

# 4: every shape, both planners
fig, axs = plt.subplots(1, 2, figsize=(13.2, 3.75), sharey=True)
fig.subplots_adjust(left=.06, right=.99, top=.78, bottom=.17, wspace=.08)
method_legend(fig)
for ci, planner in enumerate(PLANNERS):
    ax = axs[ci]
    bar_tidy(ax, (0, 108))
    ax.set_yticks(range(0, 101, 20))
    x = np.arange(7)
    for j, method in enumerate(METHODS):
        ss = [S["by_shape"][planner][s][method] for s in SHAPES]
        ax.bar(x + (j - 1) * .23, [a["mean"] for a in ss], .23, yerr=[a["sd"] for a in ss], color=BAR_COLOR[method],
               capsize=1.8, error_kw={"elinewidth": .65, "capthick": .65}, edgecolor="white", linewidth=.5)
    ax.set_xticks(x, [NAME[s] + ("" if s in ("T", "L", "Z", "plus") else "*") for s in SHAPES])
    ax.set_title(f"({chr(97 + ci)}) Shape shifts - {planner}", loc="left", pad=8)
    if ci == 0:
        ax.set_ylabel("Success Rate (%)")
fig.text(.06, .025, "Mean +/- 1 SD across test seeds 2000/2100/2200; 150 episodes per shape per method "
         "(Ours: 450). * shape not in our base model's training data.", fontsize=8.5, color="#555555")
save(fig, "shape7_per_shape", "Seven shape domains - per-shape comparison",
     "Per-shape success rates for each planner. Bars show mean success rates; error bars denote +/- 1 sample SD "
     "across test-seed blocks. " + COMMON)

# 5: Ours over five continual-learning rounds (training streams)
fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.65), sharey=True)
fig.subplots_adjust(left=.06, right=.99, top=.85, bottom=.24, wspace=.1)
for ax, g, title in zip(axs, ["all", "seen", "unseen"],
                        ["All seven shapes", f"Seen shapes ({SEEN_TXT})", f"Unseen shapes ({UNSEEN_TXT})"]):
    tidy(ax, (40, 100), both=False)
    ax.set_yticks(range(40, 101, 10))
    for planner in PLANNERS:
        ss = [R[planner][str(r)][g] for r in range(1, 6)]
        mu, sd = np.array([x["mean"] for x in ss]), np.array([x["sd"] for x in ss])
        ax.fill_between(range(1, 6), mu - sd, mu + sd, color=PLANNER_COLOR[planner], alpha=.15, lw=0)
        ax.plot(range(1, 6), mu, color=PLANNER_COLOR[planner], lw=1.8, ls="-", marker=None, gid="ours",
                label=f"Ours ({planner})")
    ax.set(xticks=range(1, 6), xlabel="Continual-adaptation round", title=title)
axs[0].set_ylabel("Success Rate (%)")
axs[2].legend(frameon=False, loc="lower right")
fig.text(.06, .025, "Mean +/- 1 SD across three chains (GD: streams S7t3, S7t3s400, S7t3s700; CEM: S7t3, S7t3b, "
         "S7t3s700). Each round: 7 shapes x 50 episodes with new test seeds; Ours keeps learning across rounds.",
         fontsize=8.3, color="#555555")
assert all(tuple(ax.get_ylim()) == (40., 100.) for ax in axs)
save(fig, "ours_five_rounds", "Ours over five continual-learning rounds - shape shifts",
     "Success of Ours on its training streams over five continual-learning rounds. Each chain runs the seven shapes in "
     "the order I, L, square, Z, small T, +, T, 50 episodes each per round, with new test seeds every round, and keeps "
     "learning throughout; 350 episodes per chain per round. Lines show the mean over three chains (GD: streams "
     "S7t3, S7t3s400, S7t3s700; CEM: S7t3, S7t3b, S7t3s700, where S7t3b is the S7t3 stream run again on a second "
     "machine); shading denotes +/- 1 sample SD across chains. Rounds use different targets, so round "
     "differences include target, seed and adaptation-history effects. These are training-stream episodes, not the "
     "held-out H7t3 targets of the other figures. All panels use a 40-100% y-axis.")


# 6-19: trajectories with prediction loss (layout of the visual-shift Figures 7-20)
def status_icon(ax, success):
    color = "#24A050" if success else "#ED4436"
    ax.add_patch(Circle((.86, .13), .105, transform=ax.transAxes, facecolor="white", edgecolor=color, lw=1, zorder=8))
    if success:
        ax.plot([.802, .842, .918], [.13, .088, .18], color=color, lw=1.3, transform=ax.transAxes, zorder=9)
    else:
        ax.plot([.817, .903], [.087, .173], color=color, lw=1.3, transform=ax.transAxes, zorder=9)
        ax.plot([.817, .903], [.173, .087], color=color, lw=1.3, transform=ax.transAxes, zorder=9)


loss_rows = []
for planner in PLANNERS:
    for case in [c for c in T["cases"] if c["planner"] == planner]:
        shape = case["shape"]
        fig = plt.figure(figsize=(14.4, 5.7))
        gs = fig.add_gridspec(4, 16, height_ratios=[1, 1, 1, 1.55], left=.073, right=.993, top=.875, bottom=.19,
                              wspace=.035, hspace=.18)
        fig.text(.073, .97, f"{NAME[shape]} / {planner}", fontsize=13, va="top")
        fig.text(.993, .97, "Fixed sample 1 | test seed 2000", fontsize=10, ha="right", va="top", color="#555555")
        frame_steps = list(range(14)) + [30, None]
        for row, method in enumerate(METHODS):
            item = case["methods"][method]
            n = item["n_steps"]
            path = ROOT / "replay" / planner / shape / method
            axes = []
            for col, step in enumerate(frame_steps):
                ax = fig.add_subplot(gs[row, col])
                axes.append(ax)
                held = step is not None and step > n
                ax.imshow(Image.open(path / ("goal.png" if step is None else f"step_{min(step, n):02d}.png")),
                          interpolation="nearest")
                if held:
                    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#D8D8D8", alpha=.88, lw=0))
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#DDDDDD")
                    sp.set_linewidth(.55)
                if row == 0:
                    ax.set_title("Goal" if step is None else str(step), fontsize=9, pad=4)
                if col == 0:
                    ax.text(.025, .98, "Replay" if method == "frozen" else "Simulator", transform=ax.transAxes,
                            va="top", color="white", fontsize=6.7, fontweight="bold",
                            bbox={"facecolor": METHOD_COLOR[method], "pad": .6, "edgecolor": "none"})
                if step is None:
                    status_icon(ax, item["success"])
                if col == 14:
                    ax.text(.5, -.13, f"{'Success' if item['success'] else 'Failure'} @ {n}", transform=ax.transAxes,
                            ha="center", fontsize=6.8, color=METHOD_COLOR[method])
            first, last = axes[0].get_position(), axes[14].get_position()
            fig.add_artist(Rectangle((first.x0, first.y0), last.x1 - first.x0, first.height, transform=fig.transFigure,
                                     fill=False, edgecolor=METHOD_COLOR[method], lw=1))
            fig.text(first.x0 - .015, first.y0 + first.height / 2, LABEL[method], color=METHOD_COLOR[method],
                     rotation=90, va="center", ha="center", fontweight="bold", fontsize=10)
        ax = fig.add_subplot(gs[3, :15])
        ax.set_facecolor("#F0F0F0")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#AAAAAA")
        ax.grid(axis="y", color="white", lw=.6)
        ax.set_axisbelow(True)
        for method in METHODS:
            item = case["methods"][method]
            assert len(item["steps"]) == len(item["loss"]) == item["n_steps"]
            ax.plot(item["steps"], item["loss"], color=METHOD_COLOR[method], lw=1.5, ls=METHOD_STYLE[method]
                    if method != "frozen" else "-", gid=method, marker=None if method == "ours" else "o", ms=3.3,
                    markeredgecolor="white", markeredgewidth=.4, label=LABEL[method])
            loss_rows.extend({"planner": planner, "shape": shape, "method": LABEL[method], "mpc_step": s,
                              "prediction_mse": v, "origin": item["loss_origin"]}
                             for s, v in zip(item["steps"], item["loss"]))
        ax.set(xlim=(0, 30), ylim=(0, None), xticks=list(range(0, 31, 2)), xlabel="MPC replanning steps",
               ylabel="Latent prediction\nloss (MSE)")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(length=2.5, labelsize=9)
        ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False, borderaxespad=0, handlelength=1.25,
                  fontsize=8)
        fig.text(.073, .071, "Prediction MSE on up to five recent transitions. Visual + proprioceptive latents; action "
                 "dimensions excluded. Frozen: same criterion, computed on its replayed trajectory.", fontsize=9,
                 color="#444444")
        fig.text(.073, .035, "Gray frames hold the terminal state. Curves stop at termination. Each method is scored "
                 "with its own weights, so raw loss levels are not directly comparable.", fontsize=9, color="#444444")
        cap = (f"Planning trajectory and latent prediction loss for the {NAME[shape]} shape with {planner}. Sample index 0 "
               "of test seed 2000 is fixed for every shape and planner, without outcome-based selection. Frozen and "
               "AdaJEPA use the official shape-shift checkpoint; Ours uses the round-5 state of chain 1 (stream S7t3), "
               "read-only. AdaJEPA and Ours rows show the frames saved during the run; the Frozen row replays the "
               "run's executed actions in the simulator (replaying AdaJEPA's actions reproduces its saved frames "
               "pixel for pixel, and the Frozen replay reproduces the run's success step). Steps 0-13 and 30 are shown; "
               "gray frames repeat a terminated trajectory. The rightmost column is the goal, with the outcome. The "
               "lower panel is the online-adaptation buffer loss evaluated before each update: sliding-window one-step "
               "latent MSE on visual and proprioceptive tokens of up to five recent transitions, action dimensions "
               "excluded. AdaJEPA and Ours values are the logged losses (adajepa_losses.csv, identical to "
               "per_iter_logs.json). Frozen never adapts, so its loss is computed afterwards with the official "
               "checkpoint on its replayed trajectory, with the same buffer, preprocessing and loss function, in the "
               "training mode of the logged loss (dropout draws from fixed diagnostic seeds); the same code reproduces "
               "AdaJEPA's logged theta0 buffer loss. Checkpoints and latent spaces differ between Ours and the "
               "baselines, so raw cross-method loss magnitude alone does not establish better predictions.")
        save(fig, f"{planner.lower()}_{shape}_prediction_loss",
             f"Planning trajectory and prediction loss - {NAME[shape]} / {planner}", cap)

# 20: MPC steps table
fig = plt.figure(figsize=(11.69, 5.6))
fig.text(.048, .93, "MPC planning steps - held-out shape-shift targets", fontsize=17, fontweight="bold")
fig.text(.048, .885, "Seven shapes x 50 targets x 3 test seeds per method and planner; identical initial/goal pairs. "
         "Ours: round-5 states of three chains, read-only.", fontsize=10, color="#555555")
cols = [("Planner", .048, "left"), ("Method", .12, "left"), ("Episodes", .31, "right"), ("Success (%)", .44, "right"),
        ("Steps, all episodes", .66, "right"), ("Steps, successes only", .93, "right")]
y0 = .80
fig.add_artist(Line2D([.048, .952], [y0 + .025, y0 + .025], color="#303030", lw=.8))
for text, x, ha in cols:
    fig.text(x, y0, text, fontsize=9.5, fontweight="bold", ha=ha, va="center")
fig.add_artist(Line2D([.048, .952], [y0 - .03, y0 - .03], color="#303030", lw=.45))
y = y0 - .075
table_rows = []
for planner in PLANNERS:
    for method in METHODS:
        r = M[planner][method]
        bold = method == "ours"
        if bold:
            fig.add_artist(Rectangle((.048, y - .025), .904, .05, color="#F0F5F1", lw=0, zorder=0))
        vals = [planner, LABEL[method], f"{r['episodes']:,}", f"{r['success_rate']['mean']:.2f}",
                f"{r['steps_all']['mean']:.2f} +/- {r['steps_all']['sd']:.2f}",
                f"{r['steps_success_only']['mean']:.2f} +/- {r['steps_success_only']['sd']:.2f}"]
        for v, (_, x, ha) in zip(vals, cols):
            fig.text(x, y, v, fontsize=10.5, ha=ha, va="center", fontweight="bold" if bold else "normal")
        table_rows.append(vals)
        y -= .065
    fig.add_artist(Line2D([.048, .952], [y + .03, y + .03], color="#D8D8D8" if planner == "GD" else "#303030",
                          lw=.45 if planner == "GD" else .8))
notes = ["Episodes per method and planner: 7 shapes x 50 targets x 3 test seeds = 1,050 (Ours: 3,150, its three "
         "chains on the same 1,050 targets).",
         "Steps are MPC replanning iterations. Mean +/- sample SD is across the three test-seed means; lower means "
         "fewer iterations.",
         "All-episode means include failures, which each used the full 30-step budget. Success-only means use each "
         "method's successful subset.",
         "Success-only subsets differ across methods; read both step columns with the success rate. Checkpoints differ "
         "between Ours and the baselines.",
         "Source: final_eval_results.npz of every run (action_len = first successful replan), checked against "
         "per_iter_logs.json for the sequential runs."]
for k, t in enumerate(notes):
    fig.text(.048, y - .035 - k * .034, t, fontsize=8.6, color="#555555")
assert y - .035 - (len(notes) - 1) * .034 > .02
save(fig, "mpc_steps_table", "MPC planning steps - shape shifts",
     "Mean number of MPC replans on the held-out shape-shift targets. " + COMMON)

# ---------------------------------------------------------------- merged PDF and indexes
with PdfPages(ROOT / "形状偏移图表合集.pdf", metadata={"Title": "Shape-shift figures",
                                                     "Author": "AdaJEPA experiment figures"}) as pdf:
    for f in FIGS:
        pdf.savefig(f)
for f in FIGS:
    plt.close(f)
with (ROOT / "data" / "prediction_loss_plot_values.csv").open("w", encoding="utf-8") as f:
    f.write("planner,shape,method,mpc_step,prediction_mse,origin\n")
    for r in loss_rows:
        f.write(f"{r['planner']},{r['shape']},{r['method']},{r['mpc_step']},{r['prediction_mse']!r},\"{r['origin']}\"\n")
(ROOT / "figure_manifest.json").write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False), encoding="utf-8")
(ROOT / "English_captions.txt").write_text("\n\n".join(f"Figure {x['number']}. {x['title']}\n{x['caption']}"
                                                       for x in MANIFEST), encoding="utf-8")
page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>形状偏移图表</title><style>body{max-width:1400px;margin:30px auto;padding:0 28px;font:16px/1.7 Arial,'
        '"Microsoft YaHei",sans-serif;color:#222}img{width:100%}figure{margin:35px 0 55px}a{color:#3268dd}'
        'figcaption{font-size:13px;color:#555}</style><h1>形状偏移图表</h1><p><a href="形状偏移图表合集.pdf">合集PDF</a> · '
        '<a href="English_captions.txt">英文图注</a> · <a href="说明.txt">说明</a></p>')
for x in MANIFEST:
    page += (f'<figure><h2>{x["number"]:02d}. {html.escape(x["title"])}</h2><a href="figures/{x["name"]}.pdf">'
             f'<img loading="lazy" src="preview/{x["name"]}.png"></a><details><summary>English caption</summary>'
             f'<figcaption>{html.escape(x["caption"])}</figcaption></details></figure>')
(ROOT / "图表索引.html").write_text(page, encoding="utf-8")
print("DONE", len(MANIFEST), "figures", flush=True)
