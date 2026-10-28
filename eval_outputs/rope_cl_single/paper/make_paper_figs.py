"""Paper figures and tables for single-domain continual learning, one set per domain (rope, granular).

Our protocol (the online stream): 50 instances in a fixed order; CEM with 300 samples, top 30, 10 iterations,
horizon 5; MPC of 3 iterations x 5 pushes; the slow learner keeps learning over three passes of the same stream.
Methods on it, paired episode by episode:
  DINO-WM          theta0, frozen (the `frozen` arm)
  AdaJEPA          test-time adaptation within each episode, reset at its end (`adajepa`)
  Ours             the slow learner alone after three passes over the stream (`cl_noada_r3`); passes 1-2
                   (`cl_noada`, `cl_noada_r2`) appear only in the per-pass figure
Reported baselines: DINO-WM (Zhou et al., 2024) Table 1 (world models) and Table 2 (encoders), on DINO-WM's own
protocol (random_state goals, CEM 30 iterations, MPC); our reproduction of DINO-WM on that protocol comes from
eval_outputs/granular_ref (5 MPC iterations, 50 instances). Rope also gets the frozen evaluation on 91 unseen
instances (heldout + heldout2): theta0 vs the slow learner at the end of pass 1 (Ours-dagger).
DINO-WM's batched run blew up FleX on granular episodes 10 and 19 (final CD ~1850, the same on a rerun of the
chunk); those two are re-planned alone (rerun_blowup.sh) and spliced in, so every arm has all 50 episodes.

The main-text set: main_qualitative (both domains, two examples each, final states with the goal tinted in),
main_table (CD on both domains, with the baselines DINO-WM reports) and main_bars (that table as bars). The
per-domain figures and tables are the appendix.

    /data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_cl_single/paper/make_paper_figs.py

Needs paper/frames/<model>_frozen_ep<e>.npz (resim_frozen.py) for the qualitative figures. Writes
paper/main_{qualitative,table,bars}.{pdf,png}, paper/main_table.tex, paper/<domain>_{qualitative,passes,bars,steps,
paired,table}.{pdf,png}, paper/tables.tex and paper/TABLES.md.
"""
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage, stats  # noqa: E402

HERE = Path(__file__).resolve().parent
CL = HERE.parent
sys.path.insert(0, str(CL))
from analyze import BLOWUP, CHUNKS, SUCCESS_CD, final_cd, start_cd  # noqa: E402
import frozen_analyze  # noqa: E402

spec = importlib.util.spec_from_file_location("granular_ref_analyze", CL.parent / "granular_ref" / "analyze.py")
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)

DOMAINS = {"rope": "dino_ours", "granular": "granular_ref"}
REPRO = {"rope": "rope_on_rope", "granular": "granular_ref"}  # eval_outputs/granular_ref cases
EXAMPLES = {"rope": (21, 2), "granular": (6, 39)}  # stream episodes shown in the qualitative figures
SOLO = {"granular_ref": (10, 19)}  # DINO-WM episodes re-planned alone after FleX blew up (rerun_blowup.sh)
ARMDIR = {"dinowm": "frozen", "adajepa": "adajepa", "ours1": "cl_noada", "ours2": "cl_noada_r2", "ours": "cl_noada_r3"}
METHODS = {"dinowm": "DINO-WM", "adajepa": "AdaJEPA", "ours": "Ours"}  # what the paper compares
PASSES = ("ours1", "ours2", "ours")  # Ours after 1, 2, 3 passes
# DINO-WM (Zhou et al., 2024): Table 1 (world models) and Table 2 (encoders), Chamfer distance (rope, granular)
REPORTED_WM = {"IRIS": (1.11, 0.37), "DreamerV3": (2.49, 1.05), "TD-MPC2": (2.52, 1.21)}
REPORTED_ENC = {"R3M": (1.13, 0.95), "ResNet": (1.08, 0.90), "DINO CLS": (0.84, 0.79)}
REPORTED_DINOWM = (0.41, 0.26)
MAIN_REPORTED = ("TD-MPC2", "DreamerV3", "IRIS", "DINO CLS")  # the reported baselines in the main table

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
C_DINOWM, C_ADA, C_OURS = "#52514e", "#eb6834", "#2a78d6"
COLOR = {"dinowm": C_DINOWM, "adajepa": C_ADA, "ours": C_OURS}
C_REPORTED = "#d5d4d0"
GOOD, BAD = "#0ca30c", "#e34948"
GOAL_TINT = np.array([242, 183, 5])  # the goal's footprint in the rollouts (AdaJEPA's gold star)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
FULL_W = 6.75  # inches, two-column text width


# ------------------------------------------------------------------ data
def step_cd(model, arm):
    """CD before any push and after every MPC iteration (5, 10, 15 pushes): (50, 4), stream order."""
    rows = []
    for k in CHUNKS:
        text = (CL / "logs" / f"{model}_{arm}_chunk_{k}.log").read_text(errors="replace")
        v = np.array([float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", text, re.M)])
        v = v.reshape(3, 10).T if arm == "frozen" else v.reshape(10, 3)  # batched: iteration-major
        assert np.allclose(v[:, -1], final_cd(model, arm, k))
        rows.append(np.column_stack([start_cd(model, k), v]))
    S = np.concatenate(rows)
    if arm == "frozen":  # the blown-up episodes, re-planned alone
        for e in SOLO.get(model, ()):
            text = (CL / "logs" / f"{model}_frozen_solo_ep{e}.log").read_text(errors="replace")
            v = [float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", text, re.M)]
            final = json.loads(open(CL / model / "frozen_solo" / f"ep{e}" / "logs.json").read().splitlines()[-1])
            assert len(v) == 3 and np.isclose(v[-1], final["final_eval/mean_chamfer_distance"])
            S[e, 1:] = v
    return S


def val_ratio(model, arm):
    """Learner / theta0 5-step error on the next episode, before training on it, along the 3 passes (150,)."""
    v = []
    for r in (1, 2, 3):
        d = arm if r == 1 else f"{arm}_r{r}"
        for k in CHUNKS:
            v += [json.loads(l).get("val_ratio") for l in open(CL / model / d / f"chunk_{k}" / "continual_episodes.jsonl")]
    return np.array([np.nan if x is None else x for x in v], float)


def unseen_rope():
    """theta0 vs the slow learner after pass 1 (frozen), on the 91 pooled unseen rope instances."""
    stream_s0 = np.concatenate([frozen_analyze.targets(CL / "stream" / f"chunk_{k}.pkl")[0] for k in CHUNKS])
    rs = [frozen_analyze.load_dir(CL / d) for d in ("heldout", "heldout2")]
    seen, F = [], {"theta0": [], "learner": []}
    for r in rs:  # the pooling of frozen_analyze.pooled: drop starts in the stream and repeats across the two
        for i, x in enumerate(r["s0"]):
            if any(np.array_equal(x, y) for y in stream_s0) or any(np.array_equal(x, y) for y in seen):
                continue
            seen.append(x)
            for a in F:
                F[a].append(r["F"]["dino_ours", a][i])
    return {a: np.array(v) for a, v in F.items()}


def load(domain):
    model = DOMAINS[domain]
    S = {key: step_cd(model, arm) for key, arm in ARMDIR.items()}
    keep = np.all([s[:, -1] < BLOWUP for s in S.values()], axis=0)
    D = {"model": model, "thr": SUCCESS_CD[domain], "keep": keep, "steps": {k: s[keep] for k, s in S.items()}}
    D["final"] = {k: s[:, -1] for k, s in D["steps"].items()}
    D["val"] = val_ratio(model, "cl_noada")
    s0, ev = gr.case_cd("mpc", REPRO[domain])
    D["repro"] = ev[-1]
    D["unseen"] = unseen_rope() if domain == "rope" else None
    return D


# ------------------------------------------------------------------ statistics
def ci95(x):
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def wilson(k, n):
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def wilcoxon_p(a, b):
    return stats.wilcoxon(a - b).pvalue


def mcnemar_p(a, b, thr):
    sa, sb = a < thr, b < thr
    x, y = int((sa & ~sb).sum()), int((~sa & sb).sum())
    return stats.binomtest(min(x, y), x + y, 0.5).pvalue if x + y else 1.0


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


# ------------------------------------------------------------------ tables
def table_rows(domain, D):
    """(group, name, cd, cd_se, sr, d_cd, p_cd, p_sr); None where a cell does not apply."""
    j = 0 if domain == "rope" else 1
    thr = D["thr"]
    rows = []
    g = "Reported by DINO-WM (its protocol)"
    for name, v in {**REPORTED_WM, **REPORTED_ENC}.items():
        rows.append((g, name + (" encoder" if name in REPORTED_ENC else ""), v[j], None, None, None, None, None))
    rows.append((g, "DINO-WM", REPORTED_DINOWM[j], None, None, None, None, None))
    r = D["repro"]
    rows.append((g, "DINO-WM, our reproduction", r.mean(), r.std(ddof=1) / np.sqrt(len(r)), (r < thr).mean(),
                 None, None, None))
    g = f"Online stream, {D['keep'].sum()} instances (our protocol)"
    base = D["final"]["dinowm"]
    for key, name in METHODS.items():
        f = D["final"][key]
        same = key == "dinowm"
        rows.append((g, name, f.mean(), f.std(ddof=1) / np.sqrt(len(f)), (f < thr).mean(),
                     None if same else (f - base).mean(), None if same else wilcoxon_p(f, base),
                     None if same else mcnemar_p(f, base, thr)))
    if D["unseen"] is not None:
        u = D["unseen"]
        g = f"Unseen instances, {len(u['theta0'])} (our protocol, frozen planning)"
        rows.append((g, "DINO-WM", u["theta0"].mean(), u["theta0"].std(ddof=1) / np.sqrt(len(u["theta0"])),
                     (u["theta0"] < thr).mean(), None, None, None))
        f = u["learner"]
        rows.append((g, "Ours†", f.mean(), f.std(ddof=1) / np.sqrt(len(f)), (f < thr).mean(),
                     (f - u["theta0"]).mean(), wilcoxon_p(f, u["theta0"]), mcnemar_p(f, u["theta0"], thr)))
    return rows


def write_tables(all_rows, main_md):
    tex = ["% Generated by eval_outputs/rope_cl_single/paper/make_paper_figs.py. Needs \\usepackage{booktabs}.", ""]
    md = ["# 论文表格", "", "由 `make_paper_figs.py` 生成。", ""] + main_md + ["## 附录：分领域的表", "",
          "CD 越低越好；SR 是成功率（rope：最终 CD < 0.5，granular：最终 CD < 0.2）。"
          "ΔCD 是与同一组实例上的 DINO-WM（θ0，冻结）逐 episode 配对的平均差，p 值：CD 用 Wilcoxon 符号秩检验，"
          "SR 用精确 McNemar 检验。", "",
          "两种协议的数值不能直接比较：DINO-WM 论文的协议是 random_state 目标、CEM 迭代 30 次、MPC；"
          "我们的协议是 50 个实例的流、CEM 迭代 10 次、MPC 3 次 × 5 推。"
          "\"DINO-WM, our reproduction\" 是我们用 DINO-WM 的协议复现的结果（50 个实例、MPC 5 次）。", "",
          "Ours：只用慢学习器，流跑完三轮后的结果（第 2、3 轮重访第 1 轮的实例）。"
          "Ours†：第 1 轮结束时的慢学习器，在从没见过的新实例上冻结规划。", ""]
    for domain, rows in all_rows.items():
        thr = SUCCESS_CD[domain]
        best = min(r[2] for r in rows if r[0].startswith("Online"))
        tex += ["\\begin{table}[t]", "\\centering", "\\small",
                f"\\caption{{\\textbf{{{domain.capitalize()}.}} Chamfer distance (CD, lower is better, mean $\\pm$ s.e.) "
                f"and success rate (SR: final CD $<$ {thr}). $\\Delta$CD: paired difference to DINO-WM on the same "
                "instances (Wilcoxon signed-rank: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$). Numbers under the two "
                "protocols are not directly comparable: DINO-WM's protocol plans toward random goal states with CEM "
                "(30 iterations) and MPC; ours runs a fixed stream of instances with CEM (10 iterations) and MPC "
                "(3 replans of 5 pushes). Ours: the slow learner after three passes over the stream, which revisit "
                "its instances" + ("; Ours$^\\dagger$: the learner after the first pass, planning frozen on instances it "
                                   "never saw" if any("†" in r[1] for r in rows) else "") + ".}",
                f"\\label{{tab:{domain}}}",
                "\\begin{tabular}{lccc}", "\\toprule",
                "Method & CD $\\downarrow$ & SR (\\%) $\\uparrow$ & $\\Delta$CD vs.\\ DINO-WM \\\\", "\\midrule"]
        md += [f"### {domain.capitalize()}", "", "| Method | CD ↓ | SR ↑ | ΔCD vs DINO-WM | p (CD) | p (SR) |",
               "|---|---|---|---|---|---|"]
        group = None
        for g, name, cd, se, sr, d, p, psr in rows:
            if g != group:
                if group is not None:
                    tex.append("\\midrule")
                tex.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{g}}}}} \\\\")
                md.append(f"| *{g}* | | | | | |")
                group = g
            cd_tex = f"{cd:.3f} $\\pm$ {se:.3f}" if se is not None else f"{cd:.2f}"
            if g.startswith("Online") and cd == best:
                cd_tex = f"\\textbf{{{cd:.3f}}} $\\pm$ {se:.3f}"
            sr_tex = f"{sr * 100:.0f}" if sr is not None else "--"
            sup = f"^{{{stars(p)}}}" if d is not None and stars(p) else ""
            d_tex = f"${d:+.3f}{sup}$" if d is not None else "--"  # math mode: a real minus sign
            name_tex = name.replace("†", "$^\\dagger$")
            tex.append(f"\\quad {name_tex} & {cd_tex} & {sr_tex} & {d_tex} \\\\")
            md.append(f"| {name} | {f'{cd:.3f} ± {se:.3f}' if se is not None else f'{cd:.2f}'} | "
                      f"{f'{sr * 100:.0f}%' if sr is not None else '—'} | {f'{d:+.3f}' if d is not None else '—'} | "
                      f"{f'{p:.2g}' if p is not None else '—'} | {f'{psr:.2g}' if psr is not None else '—'} |")
        tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        md.append("")
    (HERE / "tables.tex").write_text("\n".join(tex) + "\n")
    (HERE / "TABLES.md").write_text("\n".join(md) + "\n")


def render_table(domain, rows):
    """A booktabs-style picture of the table, for previewing it outside LaTeX (all positions in axes fractions)."""
    n = len(rows) + len({r[0] for r in rows})
    fig = plt.figure(figsize=(FULL_W, 0.165 * (n + 2)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    T = ax.transAxes
    xs, has = (0.01, 0.52, 0.70, 0.87), ("left", "center", "center", "center")
    dy = 1.0 / (n + 2)
    best = min(r[2] for r in rows if r[0].startswith("Online"))
    y = 1 - dy * 0.15
    ax.plot([0, 1], [y, y], color=INK, lw=1.0, transform=T)
    for x, h, ha in zip(xs, ("Method", "CD ↓ (mean ± s.e.)", f"SR ↑ (CD < {SUCCESS_CD[domain]})",
                             "ΔCD vs. DINO-WM"), has):
        ax.text(x, y - dy * 0.6, h, ha=ha, va="center", fontweight="bold", transform=T)
    y -= dy * 1.2
    ax.plot([0, 1], [y, y], color=INK, lw=0.6, transform=T)
    group = None
    for g, name, cd, se, sr, d, p, _ in rows:
        if g != group:
            if group is not None:
                ax.plot([0, 1], [y, y], color=MUTED, lw=0.4, transform=T)
            ax.text(xs[0], y - dy * 0.5, g, style="italic", color=INK2, va="center", transform=T)
            y -= dy
            group = g
        bold = g.startswith("Online") and cd == best
        cells = ("    " + name, f"{cd:.3f} ± {se:.3f}" if se is not None else f"{cd:.2f}",
                 f"{sr * 100:.0f}%" if sr is not None else "—", f"{d:+.3f}{stars(p)}" if d is not None else "—")
        for j, (x, c, ha) in enumerate(zip(xs, cells, has)):
            ax.text(x, y - dy * 0.5, c, ha=ha, va="center", transform=T,
                    fontweight="bold" if bold and j == 1 else "normal")
        y -= dy
    ax.plot([0, 1], [y, y], color=INK, lw=1.0, transform=T)
    fig.savefig(HERE / f"{domain}_table.pdf", bbox_inches=None, metadata={"CreationDate": None})
    fig.savefig(HERE / f"{domain}_table.png", bbox_inches=None)
    plt.close(fig)


def main_rows(Ds):
    """The main table: (key, name, {domain: (cd, se or None)}), worst to best by the mean CD over the domains. The
    baselines in MAIN_REPORTED are DINO-WM's numbers (se None); the methods in METHODS are ours."""
    rows = [(None, n, {d: ({**REPORTED_WM, **REPORTED_ENC}[n][j], None) for j, d in enumerate(Ds)})
            for n in MAIN_REPORTED]
    for key, name in METHODS.items():
        rows.append((key, name, {d: (D["final"][key].mean(), D["final"][key].std(ddof=1) / np.sqrt(len(D["final"][key])))
                                 for d, D in Ds.items()}))
    return sorted(rows, key=lambda r: -np.mean([v[0] for v in r[2].values()]))


def write_main_table(Ds, rows):
    n = {d: len(D["final"]["ours"]) for d, D in Ds.items()}
    best = {d: min(r[2][d][0] for r in rows) for d in Ds}
    tex = ["% Generated by eval_outputs/rope_cl_single/paper/make_paper_figs.py. Needs \\usepackage{booktabs}.",
           "\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{Planning on deformable objects: final Chamfer distance (CD, lower is better) between the "
           "object and the goal after 15 pushes, mean $\\pm$ s.e. over " +
           " and ".join(f"{n[d]} {d} instances" for d in Ds) + ". Ours: the slow learner after three passes over "
           "the instance stream.}",
           "\\label{tab:main}", "\\begin{tabular}{l" + "c" * len(Ds) + "}", "\\toprule",
           "Method & " + " & ".join(f"{d.capitalize()} CD $\\downarrow$" for d in Ds) + " \\\\", "\\midrule"]
    md = ["## 主表（正文）", "", "最终 Chamfer 距离（CD，越低越好），按两个任务的平均 CD 从高到低排。"
          "TD-MPC2、DreamerV3、DINO CLS、IRIS 的数取自 DINO-WM 论文（Table 1、2）；其余是我们测的，均值 ± 标准误，"
          + "、".join(f"{d} {n[d]} 个实例" for d in Ds) + "。", "",
          "| Method | " + " | ".join(f"{d.capitalize()} CD ↓" for d in Ds) + " |", "|---" * (len(Ds) + 1) + "|"]
    for key, name, v in rows:
        if key == "ours":
            tex.append("\\midrule")
        cells = []
        for d in Ds:
            cd, se = v[d]
            c = f"\\textbf{{{cd:.2f}}}" if cd == best[d] else f"{cd:.2f}"
            cells.append(c + (f"{{\\scriptsize\\,$\\pm$\\,{se:.2f}}}" if se is not None else ""))
        tex.append(name + " & " + " & ".join(cells) + " \\\\")
        md.append(f"| {name} | " + " | ".join(
            (f"**{v[d][0]:.2f}**" if v[d][0] == best[d] else f"{v[d][0]:.2f}") +
            (f" ± {v[d][1]:.2f}" if v[d][1] is not None else "") for d in Ds) + " |")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (HERE / "main_table.tex").write_text("\n".join(tex) + "\n")
    return md + [""]


def render_main_table(Ds, rows):
    """A booktabs-style picture of the main table (positions in axes fractions)."""
    fig = plt.figure(figsize=(FULL_W * 0.55, 0.19 * (len(rows) + 1.6)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    T = ax.transAxes
    xs = (0.03, 0.56, 0.84)
    dy = 1.0 / (len(rows) + 1.6)
    best = {d: min(r[2][d][0] for r in rows) for d in Ds}
    y = 1 - dy * 0.2
    ax.plot([0, 1], [y, y], color=INK, lw=1.0, transform=T)
    for x, h, ha in zip(xs, ["Method"] + [f"{d.capitalize()} CD ↓" for d in Ds], ("left", "center", "center")):
        ax.text(x, y - dy * 0.6, h, ha=ha, va="center", fontweight="bold", transform=T)
    y -= dy * 1.2
    ax.plot([0, 1], [y, y], color=INK, lw=0.6, transform=T)
    for key, name, v in rows:
        if key == "ours":
            ax.plot([0, 1], [y, y], color=INK, lw=0.6, transform=T)
        ax.text(xs[0], y - dy * 0.5, name, va="center", transform=T)
        for x, d in zip(xs[1:], Ds):
            cd, se = v[d]
            ax.text(x, y - dy * 0.5, f"{cd:.2f}" + (f" ± {se:.2f}" if se is not None else ""), ha="center",
                    va="center", transform=T, fontweight="bold" if cd == best[d] else "normal")
        y -= dy
    ax.plot([0, 1], [y, y], color=INK, lw=1.0, transform=T)
    fig.savefig(HERE / "main_table.pdf", bbox_inches=None, metadata={"CreationDate": None})
    fig.savefig(HERE / "main_table.png", bbox_inches=None)
    plt.close(fig)


# ------------------------------------------------------------------ figures
def save(fig, name):  # no creation date in the PDF: redrawing an unchanged figure leaves the file byte-identical
    fig.savefig(HERE / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(HERE / f"{name}.png")
    plt.close(fig)


def ref_line(ax, y, label, color, ls, above):
    ax.axhline(y, color=color, ls=ls, lw=0.9, zorder=1)
    ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 2 if above else -2),
                textcoords="offset points", ha="right", va="bottom" if above else "top", fontsize=6.5, color=color)


def fig_passes(domain, D):
    """Ours over the three passes: prediction error along the stream, final CD and success per pass."""
    thr = D["thr"]
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.05), gridspec_kw={"width_ratios": [1.6, 1, 1], "wspace": 0.42})
    ax = axs[0]
    v = D["val"]
    x = np.arange(len(v))
    roll = np.array([np.exp(np.nanmean(np.log(v[max(0, i - 9): i + 1]))) if i >= 4 else np.nan for i in range(len(v))])
    ax.scatter(x, v, s=4, color=C_OURS, alpha=0.3, lw=0, zorder=2)
    ax.plot(x, roll, color=C_OURS, lw=1.5, zorder=3)
    for b in (50, 100):
        ax.axvline(b - 0.5, color=GRID, lw=0.8, zorder=0)
    for k in (1, 2, 3):
        ax.text(50 * k - 25, 1.0, f"pass {k}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.5, color=INK2)
    ref_line(ax, 1.0, "DINO-WM (θ0)", C_DINOWM, "--", True)
    ax.set_xlim(-1, 150)
    ax.set_ylim(0, 1.25 if domain == "rope" else 1.15)
    ax.set_xlabel("episode in the stream")
    ax.set_ylabel("prediction error / θ0's")
    ax.set_title("(a) Ours: prediction error ↓", loc="left", pad=11)
    ax.grid(axis="y")
    for ax, metric in ((axs[1], "cd"), (axs[2], "sr")):
        m, lo, hi = [], [], []
        for key in PASSES:
            f = D["final"][key]
            if metric == "cd":
                c = ci95(f)
                m.append(f.mean()), lo.append(f.mean() - c), hi.append(f.mean() + c)
            else:
                k = int((f < thr).sum())
                a, b = wilson(k, len(f))
                m.append(100 * k / len(f)), lo.append(100 * a), hi.append(100 * b)
        x = np.arange(1, 4)
        ax.vlines(x, lo, hi, color=C_OURS, lw=1.8, alpha=0.5)
        ax.plot(x, m, "-o", color=C_OURS, ms=4.5, lw=1.4, mec="white", mew=0.6, zorder=3)
        f0, fa = D["final"]["dinowm"], D["final"]["adajepa"]
        y0, ya = (f0.mean(), fa.mean()) if metric == "cd" else (100 * (f0 < thr).mean(), 100 * (fa < thr).mean())
        ref_line(ax, y0, "DINO-WM", C_DINOWM, "--", y0 >= ya)
        ref_line(ax, ya, "AdaJEPA", C_ADA, ":", ya > y0)
        ax.set_xticks([1, 2, 3], [f"{k}\n{mi:.2f}" if metric == "cd" else f"{k}\n{mi:.0f}%"  # pass, then its value
                                  for k, mi in zip((1, 2, 3), m)])
        ax.set_xlim(0.6, 3.4)
        ax.set_xlabel("pass over the stream")
        ax.grid(axis="y")
        if metric == "cd":
            ax.set_ylabel("final CD (95% CI)")
            ax.set_title("(b) Ours: final CD ↓", loc="left", pad=11)
        else:
            ax.set_ylabel(f"success rate, CD < {thr} (%)")
            ax.set_title("(c) Ours: success rate ↑", loc="left", pad=11)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))
    save(fig, f"{domain}_passes")


def bar_group(ax, xs, vals, errs, colors, hatches, labels, fmt):
    """Bars with value labels; the tick labels pile up on the axes (ax._bar_ticks) and are set by set_bar_ticks."""
    for x, v, e, c, h in zip(xs, vals, errs, colors, hatches):
        ax.bar(x, v, width=0.72, color=c, edgecolor="white" if not h else INK2, linewidth=0 if not h else 0.4,
               hatch=h, zorder=2)
        if e is not None:
            ax.errorbar(x, v, yerr=np.array(e).reshape(2, 1), fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6,
                        zorder=3)
        top = v + (e[1] if e is not None else 0)
        ax.text(x, top, fmt(v), ha="center", va="bottom", fontsize=6, color=INK2,
                transform=ax.transData, clip_on=False)
    ticks = getattr(ax, "_bar_ticks", ([], []))
    ax._bar_ticks = (ticks[0] + list(xs), ticks[1] + list(labels))


def set_bar_ticks(ax):
    ax.set_xticks(ax._bar_ticks[0], ax._bar_ticks[1], rotation=40, ha="right", rotation_mode="anchor")


def fig_bars(domain, D):
    j = 0 if domain == "rope" else 1
    thr = D["thr"]
    rep = {**REPORTED_WM, **REPORTED_ENC, "DINO-WM": REPORTED_DINOWM}
    ours = list(METHODS)
    col, lab = COLOR, METHODS
    wide = D["unseen"] is not None
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.35),
                            gridspec_kw={"width_ratios": [2.0 if wide else 1.8, 1], "wspace": 0.22})
    # (a) CD: reported (DINO-WM's protocol) | our stream | unseen
    ax = axs[0]
    xs, x = [], 0
    names = list(rep)
    xr = np.arange(len(names))
    bar_group(ax, xr, [rep[n][j] for n in names], [None] * len(names),
              [C_REPORTED] * (len(names) - 1) + ["#a9a8a4"], ["////"] * len(names), names, lambda v: f"{v:.2f}")
    x0 = len(names) + 0.8
    xo = x0 + np.arange(len(ours))
    errs = []
    for k in ours:
        f = D["final"][k]
        errs.append((ci95(f), ci95(f)))
    bar_group(ax, xo, [D["final"][k].mean() for k in ours], errs, [col[k] for k in ours], [None] * len(ours),
              [lab[k] for k in ours], lambda v: f"{v:.2f}")
    seps = [(xr[0] - 0.5, xr[-1] + 0.5, "reported by DINO-WM\n(its protocol)"),
            (xo[0] - 0.5, xo[-1] + 0.5, f"online stream\n(n = {D['keep'].sum()})")]
    if wide:
        u = D["unseen"]
        xu = xo[-1] + 2.2 + np.arange(2)
        bar_group(ax, xu, [u["theta0"].mean(), u["learner"].mean()],
                  [(ci95(u["theta0"]),) * 2, (ci95(u["learner"]),) * 2], [C_DINOWM, C_OURS], [None, None],
                  ["DINO-WM", "Ours†"], lambda v: f"{v:.2f}")
        seps.append((xu[0] - 0.5, xu[-1] + 0.5, f"unseen\n(n = {len(u['theta0'])})"))
    for a, b, t in seps:
        ax.annotate("", xy=(a + 0.1, 1.02), xytext=(b - 0.1, 1.02), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ax.text((a + b) / 2, 1.04, t, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.3,
                color=INK2, linespacing=1.1)
    for a, b in zip(seps[:-1], seps[1:]):
        ax.axvline((a[1] + b[0]) / 2, color=GRID, lw=0.8, zorder=0)
    set_bar_ticks(ax)
    ax.set_ylabel("final Chamfer distance ↓")
    ax.set_title(f"(a) {domain.capitalize()}: Chamfer distance", loc="left", pad=26)
    ax.grid(axis="y", zorder=0)
    ax.set_xlim(xr[0] - 0.7, (xu[-1] if wide else xo[-1]) + 0.7)
    # (b) success rate: our protocol only (DINO-WM's paper reports CD only)
    ax = axs[1]
    vals, errs = [], []
    for k in ours:
        f = D["final"][k]
        s = int((f < thr).sum())
        a, b = wilson(s, len(f))
        vals.append(100 * s / len(f))
        errs.append((vals[-1] - 100 * a, 100 * b - vals[-1]))
    xs2 = np.arange(len(ours))
    labels2 = [lab[k] for k in ours]
    cols2 = [col[k] for k in ours]
    if wide:
        u = D["unseen"]
        for f, c in ((u["theta0"], C_DINOWM), (u["learner"], C_OURS)):
            s = int((f < thr).sum())
            a, b = wilson(s, len(f))
            vals.append(100 * s / len(f))
            errs.append((vals[-1] - 100 * a, 100 * b - vals[-1]))
            cols2.append(c)
        xs2 = np.concatenate([xs2, xs2[-1] + 1.8 + np.arange(2)])
        labels2 += ["DINO-WM", "Ours†"]
        ax.axvline(len(ours) + 0.15, color=GRID, lw=0.8, zorder=0)
        ax.text((xs2[-2] + xs2[-1]) / 2, 1.04, "unseen", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6.3, color=INK2)
    ax.text((xs2[0] + xs2[len(ours) - 1]) / 2, 1.04, "online stream", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6.3, color=INK2)
    bar_group(ax, xs2, vals, errs, cols2, [None] * len(vals), labels2, lambda v: f"{v:.0f}")
    set_bar_ticks(ax)
    ax.set_ylabel(f"success rate, CD < {thr} (%) ↑")
    ax.set_title("(b) Success rate", loc="left", pad=26)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", zorder=0)
    save(fig, f"{domain}_bars")


def fig_main_bars(Ds, rows):
    """The main table as bars, one panel per domain, each from worst to best; ours with 95% CIs."""
    fig, axs = plt.subplots(1, len(Ds), figsize=(FULL_W, 2.1), gridspec_kw={"wspace": 0.18})
    for n, (ax, (domain, D)) in enumerate(zip(axs, Ds.items())):
        rs = sorted(rows, key=lambda r: -r[2][domain][0])
        vals = [r[2][domain][0] for r in rs]
        errs = [(ci95(D["final"][k]),) * 2 if k else None for k, _, _ in rs]
        bar_group(ax, np.arange(len(rs)), vals, errs, [COLOR[k] if k else C_REPORTED for k, _, _ in rs],
                  [None] * len(rs), [name for _, name, _ in rs], lambda v: f"{v:.2f}")
        set_bar_ticks(ax)
        ax.set_xlim(-0.6, len(rs) - 0.4)
        ax.set_ylabel("final Chamfer distance ↓" if n == 0 else "")
        ax.set_title(f"({'ab'[n]}) {domain.capitalize()}", loc="left")
        ax.grid(axis="y", zorder=0)
    save(fig, "main_bars")


def fig_steps(domain, D):
    thr = D["thr"]
    show = [("dinowm", C_DINOWM, "--", "DINO-WM"), ("adajepa", C_ADA, ":", "AdaJEPA"), ("ours", C_OURS, "-", "Ours")]
    pushes = np.array([0, 5, 10, 15])
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W * 0.78, 2.0), gridspec_kw={"wspace": 0.32})
    for key, color, ls, label in show:
        s = D["steps"][key]
        m = s.mean(0)
        c = np.array([ci95(s[:, t]) for t in range(4)])
        axs[0].fill_between(pushes, m - c, m + c, color=color, alpha=0.12, lw=0)
        axs[0].plot(pushes, m, ls=ls, marker="o", ms=3, color=color, lw=1.3, label=label)
        sr = 100 * (s < thr).mean(0)
        ci = np.array([wilson(int((s[:, t] < thr).sum()), len(s)) for t in range(4)]) * 100
        axs[1].fill_between(pushes, ci[:, 0], ci[:, 1], color=color, alpha=0.12, lw=0)
        axs[1].plot(pushes, sr, ls=ls, marker="o", ms=3, color=color, lw=1.3, label=label)
    for ax in axs:
        ax.set_xticks(pushes)
        ax.set_xlabel("pushes executed (MPC replans every 5)")
        ax.grid(axis="y")
    axs[0].set_ylabel("Chamfer distance ↓")
    axs[0].set_title("(a) CD within an episode", loc="left")
    axs[1].set_ylabel(f"success rate, CD < {thr} (%) ↑")
    axs[1].set_title("(b) Success within an episode", loc="left")
    axs[1].set_ylim(bottom=0)
    axs[0].legend(loc="upper right", handlelength=2.2)  # CD falls from the left, so this corner stays empty
    save(fig, f"{domain}_steps")


def fig_paired(domain, D):
    """Final CD per instance: Ours against DINO-WM and against AdaJEPA, same instance and seed."""
    thr = D["thr"]
    f = D["final"]["ours"]
    log = domain == "rope"  # rope's final CDs span two orders of magnitude, granular's less than one
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W * 0.62, 2.45), gridspec_kw={"wspace": 0.38})
    vals = np.concatenate([D["final"][k] for k in METHODS])
    lo, hi = (vals.min() * 0.75, vals.max() * 1.3) if log else (0, vals.max() * 1.05)
    for n, (ax, key) in enumerate(zip(axs, ("dinowm", "adajepa"))):
        base = D["final"][key]
        better = f < base
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.7, zorder=1)
        ax.axhline(thr, color=GRID, lw=0.8, zorder=0)
        ax.axvline(thr, color=GRID, lw=0.8, zorder=0)
        ax.scatter(base[better], f[better], s=11, color=C_OURS, lw=0.4, ec="white", zorder=3, label="Ours lower")
        ax.scatter(base[~better], f[~better], s=11, color=COLOR[key], lw=0.4, ec="white", zorder=3,
                   label=f"{METHODS[key]} lower")
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            t = [x for x in (0.1, 0.2, 0.5, 1, 2) if lo <= x <= hi]
            for setter in (ax.set_xticks, ax.set_yticks):
                setter(t, [f"{x:g}" for x in t])
            ax.minorticks_off()
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{METHODS[key]} final CD")
        ax.set_ylabel("Ours final CD")
        ax.set_title(f"({'ab'[n]}) Ours vs. {METHODS[key]}", loc="left")
        ax.text(0.03, 0.97, f"Ours lower on {better.sum()} / {len(f)}\nWilcoxon p = {wilcoxon_p(f, base):.2g}",
                transform=ax.transAxes, va="top", fontsize=6.5, color=INK2)
    h = [Line2D([], [], ls="", marker="o", ms=4, color=COLOR[k], mec="white", mew=0.4) for k in METHODS]
    order = [2, 0, 1]
    fig.legend([h[i] for i in order], [f"{list(METHODS.values())[i]} lower" for i in order], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.04), handletextpad=0.2)
    fig.subplots_adjust(bottom=0.22)
    save(fig, f"{domain}_paired")


def frames(model, key, e):
    if key == "dinowm":
        return np.load(HERE / "frames" / f"{model}_frozen_ep{e}.npz")["visual"]
    k, i = divmod(e, 10)
    f = np.load(CL / model / ARMDIR[key] / f"chunk_{k}" / "episode_frames.npz")
    eps = sorted({x.split("_")[0] for x in f.files}, key=lambda s: int(s[2:]))  # adajepa restarts at ep0 per chunk
    return f[eps[i] + "_visual"]


def goal_image(model, e):
    k, i = divmod(e, 10)
    t = pickle.load(open(CL / model / "frozen" / f"chunk_{k}" / "plan_targets.pkl", "rb"))
    return np.clip(t["obs_g"]["visual"][i, 0], 0, 255).astype(np.uint8)


def object_mask(im, domain):
    """The object's pixels: rope is red, granular blue."""
    im = im.astype(int)
    return (im[..., 0] - im[..., 1] > 40) if domain == "rope" else (im[..., 2] - im[..., 0] > 40)


def object_crop(imgs, domain, aspect, pad=16, min_w=130):
    """One box around the object in every image shown, padded, widened to h / w = aspect."""
    mask = np.any([object_mask(im, domain) for im in imgs], axis=0)
    ys, xs = np.where(mask)
    H, W = mask.shape
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    w = max(xs.max() - xs.min() + 2 * pad, (ys.max() - ys.min() + 2 * pad) / aspect, min_w)
    w = min(w, W, H / aspect)
    h = w * aspect
    x0 = int(round(min(max(cx - w / 2, 0), W - w)))
    y0 = int(round(min(max(cy - h / 2, 0), H - h)))
    return slice(y0, y0 + int(round(h))), slice(x0, x0 + int(round(w)))


def fig_qualitative(domain, D):
    """Real FleX frames. Rows: methods; columns: the start, every 3rd push, and the goal. The final frame carries the
    run's final CD. DINO-WM's frames are replayed from its executed actions (resim_frozen.py)."""
    model, thr = D["model"], D["thr"]
    rows = list(METHODS.items())
    cols = [0, 3, 6, 9, 12, 15]
    aspect = 0.8  # cell height / width
    cw, gap, ggap, lab, ttl, ctl, bgap = 0.88, 0.035, 0.09, 0.34, 0.2, 0.15, 0.12  # inches
    ch = cw * aspect
    ncol = len(cols) + 1
    eps = EXAMPLES[domain]
    block = ttl + ctl + len(rows) * ch + (len(rows) - 1) * gap
    fw = lab + ncol * cw + (ncol - 1) * gap + ggap + 0.04
    fh = len(eps) * block + (len(eps) - 1) * bgap + 0.04
    fig = plt.figure(figsize=(fw, fh))
    kept = np.arange(50)[D["keep"]]
    top = fh - 0.02
    for n, e in enumerate(eps):
        idx = int(np.where(kept == e)[0][0])  # position among the kept episodes
        V = {key: frames(model, key, e) for key, _ in rows}
        g = goal_image(model, e)
        crop = object_crop([V[k][t] for k in V for t in cols] + [g], domain, aspect)
        fig.text(0.02 / fw, (top - ttl / 2) / fh,
                 f"({'ab'[n]}) {domain.capitalize()}, example {n + 1}: start CD {D['steps']['dinowm'][idx, 0]:.2f}",
                 fontsize=7.5, fontweight="bold", va="center")
        y = top - ttl - ctl
        for r, (key, label) in enumerate(rows):
            fcd = D["final"][key][idx]
            ok = fcd < thr
            for c in range(ncol):
                x = lab + c * (cw + gap) + (ggap if c == ncol - 1 else 0)
                ax = fig.add_axes([x / fw, (y - ch) / fh, cw / fw, ch / fh])
                is_goal = c == ncol - 1
                ax.imshow((g if is_goal else V[key][cols[c]])[crop], interpolation="lanczos", aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
                color, lw = (INK, 0.9) if is_goal else ((GOOD if ok else BAD), 1.2) if c == len(cols) - 1 else (MUTED, 0.5)
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(color)
                    sp.set_linewidth(lw)
                if r == 0:
                    t = "goal" if is_goal else "start" if cols[c] == 0 else f"push {cols[c]}"
                    ax.set_title(t, fontsize=7, fontweight="bold" if is_goal else "normal", pad=2)
                if c == 0:
                    ax.set_ylabel(label, fontsize=7.5, fontweight="bold", labelpad=3)
                if c == len(cols) - 1:
                    ax.text(0.03, 0.04, f"CD {fcd:.2f} {'✓' if ok else '✗'}", transform=ax.transAxes, fontsize=6.3,
                            color=GOOD if ok else BAD, fontweight="bold", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            y -= ch + gap
        top -= block + bgap
    fig.savefig(HERE / f"{domain}_qualitative.pdf", bbox_inches=None, metadata={"CreationDate": None})
    fig.savefig(HERE / f"{domain}_qualitative.png", bbox_inches=None)
    plt.close(fig)


def with_goal(im, goal, domain, alpha=0.6):
    """The frame with the goal's footprint tinted in gold wherever the object does not cover it."""
    m = ndimage.binary_dilation(object_mask(goal, domain), iterations=2) & ~object_mask(im, domain)
    out = im.astype(float)
    out[m] = (1 - alpha) * out[m] + alpha * GOAL_TINT
    return out.round().astype(np.uint8)


def badge(ax, ok):
    ax.text(0.86, 0.86, "✓" if ok else "✗", transform=ax.transAxes, ha="center", va="center", fontsize=6,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.2", fc=GOOD if ok else BAD, ec="white", lw=0.7))


def fig_main_qualitative(Ds):
    """Main-text rollouts, after AdaJEPA's Figure 4: one row per domain with its two examples; per example the start
    and the final states of DINO-WM, AdaJEPA and Ours, the goal's footprint tinted in gold, a success badge on the
    final states. Whole rendered frames, uncropped, so the table and the arm are in view. Same instances as
    <domain>_qualitative."""
    panels = [("start", "Start")] + list(METHODS.items())
    side, gap, ggap, cap, rgap = 0.17, 0.03, 0.12, 0.25, 0.06  # inches
    n_ex = max(len(e) for e in EXAMPLES.values())
    cw = (FULL_W - side - (n_ex - 1) * ggap - n_ex * (len(panels) - 1) * gap) / (n_ex * len(panels))
    fh = len(Ds) * (cw + cap) + (len(Ds) - 1) * rgap
    fig = plt.figure(figsize=(FULL_W, fh))
    letters = iter("abcdefgh")
    y = fh - cw
    for domain, D in Ds.items():
        model, thr = D["model"], D["thr"]
        fig.text(0, (y + cw / 2) / fh, domain.capitalize(), rotation=90, ha="left", va="center", fontsize=8,
                 fontweight="bold")
        kept = np.arange(50)[D["keep"]]
        for n, e in enumerate(EXAMPLES[domain]):
            letter = next(letters)
            idx = int(np.where(kept == e)[0][0])
            g = goal_image(model, e)
            imgs = {"start": frames(model, "ours", e)[0], **{k: frames(model, k, e)[-1] for k in METHODS}}
            x0 = side + n * (len(panels) * (cw + gap) - gap + ggap)
            for j, (key, name) in enumerate(panels):
                ax = fig.add_axes([(x0 + j * (cw + gap)) / FULL_W, y / fh, cw / FULL_W, cw / fh])
                ax.imshow(with_goal(imgs[key], g, domain), interpolation="lanczos")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(MUTED)
                    sp.set_linewidth(0.5)
                cd = D["steps"]["dinowm"][idx, 0] if key == "start" else D["final"][key][idx]
                if key != "start":
                    badge(ax, cd < thr)
                ax.annotate(f"({letter}{j + 1}) {name}", (0.5, 0), xycoords="axes fraction", xytext=(0, -2.5),
                            textcoords="offset points", ha="center", va="top", fontsize=6.4, fontweight="bold")
                ax.annotate(f"CD {cd:.2f}", (0.5, 0), xycoords="axes fraction", xytext=(0, -10),
                            textcoords="offset points", ha="center", va="top", fontsize=6.2, color=INK2)
        y -= cw + cap + rgap
    fig.savefig(HERE / "main_qualitative.pdf", bbox_inches=None, metadata={"CreationDate": None})
    fig.savefig(HERE / "main_qualitative.png", bbox_inches=None)
    plt.close(fig)


def export_data(domain, D, rows):
    """The numbers behind every figure, as CSV (paper/data/), so the plots can be redrawn in any tool."""
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    eps = np.arange(50)[D["keep"]]
    with open(out / f"{domain}_final_cd.csv", "w") as f:
        f.write("instance,start_cd," + ",".join(METHODS) + "\n")
        for n, e in enumerate(eps):
            f.write(f"{e},{D['steps']['dinowm'][n, 0]:.6f}," + ",".join(f"{D['final'][k][n]:.6f}" for k in METHODS)
                    + "\n")
    with open(out / f"{domain}_cd_by_push.csv", "w") as f:
        f.write("instance,method,pushes,cd\n")
        for k in METHODS:
            for n, e in enumerate(eps):
                for t, pushes in enumerate((0, 5, 10, 15)):
                    f.write(f"{e},{k},{pushes},{D['steps'][k][n, t]:.6f}\n")
    with open(out / f"{domain}_ours_by_pass.csv", "w") as f:
        f.write("instance,pass1,pass2,pass3\n")
        for n, e in enumerate(eps):
            f.write(f"{e}," + ",".join(f"{D['final'][k][n]:.6f}" for k in PASSES) + "\n")
    with open(out / f"{domain}_prediction_error.csv", "w") as f:
        f.write("stream_episode,pass,ours\n")
        for e in range(150):
            f.write(f"{e},{e // 50 + 1},{D['val'][e]:.6f}\n")
    with open(out / f"{domain}_table.csv", "w") as f:
        f.write("group,method,cd,cd_se,success_rate,delta_cd_vs_dinowm,p_wilcoxon_cd,p_mcnemar_success\n")
        for r in rows:
            f.write(",".join(f'"{x}"' if isinstance(x, str) else "" if x is None else f"{x:.6g}" for x in r) + "\n")
    if D["unseen"] is not None:
        with open(out / f"{domain}_unseen_final_cd.csv", "w") as f:
            f.write("instance,dinowm,ours_dagger\n")
            for n, (a, b) in enumerate(zip(D["unseen"]["theta0"], D["unseen"]["learner"])):
                f.write(f"{n},{a:.6f},{b:.6f}\n")


def export_frames(domain, D):
    """The raw 224 x 224 FleX frames of the qualitative examples, one PNG each (paper/raw_frames/)."""
    out = HERE / "raw_frames"
    out.mkdir(exist_ok=True)
    for n, e in enumerate(EXAMPLES[domain]):
        plt.imsave(out / f"{domain}_example{n + 1}_goal.png", goal_image(D["model"], e))
        for key in METHODS:
            for t, img in enumerate(frames(D["model"], key, e)):
                plt.imsave(out / f"{domain}_example{n + 1}_{key}_push{t:02d}.png", img)


MAIN_FIGS = [("main_qualitative", "Real-simulation rollouts (main text)"), ("main_table", "Main table preview"),
             ("main_bars", "Main table as bars")]
FIGS = [("qualitative", "Real-simulation rollouts"), ("passes", "Prediction error and results over passes"),
        ("bars", "Table as bars"), ("steps", "CD and success within an episode"), ("paired", "Per-instance paired"),
        ("table", "Table preview")]


def write_index():
    """index.html: every figure with a preview and download links; works from the unpacked folder."""
    css = ("body{font:14px/1.5 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;margin:0;background:#fafaf8;"
           "color:#0b0b0b}main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}"
           "h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid #e4e3df;padding-bottom:4px}"
           "figure{margin:16px 0;background:#fff;border:1px solid #e4e3df;border-radius:6px;padding:12px}"
           "figure img{width:100%;height:auto}figcaption{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;"
           "margin-top:6px}figcaption b{margin-right:auto}a{color:#256abf}ul{padding-left:20px}"
           ".muted{color:#52514e}")
    L = ["<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'><title>Paper Figures</title>",
         f"<style>{css}</style></head><body><main>",
         "<h1>单领域持续学习：论文图表</h1>",
         "<p class='muted'>正文三项在前，附录按 rope、granular 分开。每张图：PDF（矢量，放进 LaTeX）和 PNG（300 dpi）。"
         "说明和图注草稿见 <a href='README.md'>README.md</a>。</p>",
         "<h2>正文</h2>"]
    for stem, title in MAIN_FIGS:
        if (HERE / f"{stem}.png").exists():
            L.append(f"<figure><img src='{stem}.png' alt='{title}' loading='lazy'><figcaption><b>{title}</b>"
                     f"<a href='{stem}.pdf' download>PDF</a><a href='{stem}.png' download>PNG</a></figcaption></figure>")
    L += ["<p>主表 LaTeX：<a href='main_table.tex'>main_table.tex</a></p>",
          "<h2>表格（附录）</h2><ul><li><a href='tables.tex'>tables.tex</a>（LaTeX，booktabs，分领域）</li>"
          "<li><a href='TABLES.md'>TABLES.md</a>（Markdown，带两个 p 值）</li>"
          "<li>数据：" + " · ".join(f"<a href='data/{p.name}'>{p.name}</a>" for p in sorted((HERE / "data").glob("*.csv")))
          + "</li></ul>"]
    for domain in DOMAINS:
        L.append(f"<h2>附录：{domain.capitalize()}</h2>")
        for name, title in FIGS:
            stem = f"{domain}_{name}"
            if not (HERE / f"{stem}.png").exists():
                continue
            L.append(f"<figure><img src='{stem}.png' alt='{title}' loading='lazy'><figcaption><b>{title}</b>"
                     f"<a href='{stem}.pdf' download>PDF</a><a href='{stem}.png' download>PNG</a></figcaption></figure>")
    L.append("<h2>原始仿真帧</h2><p>定性图里每一帧的原图（224 × 224），在 <code>raw_frames/</code> 文件夹，"
             "文件名为 <code>&lt;领域&gt;_example&lt;n&gt;_&lt;方法&gt;_push&lt;t&gt;.png</code>。</p>")
    L.append("<h2>脚本</h2><ul><li><a href='make_paper_figs.py'>make_paper_figs.py</a></li>"
             "<li><a href='resim_frozen.py'>resim_frozen.py</a></li>"
             "<li><a href='rerun_blowup.sh'>rerun_blowup.sh</a></li></ul></main></body></html>")
    (HERE / "index.html").write_text("\n".join(L) + "\n")


def main():
    all_rows, Ds = {}, {}
    for domain in DOMAINS:
        D = Ds[domain] = load(domain)
        print(domain, "episodes kept:", D["keep"].sum())
        all_rows[domain] = table_rows(domain, D)
        render_table(domain, all_rows[domain])
        fig_passes(domain, D)
        fig_bars(domain, D)
        fig_steps(domain, D)
        fig_paired(domain, D)
        export_data(domain, D, all_rows[domain])
        if all((HERE / "frames" / f"{D['model']}_frozen_ep{e}.npz").exists() for e in EXAMPLES[domain]):
            fig_qualitative(domain, D)
            export_frames(domain, D)
        else:
            print(domain, ": frozen frames missing, run resim_frozen.py first")
    rows = main_rows(Ds)
    main_md = write_main_table(Ds, rows)
    render_main_table(Ds, rows)
    fig_main_bars(Ds, rows)
    if all((HERE / "frames" / f"{D['model']}_frozen_ep{e}.npz").exists() for d, D in Ds.items() for e in EXAMPLES[d]):
        fig_main_qualitative(Ds)
    write_tables(all_rows, main_md)
    write_index()
    print((HERE / "TABLES.md").read_text())


if __name__ == "__main__":
    main()
