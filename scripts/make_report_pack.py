"""Offline report pack: Markdown + PNG figures + the standalone HTML page, zipped.

Reads the page data written by build_report_page.py --dump-json, so every number
matches the web report. Figures use a Simplified-Chinese face extracted from the
system Noto Sans CJK collection (or --font).

    python scripts/make_report_pack.py --data reports/pushobj_bias_16k/report_data.json \
        --html reports/pushobj_bias_16k/report.html --out reports/pushobj_bias_16k/advisor_pack
"""
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = {"frozen": "#2a78d6", "adajepa": "#eb6834", "released": "#1baf7a", "prev": "#eda100", "new": "#7454d1",
     "paper": "#8a909c", "ink": "#151922", "muted": "#767d8c", "grid": "#e4e6eb"}
SHAPES = ["T", "L", "Z", "+", "I", "small_tee", "square"]
LABEL = {"T": "T", "L": "L", "Z": "Z", "+": "+", "I": "I★", "small_tee": "small T★", "square": "square★"}


def setup_font(path):
    if path is None:
        path = Path(tempfile.gettempdir()) / "NotoSansCJKsc-Regular.otf"
        if not path.exists():
            from fontTools.ttLib import TTCollection
            for f in TTCollection("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").fonts:
                if f["name"].getDebugName(1) == "Noto Sans CJK SC":
                    f.save(str(path))
                    break
    font_manager.fontManager.addfont(str(path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({"font.family": name, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["muted"],
                         "axes.labelcolor": C["ink"], "xtick.color": C["muted"], "ytick.color": C["muted"],
                         "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "savefig.dpi": 170, "savefig.bbox": "tight", "legend.frameon": False})


pc = lambda v: f"{v * 100:.1f}%"
p0 = lambda v: f"{v * 100:.0f}%"
pv = lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"
pp = lambda v: f"{'+' if v >= 0 else '−'}{abs(v) * 100:.1f}"


def pct_axis(ax, lo=0, hi=1):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))


def fig_vs_paper(D, out):
    I, P = D["it30"], D["paper_fig2"]["gd"]
    ids = SHAPES + ["seen", "unseen"]
    lab = [LABEL[s] for s in SHAPES] + ["已见平均", "未见平均"]
    seen, unseen = SHAPES[:4], SHAPES[4:]
    pg = lambda sub, k: np.mean([P[s][k] for s in sub]) / 100
    paper_f = [P[s][2] / 100 for s in SHAPES] + [pg(seen, 2), pg(unseen, 2)]
    paper_a = [P[s][3] / 100 for s in SHAPES] + [pg(seen, 3), pg(unseen, 3)]
    ours_f = [I["shapes"][s]["frozen"] for s in SHAPES] + [I["groups"]["seen"]["frozen"], I["groups"]["unseen"]["frozen"]]
    ours_a = [I["shapes"][s]["adajepa"] for s in SHAPES] + [I["groups"]["seen"]["adajepa"], I["groups"]["unseen"]["adajepa"]]
    x = np.arange(len(ids)) + np.array([0] * 7 + [0.6, 0.6])
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.2))
    specs = [(paper_f, -1.5, "论文图 2 冻结", "white", C["paper"], "//"), (paper_a, -0.5, "论文图 2 AdaJEPA", C["paper"], C["paper"], None),
             (ours_f, 0.5, "本次 冻结", C["frozen"], C["frozen"], None), (ours_a, 1.5, "本次 AdaJEPA", C["adajepa"], C["adajepa"], None)]
    for vals, off, name, fc, ec, hatch in specs:
        bars = ax.bar(x + off * w, vals, w * 0.92, label=name, color=fc, edgecolor=ec, hatch=hatch, linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v * 100:.0f}", ha="center", va="bottom", fontsize=7.5, color=C["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    pct_axis(ax, 0, 1.08)
    ax.axvline(6.8, color=C["grid"], lw=1)
    seeds = "、".join(f"seed {s}" for s in I["seeds"])
    ax.set_title(f"按论文主实验协议（GD、最多 {I['max_iter']} 次重规划）逐形状对照论文 Figure 2｜本次：{seeds}", fontsize=11, loc="left")
    ax.legend(ncol=4, loc="upper right", fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def fig_curves(D, out):
    I, P = D["it30"], D["paper_fig2"]["gd"]
    n = I["max_iter"]
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.8), sharey=True)
    for ax, s in zip(axes, SHAPES):
        o = I["shapes"][s]
        k = np.arange(1, n + 1)
        ax.step(k, o["frozen_curve"], where="post", color=C["frozen"], lw=1.8, label="本次 冻结")
        ax.step(k, o["adajepa_curve"], where="post", color=C["adajepa"], lw=1.8, label="本次 AdaJEPA")
        ax.plot([n], [P[s][2] / 100], "o", mfc="white", mec=C["paper"], mew=1.8, ms=6, label="论文 冻结（第 30 次）")
        ax.plot([n], [P[s][3] / 100], "D", color=C["paper"], ms=5.5, label="论文 AdaJEPA（第 30 次）")
        ax.axvline(20, color=C["grid"], lw=1, ls="--")
        ax.set_title(LABEL[s], fontsize=11)
        ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("重规划次数", fontsize=8.5)
        pct_axis(ax)
    axes[0].set_ylabel("累计成功率")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle(f"本次模型的累计成功率（{'、'.join('seed ' + str(s) for s in I['seeds'])} 平均）与论文 Figure 2 第 30 次的读数", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_frozen_seeds(D, out):
    F = D["diag"]["seeds"]["frozen"]
    seeds = F["pooled_seeds"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    groups = [(f"seed {s}", F["per_seed"][str(s)]["all"]) for s in seeds] + [("合计", F["pooled"]["all"])]
    x = np.arange(len(groups))
    for off, key, name, col in [(-0.19, "ours", "本次模型", C["new"]), (0.19, "released", "发布检查点", C["released"])]:
        bars = a1.bar(x + off, [g[1][key] for g in groups], 0.36, color=col, label=name)
        for b, g in zip(bars, groups):
            a1.text(b.get_x() + b.get_width() / 2, g[1][key] + 0.01, pc(g[1][key]), ha="center", fontsize=8)
    a1.set_xticks(x)
    a1.set_xticklabels([g[0] for g in groups])
    pct_axis(a1, 0, 0.8)
    a1.set_title("冻结规划：各测试种子的总体成功率（7 种形状、350 个目标/种子）", fontsize=10.5, loc="left")
    a1.legend(loc="upper left", fontsize=9)
    y = np.arange(len(SHAPES))[::-1]
    for dy, key, name, col in [(0.14, "ours", "本次模型", C["new"]), (-0.14, "released", "发布检查点", C["released"])]:
        v = [F["per_shape_pooled"][s][key] for s in SHAPES]
        ci = np.array([F["per_shape_pooled"][s][key + "_ci"] for s in SHAPES])
        a2.errorbar(v, y + dy, xerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], fmt="o", color=col, ms=6, capsize=0, lw=4, alpha=0.9, label=name)
    for yi, s in zip(y, SHAPES):
        a2.text(1.02, yi, pv(F["per_shape_pooled"][s]["p"]), va="center", fontsize=8.5, color=C["ink"] if F["per_shape_pooled"][s]["p"] < 0.05 else C["muted"])
    a2.set_yticks(y)
    a2.set_yticklabels([LABEL[s] for s in SHAPES])
    a2.set_xlim(0, 1.12)
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%" if v <= 1 else ""))
    a2.set_title(f"各形状 {len(seeds)} 个种子合计（每形状 {F['per_shape_pooled']['T']['n']} 个目标，95% 区间，逐例检验）", fontsize=10.5, loc="left")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_released_adajepa(D, out):
    S = {s["id"]: s for s in D["shapes"]}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(SHAPES))[::-1]
    for yi, s in zip(y, SHAPES):
        o = S[s]
        for dy, f, a, col in [(0.15, o["frozen"], o["adajepa"], C["new"]), (-0.15, o["released"], o["released_adajepa"], C["released"])]:
            ax.annotate("", xy=(a, yi + dy), xytext=(f, yi + dy), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.plot([f], [yi + dy], "o", mfc="white", mec=col, mew=2, ms=6)
    ax.plot([], [], color=C["new"], lw=2, label="本次：冻结 → AdaJEPA")
    ax.plot([], [], color=C["released"], lw=2, label="发布检查点：冻结 → AdaJEPA")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[s] for s in SHAPES])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_title("seed 100、最多 20 次重规划：两个模型的冻结与 AdaJEPA（同一批 50 个目标/形状）", fontsize=10.5, loc="left")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_runs(D, out):
    runs = D["runs"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(runs))
    for off, key, name, col in [(-0.19, "frozen", "冻结", C["frozen"]), (0.19, "adajepa", "AdaJEPA", C["adajepa"])]:
        bars = ax.bar(x + off, [r[key] for r in runs], 0.36, color=col, label=name)
        for b, r in zip(bars, runs):
            ax.text(b.get_x() + b.get_width() / 2, r[key] + 0.01, pc(r[key]), ha="center", fontsize=8)
    A = D["summary"]["all"]
    ax.axhline(A["released"], color=C["released"], ls="--", lw=1.5, label=f"发布检查点 冻结 {pc(A['released'])}")
    if A.get("released_adajepa") is not None:
        ax.axhline(A["released_adajepa"], color=C["released"], ls=":", lw=1.8, label=f"发布检查点 AdaJEPA {pc(A['released_adajepa'])}")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].replace(" · ", "\n", 1) for r in runs], fontsize=8.5)
    pct_axis(ax, 0, 0.85)
    ax.set_title("四轮实验（seed 100、最多 20 次重规划、7 种形状 350 个目标）", fontsize=10.5, loc="left")
    ax.legend(ncol=2, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_adaptivity(D, out):
    ad = D["diag"]["adaptivity"]
    col = {"released": C["released"], "replay20k": C["prev"], "bias16k": C["new"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, part, title in [(axes[0], "ours", f"旧数据的 T 验证轨迹（{ad[0]['ours']['n']} 步）"), (axes[1], "author", f"作者的 T 测试片段（{ad[0]['author']['n']} 步）")]:
        y = np.arange(len(ad))[::-1]
        v = [m[part]["next1_ratio_gmean"] for m in ad]
        ax.barh(y, np.array(v) - 1, left=1, color=[col.get(m["run"], C["muted"]) for m in ad], height=0.6)
        for yi, vi in zip(y, v):
            ax.text(vi + (0.005 if vi >= 1 else -0.005), yi, f"{vi:.3f}", va="center", ha="left" if vi >= 1 else "right", fontsize=8.5)
        ax.axvline(1, color=C["ink"], lw=1)
        ax.set_xlim(0.8, 1.22)
        ax.set_yticks(y)
        ax.set_yticklabels([m["label"] for m in ad])
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("一次适应后 / 不适应 的下一步预测误差（<1 表示变准）", fontsize=8.5)
    fig.suptitle("离线适应测试（CPU）：一次 AdaJEPA 更新能否让接下来的预测变准", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_action_stats(D, out):
    dist = D["diag"]["distribution"]
    keys = [("cos_to_block", "偏差方向与“指向物体”的余弦"), ("toward_frac", "偏差朝向物体的步数比例"), ("nn_rms", "与最近 pusht_noise 动作的偏差 (px)"),
            ("action_mag", "平均动作幅度 (px/步)"), ("contact_frac", "物体在动的步数比例"), ("mean_gap", "推杆与物体平均间隙 (px)")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2))
    shapes = [s for s in ["T", "L", "Z", "+"] if s in dist]
    x = np.arange(len(shapes))
    for ax, (k, title) in zip(axes.flat, keys):
        for off, src, name, colr in [(-0.26, "author", "作者数据", C["ink"]), (0, "old", "旧数据（原样回放）", C["prev"]), (0.26, "new", "新数据（动作偏置）", C["new"])]:
            vals = [dist[s][src][k] for s in shapes]
            if src == "old" and k in ("cos_to_block", "toward_frac"):
                vals = [0 for _ in shapes]  # no residual: the actions are pusht_noise itself
            ax.bar(x + off, vals, 0.25, color=colr, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[s] for s in shapes])
        ax.set_title(title, fontsize=10, loc="left")
    axes.flat[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("动作与接触统计（25 步窗口，按作者片段的物体位移分布重新加权）", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_training(D, out):
    T = D["training"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    it = T["iters"]
    a1.plot([r["x"] for r in it], [r["train_z"] for r in it], "-o", color=C["frozen"], ms=3, label="训练（区间平均）")
    a1.plot([r["epoch"] for r in T["epochs"]], [r["val_z"] for r in T["epochs"]], "-o", color=C["adajepa"], ms=5, label="验证（每轮结束）")
    a1.set_xlabel("训练轮数")
    a1.set_title("本次训练：潜在预测 MSE", fontsize=10.5, loc="left")
    a1.legend(fontsize=9)
    a2.plot([r["samples_m"] for r in T["prev_epochs"]], [r["val_z"] for r in T["prev_epochs"]], "-o", color=C["prev"], label=T.get("prev_label", "上一轮"))
    a2.plot([r["samples_m"] for r in T["epochs"]], [r["val_z"] for r in T["epochs"]], "-o", color=C["new"], label="本次")
    a2.set_xlabel("已训练样本数（百万个窗口）")
    a2.set_title("验证潜在 MSE（两次的验证集不同，只看趋势）", fontsize=10.5, loc="left")
    a2.legend(fontsize=9)
    for a in (a1, a2):
        a.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def markdown(D, figs, paper_fig):
    S = {s["id"]: s for s in D["shapes"]}
    A, SE, UN = D["summary"]["all"], D["summary"]["seen"], D["summary"]["unseen"]
    I, P = D.get("it30"), (D.get("paper_fig2") or {}).get("gd")
    F = D["diag"]["seeds"]["frozen"]
    T, d, pd = D["training"], D["data"], D.get("prev_data") or {}
    ad = {m["key"]: m for m in D["diag"]["adaptivity"]}
    hours = sum(r["minutes"] for r in T["epochs"]) / 60
    seen, unseen = SHAPES[:4], SHAPES[4:]
    pg = lambda sub, k: np.mean([P[s][k] for s in sub]) / 100
    seeds30 = "、".join(f"seed {s}" for s in I["seeds"]) if I else ""
    prev = next((r for r in D["runs"] if r["key"] == "replay20k"), None)
    L = []
    w = L.append
    w("# AdaJEPA PushObj 复现：训练数据加入动作偏置")
    w("")
    w(f"生成于 {D['generated_utc']}。模型 `{D['paths']['model']}`（第 {len(T['epochs'])} 轮），数据 `{D['paths']['data']}`。"
      "交互版见同目录的 `report.html`（浏览器直接打开，不需要联网）。")
    w("")
    w("## 1. 结论")
    w("")
    if I and P:
        w(f"- **按论文主实验协议（GD、最多 {I['max_iter']} 次重规划）对照 Figure 2**：AdaJEPA 已见形状 {pc(I['groups']['seen']['adajepa'])}"
          f"（论文约 {p0(pg(seen, 3))}），未见形状 {pc(I['groups']['unseen']['adajepa'])}（论文约 {p0(pg(unseen, 3))}），{sum(I['shapes'][s]['adajepa'] >= P[s][3] / 100 - 0.02 for s in SHAPES)} / 7 种形状达到或超过论文。"
          f"本次为 {seeds30}，共 {I['groups']['all']['n']} 个目标；论文为 3 个测试种子平均，数值从图上读出，约 ±2 个百分点。")
    w(f"- **冻结规划稳定超过发布检查点**：{len(F['pooled_seeds'])} 个测试种子、{F['pooled']['all']['n']:,} 个目标上 {pc(F['pooled']['all']['ours'])} 对 "
      f"{pc(F['pooled']['all']['released'])}（逐例 {pv(F['pooled']['all']['p'])}），7 种形状中 6 种显著领先，只有 I 持平。")
    w(f"- **AdaJEPA 与发布检查点逐例对照（seed 100、20 次）**：已见 {pc(SE['adajepa'])} 对 {pc(SE['released_adajepa'])}（{pv(SE['vs_released_adajepa']['p'])}，持平），"
      f"未见 {pc(UN['adajepa'])} 对 {pc(UN['released_adajepa'])}（{pv(UN['vs_released_adajepa']['p'])}）。")
    if prev:
        w(f"- **相对上一轮**（原样回放 2 万条、5 轮）：seed 100 上冻结 {pc(prev['frozen'])} → {pc(A['frozen'])}，AdaJEPA {pc(prev['adajepa'])} → {pc(A['adajepa'])}。")
    w(f"- AdaJEPA 的增益（已见 {pp(I['groups']['seen']['adajepa'] - I['groups']['seen']['frozen'])}、未见 {pp(I['groups']['unseen']['adajepa'] - I['groups']['unseen']['frozen'])} 个百分点）"
      f"小于论文（{pp(pg(seen, 3) - pg(seen, 2))}、{pp(pg(unseen, 3) - pg(unseen, 2))}），因为冻结起点高得多。" if I and P else "")
    w("")
    w("## 2. 本次改动（相对上一轮）")
    w("")
    w(table(["项目", "上一轮", "本次"], [
        ["动作", "原样回放 pusht_noise 的动作", f"每一步加 {d['action_bias_px']:g} px 指向物体重心的偏置和 σ={d['action_noise_px']:g} px 高斯噪声（按作者测试片段的动作统计标定），保存实际执行的动作"],
        ["物体初始位置", "新形状的重心对齐 T 的重心", "放在源轨迹里 T 的原位"],
        ["轨迹条数（训练 / 验证）", f"{pd.get('train', 0):,} / {pd.get('val', 0):,}", f"{d['train']['episodes']:,} / {d['val']['episodes']:,}（论文每种形状 4,000 条训练）"],
        ["每条轨迹平均接触步数", f"{pd.get('mean_contact_steps', 0):.1f}", f"{d['mean_contact_steps']:.1f}"],
        ["训练轮数", str(pd.get("epochs", "—")), f"{len(T['epochs'])}（与论文、发布检查点一致）"],
        ["decoder", "不训练", f"第 {T.get('decoder_start_epoch') or len(T['epochs'])} 轮起训练（只用于可视化，与发布检查点一致）"],
    ]))
    w("")
    w(f"不变：模型结构、训练超参数（论文表 3）、初始状态质量门槛和评估协议。训练用 {round(T['batch'] / T['gpu_batch'])} 张 RTX 3090，{hours:.1f} 小时。")
    w("")
    w("## 3. 结果")
    w("")
    if I and P:
        w(f"### 3.1 按论文主实验协议对照 Figure 2（GD，最多 {I['max_iter']} 次重规划）")
        w("")
        rows = []
        for s in SHAPES:
            o = I["shapes"][s]
            per = [f"{p0(I['per_seed'][str(sd)][s]['frozen'])} → {p0(I['per_seed'][str(sd)][s]['adajepa'])}" for sd in I["seeds"]]
            rows.append([LABEL[s], f"{P[s][2]}% → {P[s][3]}%", *per, f"{p0(o['frozen'])} → {p0(o['adajepa'])}", f"{o['adajepa_only']} / {o['frozen_only']}（{pv(o['p'])}）"])
        for g, name, sub in [("seen", "**已见平均**", seen), ("unseen", "**未见平均**", unseen), ("all", "**全部**", SHAPES)]:
            o = I["groups"][g]
            per = []
            for sd in I["seeds"]:
                ps = I["per_seed"][str(sd)]
                per.append(f"{p0(np.mean([ps[s]['frozen'] for s in sub]))} → {p0(np.mean([ps[s]['adajepa'] for s in sub]))}")
            rows.append([name, f"{pg(sub, 2) * 100:.0f}% → {pg(sub, 3) * 100:.0f}%", *per, f"**{pc(o['frozen'])} → {pc(o['adajepa'])}**", f"{o['adajepa_only']} / {o['frozen_only']}（{pv(o['p'])}）"])
        w(table(["形状", "论文图 2 冻结 → AdaJEPA", *[f"seed {sd}" for sd in I["seeds"]], "本次合计", "只有 AdaJEPA / 只有冻结"], rows))
        w("")
        w(f"冻结规划在第 20 次以后不再有新的成功；AdaJEPA 从第 20 次到第 30 次，已见 {pc(I['groups']['seen']['adajepa_20'])} → {pc(I['groups']['seen']['adajepa'])}，"
          f"未见 {pc(I['groups']['unseen']['adajepa_20'])} → {pc(I['groups']['unseen']['adajepa'])}。")
        w("")
        w(f"![本次与论文 Figure 2 逐形状对照](figures/{figs['vs_paper']})")
        w("")
        w(f"![累计成功率随重规划次数](figures/{figs['curves']})")
        w("")
        if paper_fig:
            w(f"论文原图（arXiv:2606.32026 Figure 2 上排，CC BY 4.0）：")
            w("")
            w(f"![论文 Figure 2 上排](figures/{paper_fig})")
            w("")
    w(f"### 3.2 冻结规划：{len(F['pooled_seeds'])} 个测试种子，与发布检查点逐例对照（最多 20 次重规划）")
    w("")
    rows = []
    for s in SHAPES:
        r = F["per_shape_pooled"][s]
        per = [f"{round(F['per_seed'][str(sd)]['shapes'][s]['ours'] * 50)} / {round(F['per_seed'][str(sd)]['shapes'][s]['released'] * 50)}" for sd in F["pooled_seeds"]]
        rows.append([LABEL[s], *per, pc(r["ours"]), pc(r["released"]), f"{r['ours_only']} / {r['released_only']}（{pv(r['p'])}）"])
    for g, name in [("seen", "**已见**"), ("unseen", "**未见**"), ("all", "**全部**")]:
        r = F["pooled"][g]
        rows.append([name, *["" for _ in F["pooled_seeds"]], f"**{pc(r['ours'])}**", f"**{pc(r['released'])}**", f"{r['ours_only']} / {r['released_only']}（{pv(r['p'])}）"])
    w(table(["形状", *[f"seed {sd}：本次 / 发布（/50）" for sd in F["pooled_seeds"]], "合计 本次", "合计 发布", "只有本次 / 只有发布"], rows))
    w("")
    w(f"![冻结规划多测试种子对照](figures/{figs['frozen_seeds']})")
    w("")
    w("### 3.3 AdaJEPA：与发布检查点逐例对照（seed 100，最多 20 次重规划）")
    w("")
    rows = []
    for s in SHAPES:
        o, v = S[s], S[s]["vs_released_adajepa"]
        rows.append([LABEL[s], f"{p0(o['released'])} → {p0(o['released_adajepa'])}", f"{p0(o['frozen'])} → {p0(o['adajepa'])}", f"{v['ours_only']} / {v['released_only']}（{pv(v['p'])}）"])
    for g, name in [("seen", "**已见**"), ("unseen", "**未见**"), ("all", "**全部**")]:
        G = D["summary"][g]
        v = G["vs_released_adajepa"]
        rows.append([name, f"{pc(G['released'])} → {pc(G['released_adajepa'])}", f"{pc(G['frozen'])} → {pc(G['adajepa'])}", f"{v['y_only']} / {v['x_only']}（{pv(v['p'])}）"])
    w(table(["形状", "发布检查点 冻结 → AdaJEPA", "本次 冻结 → AdaJEPA", "AdaJEPA 只有本次 / 只有发布"], rows))
    w("")
    w(f"发布检查点在同一评估管线下已见形状 {pc(SE['released'])} → {pc(SE['released_adajepa'])}，与论文一致，说明评估本身没有问题。")
    w("")
    w(f"![两个模型的 AdaJEPA](figures/{figs['released']})")
    w("")
    w("### 3.4 四轮实验")
    w("")
    w(f"![四轮实验](figures/{figs['runs']})")
    w("")
    w("## 4. 可能的提升原因")
    w("")
    w("**AdaJEPA 重新起作用：直接原因是训练数据加入了动作偏置（证据较充分）**")
    w("")
    if prev:
        w(f"- 上一轮 AdaJEPA 在 T 上只提升 {pp((prev['per_shape']['adajepa']['T'] - prev['per_shape']['frozen']['T']) / 50)} 个百分点，发布检查点在同一批目标上提升 {pp(S['T']['released_adajepa'] - S['T']['released'])}。")
    if "ours e5" in ad and "bias16k e3" in ad:
        w(f"- 离线测试（CPU）：上一轮模型做一次适应更新后，对接下来一帧的预测误差变成原来的 {ad['ours e5']['ours']['next1_ratio_gmean']:.2f} 倍（越适应越差）；"
          f"换成带偏置的数据后为 {ad['bias16k e3']['ours']['next1_ratio_gmean']:.2f} 倍，发布检查点 {ad['released (e3)']['ours']['next1_ratio_gmean']:.2f} 倍。")
    w("- 训练超参数、推理协议和代码改动都已逐项排除，两轮之间唯一的实质变化是训练数据。")
    w("- 可能的机制（推测）：规划器执行的动作几乎都是有目的地推物体，而原样回放的动作与新形状的几何无关，很多步在空中移动或擦边。"
      "适应时刚执行的片段落在训练分布之外，一步更新拟合了这段片段，却推广不到接下来的帧；加入偏置后训练数据里朝物体推的动作与规划时接近。")
    w("")
    fa = F["pooled"]["all"]
    w(f"**冻结规划比发布检查点高约 {(fa['ours'] - fa['released']) * 100:.0f} 个百分点：原因待验证**")
    w("")
    w(f"- 差距稳定：{len(F['pooled_seeds'])} 个测试种子、{fa['n']:,} 个目标上 {pc(fa['ours'])} 对 {pc(fa['released'])}，不是抽到容易目标的偶然。")
    if prev:
        w(f"- 已排除：初始状态质量门槛（上一轮同样有，冻结只有 {pc(prev['frozen'])}，与发布检查点持平）；训练轮数。")
    w(f"- 候选一：动作偏置和噪声让每条轨迹的接触步数从 {pd.get('mean_contact_steps', 0):.1f} 增加到 {d['mean_contact_steps']:.1f}，动作也更多样。")
    w("- 候选二：偏置按作者测试片段的统计量标定，训练分布可能比作者自己的训练数据更贴近测试目标；发布检查点的配置隐去了数据路径，无法核对。")
    w("- 候选三：只有 1 个训练种子。")
    w(f"- 线索：唯一没有领先的形状是 I（{pc(F['per_shape_pooled']['I']['ours'])} 对 {pc(F['per_shape_pooled']['I']['released'])}），它和训练形状差别最大。")
    w("")
    w("## 5. 问题定位与离线适应测试")
    w("")
    m = D["diag"]["action_match"]
    w(f"把作者测试片段的动作与 pusht_noise 中最接近的 25 步动作窗口比较：作者 T 的动作平均偏离 {m['T']['nn_rms_px_median']:.1f} px"
      f"（pusht_noise 自身 {m['null (pusht_noise vs other episodes)']['nn_rms_px_median']:.1f} px），偏离方向与“指向物体”的平均余弦 +{m['T']['cos_to_block_mean']:.2f}，"
      f"{m['T']['frac_residual_toward_block'] * 100:.0f}% 的步朝向物体。论文附录 A.2 只写了“contact bias”，没有给做法；本次按最简单的形式（每步朝物体重心的恒定推力加高斯噪声）"
      f"在 b × σ 网格上标定，取 b={d['action_bias_px']:g} px、σ={d['action_noise_px']:g} px。")
    w("")
    w(f"![动作与接触统计](figures/{figs['action']})")
    w("")
    w(f"![离线适应测试](figures/{figs['adapt']})")
    w("")
    w("## 6. 训练与数据")
    w("")
    w(f"![训练曲线](figures/{figs['training']})")
    w("")
    t, v = d["train"], d["val"]
    w(table(["检查项", "训练集", "验证集"], [
        ["轨迹 / 帧", f"{t['episodes']:,} / {t['frames']:,}", f"{v['episodes']:,} / {v['frames']:,}"],
        ["不合法初始状态 / agent 出画面的帧", f"{t['invalid_start_states']} / {t['agent_out_of_frame_frames']}", f"{v['invalid_start_states']} / {v['agent_out_of_frame_frames']}"],
        ["有接触的轨迹", pc(t["episodes_with_contact"]), pc(v["episodes_with_contact"])],
        ["物体单步位移 q99 / 最大 (px)", " / ".join(f"{q:.0f}" for q in t["block_step_px_q99_max"]), " / ".join(f"{q:.0f}" for q in v["block_step_px_q99_max"])],
        ["训练与验证共享的源轨迹", str(d.get("shared_sources")), ""],
    ]))
    w("")
    w("![训练数据示例](figures/dataset_examples.png)")
    w("")
    w("## 7. 局限")
    w("")
    for s_ in ["只有 1 个训练种子；单个形状 50 个目标的 95% 区间约 ±13 个百分点，逐形状比较要看合计。",
               "论文 Figure 2 的数值从图上读出，约 ±2 个百分点。论文 4.1 节写最多 20 次重规划，但 Figure 2 图注和附录 B.2 写主实验为 30 次；这里按 30 次对照 Figure 2。",
               "动作偏置按作者发布的测试片段标定：只用了动作和接触的统计量，没有用测试目标或规划结果，但这些统计量来自测试分布。",
               "偏置的具体形式是本仓库的推断，与作者数据的统计量吻合，但不能保证机制相同。",
               "发布检查点的 AdaJEPA 只在 seed 100、20 次重规划下跑过；只复现了 GD 规划，没有跑 CEM（论文中两者结果接近）。"]:
        w(f"- {s_}")
    w("")
    w("## 8. 后续工作")
    w("")
    for i, (h, t_) in enumerate([
        ("补全统计", f"再训练 2 个训练种子（每个约 {hours:.0f} 小时训练 + 2 小时评估），得到和论文同口径的平均；发布检查点也可按 30 次在 seed 200、300 上跑，得到同协议的逐例对照。"),
        ("定位冻结提升的来源", "数据消融，每组约 4 小时训练 + 15 分钟冻结评估：只加噪声（b=0、σ=10）、只加偏置（b=9、σ=0）、偏置强度 b=5 / 13、物体摆放改回对齐重心。"),
        ("建规划验证集", "从留出的源轨迹抽目标，用来挑训练轮数和 AdaJEPA 超参数，避免在测试集上调参。"),
        ("未见形状", "I 的冻结不比发布检查点好；small T 和 square 的 AdaJEPA 增益较小、不显著。可在验证集上调适应学习率、回放长度和更新步数。"),
        ("数据规模实验（可选）", "按论文改变每种形状的条数（1k / 2k / 4k）和训练形状数 K，看冻结和 AdaJEPA 增益随数据量的变化。")], 1):
        w(f"{i}. **{h}**：{t_}")
    w("")
    return "\n".join(x for x in L if x is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="reports/pushobj_bias_16k/report_data.json")
    ap.add_argument("--html", default="reports/pushobj_bias_16k/report.html")
    ap.add_argument("--examples", default="reports/pushobj_bias_16k/dataset_examples.png")
    ap.add_argument("--paper-figure", default=None, help="optional copy of the paper's Figure 2 panel (png)")
    ap.add_argument("--out", default="reports/pushobj_bias_16k/advisor_pack")
    ap.add_argument("--font", default=None)
    a = ap.parse_args()
    setup_font(a.font)
    D = json.loads((ROOT / a.data).read_text())
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    figs = {"vs_paper": "1_对照论文图2.png", "curves": "2_重规划曲线_30次.png", "frozen_seeds": "3_冻结_多测试种子.png",
            "released": "4_AdaJEPA_对照发布检查点.png", "runs": "5_四轮实验.png", "action": "6_动作与接触统计.png",
            "adapt": "7_离线适应测试.png", "training": "8_训练曲线.png"}
    if D.get("it30") and D.get("paper_fig2"):
        fig_vs_paper(D, out / "figures" / figs["vs_paper"])
        fig_curves(D, out / "figures" / figs["curves"])
    fig_frozen_seeds(D, out / "figures" / figs["frozen_seeds"])
    fig_released_adajepa(D, out / "figures" / figs["released"])
    fig_runs(D, out / "figures" / figs["runs"])
    fig_action_stats(D, out / "figures" / figs["action"])
    fig_adaptivity(D, out / "figures" / figs["adapt"])
    fig_training(D, out / "figures" / figs["training"])
    shutil.copy(ROOT / a.examples, out / "figures" / "dataset_examples.png")
    paper_fig = None
    if a.paper_figure and Path(a.paper_figure).exists():
        paper_fig = "论文Figure2上排.png"
        shutil.copy(a.paper_figure, out / "figures" / paper_fig)
    (out / "汇报.md").write_text(markdown(D, figs, paper_fig))
    shutil.copy(ROOT / a.html, out / "report.html")
    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, Path(out.name) / f.relative_to(out))
    print("wrote", out, "and", zpath, f"({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
