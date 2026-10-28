"""Single-domain continual learning over several passes of the 50-episode stream (run.sh <model> <arm> <gpu> <r>):
does planning keep improving, or has it reached its margin?

Round 1 of an arm is <arm>/, round r > 1 is <arm>_r<r>/: the same 50 instances and seeds, the slow learner carried
over, so round r is episodes 50(r-1) .. 50r-1 of one stream. Arms: cl (slow learner + AdaJEPA) and cl_noada (slow
learner only). Every round pairs episode by episode with round 1's frozen and adajepa arms (same instances, same
seeds). Each (arm, round) is a 50-long array, NaN where its chunk has not finished; a comparison uses the episodes
both sides have. An episode whose final CD exceeds analyze.BLOWUP in any arm (a FleX blow-up) is dropped everywhere.
Writes ROUNDS_<model>.md and figures in rounds/<model>/.

    /data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_cl_single/rounds.py <model>      (dino_ours, granular_ref)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import (BLOWUP, CHUNKS, MODELS, OBJ, SUCCESS_CD, final_cd, finished, paired, se,  # noqa: E402
                     start_cd, success_cmp)

OUT = Path(__file__).resolve().parent
ARMS = {"cl": "持续学习（慢学习器 + AdaJEPA）", "cl_noada": "只用慢学习器"}
ARMS_EN = {"cl": "slow learner + AdaJEPA", "cl_noada": "slow learner only"}
BASE = {"frozen": "冻结 θ0", "adajepa": "AdaJEPA"}
# categorical slots 1-3 of the reference palette (validated all-pairs) = rounds 1-3; references in neutral ink
ROUND_COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
N = 10 * len(CHUNKS)


def round_arm(arm, r):
    return arm if r == 1 else f"{arm}_r{r}"


def full(model, arm):
    """Final CD and per-episode records of one arm over the whole stream, NaN / None where a chunk is unfinished."""
    cd, recs = np.full(N, np.nan), [None] * N
    for k in CHUNKS:
        if finished(model, arm, k):
            cd[10 * k : 10 * k + 10] = final_cd(model, arm, k)
            f = OUT / model / arm / f"chunk_{k}" / "continual_episodes.jsonl"
            if f.exists():
                recs[10 * k : 10 * k + 10] = [json.loads(l) for l in open(f)]
    return cd, recs


def field(recs, key, fn=float):
    return np.array([np.nan if r is None else fn(r[key]) for r in recs], dtype=float)


def gm(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main(model="dino_ours"):
    obj = OBJ.get(model, "rope")
    thr = SUCCESS_CD[obj]
    fig_dir = OUT / "rounds" / model
    fig_dir.mkdir(parents=True, exist_ok=True)
    base = {b: full(model, b)[0] for b in BASE}
    start = np.concatenate([start_cd(model, k) for k in CHUNKS])
    ep_chunk = np.repeat(CHUNKS, 10)
    R = {}  # arm -> round -> {"cd", "val", "read", "slow_loss"} (50-long, NaN where unfinished)
    for arm in ARMS:
        for r in range(1, 10):
            cd, recs = full(model, round_arm(arm, r))
            if np.isnan(cd).all():
                break
            R.setdefault(arm, {})[r] = {"cd": cd, "val": field(recs, "val_ratio"),
                                        "read": field(recs, "read", lambda v: v is not None),
                                        "slow_loss": field(recs, "slow_loss")}
    blow = np.zeros(N, bool)
    for x in list(base.values()) + [d["cd"] for rr in R.values() for d in rr.values()]:
        blow |= np.nan_to_num(x) > BLOWUP
    for x in list(base.values()) + [d["cd"] for rr in R.values() for d in rr.values()]:
        x[blow] = np.nan
    nr = max(max(rr) for rr in R.values())

    def pair(a, b):
        m = ~np.isnan(a) & ~np.isnan(b)
        return m.sum(), f"{paired(a[m], b[m])} | {success_cmp(a[m], b[m], thr)}"

    keep = ~blow
    L = [f"# 单领域持续学习：多轮（{model}，{obj} 流）", "", f"模型：{MODELS[model]}。", "",
         f"同一条 50 个 episode 的 {obj} 流连着跑多轮，慢学习器一直接着学：第 r 轮就是整条流的第 50(r−1)～50r−1 个 episode。"
         "两组：持续学习（慢学习器 + AdaJEPA）和只用慢学习器（`planner.adapt.steps=0`），都是 `write=oracle`，"
         "mature_k 3、ratio 0.9。冻结和 AdaJEPA 只有第 1 轮的结果，但实例和种子都一样，所以每一轮都可以和它们逐 episode 配对。"
         f"主指标是最终 CD（越低越好），成功率是最终 CD < {thr} 的比例。", "",
         "注意：从第 2 轮开始，每个实例（起点和目标）慢学习器在前面的轮次里都见过，只是执行的动作和轨迹不同；"
         "所以这里看到的是在这 50 个实例上能练到的上限，不是对新实例的泛化。", ""]
    if blow.any():
        L += [f"仿真失败（某一组最终 CD > {BLOWUP:g}，FleX 把粒子甩飞了）的 episode 从所有组里去掉："
              + "、".join(f"段 {ep_chunk[i]} 第 {i % 10} 个" for i in np.flatnonzero(blow)) + "。", ""]
    L += [f"推之前的平均 CD 是 {start[keep].mean():.3f}（{keep.sum()} 个 episode）。"
          + "；".join(f"{BASE[b]}：{np.nanmean(base[b]):.3f}，成功率 {np.nanmean(base[b][keep] < thr) * 100:.0f}%" for b in BASE)
          + "。", "", "## 每一轮", "",
          "| 组 | 轮 | episode 数 | 最终 CD（均值 ± 标准误） | 中位数 | 成功率 | 读了慢学习器 | 验证比（几何平均） "
          "| 回放训练损失（中位数） |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, rr in R.items():
        for r, d in rr.items():
            f = d["cd"][~np.isnan(d["cd"])]
            done = ~np.isnan(d["val"]) | ~np.isnan(d["slow_loss"]) | (d["read"] >= 0)
            L.append(f"| {ARMS[arm]} | {r} | {len(f)} | {f.mean():.3f} ± {se(f):.3f} | {np.median(f):.3f} | "
                     f"{(f < thr).mean() * 100:.0f}%（{(f < thr).sum()}/{len(f)}） | "
                     f"{int(np.nansum(d['read']))} / {int(done.sum())} | {gm(d['val']):.3f} | {np.nanmedian(d['slow_loss']):.4f} |")
    L += ["", "验证比：慢学习器在写入这个 episode 之前（还没在它上面训练），对它的 5 步开环潜空间误差 ÷ θ0 的误差，< 1 表示比 θ0 准。"
          "回放训练损失：每写入一个 episode 后，慢学习器在回放上训练 15 步的平均损失。", "",
          "## 逐 episode 配对比较", "", "CD 差为负、成功率差为正，表示前者更好；每行只用两边都有的 episode。", "",
          "| 对比 | episode 数 | CD 平均差（95% 置信区间） | CD 前者更好 / 后者更好 | Wilcoxon p | 配对 t 检验 p "
          "| 成功率差 | 只有前者成功 / 只有后者成功 | McNemar p |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, rr in R.items():
        for r, d in rr.items():
            for b in BASE:
                n, s = pair(d["cd"], base[b])
                L.append(f"| {ARMS[arm]} 第 {r} 轮 − {BASE[b]} | {n} | {s} |")
            if r - 1 in rr:
                n, s = pair(d["cd"], rr[r - 1]["cd"])
                L.append(f"| {ARMS[arm]} 第 {r} 轮 − 第 {r - 1} 轮 | {n} | {s} |")
        if 3 in rr:
            n, s = pair(rr[max(rr)]["cd"], rr[1]["cd"])
            L.append(f"| {ARMS[arm]} 第 {max(rr)} 轮 − 第 1 轮 | {n} | {s} |")
    if len(R) == 2:
        for r in range(1, nr + 1):
            if all(r in rr for rr in R.values()):
                n, s = pair(R["cl"][r]["cd"], R["cl_noada"][r]["cd"])
                L.append(f"| 第 {r} 轮：{ARMS['cl']} − {ARMS['cl_noada']} | {n} | {s} |")
    cols = [(arm, r) for arm, rr in R.items() for r in rr]
    L += ["", "## 按段看（每段 10 个 episode，各轮的同一段是同一批实例）", "",
          "| 段 | 起点 | 冻结 θ0 | AdaJEPA | " + " | ".join(f"{ARMS[a]} 第 {r} 轮" for a, r in cols) + " |",
          "|---|---|---|---|" + "---|" * len(cols)]
    cm = lambda x, k: "—" if np.isnan(x[ep_chunk == k]).all() else f"{np.nanmean(x[ep_chunk == k]):.3f}"
    for k in CHUNKS:
        L.append(f"| {k} | {start[(ep_chunk == k) & keep].mean():.3f} | {cm(base['frozen'], k)} | {cm(base['adajepa'], k)} | "
                 + " | ".join(cm(R[a][r]["cd"], k) for a, r in cols) + " |")
    L += ["", "各段验证比的几何平均：", "", "| 段 | " + " | ".join(f"{ARMS[a]} 第 {r} 轮" for a, r in cols) + " |",
          "|---|" + "---|" * len(cols)]
    for k in CHUNKS:
        L.append(f"| {k} | " + " | ".join(f"{gm(R[a][r]['val'][ep_chunk == k]):.3f}" for a, r in cols) + " |")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    arms = list(R)

    # 1. the learner's prediction quality along the whole stream, one panel per arm
    fig, axes = plt.subplots(len(arms), 1, figsize=(8, 2.9 * len(arms)), dpi=150, sharex=True, sharey=True, squeeze=False)
    for ax, arm in zip(axes[:, 0], arms):
        style(ax)
        for r, d in R[arm].items():
            x = (r - 1) * N + np.arange(N)
            v = d["val"]
            ax.scatter(x, v, s=10, color=ROUND_COLOR[r], alpha=0.35, lw=0)
            roll = np.full(N, np.nan)
            for i in range(N):
                w = v[max(0, i - 9) : i + 1]
                if (~np.isnan(w)).sum() >= 3 and not np.isnan(v[i]):
                    roll[i] = gm(w)
            ax.plot(x, roll, color=ROUND_COLOR[r], lw=2, label=f"round {r}")
            if r > 1:
                ax.axvline((r - 1) * N - 0.5, color=GRID, lw=1)
        ax.axhline(1.0, color=MUTED, lw=1, ls="--")
        ax.text(N * nr - 1, 1.02, "theta0", color=MUTED, fontsize=8, va="bottom", ha="right")
        ax.set_xlim(-1, N * nr)
        ax.set_ylabel("val ratio (learner / theta0)", color=MUTED)
        ax.set_title(f"{ARMS_EN[arm]}: learner / theta0 error on the next unseen episode (lower is better)", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=nr)
    axes[-1, 0].set_xlabel("episode in the stream", color=MUTED)
    fig.tight_layout()
    fig.savefig(fig_dir / "val_ratio.png")
    plt.close(fig)

    # 2. planning per round: mean final CD (95% CI) and success; rows = arms; frozen / adajepa as references
    fig, axes = plt.subplots(len(arms), 2, figsize=(8, 2.9 * len(arms)), dpi=150, squeeze=False)
    for row, arm in zip(axes, arms):
        rs = list(R[arm])
        for ax, key in zip(row, ("cd", "succ")):
            style(ax)
            for r in rs:
                f = R[arm][r]["cd"]
                f = f[~np.isnan(f)]
                if key == "cd":
                    y, e = f.mean(), 1.96 * se(f)
                else:
                    p = (f < thr).mean()
                    y, e = p * 100, 196 * np.sqrt(p * (1 - p) / len(f))
                ax.errorbar([r], [y], yerr=[e], fmt="o", ms=8, color=ROUND_COLOR[r], ecolor=ROUND_COLOR[r],
                            elinewidth=2, capsize=0, mec="white", mew=2)
                ax.annotate(f"{y:.3f}" if key == "cd" else f"{y:.0f}%", (r, y), xytext=(10, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK)
            refs = {b: np.nanmean(base[b]) if key == "cd" else np.nanmean(base[b][keep] < thr) * 100 for b in BASE}
            for b, ls, name in (("frozen", "--", "frozen theta0"), ("adajepa", ":", "AdaJEPA")):
                yb = refs[b]
                ax.axhline(yb, color=MUTED, lw=1, ls=ls)
                above = yb >= max(refs.values())  # the upper reference labels above its line, the lower one below
                ax.annotate(name, (nr + 0.45, yb), xytext=(0, 2 if above else -2), textcoords="offset points",
                            color=MUTED, fontsize=8, ha="right", va="bottom" if above else "top")
            lo, hi = ax.get_ylim()  # axhline does not autoscale: make room for the references and their labels
            lo, hi = min(lo, *refs.values()), max(hi, *refs.values())
            ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
            ax.set_xticks(range(1, nr + 1))
            ax.set_xticklabels([f"round {r}" for r in range(1, nr + 1)])
            ax.set_xlim(0.5, nr + 0.5)
            ax.set_title(f"{ARMS_EN[arm]}: " + ("mean final CD (95% CI)" if key == "cd" else f"success, CD < {thr}"),
                         color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_round.png")
    plt.close(fig)

    # 3. per chunk: the same 10 instances across rounds, one panel per arm
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 3.2), dpi=150, sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        style(ax)
        ax.plot(CHUNKS, [np.nanmean(base["frozen"][ep_chunk == k]) for k in CHUNKS], color=MUTED, lw=1.5, ls="--",
                marker="o", ms=5, label="frozen theta0")
        for r, d in R[arm].items():
            ks = [k for k in CHUNKS if not np.isnan(d["cd"][ep_chunk == k]).all()]
            ax.plot(ks, [np.nanmean(d["cd"][ep_chunk == k]) for k in ks], color=ROUND_COLOR[r], lw=2, marker="o",
                    ms=7, mec="white", mew=1.5, label=f"round {r}")
        ax.set_xticks(CHUNKS)
        ax.set_xticklabels([f"chunk {k}" for k in CHUNKS])
        ax.set_title(f"{ARMS_EN[arm]}", color=INK)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    axes[0, 0].set_ylabel("mean final CD (10 episodes, lower is better)", color=MUTED)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_chunk.png")
    plt.close(fig)

    rel = f"rounds/{model}"
    L += ["", "## 图", "", f"![验证比]({rel}/val_ratio.png)", "",
          "慢学习器在下一个还没训练过的 episode 上的验证比，沿整条流（点：每个 episode；线：最近 10 个的几何平均）。", "",
          f"![每一轮]({rel}/per_round.png)", "",
          "每一轮的平均最终 CD 和成功率（95% 置信区间），虚线是第 1 轮的冻结组、点线是 AdaJEPA。", "",
          f"![按段]({rel}/per_chunk.png)", "", "按段的平均最终 CD，各轮的同一段是同一批实例。", ""]
    text = "\n".join(L) + "\n"
    (OUT / f"ROUNDS_{model}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
