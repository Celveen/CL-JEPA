"""Summarise experiment 0: how much of AdaJEPA's gain is information vs perturbation.

Reads eval_outputs/exp0_shape/s<seed>/<shape>/<arm> (scripts/exp0_run.py) and
writes reports/exp0_shape/summary.{json,md} plus success-curve figures.

Every arm plans the same targets, so comparisons are paired by episode:
  * success per arm, exact McNemar tests against the frozen F_seq and against A;
  * share of A's gain recovered by each control, (X - F) / (A - F), with a
    paired bootstrap interval (episodes resampled within each shape);
  * per-replan diagnostics from ProbeMPCPlanner: prediction error on the newest
    frame (never trained on) under the adapted vs the initial weights, the
    displacement from the initial weights and the change in predictions;
  * stall analysis of frozen failures (progress to the goal and action change
    over the last replans).

    python scripts/exp0_analyze.py --seeds 100
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["F_batch", "F_seq", "A", "A_asc", "R_tgt", "R_tgt_fix", "R_sign", "W_dom", "P_restart"]
DESC = {
    "F_batch": "冻结（批量规划）",
    "F_seq": "冻结（逐例规划，配对基准）",
    "A": "AdaJEPA",
    "A_asc": "AdaJEPA 反向一步（反信息）",
    "R_tgt": "随机目标一步（自然方向、等幅、无信息）",
    "R_tgt_fix": "随机目标一步，每 episode 固定方向",
    "R_sign": "参数空间随机符号（等参数步长）",
    "W_dom": "同形状其他 episode 的数据",
    "P_restart": "权重不变，规划器每次随机重启",
}
CONTROLS = ["A_asc", "R_tgt", "R_tgt_fix", "R_sign", "W_dom", "P_restart", "F_batch"]
MAX_ITER = 30


def finished(d):
    logs = d / "logs.json"
    return (d / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()


def load(root, seed, shape, arm):
    d = root / f"s{seed}" / shape / arm
    if not finished(d):
        return None
    z = np.load(d / "final_eval_results.npz")
    rec_path = d / "per_iter_logs.json"
    rec = [json.loads(l) for l in rec_path.read_text().splitlines() if l.strip()] if rec_path.exists() else []
    return {"succ": z["successes"].astype(bool), "len": z["action_len"].astype(float), "rec": rec}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mcnemar(x, y):
    """Exact two-sided McNemar on paired booleans. Returns (x_only, y_only, p)."""
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def share_ci(F, A, X, strata, n_boot=10000, seed=0):
    """(X - F) / (A - F) with a paired bootstrap that resamples episodes within strata."""
    rng = np.random.default_rng(seed)
    gain_a, gain_x = A.mean() - F.mean(), X.mean() - F.mean()
    point = gain_x / gain_a if gain_a != 0 else float("nan")
    idx_by = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by])
        ga = A[idx].mean() - F[idx].mean()
        if ga > 0:
            vals.append((X[idx].mean() - F[idx].mean()) / ga)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) > 100 else (float("nan"), float("nan")))
    return float(point), float(lo), float(hi), float(gain_a), float(gain_x)


def diff_ci(F, X, strata, n_boot=10000, seed=0):
    """Paired bootstrap 95% interval of mean(X) - mean(F), episodes resampled within strata."""
    rng = np.random.default_rng(seed)
    idx_by = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    d = (X.astype(float) - F.astype(float))
    vals = [d[np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by])].mean()
            for _ in range(n_boot)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def curve(lengths):
    return [float(np.mean(lengths <= k)) for k in range(1, MAX_ITER + 1)]


def diag(rec):
    """Per-arm diagnostics from per-replan records."""
    rows = [r for r in rec if "loss_new_cur" in r]
    if not rows:
        return {}
    cur = np.array([r["loss_new_cur"] for r in rows])
    ini = np.array([r["loss_new_init"] for r in rows])
    first = [r for r in rows if r["step"] == 1]
    later = [r for r in rows if r["step"] >= 3]
    out = {
        "replans": len(rows),
        # < 1: the adapted weights predict the not-yet-trained newest frame better than the initial ones
        "new_ratio_gmean": float(np.exp(np.mean(np.log(cur[ini > 0] / ini[ini > 0])))) if np.any(ini > 0) else float("nan"),
        "new_ratio_gmean_step3plus": float(np.exp(np.mean([math.log(r["loss_new_cur"] / r["loss_new_init"])
                                                          for r in later if r["loss_new_init"] > 0]))) if later else float("nan"),
        "new_improved_frac_step3plus": float(np.mean([r["loss_new_cur"] < r["loss_new_init"] for r in later])) if later else float("nan"),
        "loss_new_init_median": float(np.median(ini)),
        "func_change_median": float(np.median([r["func_change"] for r in rows])),
        "cos_own_median": float(np.nanmedian([r["cos_own"] for r in rows])) if any(
            not math.isnan(r["cos_own"]) for r in rows) else float("nan"),
        "step_pred_median": float(np.median([r["step_pred"] for r in rows])),
        "t_plan_mean": float(np.mean([r["t_plan_s"] for r in rows])),
        "t_adapt_mean": float(np.mean([r["t_adajepa_s"] for r in rows])),
    }
    for k in (5, 10, 20, 30):
        d = [r["disp_pred"] for r in rows if r["step"] == k]
        out[f"disp_pred_step{k}"] = float(np.median(d)) if d else float("nan")
    out["first_replan_loss_init_median"] = float(np.median([r["loss_new_init"] for r in first])) if first else float("nan")
    return out


def stall_stats(rec, succ):
    """For failed episodes: progress and action change over the last 10 replans."""
    by = {}
    for r in rec:
        by.setdefault(r["sample_idx"], []).append(r)
    prog, dact, plateau_step = [], [], []
    for i, rs in by.items():
        if succ[i]:
            continue
        rs.sort(key=lambda r: r["step"])
        if len(rs) < 20:
            continue
        d = np.array([r["dist4"] for r in rs])
        a = np.array([r["act"] for r in rs])
        prog.append(float(d[-11] - d[-1]))  # progress over the last 10 replans (>0 = closer)
        dact.append(float(np.mean(np.linalg.norm(np.diff(a[-11:], axis=0), axis=1))))
        best = np.minimum.accumulate(d)
        # first replan after which the best distance so far never improves by more than 2 px
        plateau_step.append(int(next(k for k in range(len(d)) if best[k] - best[-1] < 2.0) + 1))
    if not prog:
        return {}
    return {
        "n_failed": len(prog),
        "progress_last10_median_px": float(np.median(prog)),
        "frac_no_progress_last10": float(np.mean(np.abs(np.array(prog)) < 5.0)),
        "action_change_last10_median": float(np.median(dact)),
        "plateau_step_median": float(np.median(plateau_step)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--shapes", nargs="+", default=["T", "I", "square"])
    ap.add_argument("--root", default="eval_outputs/exp0_shape")
    ap.add_argument("--out", default="reports/exp0_shape")
    a = ap.parse_args()
    root, out = ROOT / a.root, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    data = {}  # (seed, shape) -> arm -> run
    for seed in a.seeds:
        for shape in a.shapes:
            runs = {arm: load(root, seed, shape, arm) for arm in ARMS}
            data[(seed, shape)] = {k: v for k, v in runs.items() if v is not None}
    arms = [arm for arm in ARMS if all(arm in runs for runs in data.values())]
    if "F_seq" not in arms or "A" not in arms:
        raise SystemExit("need F_seq and A finished for every (seed, shape) to summarise")

    def stack(arm, keys):
        return np.concatenate([data[k][arm]["succ"] for k in keys]), np.concatenate([data[k][arm]["len"] for k in keys])

    groups = {f"{shape}": [(s, shape) for s in a.seeds] for shape in a.shapes}
    groups["all"] = list(data)
    summary = {"seeds": a.seeds, "shapes": a.shapes, "arms": arms, "groups": {}}
    for g, keys in groups.items():
        strata = np.concatenate([np.full(len(data[k]["F_seq"]["succ"]), i) for i, k in enumerate(keys)])
        F, _ = stack("F_seq", keys)
        A, _ = stack("A", keys)
        res = {"n": int(len(F)), "arms": {}}
        for arm in arms:
            X, L = stack(arm, keys)
            k = int(X.sum())
            entry = {"success": float(X.mean()), "k": k, "ci": wilson(k, len(X)), "curve": curve(L),
                     "steps_to_success_median": float(np.median(L[X])) if X.any() else float("nan")}
            if arm != "F_seq":
                entry["vs_F_seq"] = dict(zip(("only_arm", "only_F", "p"), mcnemar(X, F)))
                entry["diff_vs_F_ci"] = diff_ci(F, X, strata)
            if arm not in ("A", "F_seq"):
                entry["vs_A"] = dict(zip(("only_arm", "only_A", "p"), mcnemar(X, A)))
                point, lo, hi, ga, gx = share_ci(F, A, X, strata)
                entry["share_of_A_gain"] = {"point": point, "ci": [lo, hi], "gain_A": ga, "gain_arm": gx}
            res["arms"][arm] = entry
        summary["groups"][g] = res

    summary["diagnostics"] = {}
    summary["stalls"] = {}
    for arm in arms:
        rec_all = [r for k in data for r in data[k][arm]["rec"]]
        if rec_all:
            summary["diagnostics"][arm] = diag(rec_all)
        st = [stall_stats(data[k][arm]["rec"], data[k][arm]["succ"]) for k in data if data[k][arm]["rec"]]
        st = [s for s in st if s]
        if st:
            n = sum(s["n_failed"] for s in st)
            summary["stalls"][arm] = {
                "n_failed": n,
                **{key: float(np.average([s[key] for s in st], weights=[s["n_failed"] for s in st]))
                   for key in st[0] if key != "n_failed"},
            }

    tag = "_".join(map(str, a.seeds))
    (out / f"summary_s{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_markdown(summary, out / f"summary_s{tag}.md")
    plot_curves(summary, out / f"curves_s{tag}.png")
    plot_gains(summary, out / f"gains_s{tag}.png")
    print((out / f"summary_s{tag}.md").read_text())


def fmt_p(p):
    return f"{p:.2g}" if p >= 0.001 else "<0.001"


def write_markdown(s, path):
    L = [f"# 实验 0（shape 线）：AdaJEPA 的提升来自信息还是扰动", "",
         f"测试种子 {s['seeds']}，形状 {s['shapes']}，每个 (种子, 形状) 50 个目标；所有组规划同一批目标，逐例配对。", ""]
    groups = list(s["groups"])
    L += ["## 成功率", "", "| 组 | 说明 | " + " | ".join(groups) + " |", "|---|---|" + "---:|" * len(groups)]
    for arm in s["arms"]:
        cells = []
        for g in groups:
            e = s["groups"][g]["arms"][arm]
            cells.append(f"{e['success']*100:.1f}% ({e['k']}/{s['groups'][g]['n']})")
        L.append(f"| {arm} | {DESC[arm]} | " + " | ".join(cells) + " |")
    L += ["", "## 各对照恢复了 AdaJEPA 增益的多少（全部形状合计）", "",
          "占比 = (对照 − F_seq) / (A − F_seq)，配对 bootstrap 95% 区间；p 为精确 McNemar。", "",
          "| 组 | 占比 | 95% 区间 | 对照 − F_seq（95% 区间） | vs F_seq（仅对照/仅F，p） | vs A（仅对照/仅A，p） |", "|---|---:|---|---:|---|---|"]
    allg = s["groups"]["all"]
    gainA = allg["arms"]["A"]["success"] - allg["arms"]["F_seq"]["success"]
    L.insert(L.index("## 各对照恢复了 AdaJEPA 增益的多少（全部形状合计）") + 2,
             f"A − F_seq = {gainA*100:+.1f} 个百分点（{allg['arms']['A']['vs_F_seq']['only_arm']} / "
             f"{allg['arms']['A']['vs_F_seq']['only_F']}，p={fmt_p(allg['arms']['A']['vs_F_seq']['p'])}）。")
    for arm in s["arms"]:
        e = allg["arms"][arm]
        if "share_of_A_gain" not in e:
            continue
        sh = e["share_of_A_gain"]
        dci = e["diff_vs_F_ci"]
        L.append(f"| {arm} | {sh['point']:.2f} | [{sh['ci'][0]:.2f}, {sh['ci'][1]:.2f}] | "
                 f"{sh['gain_arm']*100:+.1f} pp [{dci[0]*100:+.1f}, {dci[1]*100:+.1f}] | "
                 f"{e['vs_F_seq']['only_arm']}/{e['vs_F_seq']['only_F']}，p={fmt_p(e['vs_F_seq']['p'])} | "
                 f"{e['vs_A']['only_arm']}/{e['vs_A']['only_A']}，p={fmt_p(e['vs_A']['p'])} |")
    L += ["", "## 按形状的占比", "", "| 组 | " + " | ".join(g for g in groups if g != "all") + " |",
          "|---|" + "---:|" * (len(groups) - 1)]
    for arm in s["arms"]:
        if "share_of_A_gain" not in allg["arms"][arm]:
            continue
        cells = []
        for g in groups:
            if g == "all":
                continue
            sh = s["groups"][g]["arms"][arm]["share_of_A_gain"]
            cells.append(f"{sh['point']:.2f} [{sh['ci'][0]:.2f}, {sh['ci'][1]:.2f}]")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", "## 每次重规划的诊断（所有形状、所有重规划）", "",
          "新帧误差比 = 当前权重 / 初始权重 在最新一帧（尚未用于训练）上的预测误差，几何平均；<1 表示适应让模型对没见过的新数据预测更准。", "",
          "| 组 | 新帧误差比 | 第3次起 | 第3次起变准的比例 | 预测改变量中位数 | 与本 episode 梯度的余弦 | 第10次时参数位移 | 第30次时参数位移 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, d in s["diagnostics"].items():
        L.append(f"| {arm} | {d['new_ratio_gmean']:.3f} | {d['new_ratio_gmean_step3plus']:.3f} | "
                 f"{d['new_improved_frac_step3plus']:.2f} | {d['func_change_median']:.2e} | {d['cos_own_median']:+.3f} | "
                 f"{d['disp_pred_step10']:.2f} | {d['disp_pred_step30']:.2f} |")
    L += ["", "## 失败 episode 是否卡住（最后 10 次重规划）", "",
          "| 组 | 失败数 | 最后10次的进展中位数(px) | 几乎无进展的比例(<5px) | 动作变化中位数 | 最好距离不再改善的起点(中位数) |",
          "|---|---:|---:|---:|---:|---:|"]
    for arm, d in s["stalls"].items():
        L.append(f"| {arm} | {d['n_failed']} | {d['progress_last10_median_px']:.1f} | {d['frac_no_progress_last10']:.2f} | "
                 f"{d['action_change_last10_median']:.3f} | {d['plateau_step_median']:.1f} |")
    path.write_text("\n".join(L) + "\n")


def plot_curves(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(s["groups"])
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    style = {"F_seq": ("#444444", "-"), "F_batch": ("#999999", ":"), "A": ("#d62728", "-"),
             "A_asc": ("#9467bd", "--"), "R_tgt": ("#1f77b4", "-"), "R_tgt_fix": ("#17becf", "--"),
             "R_sign": ("#8c564b", ":"), "W_dom": ("#2ca02c", "-"), "P_restart": ("#ff7f0e", "-.")}
    x = np.arange(1, MAX_ITER + 1)
    for ax, g in zip(axes, groups):
        for arm in s["arms"]:
            c, ls = style[arm]
            ax.plot(x, np.array(s["groups"][g]["arms"][arm]["curve"]) * 100, color=c, ls=ls, lw=1.8, label=arm)
        ax.set_title(f"{g} (n={s['groups'][g]['n']})")
        ax.set_xlabel("MPC step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cumulative success (%)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_gains(s, path):
    """Success change vs frozen (paired CI) next to the effect on not-yet-trained-frame prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["A_asc", "R_sign", "R_tgt", "R_tgt_fix", "P_restart", "W_dom", "A"]
    order = [a for a in order if a in s["arms"]]
    g = s["groups"]["all"]["arms"]
    colors = {"A": "#d62728", "A_asc": "#9467bd", "W_dom": "#2ca02c", "P_restart": "#ff7f0e"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    y = np.arange(len(order))
    for i, arm in enumerate(order):
        d = (g[arm]["success"] - g["F_seq"]["success"]) * 100
        lo, hi = (np.array(g[arm]["diff_vs_F_ci"]) * 100)
        c = colors.get(arm, "#1f77b4")
        ax1.barh(i, d, color=c, alpha=0.85)
        ax1.plot([lo, hi], [i, i], color="black", lw=1.2)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(order)
    ax1.set_xlabel("success - frozen F_seq (pp), paired 95% CI")
    ax1.set_title(f"planning success (n={s['groups']['all']['n']})")
    ax1.grid(axis="x", alpha=0.3)
    diag = s["diagnostics"]
    for i, arm in enumerate(order):
        r = diag.get(arm, {}).get("new_ratio_gmean_step3plus", float("nan"))
        if not np.isnan(r):
            ax2.barh(i, r, color=colors.get(arm, "#1f77b4"), alpha=0.85)
            ax2.text(r * 1.08, i, f"{r:.2f}", va="center", fontsize=8)
    ax2.axvline(1, color="black", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_yticks(y)
    ax2.set_yticklabels(order)
    ax2.set_xlabel("error on newest (untrained) frame, current / initial weights")
    ax2.set_title("prediction of what happens next (<1 = better)")
    ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
