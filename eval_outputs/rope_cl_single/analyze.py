"""Single-domain continual learning on rope: frozen vs AdaJEPA vs CL, episode by episode (rope_cl_single).

Per-episode final Chamfer distance (CD) is parsed from the logs ("CD:" lines, one per MPC iteration and episode;
resim_from_start=false, so the last iteration's CD is the final one) and checked against logs.json. Only chunks
finished by all three arms of a model are used. Writes RESULTS.md next to this file.

    /data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_cl_single/analyze.py
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from scipy import stats

OUT = Path(__file__).resolve().parent
CHUNKS = [0, 1, 2, 3, 4]
ITERS = 3
MODELS = {"dino_ours": "我们训的 DINO-WM", "B_h3cos_e50": "B：ResNet，history 3 + 拉直（epoch 50）"}
ARMS = {"frozen": "冻结 θ0", "adajepa": "AdaJEPA（每个 episode 重置）", "cl": "持续学习（单领域慢学习器 + AdaJEPA）",
        "cl_noada": "只用慢学习器（不加 AdaJEPA）"}
CORE = ["frozen", "adajepa", "cl"]  # chunks are reported once these three finished them; other arms join when done
SUCCESS_CD = 0.5  # rope has no success rate (the env's test is CD < 0, never true); reports count final CD < 0.5
N50 = OUT.parent / "rope_dset_n50"  # same 50 instances, frozen with resim_from_start=true


def chamfer_distance(x, y):  # env/deformable_env/FlexEnvWrapper.py, first 3 dims
    x = x[:, :, None, :3].repeat(1, 1, y.size(1), 1)
    y = y[:, None, :, :3].repeat(1, x.size(1), 1, 1)
    dis = torch.norm(x - y, 2, dim=3)
    return torch.mean(torch.min(dis, dim=2)[0]) + torch.mean(torch.min(dis, dim=1)[0])


def finished(model, arm, k):
    f = OUT / model / arm / f"chunk_{k}" / "logs.json"
    return f.exists() and "final_eval/" in f.read_text()


def final_cd(model, arm, k):
    """Final CD of the chunk's 10 episodes, in episode order."""
    text = (OUT / "logs" / f"{model}_{arm}_chunk_{k}.log").read_text(errors="replace")
    v = np.array([float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", text, re.M)])
    if len(v) != 10 * ITERS:
        raise SystemExit(f"{model} {arm} chunk {k}: {len(v)} CD values, expected {10 * ITERS}")
    # batched (frozen): iteration-major; per-sample planners (adajepa, cl): episode-major
    fin = v.reshape(ITERS, 10)[-1] if arm == "frozen" else v.reshape(10, ITERS)[:, -1]
    rows = [json.loads(l) for l in open(OUT / model / arm / f"chunk_{k}" / "logs.json")]
    logged = [r for r in rows if "final_eval/mean_chamfer_distance" in r][-1]["final_eval/mean_chamfer_distance"]
    assert abs(fin.mean() - logged) < 1e-6, (model, arm, k, fin.mean(), logged)
    return fin


def start_cd(k):
    t = pickle.load(open(OUT / "stream" / f"chunk_{k}.pkl", "rb"))
    g = np.asarray(t["state_g"], dtype=np.float64)
    s0 = np.asarray(t["state_0"], dtype=np.float64).reshape(len(g), g.shape[1], g.shape[2])
    return np.array([float(chamfer_distance(torch.tensor(s0[i : i + 1]), torch.tensor(g[i : i + 1]))) for i in range(len(g))])


def cl_records(model, chunks):
    recs = []
    for k in chunks:
        recs += [json.loads(l) for l in open(OUT / model / "cl" / f"chunk_{k}" / "continual_episodes.jsonl")]
    return recs


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def paired(a, b):
    d = a - b
    ci = stats.t.ppf(0.975, len(d) - 1) * se(d)
    p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
    return (f"{d.mean():+.3f}（{d.mean() - ci:+.3f}，{d.mean() + ci:+.3f}） | {(d < 0).sum()} / {(d > 0).sum()} | "
            f"{p_w:.3g} | {stats.ttest_rel(a, b).pvalue:.3g}")


def success_cmp(a, b):
    """Success-rate difference (CD < SUCCESS_CD) in points and the exact McNemar test on the discordant pairs."""
    sa, sb = a < SUCCESS_CD, b < SUCCESS_CD
    only_a, only_b = int((sa & ~sb).sum()), int((~sa & sb).sum())
    p = stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return f"{(sa.mean() - sb.mean()) * 100:+.0f} 个百分点 | {only_a} / {only_b} | {p:.3g}"


def report(model):
    chunks = [k for k in CHUNKS if all(finished(model, a, k) for a in CORE)]
    if not chunks:
        return [f"## {MODELS[model]}", "", "还没有三组都跑完的段。", ""]
    arms = CORE + [a for a in ARMS if a not in CORE and all(finished(model, a, k) for k in chunks)]
    F = {a: np.concatenate([final_cd(model, a, k) for k in chunks]) for a in arms}
    start = np.concatenate([start_cd(k) for k in chunks])
    n = len(start)
    L = [f"## {MODELS[model]}", "",
         f"用了 {len(chunks)} 段、共 {n} 个 episode（段 {chunks}）。推之前的平均 CD 是 {start.mean():.3f}。"
         f"主指标是最终 CD，越低越好；成功率是最终 CD < {SUCCESS_CD} 的 episode 比例"
         "（rope 环境本身的判定是 CD < 0，永远不成立，所以这个门槛是我们定的）。", "",
         f"| 组 | 最终 CD（均值 ± 标准误） | 中位数 | 成功率（CD < {SUCCESS_CD}） |", "|---|---|---|---|"]
    for a in arms:
        f = F[a]
        L.append(f"| {ARMS[a]} | {f.mean():.3f} ± {se(f):.3f} | {np.median(f):.3f} | "
                 f"{(f < SUCCESS_CD).mean() * 100:.0f}%（{(f < SUCCESS_CD).sum()}/{n}） |")
    L += ["", "**逐 episode 配对比较**（CD 差为负、成功率差为正，表示前者更好）：", "",
          "| 对比 | CD 平均差（95% 置信区间） | CD 前者更好 / 后者更好 | Wilcoxon p | 配对 t 检验 p "
          "| 成功率差 | 只有前者成功 / 只有后者成功 | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    pairs = [("cl", "adajepa"), ("cl", "frozen"), ("adajepa", "frozen")]
    if "cl_noada" in arms:
        pairs += [("cl_noada", "frozen"), ("cl", "cl_noada")]
    for a, b in pairs:
        L.append(f"| {ARMS[a]} − {ARMS[b]} | {paired(F[a], F[b])} | {success_cmp(F[a], F[b])} |")
    L += ["", "**按段看**（每段 10 个 episode，段号就是在流里的先后顺序；持续学习组的慢学习器越往后见过的 episode 越多）：", "",
          "| 段 | 起点 | " + " | ".join(ARMS[a] for a in arms) + " | 持续学习 − AdaJEPA |", "|---|---|" + "---|" * (len(arms) + 1)]
    for j, k in enumerate(chunks):
        sl = slice(10 * j, 10 * j + 10)
        L.append(f"| {k} | {start[sl].mean():.3f} | " + " | ".join(f"{F[a][sl].mean():.3f}" for a in arms) +
                 f" | {(F['cl'][sl] - F['adajepa'][sl]).mean():+.3f} |")
    recs = cl_records(model, chunks)
    read = np.array([r["read"] is not None for r in recs])
    vr = np.array([r["val_ratio"] for r in recs], dtype=float)
    L += ["", "**慢学习器本身学到了没有**（预测层面，比规划结果的噪声小得多）：", ""]
    first_read = next((r["episode"] for r in recs if r["read"] is not None), None)
    L.append(f"- 读取了慢学习器的 episode：{read.sum()} / {n}；第一次读取是在流里的第 {first_read} 个 episode（从 0 数）。"
             if first_read is not None else f"- 慢学习器一直没有通过成熟门，{n} 个 episode 全部用的是 θ0。")
    ok = ~np.isnan(vr)
    if ok.any():
        gm = lambda x: float(np.exp(np.mean(np.log(x))))
        L.append(f"- 验证比（慢学习器在一个它还没训练过的新 episode 上的 5 步开环潜空间误差 ÷ θ0 的误差，< 1 表示比 θ0 预测得准）："
                 f"共 {ok.sum()} 次，几何平均 {gm(vr[ok]):.3f}，中位数 {np.median(vr[ok]):.3f}，< 1 的有 {(vr[ok] < 1).sum()} 次。")
        rows = []
        for j, k in enumerate(chunks):
            v = vr[10 * j : 10 * j + 10]
            v = v[~np.isnan(v)]
            if len(v):
                rows.append(f"段 {k}：{gm(v):.3f}")
        L.append("- 各段验证比的几何平均：" + "，".join(rows) + "。")
    if read.any() and (~read).any():
        L += ["", "持续学习组按「这个 episode 读没读慢学习器」分开，和 AdaJEPA 组在同一批 episode 上配对比较：", "",
              "| 子集 | episode 数 | 持续学习 − AdaJEPA（95% 置信区间） | 前者更好 / 后者更好 | Wilcoxon p | 配对 t 检验 p |",
              "|---|---|---|---|---|---|"]
        for label, m in [("读了慢学习器", read), ("用的是 θ0", ~read)]:
            if m.sum() >= 3:
                L.append(f"| {label} | {m.sum()} | {paired(F['cl'][m], F['adajepa'][m])} |")
        if read.sum() >= 3:
            L += ["", "同一批「读了慢学习器」的 episode 上，和冻结组比：", "",
                  "| 对比 | episode 数 | 平均差（95% 置信区间） | 前者更好 / 后者更好 | Wilcoxon p | 配对 t 检验 p |",
                  "|---|---|---|---|---|---|"]
            for a in ("cl", "adajepa"):
                L.append(f"| {ARMS[a]} − 冻结 θ0 | {read.sum()} | {paired(F[a][read], F['frozen'][read])} |")
    fast = []
    for a in ("adajepa", "cl"):
        cur, ini = [], []
        for k in chunks:
            for line in open(OUT / model / a / f"chunk_{k}" / "per_iter_logs.json"):
                r = json.loads(line)
                if "loss_new_cur" in r and r["step"] > 1:  # replan >= 2: the weights have seen earlier transitions
                    cur.append(r["loss_new_cur"])
                    ini.append(r["loss_new_init"])
        if cur:
            q = np.array(cur) / np.array(ini)
            fast.append(f"{ARMS[a]} {np.exp(np.mean(np.log(q))):.3f}（{(q < 1).sum()}/{len(q)} 次比 θ0 准）")
    if fast:
        L += ["", "- episode 内、在用新转移更新之前，当前权重对这条新转移的 1 步预测误差 ÷ θ0 的（几何平均）：" + "；".join(fast) + "。"]
    ref = [N50 / "logs" / f"{model}_seed{s}.log" for s in (99, 100, 101, 102, 103)]
    if all(f.exists() for f in ref) and len(chunks) == 5:
        resim = np.concatenate([np.array([float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", f.read_text(errors="replace"), re.M)])
                                .reshape(4, 10)[-1] for f in ref])
        L += ["", f"- 协议对照：同一个模型、同一批 50 个实例、都冻结，`rope_dset_n50` 每次迭代都从头重新仿真全部推动，"
              f"平均最终 CD 是 {resim.mean():.3f}；这里的冻结组接着仿真，是 {F['frozen'].mean():.3f}。"
              f"逐实例配对（从头重新仿真 − 接着仿真）：{paired(resim, F['frozen'])}"
              "（依次是平均差和 95% 置信区间、前者更好 / 后者更好、Wilcoxon p、配对 t 检验 p）。"
              "FleX 不确定，从头重新仿真时，执行到下一个计划的状态和计划所依据的状态并不相同。"]
    return L + [""]


def main():
    L = ["# rope 单领域在线持续学习：结果", ""]
    for model in MODELS:
        L += report(model)
    text = "\n".join(L) + "\n"
    (OUT / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
