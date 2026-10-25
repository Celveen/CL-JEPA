"""Held-out test of the single-domain slow learner on rope: theta0 vs learner, frozen, on 50 new instances.

Per-instance final Chamfer distance (CD) is parsed from the logs (batched MPC, resim_from_start=false: 3 MPC
iterations x 10 instances, iteration-major; the last iteration's CD is the final one) and checked against
logs.json. Also checks that no held-out start state is one of the stream's 50. Writes RESULTS.md here.

    /data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_cl_single/heldout/analyze.py
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from scipy import stats

OUT = Path(__file__).resolve().parent
STREAM = OUT.parent / "stream"
SEEDS = [104, 105, 106, 107, 108]
MODELS = {"dino_ours": "我们训的 DINO-WM", "B_h3cos_e50": "B：ResNet，history 3 + 拉直（epoch 50）"}
ARMS = {"theta0": "θ0（训练好的原模型）", "learner": "慢学习器（在 50 个 episode 的流上学完）"}
SUCCESS_CD = 0.5  # rope has no success rate (the env's test is CD < 0, never true); reports count final CD < 0.5


def chamfer_distance(x, y):  # env/deformable_env/FlexEnvWrapper.py, first 3 dims
    x = x[:, :, None, :3].repeat(1, 1, y.size(1), 1)
    y = y[:, None, :, :3].repeat(1, x.size(1), 1, 1)
    dis = torch.norm(x - y, 2, dim=3)
    return torch.mean(torch.min(dis, dim=2)[0]) + torch.mean(torch.min(dis, dim=1)[0])


def finished(model, arm, seed):
    f = OUT / model / arm / f"seed{seed}" / "logs.json"
    return f.exists() and "final_eval/" in f.read_text()


def final_cd(model, arm, seed):
    text = (OUT / "logs" / f"{model}_{arm}_seed{seed}.log").read_text(errors="replace")
    v = np.array([float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", text, re.M)])
    if len(v) != 30:
        raise SystemExit(f"{model} {arm} seed{seed}: {len(v)} CD values, expected 30")
    fin = v.reshape(3, 10)[-1]
    rows = [json.loads(l) for l in open(OUT / model / arm / f"seed{seed}" / "logs.json")]
    logged = [r for r in rows if "final_eval/mean_chamfer_distance" in r][-1]["final_eval/mean_chamfer_distance"]
    assert abs(fin.mean() - logged) < 1e-6, (model, arm, seed, fin.mean(), logged)
    return fin


def targets(path):
    t = pickle.load(open(path, "rb"))
    g = np.asarray(t["state_g"], dtype=np.float64)
    return np.asarray(t["state_0"], dtype=np.float64).reshape(len(g), g.shape[1], g.shape[2]), g


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def paired(a, b):
    d = a - b
    ci = stats.t.ppf(0.975, len(d) - 1) * se(d)
    return (f"{d.mean():+.3f}（{d.mean() - ci:+.3f}，{d.mean() + ci:+.3f}） | {(d < 0).sum()} / {(d > 0).sum()} | "
            f"{stats.wilcoxon(d).pvalue:.3g} | {stats.ttest_rel(a, b).pvalue:.3g}")


def success_cmp(a, b):
    """Success-rate difference (CD < SUCCESS_CD) in points and the exact McNemar test on the discordant pairs."""
    sa, sb = a < SUCCESS_CD, b < SUCCESS_CD
    only_a, only_b = int((sa & ~sb).sum()), int((~sa & sb).sum())
    p = stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return f"{(sa.mean() - sb.mean()) * 100:+.0f} 个百分点 | {only_a} / {only_b} | {p:.3g}"


def main():
    shards = [s for s in SEEDS if all(finished(m, a, s) for m in MODELS for a in ARMS)]
    if not shards:
        raise SystemExit("no shard finished by every model and arm yet")
    held = [targets(OUT / "targets" / f"seed{s}.pkl") for s in shards]
    s0 = np.concatenate([h[0] for h in held])
    g = np.concatenate([h[1] for h in held])
    start = np.array([float(chamfer_distance(torch.tensor(s0[i : i + 1]), torch.tensor(g[i : i + 1]))) for i in range(len(g))])
    stream = np.concatenate([targets(STREAM / f"chunk_{k}.pkl")[0] for k in range(5)])
    seen = np.array([any(np.array_equal(x, y) for y in stream) for x in s0])  # start states the learner has seen
    overlap = int(seen.sum())
    n = len(start)
    L = ["# rope 留出集测试：慢学习器 vs θ0（冻结规划）", "",
         f"{len(shards)} 份、共 {n} 个新实例（种子 {shards}）。所有模型和组用的是同一份起点和目标（`targets/`）。"
         f"其中和持续学习流里那 50 个实例起点相同的有 {overlap} 个。推之前的平均 CD 是 {start.mean():.3f}。"
         f"主指标是最终 CD，越低越好；成功率是最终 CD < {SUCCESS_CD} 的实例比例"
         "（rope 环境本身的判定是 CD < 0，永远不成立，所以这个门槛是我们定的）。", ""]
    for m, mname in MODELS.items():
        F = {a: np.concatenate([final_cd(m, a, s) for s in shards]) for a in ARMS}
        L += [f"## {mname}", "",
              f"| 组 | 最终 CD（均值 ± 标准误） | 中位数 | 成功率（CD < {SUCCESS_CD}） |", "|---|---|---|---|"]
        for a, aname in ARMS.items():
            f = F[a]
            L.append(f"| {aname} | {f.mean():.3f} ± {se(f):.3f} | {np.median(f):.3f} | "
                     f"{(f < SUCCESS_CD).mean() * 100:.0f}%（{(f < SUCCESS_CD).sum()}/{n}） |")
        L += ["", "| 对比 | CD 平均差（95% 置信区间） | CD 前者更好 / 后者更好 | Wilcoxon p | 配对 t 检验 p "
              "| 成功率差 | 只有前者成功 / 只有后者成功 | McNemar p |", "|---|---|---|---|---|---|---|---|",
              f"| 慢学习器 − θ0 | {paired(F['learner'], F['theta0'])} | {success_cmp(F['learner'], F['theta0'])} |"]
        if overlap:
            u = ~seen
            L.append(f"| 慢学习器 − θ0，去掉和流重叠的 {overlap} 个实例（n = {u.sum()}） | {paired(F['learner'][u], F['theta0'][u])} "
                     f"| {success_cmp(F['learner'][u], F['theta0'][u])} |")
        q = {a: np.percentile(F[a], [25, 50, 75]) for a in ARMS}
        L += ["", f"- 最终 CD 的四分位数（25% / 50% / 75%）：θ0 {q['theta0'][0]:.3f} / {q['theta0'][1]:.3f} / {q['theta0'][2]:.3f}，"
              f"慢学习器 {q['learner'][0]:.3f} / {q['learner'][1]:.3f} / {q['learner'][2]:.3f}；"
              f"CD > 1.0 的大失败：θ0 {(F['theta0'] > 1).sum()} 个，慢学习器 {(F['learner'] > 1).sum()} 个。",
              "- 各份的平均最终 CD（θ0 → 慢学习器）：" + "，".join(
                  f"{s}：{F['theta0'][10 * j:10 * j + 10].mean():.2f} → {F['learner'][10 * j:10 * j + 10].mean():.2f}"
                  for j, s in enumerate(shards)) + "。", ""]
    text = "\n".join(L) + "\n"
    (OUT / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
