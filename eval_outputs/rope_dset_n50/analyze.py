"""Paired comparison of the two DINO-WM rope models on the 50 shared instances (rope_dset_n50).

Per-instance Chamfer distances (CD) are parsed from the logs ("CD:" lines: 3 MPC iterations + the final
evaluation, 10 instances each) and checked against logs.json. Start CD (before any push) is computed from the
shared goal files. Writes RESULTS.md next to this file.

    /data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_dset_n50/analyze.py
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from scipy import stats

OUT = Path(__file__).resolve().parent
SEEDS = [99, 100, 101, 102, 103]
MODELS = {"dino_ours": "我们训的 DINO-WM", "ddpwm": "DDP-WM 提供的 DINO-WM",
          "A_h1_e100": "A：ResNet，history 1（epoch 100）", "B_h3cos_e50": "B：ResNet，history 3 + 拉直（epoch 50）"}
OLD_LOG = OUT.parent / "gen_logs" / "rope_dset_dino_ours.log"  # earlier 10-instance run = shard seed 99
OLD_TARGETS = OUT.parent / "rope_dset" / "dino_ours" / "plan_targets.pkl"


def chamfer_distance(x, y):  # env/deformable_env/FlexEnvWrapper.py, first 3 dims
    x = x[:, :, None, :3].repeat(1, 1, y.size(1), 1)
    y = y[:, None, :, :3].repeat(1, x.size(1), 1, 1)
    dis = torch.norm(x - y, 2, dim=3)
    return torch.mean(torch.min(dis, dim=2)[0]) + torch.mean(torch.min(dis, dim=1)[0])


def cd_blocks(text, where):
    v = [float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", text, re.M)]
    if len(v) != 40:
        raise SystemExit(f"{where}: {len(v)} CD values, expected 40 (3 MPC iterations + final, 10 instances)")
    return np.array(v).reshape(4, 10)


def finished(model, seed):
    f = OUT / model / f"seed{seed}" / "logs.json"
    return f.exists() and "final_eval/" in f.read_text()


def load(model, seed):
    b = cd_blocks((OUT / "logs" / f"{model}_seed{seed}.log").read_text(errors="replace"), f"{model} seed{seed}")
    rows = [json.loads(l) for l in open(OUT / model / f"seed{seed}" / "logs.json")]
    fin = [r for r in rows if "final_eval/mean_chamfer_distance" in r][-1]["final_eval/mean_chamfer_distance"]
    assert abs(b[-1].mean() - fin) < 1e-6, (model, seed, b[-1].mean(), fin)
    return b


def start_cd(seed):
    t = pickle.load(open(OUT / "targets" / f"seed{seed}.pkl", "rb"))
    s0 = np.asarray(t["state_0"], dtype=np.float64)
    g = np.asarray(t["state_g"], dtype=np.float64)
    s0 = s0.reshape(len(s0), g.shape[1], g.shape[2])
    return np.array([float(chamfer_distance(torch.tensor(s0[i : i + 1]), torch.tensor(g[i : i + 1]))) for i in range(len(g))])


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


SUCCESS_CD = 0.5  # rope has no success rate (the env's test is CD < 0, never true); reports count final CD < 0.5


def success_cmp(a, b):
    """Success-rate difference (CD < SUCCESS_CD) in points and the exact McNemar test on the discordant pairs."""
    sa, sb = a < SUCCESS_CD, b < SUCCESS_CD
    only_a, only_b = int((sa & ~sb).sum()), int((~sa & sb).sum())
    p = stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return f"{(sa.mean() - sb.mean()) * 100:+.0f} 个百分点 | {only_a} / {only_b} | {p:.3g}"


def main():
    done = [m for m in MODELS if all(finished(m, s) for s in SEEDS)]  # models whose 5 shards are all finished
    B = {m: np.concatenate([load(m, s) for s in SEEDS], axis=1) for m in done}  # (4, 50)
    start = np.concatenate([start_cd(s) for s in SEEDS])
    final = {m: B[m][-1] for m in done}
    best = {m: B[m][:3].min(0) for m in done}
    n = len(start)
    L = [f"# rope，数据集目标协议，50 个共享实例：结果", "",
         f"每个实例的起点和目标在所有模型之间逐位相同（`targets/`）。推之前的平均 CD 是 {start.mean():.3f}。"
         f"主指标是最终 CD，越低越好；成功率是最终 CD < {SUCCESS_CD} 的实例比例"
         "（rope 环境本身的判定是 CD < 0，永远不成立，所以这个门槛是我们定的）。", "",
         f"| 模型 | 最终 CD（均值 ± 标准误） | 中位数 | 成功率（CD < {SUCCESS_CD}） | 3 次迭代里取最好 | 比起点更接近目标的实例数 |",
         "|---|---|---|---|---|---|"]
    for m in done:
        L.append(f"| {MODELS[m]} | {final[m].mean():.3f} ± {se(final[m]):.3f} | {np.median(final[m]):.3f} | "
                 f"{(final[m] < SUCCESS_CD).mean() * 100:.0f}%（{(final[m] < SUCCESS_CD).sum()}/{n}） | "
                 f"{best[m].mean():.3f} ± {se(best[m]):.3f} | {(final[m] < start).sum()} / {n} |")
    L += ["", "**逐实例配对比较**（我们训的 DINO-WM − 另一个模型；CD 差为负、成功率差为正，表示我们训的更好）：", "",
          "| 对比 | 指标 | CD 平均差（95% 置信区间） | CD 我们训的更好 / 对方更好 | Wilcoxon 符号秩检验 p | 配对 t 检验 p "
          "| 成功率差 | 只有我们训的成功 / 只有对方成功 | McNemar p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for other in [m for m in done if m != "dino_ours"]:
        for label, D in [("最终 CD", final), ("3 次迭代里取最好", best)]:
            d = D["dino_ours"] - D[other]
            ci = stats.t.ppf(0.975, n - 1) * se(d)
            L.append(f"| 对 {MODELS[other]} | {label} | {d.mean():+.3f}（{d.mean() - ci:+.3f}，{d.mean() + ci:+.3f}） | "
                     f"{(d < 0).sum()} / {(d > 0).sum()} | {stats.wilcoxon(d).pvalue:.3g} | "
                     f"{stats.ttest_rel(D['dino_ours'], D[other]).pvalue:.3g} | {success_cmp(D['dino_ours'], D[other])} |")
    L += ["", "**各份的最终 CD 均值**（每份 10 个实例）：", "",
          "| 份（种子） | 起点 | " + " | ".join(MODELS[m] for m in done) + " |", "|---|---|" + "---|" * len(done)]
    for k, s in enumerate(SEEDS):
        sl = slice(10 * k, 10 * k + 10)
        L.append(f"| {s} | {start[sl].mean():.3f} | " + " | ".join(f"{final[m][sl].mean():.3f}" for m in done) + " |")
    d = final["dino_ours"] - final["ddpwm"]
    sd = d.std(ddof=1)
    need = {delta: int(np.ceil(((1.96 + 0.84) * sd / delta) ** 2)) for delta in (0.1, 0.2, 0.3)}
    worse = {m: int((B[m][-1] > best[m] + 1e-9).sum()) for m in done}
    L += ["", "**评测噪声**：", "",
          f"- 两个模型在同一实例上的最终 CD 相关系数 r = {np.corrcoef(final['dino_ours'], final['ddpwm'])[0, 1]:.2f}，"
          f"逐实例配对差的标准差是 {sd:.3f}。按这个噪声估算（双侧 5%、80% 把握），要分辨 0.1、0.2、0.3 的平均差，"
          f"大约分别需要 {need[0.1]}、{need[0.2]}、{need[0.3]} 个实例。",
          f"- 「3 次迭代里取最好」是事后按真实距离挑出来的，偏乐观：最终状态比自己最好那次迭代更差的实例数——"
          + "，".join(f"{MODELS[m]} {worse[m]}/{n}" for m in done) + "。"]
    if OLD_LOG.exists() and OLD_TARGETS.exists():
        old = cd_blocks(OLD_LOG.read_text(errors="replace"), "old run")[-1]
        new = final["dino_ours"][:10]
        go = np.asarray(pickle.load(open(OLD_TARGETS, "rb"))["state_g"], dtype=np.float64)
        gn = np.asarray(pickle.load(open(OUT / "targets" / "seed99.pkl", "rb"))["state_g"], dtype=np.float64)
        gcd = max(float(chamfer_distance(torch.tensor(go[i : i + 1]), torch.tensor(gn[i : i + 1]))) for i in range(len(gn)))
        r = np.corrcoef(old, new)[0, 1]
        L += [f"- 复现检查：种子 99 这一份就是旧 10 实例 run 的实例。起点相同，重新回放的目标和旧的最多只差 CD {gcd:.3f}，几乎一样。"
              f"但我们训的模型在这 10 个实例上的最终 CD，旧 run 均值 {old.mean():.3f}、这次 {new.mean():.3f}；"
              f"逐实例相关系数只有 r = {r:.2f}，逐实例差的绝对值中位数达 {np.median(np.abs(old - new)):.3f}。"
              "也就是说，同一个模型在同一道题上重跑，结果也能差很多（CEM 采样、GPU 浮点和 FleX 的不确定性在 15 次推动中被放大），"
              "只看 10 个实例的比较基本分不出好坏。"]
    text = "\n".join(L) + "\n"
    (OUT / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
