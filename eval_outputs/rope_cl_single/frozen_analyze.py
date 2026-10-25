"""Frozen rope evaluations of theta0 vs the final single-domain slow learner (output dirs of frozen_eval.sh).

    /data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_cl_single/frozen_analyze.py <dir> [<dir> ...]

Each dir gets its own RESULTS.md. Instances whose start state is one of the stream's 50 (which the learner trained
on) are excluded from 'new' dirs, and instances repeated across dirs are counted once; with two or more 'new' dirs
the pooled comparison goes to HELDOUT_POOLED.md next to this file. In a 'stream' dir theta0 reruns ../<model>/frozen
exactly, so the two theta0 runs differ only by run-to-run noise.
Main metric: final Chamfer distance (CD, lower is better); success rate = final CD < 0.5 (the rope env's own test,
CD < 0, is never true). Per-instance final CD comes from the logs (batched MPC, resim_from_start=false: 3 MPC
iterations x 10 instances, iteration-major) and is checked against logs.json.
"""
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

CL = Path(__file__).resolve().parent
MODELS = {"dino_ours": "我们训的 DINO-WM", "B_h3cos_e50": "B：ResNet，history 3 + 拉直（epoch 50）"}
SUCCESS_CD = 0.5


def chamfer_distance(x, y):  # env/deformable_env/FlexEnvWrapper.py, first 3 dims
    x = x[:, :, None, :3].repeat(1, 1, y.size(1), 1)
    y = y[:, None, :, :3].repeat(1, x.size(1), 1, 1)
    dis = torch.norm(x - y, 2, dim=3)
    return torch.mean(torch.min(dis, dim=2)[0]) + torch.mean(torch.min(dis, dim=1)[0])


def parse_final(log, logs_json=None):
    v = np.array([float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", Path(log).read_text(errors="replace"), re.M)])
    if len(v) != 30:
        raise SystemExit(f"{log}: {len(v)} CD values, expected 30")
    fin = v.reshape(3, 10)[-1]
    if logs_json is not None:
        rows = [json.loads(l) for l in open(logs_json)]
        logged = [r for r in rows if "final_eval/mean_chamfer_distance" in r][-1]["final_eval/mean_chamfer_distance"]
        assert abs(fin.mean() - logged) < 1e-6, (log, fin.mean(), logged)
    return fin


def targets(path):
    t = pickle.load(open(path, "rb"))
    g = np.asarray(t["state_g"], dtype=np.float64)
    return np.asarray(t["state_0"], dtype=np.float64).reshape(len(g), g.shape[1], g.shape[2]), g


def start_cd(s0, g):
    return np.array([float(chamfer_distance(torch.tensor(s0[i : i + 1]), torch.tensor(g[i : i + 1]))) for i in range(len(g))])


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def paired(a, b):
    d = a - b
    ci = stats.t.ppf(0.975, len(d) - 1) * se(d)
    sa, sb = a < SUCCESS_CD, b < SUCCESS_CD
    only_a, only_b = int((sa & ~sb).sum()), int((~sa & sb).sum())
    p_mc = stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return (f"{d.mean():+.3f}（{d.mean() - ci:+.3f}，{d.mean() + ci:+.3f}） | {(d < 0).sum()} / {(d > 0).sum()} | "
            f"{stats.wilcoxon(d).pvalue:.3g} | {stats.ttest_rel(a, b).pvalue:.3g} | "
            f"{(sa.mean() - sb.mean()) * 100:+.0f} 个百分点 | {only_a} / {only_b} | {p_mc:.3g}")


HEAD = ("| 对比 | n | CD 平均差（95% 置信区间） | CD 前者更好 / 后者更好 | Wilcoxon p | 配对 t 检验 p "
        "| 成功率差 | 只有前者成功 / 只有后者成功 | McNemar p |\n|---|---|---|---|---|---|---|---|---|")


def arm_row(name, f):
    return (f"| {name} | {f.mean():.3f} ± {se(f):.3f} | {np.median(f):.3f} | "
            f"{(f < SUCCESS_CD).mean() * 100:.0f}%（{(f < SUCCESS_CD).sum()}/{len(f)}） | {(f > 1).sum()} |")


ARM_HEAD = f"| 组 | 最终 CD（均值 ± 标准误） | 中位数 | 成功率（CD < {SUCCESS_CD}） | CD > 1.0 的大失败 |\n|---|---|---|---|---|"


def load_dir(d):
    d = Path(d).resolve()
    meta = json.loads((d / "meta.json").read_text())
    seeds, mode = meta["seeds"], meta["mode"]
    tg = [targets(CL / "stream" / f"chunk_{j}.pkl") if mode == "stream" else targets(d / "targets" / f"seed{s}.pkl")
          for j, s in enumerate(seeds)]
    s0, g = np.concatenate([t[0] for t in tg]), np.concatenate([t[1] for t in tg])
    F = {}
    for m in MODELS:
        for a in ("theta0", "learner"):
            F[m, a] = np.concatenate([parse_final(d / "logs" / f"{m}_{a}_seed{s}.log", d / m / a / f"seed{s}" / "logs.json")
                                      for s in seeds])
        if mode == "stream":  # the first theta0 run: the stream's frozen arm
            F[m, "theta0_first"] = np.concatenate([parse_final(CL / "logs" / f"{m}_frozen_chunk_{j}.log") for j in range(5)])
    return {"dir": d, "mode": mode, "seeds": seeds, "s0": s0, "start": start_cd(s0, g), "F": F}


def report_dir(r, stream_s0):
    L = [f"# rope 冻结评估：θ0 vs 慢学习器（`{r['dir'].name}`）", "",
         f"实例：{'持续学习流里的 50 个实例（慢学习器训练时见过它们）' if r['mode'] == 'stream' else '新抽的实例'}，"
         f"种子 {r['seeds']}，共 {len(r['start'])} 个，推之前平均 CD {r['start'].mean():.3f}。"
         f"冻结批量规划，接着仿真。慢学习器 = 流跑完后的 `<模型>/cl/chunk_4/continual_state.pt`。"
         f"主指标是最终 CD（越低越好）；成功率是最终 CD < {SUCCESS_CD} 的比例。", ""]
    keep = np.ones(len(r["start"]), bool)
    if r["mode"] == "new":
        keep = np.array([not any(np.array_equal(x, y) for y in stream_s0) for x in r["s0"]])
        L += [f"其中起点和流里重合的有 {(~keep).sum()} 个，下面的比较都去掉了它们（n = {keep.sum()}）。", ""]
    for m, mname in MODELS.items():
        F = {k[1]: v[keep] for k, v in r["F"].items() if k[0] == m}
        L += [f"## {mname}", "", ARM_HEAD, arm_row("θ0", F["theta0"]), arm_row("慢学习器", F["learner"])]
        if "theta0_first" in F:
            L.append(arm_row("θ0（第一次运行，即流的冻结组）", F["theta0_first"]))
        L += ["", HEAD, f"| 慢学习器 − θ0 | {keep.sum()} | {paired(F['learner'], F['theta0'])} |"]
        if "theta0_first" in F:
            L += [f"| 慢学习器 − θ0（第一次运行） | {keep.sum()} | {paired(F['learner'], F['theta0_first'])} |",
                  f"| θ0 重跑 − θ0 第一次运行（纯运行噪声） | {keep.sum()} | {paired(F['theta0'], F['theta0_first'])} |"]
        L.append("")
    text = "\n".join(L) + "\n"
    (r["dir"] / "RESULTS.md").write_text(text)
    return text


def pooled(rs, stream_s0):
    seen_s0, idx = [], []
    for r in rs:
        k = []
        for i, x in enumerate(r["s0"]):
            if any(np.array_equal(x, y) for y in stream_s0) or any(np.array_equal(x, y) for y in seen_s0):
                continue
            seen_s0.append(x)
            k.append(i)
        idx.append(np.array(k, int))
    n = sum(len(k) for k in idx)
    L = ["# rope 留出集合并：θ0 vs 慢学习器（冻结规划）", "",
         f"合并 {', '.join('`' + r['dir'].name + '`' for r in rs)}，去掉起点和流重合的、以及两份之间重复的实例后共 {n} 个新实例。"
         f"主指标是最终 CD（越低越好）；成功率是最终 CD < {SUCCESS_CD} 的比例。", ""]
    for m, mname in MODELS.items():
        F = {a: np.concatenate([r["F"][m, a][k] for r, k in zip(rs, idx)]) for a in ("theta0", "learner")}
        L += [f"## {mname}", "", ARM_HEAD, arm_row("θ0", F["theta0"]), arm_row("慢学习器", F["learner"]), "", HEAD,
              f"| 慢学习器 − θ0 | {n} | {paired(F['learner'], F['theta0'])} |"]
        for r, k in zip(rs, idx):
            L.append(f"| 慢学习器 − θ0，只看 `{r['dir'].name}` | {len(k)} | {paired(r['F'][m, 'learner'][k], r['F'][m, 'theta0'][k])} |")
        L.append("")
    text = "\n".join(L) + "\n"
    (CL / "HELDOUT_POOLED.md").write_text(text)
    return text


def main():
    stream_s0 = np.concatenate([targets(CL / "stream" / f"chunk_{k}.pkl")[0] for k in range(5)])
    rs = [load_dir(d) for d in sys.argv[1:]]
    for r in rs:
        print(report_dir(r, stream_s0))
    new = [r for r in rs if r["mode"] == "new"]
    if len(new) >= 2:
        print(pooled(new, stream_s0))


if __name__ == "__main__":
    main()
