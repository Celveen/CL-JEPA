"""Granular on its own: our rope DINO-WM (theta0, frozen) on 50 granular instances (granular_single/).

Per-instance final Chamfer distance (CD; batched MPC, resim_from_start=false: 3 iterations x 10 instances,
iteration-major) is parsed from the logs and checked against logs.json; the start CD (before any push) is computed
from the target files on the GPU (12774 particles). Granular's CD scale differs from rope's, so the report gives the
share of the start-to-goal distance closed besides the final CD, and how many instances start below the rope success
threshold (0.5). Writes granular_single/RESULTS.md.

    CUDA_VISIBLE_DEVICES=7 /data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_granular/granular_single_analyze.py
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
SEEDS = [301, 302, 303, 304, 305]


def cd(a, b, dev):  # FlexEnvWrapper.chamfer_distance (first 3 dims), via cdist
    a = torch.tensor(a[:, :3], device=dev, dtype=torch.float64)
    b = torch.tensor(b[:, :3], device=dev, dtype=torch.float64)
    d = torch.cdist(a, b)
    return float(d.min(1)[0].mean() + d.min(0)[0].mean())


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    start, final, s0_all = [], [], []
    for s in SEEDS:
        t = pickle.load(open(OUT / "targets" / f"granular_seed{s}.pkl", "rb"))
        g = np.asarray(t["state_g"], dtype=np.float64)
        s0 = np.asarray(t["state_0"], dtype=np.float64).reshape(g.shape)
        start += [cd(s0[i], g[i], dev) for i in range(len(g))]
        s0_all += [x.tobytes() for x in s0]
        text = (OUT / "logs" / f"granular_single_seed{s}.log").read_text(errors="replace")
        v = np.array([float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", text, re.M)])
        assert len(v) == 30, (s, len(v))
        fin = v.reshape(3, 10)[-1]
        rows = [json.loads(l) for l in open(OUT / "granular_single" / f"seed{s}" / "logs.json")]
        logged = [r for r in rows if "final_eval/mean_chamfer_distance" in r][-1]["final_eval/mean_chamfer_distance"]
        assert abs(fin.mean() - logged) < 1e-6, (s, fin.mean(), logged)
        final += list(fin)
    start, final = np.array(start), np.array(final)
    closed = 1 - final / start
    n = len(final)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    L = ["# granular 单独测试：我们训的 rope DINO-WM（θ0，冻结）在 granular 上规划", "",
         f"{n} 个 granular 实例（种子 {SEEDS}，其中起点各不相同的有 {len(set(s0_all))} 个）；冻结批量规划，接着仿真，"
         "CEM 300 样本、取前 30、10 次迭代，MPC 3 次迭代 × 每次 5 推。CD 越低越好。", "",
         "| | 均值 ± 标准误 | 中位数 | 范围 |", "|---|---|---|---|",
         f"| 推之前的 CD（起点到目标） | {start.mean():.3f} ± {se(start):.3f} | {np.median(start):.3f} | {start.min():.3f} ～ {start.max():.3f} |",
         f"| 最终 CD | {final.mean():.3f} ± {se(final):.3f} | {np.median(final):.3f} | {final.min():.3f} ～ {final.max():.3f} |",
         f"| 缩短的距离占比（1 − 最终 / 起点） | {closed.mean() * 100:.0f}% | {np.median(closed) * 100:.0f}% | "
         f"{closed.min() * 100:.0f}% ～ {closed.max() * 100:.0f}% |", "",
         f"- 比起点更接近目标的实例：{(final < start).sum()} / {n}；变得更远的：{(final > start).sum()} / {n}。",
         f"- granular 的 CD 尺度和 rope 不同（rope 推之前平均约 1.64）：这里有 {(start < 0.5).sum()} / {n} 个实例推之前就已经 CD < 0.5，"
         "所以 rope 用的成功门槛 0.5 不适合 granular，需要另定。", ""]
    for q in (0.1, 0.15, 0.2, 0.3):
        L.append(f"- 最终 CD < {q}：{(final < q).mean() * 100:.0f}%（推之前就 < {q} 的有 {(start < q).sum()} 个）")
    text = "\n".join(L) + "\n"
    (OUT / "granular_single" / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
