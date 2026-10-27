"""DINO-WM's deformable protocol (eval.sh): mean Chamfer distance (CD) before any push and after every replan, per case,
next to DINO-WM's reported numbers (Table 8, MPC: rope 0.41, granular 0.26, on 10 instances). Writes RESULTS.md.

The start CD is recomputed from plan_targets.pkl (state_0 vs state_g) with the env's chamfer_distance; the CD after each
replan is parsed from the log ("CD:" lines, 10 per evaluation). mpc (resim_from_start=false) prints one evaluation per
MPC iteration and the final evaluation reuses the last one; openloop prints the planner's evaluation and then the final
one, which re-simulates the same 5 pushes (FleX is not deterministic), so only the first is used.

    /data/zjc/envs/cljepa/bin/python -I eval_outputs/granular_ref/analyze.py
"""
import pickle
import re
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
SEEDS = (99, 100, 101, 102, 103)
CASES = {"rope_on_rope": ("我们训的 rope DINO-WM", "rope", 0.41),
         "ddpwm_on_rope": ("DDP-WM 提供的 rope DINO-WM", "rope", 0.41),
         "rope_on_granular": ("我们训的 rope DINO-WM（θ0）", "granular", None),
         "granular_ref": ("granular 参照模型（只用 granular 训练）", "granular", 0.26)}


def chamfer(x, y):
    x, y = torch.tensor(x)[None, :, :3].double(), torch.tensor(y)[None, :, :3].double()
    d = torch.cdist(x, y)
    return float(d.min(2)[0].mean() + d.min(1)[0].mean())


def case_cd(proto, case):
    """(start CD (n,), CD after each evaluation (k, n)), or None while a seed is unfinished."""
    starts, evals = [], []
    for s in SEEDS:
        log = OUT / "logs" / f"{proto}_{case}_seed{s}.log"
        if not log.exists() or "EXIT=0" not in log.read_text(errors="replace"):
            return None
        t = pickle.load(open(OUT / proto / case / f"seed{s}" / "plan_targets.pkl", "rb"))
        st0 = t["state_0"].reshape(len(t["state_0"]), -1, 4)
        starts += [chamfer(t["state_g"][i], st0[i]) for i in range(len(st0))]
        v = [float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", log.read_text(errors="replace"), re.M)]
        evals.append(np.array(v).reshape(-1, len(st0)))
    k = min(e.shape[0] for e in evals)
    return np.array(starts), np.concatenate([e[:k] for e in evals], axis=1)


def main():
    L = ["# DINO-WM 协议下的 rope / granular 规划", "",
         "目标：`random_state`（起点取自验证集；rope 的目标是起点旋转加平移，granular 的目标是起点缩放 0.6～0.9 倍再平移）。"
         "CEM：horizon 5、300 个样本、取前 30、迭代 30 次。5 个种子 × 10 个实例 = 50 个实例。CD 越低越好。", ""]
    for proto, title, note in (("mpc", "MPC（每次执行 5 推后重规划，共 5 次、25 推）",
                                "DINO-WM 的 Table 8 是 MPC 的结果，但论文没写重规划多少次（官方配置 max_iter 为 null）。"),
                               ("openloop", "开环（规划一次，执行 5 推）", "README 的示例命令；论文里开环 CEM 在 rope / granular 上标的是 NA。")):
        rows = []
        for case, (name, obj, paper) in CASES.items():
            r = case_cd(proto, case)
            if r is None:
                continue
            s0, ev = r
            after = ev if proto == "mpc" else ev[:1]
            rows.append((name, obj, s0, after, paper))
        if not rows:
            continue
        k = max(a.shape[0] for _, _, _, a, _ in rows)
        L += [f"## {title}", "", note, "",
              "| 模型 | 环境 | 推之前 | " + " | ".join(f"{5 * (j + 1)} 推后" for j in range(k)) + " | 比起点更近 | 论文 |",
              "|---|---|---|" + "---|" * k + "---|---|"]
        for name, obj, s0, a, paper in rows:
            cells = [f"{a[j].mean():.3f}" if j < a.shape[0] else "—" for j in range(k)]
            L.append(f"| {name} | {obj} | {s0.mean():.3f} | " + " | ".join(cells) +
                     f" | {int((a[-1] < s0).sum())} / {len(s0)} | {paper if paper is not None else '—'} |")
        L += [""]
        for name, obj, s0, a, paper in rows:
            fin = a[-1]
            L.append(f"- {name}（{obj}）：最终 CD {fin.mean():.3f} ± {fin.std(ddof=1) / np.sqrt(len(fin)):.3f}，"
                     f"中位数 {np.median(fin):.3f}，范围 {fin.min():.3f}～{fin.max():.3f}；"
                     f"按种子（各 10 个）：{', '.join(f'{fin[i * 10:(i + 1) * 10].mean():.2f}' for i in range(len(SEEDS)))}")
        L += [""]
    text = "\n".join(L) + "\n"
    (OUT / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
