"""Multi-domain continual learning on a rope + granular stream (rope_granular): does routing tell the domains apart,
and does planning improve per domain?

Routing comes from continual_episodes.jsonl (label-free arm `free`; `oracle` uses the true tags). Per-episode final
Chamfer distance (CD) is parsed from the logs (frozen: batched, iteration-major; free / oracle: per-sample,
episode-major; resim_from_start=false, so the last MPC iteration's CD is the final one) and checked against
logs.json. Only chunks finished by frozen and the arm are compared. Main metric: final CD (lower is better); success
rate = final CD < 0.5 (our threshold, chosen on rope; the env's own test is CD < 0, never true). Writes RESULTS.md.

    /data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_granular/analyze.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent
CHUNKS = [(0, "rope"), (1, "granular"), (2, "rope"), (3, "granular"), (4, "rope"), (5, "granular")]
ARMS = {"frozen": "冻结 θ0", "free": "无标签路由（cls6a_free）", "oracle": "真实标签（每个领域一个慢学习器）"}
SUCCESS_CD = 0.5
FAMILIAR_THR = json.loads((OUT / "familiarity.json").read_text())["familiar_thr_rope_p90"]


def done(arm, k, obj):
    f = OUT / arm / f"chunk_{k}_{obj}" / "logs.json"
    return f.exists() and "final_eval/" in f.read_text()


def final_cd(arm, k, obj):
    text = (OUT / "logs" / f"{arm}_chunk_{k}_{obj}.log").read_text(errors="replace")
    v = np.array([float(x) for x in re.findall(r"^CD:\s+([0-9.eE+-]+)", text, re.M)])
    if len(v) != 30:
        raise SystemExit(f"{arm} chunk {k}: {len(v)} CD values, expected 30")
    fin = v.reshape(3, 10)[-1] if arm == "frozen" else v.reshape(10, 3)[:, -1]
    rows = [json.loads(l) for l in open(OUT / arm / f"chunk_{k}_{obj}" / "logs.json")]
    logged = [r for r in rows if "final_eval/mean_chamfer_distance" in r][-1]["final_eval/mean_chamfer_distance"]
    assert abs(fin.mean() - logged) < 1e-6, (arm, k, fin.mean(), logged)
    return fin


def records(arm, chunks):
    return [json.loads(l) for k, obj in chunks for l in open(OUT / arm / f"chunk_{k}_{obj}" / "continual_episodes.jsonl")]


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def paired(a, b):
    d = a - b
    ci = stats.t.ppf(0.975, len(d) - 1) * se(d)
    sa, sb = a < SUCCESS_CD, b < SUCCESS_CD
    only_a, only_b = int((sa & ~sb).sum()), int((~sa & sb).sum())
    p_mc = stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
    return (f"{d.mean():+.3f}（{d.mean() - ci:+.3f}，{d.mean() + ci:+.3f}） | {(d < 0).sum()} / {(d > 0).sum()} | {p_w:.3g} | "
            f"{(sa.mean() - sb.mean()) * 100:+.0f} 个百分点 | {only_a} / {only_b} | {p_mc:.3g}")


def routing(arm, chunks):
    R = records(arm, chunks)
    L = [f"## 路由：{ARMS[arm]}", "",
         f"{len(R)} 个 episode（段 {[k for k, _ in chunks]}）。熟悉度门槛 familiar_thr = {FAMILIAR_THR:.3f}"
         "（θ0 在 rope 验证窗口上熟悉度分数的 90 分位数，`familiarity.json`）。", ""]
    tags = sorted({r["tag"] for r in R})
    L += ["**熟悉度门**（θ0 前 3 次转移的误差）：", "", "| 领域 | episode 数 | 判为熟悉（≤ 门槛） | 分数中位数 | 分数范围 |", "|---|---|---|---|---|"]
    for t in tags:
        f = np.array([r["familiar_score"] for r in R if r["tag"] == t], dtype=float)
        L.append(f"| {t} | {len(f)} | {(f <= FAMILIAR_THR).sum()} | {np.median(f):.3f} | {f.min():.3f} ～ {f.max():.3f} |")
    L += ["", "**写入**（每个 episode 结束时写到哪个领域、按什么规则）：", "", "| 领域 | 写入规则：次数 |", "|---|---|"]
    for t in tags:
        c = Counter((r["write_rule"], r["write"]) for r in R if r["tag"] == t)
        L.append(f"| {t} | " + "，".join(f"{rule}→{w}：{n}" for (rule, w), n in sorted(c.items(), key=lambda x: -x[1])) + " |")
    last = R[-1]["domains"]
    merged = [m for r in R for m in r.get("merges", [])]
    absorbed = [x for r in R for x in r.get("absorbs", [])]
    L += ["", f"**流结束时的领域**（至少 2 个 episode 的；合并 {len(merged)} 次，吸收 {len(absorbed)} 次）：", "",
          "| 领域 | episode 数 | 来源（真实标签：个数） | 有学习器 | 已成熟 | 最近的验证比 |", "|---|---|---|---|---|---|"]
    for dk, d in last.items():
        if d["n"] == 0:
            continue
        L.append(f"| {dk} | {d['n']} | {d['tags']} | {'是' if d['learner'] else '否'} | {'是' if d['mature'] else '否'} | "
                 f"{', '.join(f'{v:.2f}' for v in d['val_last']) or '—'} |")
    L += ["", "**读取**（规划时用 θ0 还是哪个领域的学习器）：", "", "| 领域 | 读取：次数 |", "|---|---|"]
    for t in tags:
        c = Counter(str(r["read"]) for r in R if r["tag"] == t)
        L.append(f"| {t} | " + "，".join(f"{('θ0' if k == 'None' else k)}：{n}" for k, n in sorted(c.items(), key=lambda x: -x[1])) + " |")
    return L + [""]


def planning(chunks_by_arm):
    L = ["## 规划结果（最终 CD 越低越好；成功率 = 最终 CD < 0.5）", ""]
    for arm in ("free", "oracle"):
        chunks = chunks_by_arm.get(arm, [])
        if not chunks:
            continue
        for obj in ("rope", "granular"):
            cs = [(k, o) for k, o in chunks if o == obj]
            if not cs:
                continue
            fa = np.concatenate([final_cd(arm, k, o) for k, o in cs])
            fz = np.concatenate([final_cd("frozen", k, o) for k, o in cs])
            L += [f"**{ARMS[arm]}，{obj}**（{len(cs)} 段、{len(fa)} 个 episode）：", "",
                  "| 组 | 最终 CD（均值 ± 标准误） | 中位数 | 成功率 |", "|---|---|---|---|",
                  f"| 冻结 θ0 | {fz.mean():.3f} ± {se(fz):.3f} | {np.median(fz):.3f} | {(fz < SUCCESS_CD).mean() * 100:.0f}% |",
                  f"| {ARMS[arm]} | {fa.mean():.3f} ± {se(fa):.3f} | {np.median(fa):.3f} | {(fa < SUCCESS_CD).mean() * 100:.0f}% |", "",
                  "| 对比 | CD 平均差（95% 置信区间） | CD 前者更好 / 后者更好 | Wilcoxon p | 成功率差 | 只有前者成功 / 只有后者成功 | McNemar p |",
                  "|---|---|---|---|---|---|---|", f"| {ARMS[arm]} − 冻结 θ0 | {paired(fa, fz)} |", "",
                  "按轮次（每轮 10 个 episode）的平均最终 CD，冻结 → 该组：" +
                  "，".join(f"第 {j + 1} 轮 {final_cd('frozen', k, o).mean():.3f} → {final_cd(arm, k, o).mean():.3f}" for j, (k, o) in enumerate(cs)) + "。", ""]
    return L


def main():
    L = ["# rope + granular 多领域持续学习：结果", "",
         "θ0 是我们训的 rope DINO-WM（`rope_dino_h1_fs1_e100_seed0`），所以 rope 是它自己的领域，granular 是新领域。"
         "流：6 段、每段 10 个 episode，rope / granular 交替（rope 种子 99、100、101，granular 种子 301、302、303）。", ""]
    chunks_by_arm = {}
    for arm in ("free", "oracle"):
        cs = [(k, o) for k, o in CHUNKS if done(arm, k, o) and done("frozen", k, o)]
        chunks_by_arm[arm] = cs
        if cs:
            L += routing(arm, cs)
    L += planning(chunks_by_arm)
    text = "\n".join(L) + "\n"
    (OUT / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
