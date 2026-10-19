"""Fixed-setting sweep of the two-stage label-free router on cached signatures (scripts/cont_route_screen.py).

Stage 1: familiarity gate (theta0's first-3-transition error above a percentile of its
training-validation errors). Stage 2: leader clustering with a confirmation pool on the
fast-delta fingerprint. One setting must work on every stream, so settings are ranked by
their worst stream. Then, for the chosen setting, a read router is trained on the router's
own domain labels (no ground truth): from the first 3 transitions' residual signature it
predicts the domain an unfamiliar return-visit episode belongs to (nearest centroid vs a
small MLP); a read counts as right when that domain's majority shape is the episode's.

    python scripts/cont_route_sweep.py --streams S7 S2
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cont_route_screen import classify, cluster, score  # noqa: E402

SEEN = {"T", "L", "Z", "+"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=["S7", "S2"])
    ap.add_argument("--dir", default="reports/cont_S7/route_screen")
    ap.add_argument("--sig", default="delta")
    a = ap.parse_args()
    d = ROOT / a.dir
    C = {s: np.load(d / f"cache_{s}.npz") for s in a.streams}
    cal = C[a.streams[0]]["cal"]
    grid = list(itertools.product((85, 90, 95), (2, 3), (0.04, 0.05, 0.06, 0.08, 0.10, 0.12), (0.03, 0.05, 0.08, 0.12)))
    res = []
    for q, m, tj, tn in grid:
        thr = float(np.percentile(cal, q))
        per = {}
        for s, c in C.items():
            tags = [str(t) for t in c["tags"]]
            unf = [i for i in range(len(tags)) if c["fam"][i] > thr]
            sc = score(cluster(c[f"G_{a.sig}"], unf, tj, tn, m), tags, unf)
            n_shapes = len({tags[i] for i in unf if tags[i] not in SEEN})
            per[s] = {**sc, "excess": sc["n_domains_ge3"] - n_shapes}
        worst = min(per[s]["purity"] - 0.03 * max(per[s]["excess"], 0) - 0.3 * max(0.85 - per[s]["coverage"], 0) for s in per)
        res.append({"q": q, "m": m, "tau_join": tj, "tau_new": tn, "worst": worst, "per": per})
    res.sort(key=lambda r: -r["worst"])
    L = [f"# 两段式无标签路由：固定参数扫描（指纹 {a.sig}）", "",
         "按最差流排序（纯度，减去多余领域和覆盖率不足的惩罚）。每格：领域数（≥3 个 episode 的）/ 纯度 / 覆盖率 / 碎片。", "",
         "| 熟悉度门槛 | m | tau_join | tau_new | " + " | ".join(a.streams) + " |", "|---|---:|---:|---:|" + "---|" * len(a.streams)]
    for r in res[:12]:
        L.append(f"| {r['q']} 分位 | {r['m']} | {r['tau_join']} | {r['tau_new']} | " + " | ".join(
            f"{r['per'][s]['n_domains']}（{r['per'][s]['n_domains_ge3']}）/ {r['per'][s]['purity']:.2f} / {r['per'][s]['coverage']:.2f} / {r['per'][s]['frag_mean']:.1f}"
            for s in a.streams) + " |")
    best = res[0]
    L += ["", f"选定：{best['q']} 分位门槛，m={best['m']}，tau_join={best['tau_join']}，tau_new={best['tau_new']}。各流领域构成：", ""]
    for s in a.streams:
        L.append(f"- {s}：{best['per'][s]['composition']}")

    # read router trained on the router's own labels: first-visit assignments -> return-visit reads
    L += ["", "## 读取：用路由器自己分出的领域当标签，训练早期读取器（前 3 步残差指纹）", "",
          "| 流 | 回访中判为不熟悉的 episode | 最近质心读对 | MLP 读对 |", "|---|---:|---:|---:|"]
    thr = float(np.percentile(cal, best["q"]))
    for s, c in C.items():
        tags = [str(t) for t in c["tags"]]
        vis = c["visit"]
        unf = [i for i in range(len(tags)) if c["fam"][i] > thr]
        assign = cluster(c[f"G_{a.sig}"], [i for i in unf if vis[i] == 1], best["tau_join"], best["tau_new"], best["m"])
        tr = [i for i, dm in assign.items() if dm is not None]
        ytr = [f"d{assign[i]}" for i in tr]
        if len(set(ytr)) < 2:
            continue
        major = {}
        for i, y in zip(tr, ytr):
            major.setdefault(y, []).append(tags[i])
        major = {y: max(set(v), key=v.count) for y, v in major.items()}
        te = [i for i in unf if vis[i] >= 2]
        G = c["G_resid3"]
        out = {}
        for kind in ("centroid", "mlp"):
            pred = classify(G, tr, te, ytr, kind)
            out[kind] = float(np.mean([major[p] == tags[i] for p, i in zip(pred, te)]))
        L.append(f"| {s} | {len(te)} | {out['centroid'] * 100:.0f}% | {out['mlp'] * 100:.0f}% |")
    (d / "sweep.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
