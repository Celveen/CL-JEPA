"""Why do slow learners help prediction more than planning? Diagnostics for a continual stream.

  * failure anatomy: for failed episodes, how close they got (max of position error / 20 px and
    angle error / (pi/9), the success thresholds), whether they stalled, and which of object,
    pusher and angle is off at the end;
  * prediction gain vs outcome: episodes grouped by the newest-frame error ratio at replans 2-3
    (episode-start weights / theta0), with paired flips against the reference arm;
  * post-hoc screen of a per-episode evidence read gate: take the arm's outcome when its early
    ratio is below tau, else the reference's (approximate: the arm read its learner from replan 1).

    python scripts/cont_diagnose.py --stream S7 --arms cls_oracle cls3_oracle cls3_free > reports/cont_S7/diagnostics.txt
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cont_analyze import mcnemar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEN = {"T", "L", "Z", "+"}  # training shapes of the base model
ANG = math.pi / 9


def episodes(root, arm, blocks):
    visit, cnt, out = [], {}, []
    for b in blocks:
        visit.append(cnt.get(b["shape"], 0) + 1)
        cnt[b["shape"]] = visit[-1]
    for k, b in enumerate(blocks):
        d = root / arm / b["chunk"]
        it = [json.loads(l) for l in open(d / "per_iter_logs.json") if l.strip()]
        succ = np.load(d / "final_eval_results.npz")["successes"].astype(bool)
        for i in range(b["n"]):
            rows = [r for r in it if r["sample_idx"] == i and "dist4" in r]
            miss = [max(r["dist4"] / 20, r["ang"] / ANG) for r in rows]
            first = [r for r in rows if 2 <= r.get("step", 99) <= 3 and "loss_new_start" in r]
            out.append({"shape": b["shape"], "visit": visit[k], "succ": bool(succ[i]),
                        "min_miss": min(miss), "last_miss": miss[-1],
                        "stalled": len(miss) > 11 and miss[-11] - miss[-1] < 0.1,
                        "obj": rows[-1]["dist_obj"], "agent": rows[-1]["dist_agent"], "ang": rows[-1]["ang"],
                        "ratio": float(np.exp(np.mean([math.log(r["loss_new_start"] / r["loss_new_init"]) for r in first])))
                        if first else float("nan")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S7")
    ap.add_argument("--arms", nargs="+", default=["cls_oracle", "cls3_oracle", "cls3_free"])
    ap.add_argument("--ref", default="w0")
    a = ap.parse_args()
    root = ROOT / f"eval_outputs/cont_{a.stream}"
    blocks = json.loads((ROOT / "data/cont_streams" / a.stream / "manifest.json").read_text())["blocks"]
    E = {n: episodes(root, n, blocks) for n in [a.ref] + a.arms}
    s0 = np.array([e["succ"] for e in E[a.ref]])

    print("== failure anatomy (failed episodes); miss = max(dist4 / 20 px, angle / 20 deg), success needs miss < 1")
    for n in [a.ref] + a.arms:
        for grp, f in (("unseen", lambda e: e["shape"] not in SEEN), ("seen", lambda e: e["shape"] in SEEN)):
            F = [e for e in E[n] if f(e) and not e["succ"]]
            if not F:
                continue
            col = lambda key: np.array([e[key] for e in F])
            print(f"{n:12s} {grp:6s} fails={len(F):3d}  closest<1.5x {np.mean(col('min_miss') < 1.5):.2f}  "
                  f"final<2x {np.mean(col('last_miss') < 2):.2f}  final>4x {np.mean(col('last_miss') > 4):.2f}  "
                  f"stalled(last 10) {np.mean(col('stalled')):.2f}  | final median: object {np.median(col('obj')):.0f}px "
                  f"pusher {np.median(col('agent')):.0f}px angle {np.degrees(np.median(col('ang'))):.0f}deg")

    print("\n== prediction gain vs paired outcome (unseen shapes); ratio = newest-frame error at replans 2-3, start weights / theta0")
    for n in a.arms:
        r = np.array([e["ratio"] for e in E[n]])
        s = np.array([e["succ"] for e in E[n]])
        for v in (1, 2):
            base = np.array([e["visit"] == v and e["shape"] not in SEEN for e in E[n]])
            for lo, hi in ((0, 0.6), (0.6, 0.9), (0.9, np.inf)):
                m = base & (r >= lo) & (r < hi)
                if m.any():
                    print(f"{n:12s} visit{v} ratio [{lo},{hi}) n={m.sum():3d}  arm {s[m].mean() * 100:5.1f}%  "
                          f"{a.ref} {s0[m].mean() * 100:5.1f}%  gained {int((s & ~s0)[m].sum())} lost {int((~s & s0)[m].sum())}")

    print(f"\n== post-hoc evidence read gate: arm outcome if early ratio < tau else {a.ref} outcome")
    for n in a.arms:
        r = np.nan_to_num(np.array([e["ratio"] for e in E[n]]), nan=np.inf)
        s = np.array([e["succ"] for e in E[n]])
        vis = np.array([e["visit"] for e in E[n]])
        un = np.array([e["shape"] not in SEEN for e in E[n]])
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9, np.inf):
            use = r < tau
            g = np.where(use, s, s0)
            b, c, p = mcnemar(g, s0)
            print(f"{n:12s} tau={tau:<4} learner used {use.mean() * 100:3.0f}%  all {g.mean() * 100:5.1f}% "
                  f"({a.ref} {s0.mean() * 100:.1f}; {b}/{c} p={p:.2g})  unseen-return {g[(vis == 2) & un].mean() * 100:5.1f}% "
                  f"({a.ref} {s0[(vis == 2) & un].mean() * 100:.1f})  seen {g[~un].mean() * 100:5.1f}% ({a.ref} {s0[~un].mean() * 100:.1f})")


if __name__ == "__main__":
    main()
