"""Summarise stall-escape arms (scripts/escape_run.py) against the mode=none baseline.

Arms pair with `none` exactly up to each episode's first escape (planning/escape.py),
so episodes are split by what the baseline did with them:

  * baseline failures: how many does the arm rescue?
  * late baseline successes (after replan `start`): how many does it break?
  * early baseline successes: no escape can fire before `start`, so these must agree
    (sanity check of the pairing).

Escape statistics: how often and when escapes fire, what a kick does to the pusher and
the object (privileged state, analysis only), how long after the last escape rescues
succeed, and whether unrescued failures were still improving at the step limit (does a
longer budget look worthwhile?). The `none` arm is also compared with the earlier AdaJEPA
run (eval_outputs/cont_<stream>/w0), which used a different random stream, to show how
much AdaJEPA itself varies between runs.

    python scripts/escape_analyze.py --stream S7
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cont_analyze import finished, jsonl, mcnemar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEN = {"T", "L", "Z", "+"}
ANG = math.pi / 9
LABEL = {"none": "AdaJEPA (none)", "K1_5": "K1 random 5px", "K1_10": "K1 random 10px", "K2": "K2 toward pusher goal", "K2P": "K2P toward goal + proprio x10",
         "K3": "K3 best-of-8 restart", "KS": "KS object first, then K2P", "K2P7": "K2P with up to 7 escapes",
         "K2P_7x15": "K2P (7 escapes, 15 px)", "K2P_short2": "K2P, boost only kick + 2 replans", "K2P_short1": "K2P, boost only kick + 1 replan", "K2P_short2_ws_cap60": "K2P short2 + workspace clip + cap 60px", "K2P_short2_ws_cap70": "K2P short2 + workspace clip + cap 70px", "K2P_short2_ws": "K2P short2 + workspace clip", "none_ws": "AdaJEPA + workspace clip", "KS_7x15": "KS (2 object escapes, 7 escapes, 15 px)", "w0_old": "AdaJEPA earlier run"}


def load(d, blocks, truncate=0):
    """Episodes of one arm; truncate > 0 cuts every episode at that many replans (exact for runs that
    differ only in max_iter, since everything before the cut is identical)."""
    eps = []
    for b in blocks:
        c = d / b["chunk"]
        if not finished(c):
            return None
        z = np.load(c / "final_eval_results.npz")
        it = jsonl(c / "per_iter_logs.json")
        for i in range(b["n"]):
            rows = [r for r in it if r["sample_idx"] == i]
            ln = float(z["action_len"][i])
            if truncate and ln > truncate:
                ln, rows = float("inf"), rows[:truncate]
            eps.append({"chunk": b["chunk"], "shape": b["shape"], "succ": bool(np.isfinite(ln)),
                        "len": ln, "rows": rows, "goal": z["state_g"][i]})
    return eps


def boot_ci(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(x), size=(n, len(x)))
    d = (x[idx].mean(1) - y[idx].mean(1)) * 100
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def miss(r):
    return max(r["dist4"] / 20, r["ang"] / ANG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S7")
    ap.add_argument("--arms", nargs="+", default=["none", "K2", "K2P", "K3"])
    ap.add_argument("--start", type=int, default=10, help="escape.start used by the runs")
    ap.add_argument("--max-iter", type=int, default=30, help="planner.max_iter used by the runs (failures count as this)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=None, help="default eval_outputs/escape_<stream>")
    ap.add_argument("--arm-dir", nargs="*", default=[], help="arm=dir for arms that live elsewhere")
    ap.add_argument("--truncate", type=int, default=0, help="cut every episode at this many replans")
    a = ap.parse_args()
    root = ROOT / (a.root or f"eval_outputs/escape_{a.stream}")
    dirs = {arm: root / arm for arm in a.arms}
    dirs.update({kv.split("=", 1)[0]: ROOT / kv.split("=", 1)[1] for kv in a.arm_dir})
    out = ROOT / (a.out or f"reports/escape_{a.stream}")
    out.mkdir(parents=True, exist_ok=True)
    blocks = json.loads((ROOT / "data/cont_streams" / a.stream / "manifest.json").read_text())["blocks"]
    a.arms = [arm for arm in a.arms if dirs[arm].is_dir()]
    blocks = [b for b in blocks if all(finished(dirs[arm] / b["chunk"]) for arm in a.arms)]  # finished in every arm
    E = {arm: load(dirs[arm], blocks, a.truncate) for arm in a.arms}
    old = load(ROOT / f"eval_outputs/cont_{a.stream}/w0", blocks, a.truncate)
    if old is not None:
        E["w0_old"] = old
    assert "none" in E, "the none arm is the pairing baseline"
    base = E["none"]
    s0 = np.array([e["succ"] for e in base])
    late = np.array([e["succ"] and e["len"] > a.start for e in base])
    early = s0 & ~late
    unseen = np.array([e["shape"] not in SEEN for e in base])
    budgets = [10, 20, 30] + ([a.max_iter] if a.max_iter > 30 else [])
    L = [f"# 卡住时的逃逸策略（{a.stream}，{len(blocks)} 段 {len(base)} 个 episode，与 AdaJEPA 逐例配对）", "",
         f"最多 {a.max_iter} 次重规划；卡住检测从第 {a.start + 1} 次重规划起（推杆最近 3 次重规划的位移 < 10 px），"
         "各组的干预参数见 scripts/escape_run.py 的 ARMS。各组在第一次干预之前与 none 完全一致（每个 episode 固定随机种子）。", "",
         "## 总表", "",
         "| 组 | 成功率 | 与 none 的差（95% 区间；仅该组/仅 none，p） | 救回 none 的失败 | 弄坏 none 的晚成功 | 早成功一致 | "
         + " | ".join(f"≤{k} 次" for k in budgets) + f" | 平均步数（失败记 {a.max_iter}） |",
         "|---|---:|---|---:|---:|---:|" + "---:|" * (len(budgets) + 1)]
    summary = {"n": len(base), "arms": {}}
    for arm, eps in E.items():
        s = np.array([e["succ"] for e in eps])
        ln = np.array([e["len"] for e in eps])
        b, c, p = mcnemar(s, s0)
        lo, hi = boot_ci(s.astype(float), s0.astype(float))
        rescued = int((s & ~s0).sum())
        broken = int((~s & late).sum())
        early_ok = float((s[early] == s0[early]).mean())
        row = {"success": float(s.mean() * 100), "diff": float((s.mean() - s0.mean()) * 100), "ci": [lo, hi],
               "only_arm": b, "only_none": c, "p": p, "rescued": rescued, "n_fail": int((~s0).sum()),
               "broken_late": broken, "n_late": int(late.sum()), "early_agree": early_ok,
               "budget": {k: float(np.mean(ln <= k) * 100) for k in budgets},
               "steps": float(np.mean(np.where(s, ln, a.max_iter))),
               "unseen": float(s[unseen].mean() * 100), "seen": float(s[~unseen].mean() * 100)}
        summary["arms"][arm] = row
        diff = "—" if arm == "none" else f"{row['diff']:+.1f} [{lo:+.1f}, {hi:+.1f}]（{b}/{c}，p={p:.2g}）"
        L.append(f"| {LABEL.get(arm, arm)} | {row['success']:.1f}% | {diff} | {rescued}/{row['n_fail']} | {broken}/{row['n_late']} | "
                 f"{early_ok * 100:.0f}% | " + " | ".join(f"{row['budget'][k]:.0f}%" for k in budgets) + f" | {row['steps']:.1f} |")
    L += ["", "| 组 | 没见过的形状 | 见过的形状 |", "|---|---:|---:|"]
    for arm, row in summary["arms"].items():
        L.append(f"| {LABEL.get(arm, arm)} | {row['unseen']:.1f}% | {row['seen']:.1f}% |")
    shapes = list(dict.fromkeys(e["shape"] for e in base))
    L += ["", "| 组 | " + " | ".join(f"{sh}{'（见过）' if sh in SEEN else ''}" for sh in shapes) + " |", "|---|" + "---:|" * len(shapes)]
    for arm, eps in E.items():
        cells = []
        for sh in shapes:
            m = np.array([e["shape"] == sh for e in eps])
            ss = np.array([e["succ"] for e in eps])[m]
            cells.append(f"{ss.mean() * 100:.0f}%" + ("" if arm == "none" else f"（{int((ss & ~s0[m]).sum())}/{int((~ss & s0[m]).sum())}）"))
        L.append(f"| {LABEL.get(arm, arm)} | " + " | ".join(cells) + " |")

    # escape statistics
    L += ["", "## 逃逸统计", "",
          "| 组 | 触发过的 episode | 每个触发 episode 的次数 | 首次触发步（中位数） | 推杆被推开（px，中位数） | 3 次重规划内物体动了（>2 px） | 救回的 episode 在最后一次逃逸后几步成功（中位数） | 救回发生在最后 5 次重规划内 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in a.arms:
        if arm == "none" or arm not in E:
            continue
        eps = E[arm]
        trig = [e for e in eps if any("escape" in r for r in e["rows"])]
        n_esc = [sum("escape" in r for r in e["rows"]) for e in trig]
        first = [next(r["escape"]["iter"] for r in e["rows"] if "escape" in r) + 1 for e in trig]
        moved, obj_moved = [], []
        for e in trig:
            rows = e["rows"]
            for j, r in enumerate(rows):
                if "escape" not in r:
                    continue
                esc = r["escape"]
                moved.append(float(np.linalg.norm(np.subtract(esc["pusher_after"], esc["pusher_before"]))))
                before = np.array(rows[j - 1]["state"]) if j > 0 else None
                after = [np.array(x["state"]) for x in rows[j : j + 3]]
                if before is not None:
                    obj_moved.append(max(np.linalg.norm(s_[2:4] - before[2:4]) for s_ in after) > 2)
        s = np.array([e["succ"] for e in eps])
        lat, tail = [], 0
        for e, ok, ok0 in zip(eps, s, s0):
            if ok and not ok0:
                its = [r["escape"]["iter"] for r in e["rows"] if "escape" in r]
                if its:
                    lat.append(e["len"] - (its[-1] + 1))
                tail += e["len"] > a.max_iter - 5
        L.append(f"| {LABEL.get(arm, arm)} | {len(trig)} | {np.mean(n_esc):.1f} | {np.median(first):.0f} | {np.median(moved):.0f} | "
                 f"{np.mean(obj_moved) * 100:.0f}% | {np.median(lat) if lat else float('nan'):.0f} | {tail} |")
        summary["arms"][arm].update({"n_triggered": len(trig), "escapes_per_triggered": float(np.mean(n_esc)),
                                     "first_escape_step_median": float(np.median(first)),
                                     "pusher_moved_median": float(np.median(moved)),
                                     "object_moved_frac": float(np.mean(obj_moved)),
                                     "rescue_latency": lat, "rescues_after_25": tail})

    # were unrescued failures still improving at the limit?
    L += ["", "## 仍失败的 episode 在步数上限时是否还在进步（加长预算是否值得）", "",
          "| 组 | 仍失败 | 最后 5 步里 miss 还在下降（>0.1） | 结束时 miss < 1.5（差一点） | 用满 5 次逃逸 |", "|---|---:|---:|---:|---:|"]
    for arm in a.arms:
        if arm not in E:
            continue
        fails = [e for e in E[arm] if not e["succ"] and len(e["rows"]) >= 6]
        impr = np.mean([miss(e["rows"][-6]) - miss(e["rows"][-1]) > 0.1 for e in fails])
        near = np.mean([miss(e["rows"][-1]) < 1.5 for e in fails])
        used = np.mean([sum("escape" in r for r in e["rows"]) >= 5 for e in fails])
        L.append(f"| {LABEL.get(arm, arm)} | {len(fails)} | {impr * 100:.0f}% | {near * 100:.0f}% | {used * 100:.0f}% |")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
