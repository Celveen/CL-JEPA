"""Per-round success of the five v6a runs and their AdaJEPA baselines (5 rounds x 7 shapes x 50 episodes).

A round counts once all 7 of its chunks are finished. Writes round_success.md / .json next to this file and
appends one line to round_success.log the first time each (run, arm, round) completes.

    python eval_outputs/gen_logs/round_tracker.py
"""
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEN = {"T", "L", "Z", "+"}
RUNS = [  # label, planner, stream (5 rounds), output root
    ("GD · S7t3s400", "gd", "S7t5s400", "eval_outputs/cont_S7t3s400"),
    ("GD · S7t3s700", "gd", "S7t5s700", "eval_outputs/cont_S7t3s700"),
    ("CEM · S7t3", "cem", "S7t5", "eval_outputs/cont_S7t3_cem"),
    ("CEM · S7t3s400", "cem", "S7t5s400", "eval_outputs/cont_S7t3s400_cem"),
    ("CEM · S7t3s700", "cem", "S7t5s700", "eval_outputs/cont_S7t3s700_cem"),
]
ARMS = {"cls6a_free": "v6a", "w0": "AdaJEPA"}


def mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def chunk_succ(d):
    logs = d / "logs.json"
    if not ((d / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()):
        return None
    return np.load(d / "final_eval_results.npz")["successes"].astype(int)


def main():
    out, rows_md = [], []
    log_path, state_path = HERE / "round_success.log", HERE / "round_success.state.json"
    logged = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    now = dt.datetime.now().strftime("%F %T")
    for label, planner, stream, root in RUNS:
        blocks = json.loads((ROOT / "data/cont_streams" / stream / "manifest.json").read_text())["blocks"]
        for r in range(5):
            rb = blocks[7 * r: 7 * r + 7]
            rec = {"run": label, "planner": planner, "round": r + 1, "seed": rb[0]["seed"]}
            arms = {}
            for arm, name in ARMS.items():
                s = [chunk_succ(ROOT / root / arm / b["chunk"]) for b in rb]
                if any(x is None for x in s):
                    arms[arm] = None
                    continue
                seen = np.concatenate([x for x, b in zip(s, rb) if b["shape"] in SEEN])
                unseen = np.concatenate([x for x, b in zip(s, rb) if b["shape"] not in SEEN])
                allv = np.concatenate(s)
                arms[arm] = allv
                rec[name] = {"all": round(100 * allv.mean(), 1), "seen": round(100 * seen.mean(), 1),
                             "unseen": round(100 * unseen.mean(), 1), "n": int(len(allv))}
                key = f"{label}|{arm}|{r + 1}"
                if key not in logged:
                    logged.add(key)
                    with open(log_path, "a") as f:
                        f.write(f"{now}\t{label}\t{name}\tround {r + 1} (seed {rb[0]['seed']})\t"
                                f"all {rec[name]['all']}%\tseen {rec[name]['seen']}%\tunseen {rec[name]['unseen']}%\n")
            if arms["cls6a_free"] is not None and arms["w0"] is not None:
                x, y = arms["cls6a_free"], arms["w0"]
                b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
                rec["diff"] = {"all": round(100 * (x.mean() - y.mean()), 1), "only_v6a": b, "only_w0": c, "p": mcnemar_p(b, c)}
            out.append(rec)
            v, w, d = rec.get("v6a"), rec.get("AdaJEPA"), rec.get("diff")
            cell = lambda z: f"{z['all']:.1f}% (见过 {z['seen']:.1f} / 没见过 {z['unseen']:.1f})" if z else "—"
            dcell = f"{d['all']:+.1f}（{d['only_v6a']}/{d['only_w0']}，p={d['p']:.2g}）" if d else "—"
            rows_md.append(f"| {label} | {r + 1} | {rec['seed']} | {cell(v)} | {cell(w)} | {dcell} |")
    state_path.write_text(json.dumps(sorted(logged)))
    (HERE / "round_success.json").write_text(json.dumps({"updated": now, "rounds": out}, ensure_ascii=False, indent=1))
    md = [f"# 每轮成功率（更新于 {now}）", "",
          "每轮 = 7 个形状 × 50 个 episode = 350 个；第 1–3 轮来自上一批运行，第 4–5 轮接着第 3 轮结束时的状态续跑。",
          "差 = v6a − AdaJEPA（百分点），括号里是仅 v6a 成功 / 仅 AdaJEPA 成功的 episode 数和 McNemar 精确 p。", "",
          "| 实验 | 轮 | 种子 | v6a | AdaJEPA | 差 |", "|---|---:|---:|---|---|---|", *rows_md]
    (HERE / "round_success.md").write_text("\n".join(md) + "\n")
    done = sum(1 for o in out if "v6a" in o and "AdaJEPA" in o)
    print(f"{now}: {done}/25 rounds complete for both arms")


if __name__ == "__main__":
    main()
