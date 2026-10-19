"""Summarise a continual (fast/slow) stream: online performance, probes, consolidation signals.

Reads eval_outputs/cont_<stream>/ (scripts/cont_run_stream.py) and writes
reports/cont_<stream>/summary.{json,md} plus figures.

  * online: success and steps per chunk for every w, paired with w=0 (AdaJEPA)
    episode by episode (same targets, same order);
  * probes: frozen theta0 + slow after each chunk on held-out targets of each
    probe shape (accumulation, interference, forgetting);
  * slow effect: at each new episode, prediction error on its newest frames under
    the episode-start weights (theta0 + slow) vs theta0 -- does what was
    consolidated already help before any fast update?
  * consolidation signals per episode: size of delta, cosine to the previous
    delta and to slow, transfer of delta to remembered episodes (same / other shape).

    python scripts/cont_analyze.py --stream S2
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def finished(d):
    logs = d / "logs.json"
    return (d / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()


def jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def mcnemar(x, y):
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    n = b + c
    if n == 0:
        return b, c, 1.0
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def gmean_ratio(num, den):
    pairs = [(a, b) for a, b in zip(num, den) if a > 0 and b > 0]
    return float(np.exp(np.mean([math.log(a / b) for a, b in pairs]))) if pairs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S2")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", nargs="*", default=[], help="arms to include, in this order (default: all)")
    a = ap.parse_args()
    root = ROOT / (a.root or f"eval_outputs/cont_{a.stream}")
    out = ROOT / (a.out or f"reports/cont_{a.stream}")
    out.mkdir(parents=True, exist_ok=True)
    blocks = json.loads((ROOT / "data/cont_streams" / a.stream / "manifest.json").read_text())["blocks"]
    found = [d.name for d in root.iterdir() if d.is_dir() and (d / blocks[0]["chunk"]).exists()]
    fixed = sorted([n for n in found if n.startswith("w")], key=lambda n: float(n[1:]))
    arms = fixed + sorted(n for n in found if not n.startswith("w"))
    if a.arms:
        arms = [n for n in a.arms if n in found]

    online, probes, signals = {}, {}, {}
    for arm in arms:
        chunks = []
        for k, b in enumerate(blocks):
            d = root / arm / b["chunk"]
            if not finished(d):
                break
            z = np.load(d / "final_eval_results.npz")
            eps = jsonl(d / "continual_episodes.jsonl")
            it = jsonl(d / "per_iter_logs.json")
            chunks.append({"k": k, "shape": b["shape"], "succ": z["successes"].astype(bool),
                           "len": z["action_len"].astype(float), "eps": eps, "iters": it})
        online[arm] = chunks
        probes[arm] = {}
        for k in range(len(blocks)):
            for shape_dir in sorted((root / arm / f"probe_after_{k:02d}").glob("*")):
                if finished(shape_dir):
                    s = np.load(shape_dir / "final_eval_results.npz")["successes"].astype(bool)
                    probes[arm].setdefault(shape_dir.name, {})[k] = float(s.mean())
    base = {}
    for shape_dir in sorted((root / "probe_base").glob("*")):
        if finished(shape_dir):
            base[shape_dir.name] = float(np.load(shape_dir / "final_eval_results.npz")["successes"].mean())

    n_common = min(len(online[arm]) for arm in arms) if arms else 0
    summary = {"stream": a.stream, "blocks": [(b["chunk"], b["shape"]) for b in blocks], "arms": arms,
               "chunks_done": n_common, "online": {}, "probes": {"base": base, **probes}, "signals": {}}
    ref = online.get("w0", [])
    for arm in arms:
        rows = []
        for c in online[arm][:n_common]:
            row = {"chunk": c["k"], "shape": c["shape"], "success": float(c["succ"].mean()),
                   "budget": {b: float(np.mean(c["len"] <= b)) for b in (10, 15, 20, 30)},
                   "steps_median_success": float(np.median(c["len"][c["succ"]])) if c["succ"].any() else float("nan"),
                   "steps_mean_all": float(np.mean(np.where(c["succ"], c["len"], 30)))}
            if arm != "w0" and ref:
                row["vs_w0"] = dict(zip(("only_arm", "only_w0", "p"), mcnemar(c["succ"], ref[c["k"]]["succ"])))
            # slow effect: newest-frame error at episode start weights vs theta0, first replans of each episode
            # replans 2-3: replan 1's transition is the one memory arms choose their memory on
            first = [r for r in c["iters"] if 2 <= r.get("step", 99) <= 3 and "loss_new_start" in r]
            row["slow_effect_ratio_first3"] = gmean_ratio([r["loss_new_start"] for r in first], [r["loss_new_init"] for r in first])
            later = [r for r in c["iters"] if "loss_new_cur" in r]
            row["fast_effect_ratio"] = gmean_ratio([r["loss_new_cur"] for r in later], [r["loss_new_start"] for r in later])
            rows.append(row)
        allx = np.concatenate([c["succ"] for c in online[arm][:n_common]]) if n_common else np.array([], bool)
        alll = np.concatenate([c["len"] for c in online[arm][:n_common]]) if n_common else np.array([])
        tot = {"success": float(allx.mean()) if len(allx) else float("nan"), "n": int(len(allx)), "chunks": rows,
               "budget": {b: float(np.mean(alll <= b)) for b in (10, 15, 20, 30)} if n_common else {}}
        if arm != "w0" and ref and n_common:
            refx = np.concatenate([c["succ"] for c in ref[:n_common]])
            tot["vs_w0"] = dict(zip(("only_arm", "only_w0", "p"), mcnemar(allx, refx)))
        summary["online"][arm] = tot
        eps = [e for c in online[arm][:n_common] for e in c["eps"]]
        if eps and "read" in eps[0]:  # memory arms: what was read and where it was written
            reads = {}
            for c in online[arm][:n_common]:
                cnt = {}
                for e in c["eps"]:
                    key = "θ0" if e["read"] is None else str(e["read"])
                    cnt[key] = cnt.get(key, 0) + 1
                reads[f"{c['k']}:{c['shape']}"] = cnt
            writes = {}
            for e in eps:
                writes.setdefault(e["write"], {}).setdefault(e["tag"], 0)
                writes[e["write"]][e["tag"]] += 1
            writes.pop(None, None)  # episodes the arm chose not to consolidate
            big = {k: v for k, v in writes.items() if sum(v.values()) >= 3}
            n_written = sum(sum(v.values()) for v in writes.values())
            purity = sum(max(v.values()) for v in writes.values()) / max(n_written, 1)
            # read accuracy: a learner/memory counts as right when its majority tag is the episode's shape;
            # theta0 counts as right only for the training shape T
            major = {k: max(v, key=v.get) for k, v in writes.items()}
            acc = {}
            for c in online[arm][:n_common]:
                ok = [(e["read"] is None and e["tag"] == "T") or (e["read"] is not None and major.get(e["read"]) == e["tag"])
                      for e in c["eps"]]
                acc[f"{c['k']}:{c['shape']}"] = float(np.mean(ok))
            summary.setdefault("routing", {})[arm] = {
                "reads": reads, "writes_ge3": big, "write_purity": purity, "n_written": n_written,
                "read_accuracy": acc,
                "n_memories": eps[-1].get("n_memories", len(writes)),
                "slow_loss_last": [e.get("slow_loss") for e in eps[-5:]],
            }
        if eps:
            mean = lambda key: float(np.nanmean([e[key] for e in eps if key in e and e[key] is not None]))
            summary["signals"][arm] = {
                "delta_pred_mean": mean("delta_pred"), "cos_prev_mean": mean("cos_prev"),
                "cos_slow_mean": mean("cos_slow"), "transfer_same_mean": mean("transfer_same"),
                "transfer_other_mean": mean("transfer_other"),
                "slow_pred_final": eps[-1]["slow_pred_after"], "slow_enc_final": eps[-1]["slow_enc_after"],
                "slow_pred_curve": [e["slow_pred_after"] for e in eps],
            }

    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    write_markdown(summary, out / "summary.md")
    plot(summary, out)
    print((out / "summary.md").read_text())


def label(arm):
    return f"w={arm[1:]}" if arm.startswith("w") else arm


def write_markdown(s, path):
    arms, blocks = s["arms"], s["blocks"][: s["chunks_done"]]
    L = [f"# 持续学习 {s['stream']}：快慢权重（slow ← slow + w·Δ）", "",
         f"流：{' → '.join(f'{sh}×20' for _, sh in s['blocks'])}；已完成 {s['chunks_done']} 段。所有 w 走同一批目标、同一顺序，逐 episode 与 w=0（AdaJEPA）配对。", "",
         "## 在线成功率（每段 20 个 episode）", "",
         "| 组 | " + " | ".join(f"{k}:{sh}" for k, (_, sh) in enumerate(blocks)) + " | 合计 | 对 w=0（仅此/仅w0，p） |",
         "|---|" + "---:|" * (len(blocks) + 2)]
    for arm in arms:
        o = s["online"][arm]
        cells = [f"{r['success']*100:.0f}%" for r in o["chunks"]]
        vs = o.get("vs_w0")
        vs = f"{vs['only_arm']}/{vs['only_w0']}，p={vs['p']:.2g}" if vs else "—"
        L.append(f"| {label(arm)} | " + " | ".join(cells) + f" | {o['success']*100:.1f}% | {vs} |")
    L += ["", "## 累计成功率随步数预算（全部 100 个 episode）", "", "| 组 | ≤10 步 | ≤15 步 | ≤20 步 | ≤30 步 |", "|---|---:|---:|---:|---:|"]
    for arm in arms:
        bd = s["online"][arm].get("budget", {})
        if bd:
            L.append(f"| {label(arm)} | " + " | ".join(f"{bd[b]*100:.0f}%" for b in (10, 15, 20, 30)) + " |")
    L += ["", "## 平均步数（失败记 30）", "", "| 组 | " + " | ".join(f"{k}:{sh}" for k, (_, sh) in enumerate(blocks)) + " |",
          "|---|" + "---:|" * len(blocks)]
    for arm in arms:
        L.append(f"| {label(arm)} | " + " | ".join(f"{r['steps_mean_all']:.1f}" for r in s["online"][arm]["chunks"]) + " |")
    L += ["", "## 慢权重本身的作用：新 episode 第 2–3 次重规划，最新帧误差（episode 开始权重 / 初始权重，<1 表示慢权重已经让预测更准）", "",
          "| 组 | " + " | ".join(f"{k}:{sh}" for k, (_, sh) in enumerate(blocks)) + " |", "|---|" + "---:|" * len(blocks)]
    for arm in arms:
        L.append(f"| {label(arm)} | " + " | ".join(f"{r['slow_effect_ratio_first3']:.2f}" for r in s["online"][arm]["chunks"]) + " |")
    shapes = sorted(s["probes"]["base"])
    if shapes:
        L += ["", "## 冻结探针（θ0 + slow，不做快适应；每形状 50 个保留目标）", "",
              "| 组 | 形状 | θ0 | " + " | ".join(f"段{k}后" for k in range(len(s["blocks"]))) + " |",
              "|---|---|---:|" + "---:|" * len(s["blocks"])]
        for arm in arms:
            if arm == "w0":
                continue
            for sh in shapes:
                curve = s["probes"].get(arm, {}).get(sh, {})
                cells = [f"{curve[str(k)] * 100:.0f}%" if str(k) in curve else (f"{curve[k] * 100:.0f}%" if k in curve else "")
                         for k in range(len(s["blocks"]))]
                L.append(f"| {label(arm)} | {sh} | {s['probes']['base'][sh]*100:.0f}% | " + " | ".join(cells) + " |")
    if s.get("routing"):
        L += ["", "## 记忆组的路由", ""]
        for arm, r in s["routing"].items():
            L.append(f"- **{arm}**：共 {r['n_memories']} 份记忆/领域，写入 {r['n_written']} 个 episode，写入纯度 {r['write_purity']:.2f}；≥3 个 episode 的构成 {r['writes_ge3']}")
            L.append("  - 各段读取：" + "；".join(f"{k} {v}" for k, v in r["reads"].items()))
            L.append("  - 各段读对比例：" + "；".join(f"{k} {v:.0%}" for k, v in r["read_accuracy"].items()))
    if s["signals"]:
        L += ["", "## 写入信号（每个 episode 的快权重改动 Δ）", "",
              "| 组 | Δ 大小（predictor） | 与上一个 Δ 的余弦 | 与慢权重的余弦 | 对同形状记忆的迁移 | 对其他形状记忆的迁移 | 最终慢权重大小 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, g in s["signals"].items():
            L.append(f"| {label(arm)} | {g['delta_pred_mean']:.3f} | {g['cos_prev_mean']:+.3f} | {g['cos_slow_mean']:+.3f} | "
                     f"{g['transfer_same_mean']:+.3f} | {g['transfer_other_mean']:+.3f} | {g['slow_pred_final']:.2f} |")
    path.write_text("\n".join(L) + "\n")


def plot(s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms, nb = s["arms"], s["chunks_done"]
    if not nb:
        return
    colors = dict(zip(arms, plt.cm.viridis(np.linspace(0, 0.9, len(arms)))))
    labels = [f"{k}:{sh}" for k, (_, sh) in enumerate(s["blocks"][:nb])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    x = np.arange(nb)
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        rows = s["online"][arm]["chunks"]
        axes[0].bar(x + i * width - 0.4 + width / 2, [r["success"] * 100 for r in rows], width, color=colors[arm], label=label(arm))
        axes[1].plot(x, [r["steps_mean_all"] for r in rows], "o-", color=colors[arm], label=label(arm))
        axes[2].plot(x, [r["slow_effect_ratio_first3"] for r in rows], "o-", color=colors[arm], label=label(arm))
    for ax, title in zip(axes, ["online success per chunk (%)", "mean MPC steps (fail = 30)",
                                "newest-frame error at episode start / theta0"]):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[2].axhline(1, color="black", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "online.png", dpi=130)

    shapes = sorted(s["probes"]["base"])
    if shapes:
        fig, axes = plt.subplots(1, len(shapes), figsize=(3.6 * len(shapes), 3.4), sharey=True)
        for ax, sh in zip(np.atleast_1d(axes), shapes):
            for arm in arms:
                if arm == "w0":
                    continue
                curve = s["probes"].get(arm, {}).get(sh, {})
                ks = sorted(int(k) for k in curve)
                ys = [s["probes"]["base"][sh] * 100] + [curve[k] * 100 for k in ks]
                ax.plot([-1] + ks, ys, "o-", color=colors[arm], label=label(arm))
            ax.axhline(s["probes"]["base"][sh] * 100, color="black", lw=0.8, ls=":")
            ax.set_title(f"probe {sh}")
            ax.set_xlabel("after chunk (-1 = theta0)")
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("frozen success (%)")
        np.atleast_1d(axes)[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "probes.png", dpi=130)


if __name__ == "__main__":
    main()
