"""Per-visit report for a long continual stream in which every shape is visited more than once.

Arms are paired episode by episode with a reference arm (AdaJEPA, w0) and with
frozen theta0 run on the same chunks. Episodes are grouped by visit (first visit
of a shape vs return), by seen (training) vs unseen shapes, and by shape; the
CLS arms' gates are summarised from their per-episode records:

  * read: what each episode planned with -- theta0, the learner of its own shape,
    or another shape's learner (a learner's shape = majority tag of the episodes
    written to it so far, i.e. known only to the analysis);
  * maturity: when each domain's learner first passed the validation gate, and
    its validation ratios (5-step error vs theta0 on unseen episodes);
  * write: domain composition and purity, routing rule, rejections by the write
    gate and whether the rejected domain was indeed another shape.

    python scripts/cont_rounds.py --stream S7 --arms w0 cls3_free cls3_oracle cls2_free cls_oracle \
        --ablations cls3_free:cls2_free cls3_oracle:cls_oracle
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cont_analyze import finished, gmean_ratio, jsonl, mcnemar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEN = {"T", "L", "Z", "+"}  # training shapes of the base model (data/pushobj_bias_16k/generation_config.json)
BUDGETS = (5, 10, 15, 20, 30)
LABELS = {"w0": "AdaJEPA (w0)", "frozen": "frozen θ0"}
SUP = {1: "¹", 2: "²", 3: "³", 4: "⁴"}


def load_arm(root, arm, blocks):
    chunks = []
    for k, b in enumerate(blocks):
        d = root / arm / b["chunk"]
        if not finished(d):
            break
        z = np.load(d / "final_eval_results.npz")
        chunks.append({"k": k, "shape": b["shape"], "succ": z["successes"].astype(bool),
                       "len": z["action_len"].astype(float), "eps": jsonl(d / "continual_episodes.jsonl"),
                       "iters": jsonl(d / "per_iter_logs.json")})
    return chunks


def boot_ci(x, y, n=10000, seed=0):
    """95% paired-bootstrap interval of mean(x) - mean(y), in points."""
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(x), size=(n, len(x)))
    d = (x[idx].mean(1) - y[idx].mean(1)) * 100
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare(x, y):
    b, c, p = mcnemar(x, y)
    lo, hi = boot_ci(x.astype(float), y.astype(float))
    return {"diff": float((x.mean() - y.mean()) * 100), "only_a": b, "only_b": c, "p": p, "ci": [lo, hi]}


def gate_records(chunks, oracle):
    """Per-episode gate view of a CLS arm, in stream order."""
    counts, out = {}, []  # counts: domain -> {tag: n written so far}
    major = lambda dk: max(counts[dk], key=counts[dk].get) if counts.get(dk) else None
    for c in chunks:
        for e in c["eps"]:
            tag, read = e["tag"], e.get("read")
            if read is None:
                kind = "theta0"
            else:
                owner = read if oracle else major(read)
                kind = "own" if owner == tag else "other"
            rej = [{"domain": dk, "domain_shape": major(dk), "ratio": e.get("rejected_ratios", {}).get(dk)}
                   for dk in e.get("rejected_by", [])]
            out.append({"k": c["k"], "tag": tag, "read": read, "read_kind": kind, "write": e.get("write"),
                        "rule": e.get("write_rule"), "val_ratio": e.get("val_ratio"), "rejected": rej,
                        "write_shape_before": major(e["write"]) if e.get("write") is not None else None,
                        "mature": [dk if oracle else f"{dk}({major(dk)})"
                                   for dk, d in e.get("domains", {}).items() if d.get("mature", d.get("learner"))],
                        "success": e["success"], "steps": e["steps"]})
            if e.get("write") is not None:
                counts.setdefault(e["write"], {})
                counts[e["write"]][tag] = counts[e["write"]].get(tag, 0) + 1
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S7")
    ap.add_argument("--arms", nargs="+", default=["w0", "cls3_free", "cls3_oracle", "cls2_free", "cls_oracle"])
    ap.add_argument("--ref", default="w0")
    ap.add_argument("--frozen", default="frozen", help="arm dir of frozen theta0 on the same chunks ('' to skip)")
    ap.add_argument("--ablations", nargs="*", default=[], help="a:b pairs compared directly (gate on vs off)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = ROOT / (a.root or f"eval_outputs/cont_{a.stream}")
    out = ROOT / (a.out or f"reports/cont_{a.stream}")
    out.mkdir(parents=True, exist_ok=True)
    blocks = json.loads((ROOT / "data/cont_streams" / a.stream / "manifest.json").read_text())["blocks"]

    names = list(a.arms) + ([a.frozen] if a.frozen else [])
    data = {n: load_arm(root, n, blocks) for n in names}
    data = {n: v for n, v in data.items() if v}
    n_done = min(len(v) for v in data.values())
    names = [n for n in names if n in data]
    blocks = [dict(b) for b in blocks[:n_done]]
    for k, b in enumerate(blocks):  # episodes every arm finished in this chunk (fewer than n in --max-episodes smoke runs)
        b["n"] = min(len(data[n][k]["succ"]) for n in names)
        for n in names:
            c = data[n][k]
            c["succ"], c["len"], c["eps"] = c["succ"][: b["n"]], c["len"][: b["n"]], c["eps"][: b["n"]]

    # episode-level arrays over the finished part of the stream
    visit, seen_before = [], {}
    for b in blocks:
        visit.append(seen_before.get(b["shape"], 0))
        seen_before[b["shape"]] = visit[-1] + 1
    ep_shape = np.concatenate([[b["shape"]] * b["n"] for b in blocks])
    ep_visit = np.concatenate([[v] * b["n"] for b, v in zip(blocks, visit)])
    ep_chunk = np.concatenate([[k] * b["n"] for k, b in enumerate(blocks)])
    succ = {n: np.concatenate([c["succ"] for c in data[n][:n_done]]) for n in names}
    length = {n: np.concatenate([c["len"] for c in data[n][:n_done]]) for n in names}

    groups = {"all": np.ones(len(ep_shape), bool)}
    for v in sorted(set(ep_visit)):
        groups[f"visit{v + 1}"] = ep_visit == v
        groups[f"seen_visit{v + 1}"] = (ep_visit == v) & np.isin(ep_shape, list(SEEN))
        groups[f"unseen_visit{v + 1}"] = (ep_visit == v) & ~np.isin(ep_shape, list(SEEN))
    shapes = list(dict.fromkeys(b["shape"] for b in blocks))
    for sh in shapes:
        for v in sorted(set(ep_visit)):
            m = (ep_shape == sh) & (ep_visit == v)
            if m.any():
                groups[f"{sh}|visit{v + 1}"] = m

    res = {"stream": a.stream, "blocks": [(b["chunk"], b["shape"], v + 1) for b, v in zip(blocks, visit)],
           "arms": names, "ref": a.ref, "n_episodes": int(len(ep_shape)), "groups": {}, "budget": {}, "steps": {},
           "ablations": {}, "gates": {}, "slow_effect": {}}
    groups = {g: m for g, m in groups.items() if m.any()}
    for g, m in groups.items():
        row = {}
        for n in names:
            row[n] = {"success": float(succ[n][m].mean() * 100), "n": int(m.sum())}
            if a.ref in succ and n != a.ref:
                row[n]["vs_ref"] = compare(succ[n][m], succ[a.ref][m])
        res["groups"][g] = row
    visit_groups = [f"visit{v + 1}" for v in sorted(set(ep_visit))]
    for g in ["all"] + visit_groups:
        if g not in groups:
            continue
        m = groups[g]
        res["budget"][g] = {n: {b: float(np.mean(length[n][m] <= b) * 100) for b in BUDGETS} for n in names}
        res["steps"][g] = {n: float(np.mean(np.where(succ[n][m], length[n][m], 30))) for n in names}
    for pair in a.ablations:
        x, y = pair.split(":")
        if x in succ and y in succ:
            res["ablations"][pair] = {g: compare(succ[x][groups[g]], succ[y][groups[g]])
                                      for g in ["all"] + visit_groups if g in groups}

    # slow effect: newest-frame error at the weights the episode starts its 2nd-3rd replan with, over theta0
    for n in names:
        rows = []
        for c in data[n][:n_done]:
            first = [r for r in c["iters"] if 2 <= r.get("step", 99) <= 3 and "loss_new_start" in r]
            rows.append(gmean_ratio([r["loss_new_start"] for r in first], [r["loss_new_init"] for r in first]))
        res["slow_effect"][n] = rows

    # gates of the CLS arms
    for n in names:
        eps0 = data[n][0]["eps"]
        if not eps0 or "route" not in eps0[0]:
            continue
        oracle = eps0[0].get("write_rule") == "oracle"
        recs, counts = gate_records(data[n][:n_done], oracle)
        g = {"oracle": oracle, "per_chunk": [], "domains": {}, "rejections": [], "read_success": {}}
        for k, b in enumerate(blocks):
            rr = [r for r in recs if r["k"] == k]
            g["per_chunk"].append({
                "chunk": k, "shape": b["shape"], "visit": visit[k] + 1,
                "reads": {kind: sum(r["read_kind"] == kind for r in rr) for kind in ("theta0", "own", "other")},
                "rules": {rule: sum(r["rule"] == rule for r in rr) for rule in ("oracle", "direction", "learner", "none")},
                "n_rejected_eps": sum(bool(r["rejected"]) for r in rr),
                "mature_end": rr[-1]["mature"] if rr else []})
        written = sum(sum(v.values()) for v in counts.values())
        g["write_purity"] = sum(max(v.values()) for v in counts.values()) / max(written, 1)
        g["n_written"] = written
        g["n_domains"] = len(counts)
        g["n_domains_ge3"] = sum(sum(v.values()) >= 3 for v in counts.values())
        last = data[n][n_done - 1]["eps"][-1]["domains"] if data[n][n_done - 1]["eps"] else {}
        first_mature = {}
        for i, r in enumerate(recs):
            for m in r["mature"]:
                first_mature.setdefault(m.split("(")[0], i)
        for dk, cnt in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
            vals = [r["val_ratio"] for r in recs if r["write"] == dk and r["val_ratio"] is not None
                    and not (isinstance(r["val_ratio"], float) and np.isnan(r["val_ratio"]))]
            vis = [r["k"] for r in recs if r["write"] == dk]
            g["domains"][dk] = {"tags": cnt, "n": sum(cnt.values()), "learner": last.get(dk, {}).get("learner", False),
                                "mature_now": last.get(dk, {}).get("mature", False),
                                "first_mature_episode": first_mature.get(dk),
                                "val_ratio_median": float(np.median(vals)) if vals else None,
                                "val_ratio_last3": vals[-3:], "chunks": sorted(set(vis))}
        for r in recs:
            for rj in r["rejected"]:
                g["rejections"].append({"chunk": r["k"], "tag": r["tag"], **rj,
                                        "correct": rj["domain_shape"] is not None and rj["domain_shape"] != r["tag"]})
        # success by what the episode read, paired with the reference on the same episodes
        if a.ref in succ:
            kinds = np.array([r["read_kind"] for r in recs])
            for v in sorted(set(ep_visit)):
                for kind in ("theta0", "own", "other"):
                    m = (kinds == kind) & (ep_visit == v)
                    if m.any():
                        g["read_success"][f"visit{v + 1}|{kind}"] = {
                            "n": int(m.sum()), "arm": float(succ[n][m].mean() * 100),
                            "ref": float(succ[a.ref][m].mean() * 100), **dict(zip(("only_arm", "only_ref", "p"),
                                                                                 mcnemar(succ[n][m], succ[a.ref][m])))}
        res["gates"][n] = g

    (out / "rounds.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    write_markdown(res, out / "rounds.md")
    plot(res, succ, ep_chunk, blocks, visit, out)
    print((out / "rounds.md").read_text())


def lab(n):
    return LABELS.get(n, n)


def fmt_cmp(c):
    return f"{c['diff']:+.1f} [{c['ci'][0]:+.0f},{c['ci'][1]:+.0f}] ({c['only_a']}/{c['only_b']}, p={c['p']:.2g})"


def write_markdown(r, path):
    arms, ref = r["arms"], r["ref"]
    L = [f"# 持续学习长流 {r['stream']}：每个形状访问两次", "",
         "流：" + " → ".join(f"{sh}{SUP.get(v, v)}" for _, sh, v in r["blocks"])
         + f"（共 {r['n_episodes']} 个 episode；上标是第几次访问该形状）。所有组走同一批目标、同一顺序，逐 episode 配对。",
         f"见过的形状（训练集）：{', '.join(sorted(SEEN))}；没见过：其余。", "",
         f"差值 = 该组 − {lab(ref)}，单位百分点；方括号是 95% 配对 bootstrap 区间；括号里是（仅该组成功/仅参照成功，McNemar 精确 p）。", "",
         "## 总表", "", "| 组 | " + " | ".join(lab(n) for n in arms) + " |", "|---|" + "---:|" * len(arms)]
    visits = sorted({v for _, _, v in r["blocks"]})
    vname = lambda v: "首次" if v == 1 else ("回访" if len(visits) == 2 else f"第 {v} 次")
    names = {"all": "全部"}
    for v in visits:
        names[f"visit{v}"] = vname(v) + ("访问" if v == 1 else "")
        names[f"seen_visit{v}"] = f"见过·{vname(v)}"
        names[f"unseen_visit{v}"] = f"没见过·{vname(v)}"
    main_groups = [g for g in ["all"] + [f"visit{v}" for v in visits] + [f"seen_visit{v}" for v in visits]
                   + [f"unseen_visit{v}" for v in visits] if g in r["groups"]]
    for g in main_groups:
        row = r["groups"][g]
        L.append(f"| {names[g]}（n={row[arms[0]]['n']}） | " + " | ".join(f"{row[n]['success']:.1f}%" for n in arms) + " |")
    L += ["", f"## 与 {lab(ref)} 的配对差", "", "| 组 | " + " | ".join(lab(n) for n in arms if n != ref) + " |",
          "|---|" + "---|" * (len(arms) - 1)]
    for g in main_groups:
        row = r["groups"][g]
        L.append(f"| {names[g]} | " + " | ".join(fmt_cmp(row[n]["vs_ref"]) for n in arms if n != ref) + " |")
    if r["ablations"]:
        L += ["", "## 直接对比（同一流上逐 episode 配对：前者 − 后者）", "",
              "| 对比 | " + " | ".join(names[g] for g in ["all"] + [f"visit{v}" for v in visits]) + " |", "|---|" + "---|" * (len(visits) + 1)]
        for pair, rows in r["ablations"].items():
            L.append(f"| {pair.replace(':', ' − ')} | " + " | ".join(fmt_cmp(rows[g]) if g in rows else "" for g in ["all"] + [f"visit{v}" for v in visits]) + " |")
    shapes = list(dict.fromkeys(sh for _, sh, _ in r["blocks"]))
    L += ["", "## 各形状（" + " / ".join(vname(v) for v in visits) + "）", "", "| 形状 | " + " | ".join(lab(n) for n in arms) + " |", "|---|" + "---:|" * len(arms)]
    for sh in shapes:
        cells = []
        for n in arms:
            vals = [r["groups"][f"{sh}|visit{v}"][n]["success"] for v in visits if f"{sh}|visit{v}" in r["groups"]]
            cells.append(" / ".join(f"{x:.0f}" for x in vals))
        L.append(f"| {sh}{'（见过）' if sh in SEEN else ''} | " + " | ".join(cells) + " |")
    for g in ["all"] + [f"visit{v}" for v in visits]:
        if g in r["budget"]:
            L += ["", f"## 步数预算内的成功率（{names[g]}）", "", "| 组 | " + " | ".join(f"≤{b} 步" for b in BUDGETS) + " | 平均步数（失败记 30） |",
                  "|---|" + "---:|" * (len(BUDGETS) + 1)]
            for n in arms:
                bd = r["budget"][g][n]
                L.append(f"| {lab(n)} | " + " | ".join(f"{bd[b]:.0f}%" for b in BUDGETS) + f" | {r['steps'][g][n]:.1f} |")
    L += ["", "## 新 episode 第 2–3 次重规划时最新帧误差（所用权重 / θ0；<1 = 慢学习器已让预测更准）", "",
          "| 组 | " + " | ".join(f"{k}:{sh}{v}" for k, (_, sh, v) in enumerate(r["blocks"])) + " |", "|---|" + "---:|" * len(r["blocks"])]
    for n in arms:
        if n in r["slow_effect"] and n != "frozen":
            L.append(f"| {lab(n)} | " + " | ".join(f"{x:.2f}" for x in r["slow_effect"][n]) + " |")
    for n, g in r["gates"].items():
        L += ["", f"## 门控：{n}（{'按真实形状写入' if g['oracle'] else '无标签路由'}）", "",
              f"写入 {g['n_written']} 个 episode，{g['n_domains']} 个领域（≥3 个 episode 的 {g['n_domains_ge3']} 个），写入纯度 {g['write_purity']:.2f}。", "",
              "| 段 | 读 θ0 / 本形状学习器 / 其他形状学习器 | 写入规则 | 被写入门拒绝的 episode | 段末已成熟的学习器 |", "|---|---|---|---:|---|"]
        for c in g["per_chunk"]:
            rules = ", ".join(f"{k} {v}" for k, v in c["rules"].items() if v)
            L.append(f"| {c['chunk']}:{c['shape']}{c['visit']} | {c['reads']['theta0']} / {c['reads']['own']} / {c['reads']['other']} | "
                     f"{rules} | {c['n_rejected_eps']} | {', '.join(c['mature_end']) or '—'} |")
        L += ["", "| 领域 | 构成 | 首次成熟（流中第几个 episode） | 验证比中位数 | 最近 3 次验证比 | 出现的段 |", "|---|---|---:|---:|---|---|"]
        for dk, d in g["domains"].items():
            if d["n"] < 2:
                continue
            fm = "—" if d["first_mature_episode"] is None else d["first_mature_episode"]
            vm = "—" if d["val_ratio_median"] is None else f"{d['val_ratio_median']:.2f}"
            last3 = ", ".join(f"{x:.2f}" for x in d["val_ratio_last3"]) or "—"
            L.append(f"| {dk} | {d['tags']} | {fm} | {vm} | {last3} | {d['chunks']} |")
        if g["rejections"]:
            ok = sum(x["correct"] for x in g["rejections"])
            L += ["", f"写入门拒绝 {len(g['rejections'])} 次，其中 {ok} 次被拒绝的领域确实属于别的形状（按该领域当时的多数标签）。"]
        if g["read_success"]:
            L += ["", f"| 读取情况 | n | 该组成功率 | 同批 episode 上 {lab(ref)} | 仅该组/仅参照，p |", "|---|---:|---:|---:|---|"]
            for k, v in g["read_success"].items():
                L.append(f"| {k} | {v['n']} | {v['arm']:.0f}% | {v['ref']:.0f}% | {v['only_arm']}/{v['only_ref']}，p={v['p']:.2g} |")
    path.write_text("\n".join(L) + "\n")


def plot(r, succ, ep_chunk, blocks, visit, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms, ref = r["arms"], r["ref"]
    colors = dict(zip(arms, plt.cm.tab10(np.arange(len(arms)))))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=False)
    x = np.arange(len(blocks))
    width = 0.8 / len(arms)
    for i, n in enumerate(arms):
        rates = [succ[n][ep_chunk == k].mean() * 100 for k in x]
        axes[0].bar(x + i * width - 0.4 + width / 2, rates, width, color=colors[n], label=lab(n))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{k}:{b['shape']}{v + 1}" for k, (b, v) in enumerate(zip(blocks, visit))])
    axes[0].set_ylabel("online success (%)")
    axes[0].legend(fontsize=8, ncol=len(arms))
    axes[0].grid(alpha=0.3, axis="y")
    if ref in succ:
        for n in arms:
            if n != ref:
                axes[1].plot(np.cumsum(succ[n].astype(int) - succ[ref].astype(int)), color=colors[n], label=lab(n))
        for k in range(1, len(blocks)):
            axes[1].axvline(k * blocks[0]["n"], color="grey", lw=0.5, ls=":")
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_xlabel("episode in stream")
        axes[1].set_ylabel(f"cumulative successes − {lab(ref)}")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "rounds.png", dpi=130)


if __name__ == "__main__":
    main()
