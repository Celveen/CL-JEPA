"""Offline design of validated domain merging for the label-free router (no planning).

Replays a recorded w0 stream in order. Episodes are routed exactly as v4 writes them offline
(familiarity gate + confirmation-pool clustering of fast-change fingerprints, from the cached
signatures of scripts/cont_route_screen.py). Each domain trains a slow learner as CLSMPCPlanner
does (whole predictor, theta0 latents, 15 updates per written episode, catch-up at creation).
Before a new episode is trained on, EVERY existing learner is validated on it (5-step open-loop
latent error over theta0's), so for each pair of domains we see how well A's learner predicts
B's unseen episodes compared with B's own learner -- the evidence a merge rule can use.

Merge rule (validated, symmetric): A and B merge when, over the last k new episodes of each,
A's learner predicts B's episodes better than theta0 and within (1 + eps) of B's own learner,
AND vice versa. The merged
domain keeps the learner with more data, takes the union of the replays and catches up on it.

    python scripts/cont_merge_screen.py --stream S7 --k 3 --eps 0.10 0.20
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cont_offline import load_stream, load_world  # noqa: E402
from cont_route_screen import cluster  # noqa: E402


class Slow:
    """Learner bookkeeping mirroring CLSMPCPlanner (predictor-only, theta0 latents)."""

    def __init__(self, wm, lr=1e-4, steps=15, batch=8, seed=0):
        self.wm, self.lr, self.steps, self.batch = wm, lr, steps, batch
        self.drop = wm.action_dim
        self.rng = np.random.default_rng(seed)

    def predict(self, m, z):
        T = z.shape[1]
        return rearrange(m(rearrange(z, "b t p d -> b (t p) d")), "b (t p) d -> b t p d", t=T)

    def loss(self, m, z):
        k = z.shape[1] - 1
        w = min(self.wm.num_hist, k)
        return torch.stack([torch.mean((self.predict(m, z[:, t : t + w])[..., :-self.drop] - z[:, t + 1 : t + 1 + w][..., :-self.drop]) ** 2)
                            for t in range(k - w + 1)]).mean()

    @torch.no_grad()
    def rollout_err(self, m, z, h=5):
        k = z.shape[1] - 1
        h = min(h, k)
        starts = list(range(k - h + 1))
        seq = torch.cat([z[:, t : t + 1] for t in starts], dim=0)
        for j in range(h):
            nxt = self.predict(m, seq[:, -self.wm.num_hist :])[:, -1:]
            act = torch.cat([z[:, t + j + 1 : t + j + 2, ..., -self.drop :] for t in starts], dim=0)
            seq = torch.cat([seq, torch.cat([nxt[..., :-self.drop], act], dim=-1)], dim=1)
        tgt = torch.cat([z[:, t + 1 : t + 1 + h] for t in starts], dim=0)
        return float(torch.mean((seq[:, 1:][..., :-self.drop] - tgt[..., :-self.drop]) ** 2))

    def new_learner(self):
        m = copy.deepcopy(self.wm.predictor)
        for p in m.parameters():
            p.requires_grad_(True)
        return {"m": m, "opt": torch.optim.Adam(m.parameters(), lr=self.lr)}

    def train(self, L, replay, steps):
        m = L["m"]
        m.train()
        for _ in range(steps):
            L["opt"].zero_grad()
            idx = self.rng.integers(len(replay), size=min(self.batch, len(replay)))
            losses = []
            for i in idx:
                z = replay[i]
                k = z.shape[1] - 1
                n = min(5, k)
                t0 = int(self.rng.integers(0, k - n + 1))
                losses.append(self.loss(m, z[:, t0 : t0 + n + 1]))
            torch.stack(losses).mean().backward()
            L["opt"].step()
        m.eval()


def run(eps, Z, assign, S, theta0, k, eps_merge, m_found=3, min_n=3, merge=True):
    """Stream replay; returns per-episode validation records, the merge log and the final domains.

    Pending founders join their domain when it is founded (when its m_found-th member arrives),
    as online; every learner is validated on each new episode before any training on it.
    """
    found = {}
    for d in {v for v in assign.values() if v is not None}:
        members = sorted(i for i, v in assign.items() if v == d)
        found[d] = members[min(m_found, len(members)) - 1]
    events = sorted((max(i, found[v]), i) for i, v in assign.items() if v is not None)
    dom, recs, merges, alias = {}, [], [], {}
    for t, i in events:
        d = alias.get(assign[i], assign[i])
        z = Z[i]
        e0 = S.rollout_err(theta0, z)
        ratios = {did: S.rollout_err(D["L"]["m"], z) / max(e0, 1e-12) for did, D in dom.items() if D["L"] is not None}
        recs.append({"i": i, "t": t, "tag": eps[i]["tag"], "dom": d, "ratios": ratios})
        D = dom.setdefault(d, {"members": [], "replay": [], "L": None, "vals": []})
        D["vals"].append((t, i, ratios))
        D["members"].append(i)
        D["replay"].append(z)
        if len(D["members"]) >= min_n:
            if D["L"] is None:
                D["L"] = S.new_learner()
                S.train(D["L"], D["replay"], S.steps * len(D["replay"]))
            else:
                S.train(D["L"], D["replay"], S.steps)
        if not merge:
            continue
        ids = sorted(x for x in dom if dom[x]["L"] is not None)
        for a in ids:
            for b in ids:
                if a >= b or a not in dom or b not in dom:
                    continue
                A, B = dom[a], dom[b]
                on_a = [(r[b], r[a]) for _, _, r in sorted(A["vals"]) if a in r and b in r][-k:]  # (B's, A's own) on A's episodes
                on_b = [(r[a], r[b]) for _, _, r in sorted(B["vals"]) if a in r and b in r][-k:]  # (A's, B's own) on B's episodes
                if len(on_a) < k or len(on_b) < k:
                    continue
                gm = lambda xs: math.exp(np.mean(np.log(np.maximum(xs, 1e-12))))
                # each learner must transfer to the other's unseen episodes: better than theta0 there, and
                # within (1 + eps) of that domain's own learner
                if (gm([x for x, _ in on_a]) <= min(1.0, (1 + eps_merge) * gm([y for _, y in on_a]))
                        and gm([x for x, _ in on_b]) <= min(1.0, (1 + eps_merge) * gm([y for _, y in on_b]))):
                    keep, gone = (a, b) if len(A["members"]) >= len(B["members"]) else (b, a)
                    K, G = dom[keep], dom.pop(gone)
                    K["members"] += G["members"]
                    K["replay"] += G["replay"]
                    K["vals"] += [(tt, ii, {(keep if o == gone else o): v for o, v in r.items()}) for tt, ii, r in G["vals"]]
                    S.train(K["L"], K["replay"], S.steps * len(G["replay"]))
                    for D2 in dom.values():
                        D2["vals"] = [(tt, ii, {o: v for o, v in r.items() if o != gone}) for tt, ii, r in D2["vals"]]
                    for x, y in list(alias.items()):
                        if y == gone:
                            alias[x] = keep
                    alias[gone] = keep
                    merges.append({"at": t, "keep": keep, "gone": gone,
                                   "tags": {tg: [eps[j]["tag"] for j in K["members"]].count(tg) for tg in {eps[j]["tag"] for j in K["members"]}}})
    final = {did: {tg: [eps[j]["tag"] for j in D["members"]].count(tg) for tg in {eps[j]["tag"] for j in D["members"]}} for did, D in dom.items()}
    return recs, merges, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S7")
    ap.add_argument("--cache", default="reports/cont_S7/route_screen")
    ap.add_argument("--q", type=float, default=90)
    ap.add_argument("--tau-join", type=float, default=0.10)
    ap.add_argument("--tau-new", type=float, default=0.05)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.10, 0.20])
    ap.add_argument("--out", default="reports/cont_S7/merge_screen")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre, _ = load_world(ROOT / "checkpoints/pushobj_bias_16k_e3_seed0", "3", dev)
    _, eps = load_stream(ROOT / f"eval_outputs/cont_{a.stream}/w0", a.stream, pre, dev)
    c = np.load(ROOT / a.cache / f"cache_{a.stream}.npz")
    tags = [str(t) for t in c["tags"]]
    assert tags == [e["tag"] for e in eps], "cache and stream episodes differ"
    thr = float(np.percentile(c["cal"], a.q))
    unf = [i for i in range(len(eps)) if c["fam"][i] > thr]
    assign = cluster(c["G_delta"], unf, a.tau_join, a.tau_new, a.m)
    with torch.no_grad():
        Z = {i: wm.encode(eps[i]["obs"], torch.cat([eps[i]["act"], torch.zeros_like(eps[i]["act"][:, :1])], dim=1)) for i in unf}
    theta0 = wm.predictor
    L = [f"# 有验证的领域合并：离线设计（流 {a.stream}）", "",
         f"写入沿用 v4：熟悉度门槛 {a.q:g} 分位（{thr:.4f}），确认池 m={a.m}，tau_join {a.tau_join}，tau_new {a.tau_new}。"
         "每个领域按在线规则训练慢学习器；每来一个新 episode，先用所有已有学习器在它上面做验证（5 步开环误差 / θ0），再训练。", ""]
    res = {}
    for label, merge, e in [("不合并", False, 0.0)] + [(f"合并 eps={x:g}", True, x) for x in a.eps]:
        recs, merges, final = run(eps, Z, assign, Slow(wm), theta0, a.k, e, m_found=a.m, merge=merge)
        res[label] = {"merges": merges, "final": final}
        if not merge:
            # cross-validation matrix: rows = episode's domain, cols = learner's domain (geometric mean ratio)
            doms = sorted({r["dom"] for r in recs})
            comp = {d: {} for d in doms}
            for r in recs:
                comp[r["dom"]][r["tag"]] = comp[r["dom"]].get(r["tag"], 0) + 1
            name = {d: f"d{d}(" + ",".join(f"{t}{n}" for t, n in sorted(comp[d].items(), key=lambda kv: -kv[1])) + ")" for d in doms}
            L += ["## 交叉验证矩阵（不合并时）", "", "行 = 新 episode 所属领域，列 = 用哪个领域的学习器去预测它；数值 = 5 步误差 / θ0 的几何平均（越小越好），括号里是样本数。", "",
                  "| episode 所属 \\ 学习器 | " + " | ".join(name[d] for d in doms) + " |", "|---|" + "---:|" * len(doms)]
            for rd in doms:
                cells = []
                for ld in doms:
                    v = [r["ratios"][ld] for r in recs if r["dom"] == rd and ld in r["ratios"]]
                    cells.append(f"{math.exp(np.mean(np.log(v))):.2f}（{len(v)}）" if v else "—")
                L.append(f"| {name[rd]} | " + " | ".join(cells) + " |")
        L += ["", f"## {label}", "", f"合并记录：{merges if merges else '无'}", "", f"最终领域构成：{final}", ""]
    (out / "merge_screen.md").write_text("\n".join(L) + "\n")
    (out / "merge_screen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
