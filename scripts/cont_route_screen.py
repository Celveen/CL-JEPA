"""Offline screen of label-free domain routing on recorded streams (no planning).

Input: w0 (AdaJEPA) stream runs saved with deltas and frames. Per episode we compute
candidate signals at theta0 and replay the write rule in stream order:

  familiarity  theta0's newest-frame error over the first 3 transitions (new3). The
               threshold is a percentile of the same statistic on theta0's own
               training-distribution validation episodes (no stream labels), so
               "familiar" means "theta0 already predicts this as well as its training data".
  signatures   for clustering the unfamiliar episodes:
               delta  - AdaJEPA's fast change over the episode (the current fingerprint)
               grad   - gradient of the 1-step latent loss at theta0 on the episode
                        (same tensors as AdaJEPA adapts; fixed reference point, no Adam sign)
               resid  - theta0's mean latent prediction residual over the episode
               resid3 - the same over the first 3 transitions only (available for reads)
               enc    - mean visual latent of the episode's frames (theta0 encoder),
                        centred on the training-validation mean
  write rule   leader clustering with a confirmation pool: an unfamiliar episode joins
               the domain whose centroid it is most aligned with (cos >= tau_join);
               otherwise it waits in a pending pool, and a new domain is created only
               when m pending episodes are mutually aligned (cos >= tau_new). m = 1 is
               the current rule (any unmatched episode founds a domain).

Also reported: how separable shapes are for each signature with true labels (nearest
centroid, 1-NN, logistic regression, small MLP; trained on first visits, tested on
returns) -- the upper bound a learned router (e.g. an MLP) could reach.

    python scripts/cont_route_screen.py --streams S7 S2 --out reports/cont_S7/route_screen
"""
import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cont_offline import load_stream, load_world  # noqa: E402

SEEN = {"T", "L", "Z", "+"}
VIS = 384  # visual channels of the concatenated latent (then proprio 10, action 10)


def latents(wm, obs, act):
    return wm.encode(obs, torch.cat([act, torch.zeros_like(act[:, :1])], dim=1))


def new3(wm, z):
    drop = wm.action_dim
    k = z.shape[1] - 1
    errs = [float(torch.mean((wm.predict(z[:, max(0, t - wm.num_hist) : t])[:, -1:, ..., :-drop] - z[:, t : t + 1, ..., :-drop]) ** 2))
            for t in range(1, min(3, k) + 1)]
    return float(np.exp(np.mean(np.log(errs))))


def residuals(wm, z, upto=None):
    """theta0 residuals z_{t+1} - pred(z_{<=t}) over obs channels, one per transition."""
    drop = wm.action_dim
    k = z.shape[1] - 1 if upto is None else min(upto, z.shape[1] - 1)
    return torch.stack([(z[:, t : t + 1, ..., :-drop] - wm.predict(z[:, max(0, t - wm.num_hist) : t])[:, -1:, ..., :-drop]).flatten()
                        for t in range(1, k + 1)])


def grad_signature(wm, params, obs, act):
    for p in params:
        p.requires_grad_(True)
    z = latents(wm, obs, act)
    drop, k = wm.action_dim, z.shape[1] - 1
    w = min(wm.num_hist, k)
    loss = torch.stack([torch.mean((wm.predict(z[:, t : t + w])[..., :-drop] - z[:, t + 1 : t + 1 + w][..., :-drop]) ** 2)
                        for t in range(k - w + 1)]).mean()
    g = torch.autograd.grad(loss, params)
    for p in params:
        p.requires_grad_(False)
    return torch.cat([x.flatten() for x in g]).detach()


@torch.no_grad()
def calibration(wm, pre, data_path, n_per_shape, fs, seed=0):
    """theta0's new3 and mean visual latent on training-distribution validation windows."""
    from datasets.pusht_dset import PushTDataset
    ds = PushTDataset(data_path=str(data_path), transform=pre.transform if hasattr(pre, "transform") else None)
    rng = random.Random(seed)
    by_shape = {}
    for i, s in enumerate(ds.shapes):
        by_shape.setdefault(s, []).append(i)
    vals, enc = [], []
    dev = next(wm.parameters()).device
    for s, idx in by_shape.items():
        for i in rng.sample(idx, min(n_per_shape, len(idx))):
            L = ds.get_seq_length(i)
            if L < 3 * fs + 1:
                continue
            t0 = rng.randrange(0, L - 3 * fs)
            frames = [t0 + j * fs for j in range(4)]
            obs, _, _, _ = ds.get_frames(i, frames)
            act = ds.actions[i, t0 : t0 + 3 * fs].reshape(3, fs * ds.action_dim)
            obs = {"visual": obs["visual"][None].to(dev), "proprio": obs["proprio"][None].to(dev)}
            z = latents(wm, obs, act[None].to(dev))
            vals.append(new3(wm, z))
            enc.append(z[0, :, 0, :VIS].mean(0).cpu())
    return np.array(vals), torch.stack(enc).mean(0)


def unit(x):
    return x / (x.norm() + 1e-12)


def cluster(G, order, tau_join, tau_new, m):
    """Leader clustering with a confirmation pool on a Gram matrix of unit signatures.

    cos(episode, domain centroid) = sum of its dots with the members / norm of the members' sum.
    Returns {episode index: domain id or None (pending, not written)}.
    """
    members, pending, out = [], [], {}
    for i in order:
        if members:
            cs = [G[i, M].sum() / math.sqrt(max(G[np.ix_(M, M)].sum(), 1e-12)) for M in members]
            j = int(np.argmax(cs))
            if cs[j] >= tau_join:
                members[j].append(i)
                out[i] = j
                continue
        nb = [q for q in pending if G[i, q] >= tau_new]
        if len(nb) + 1 >= m:
            group = (nb[: m - 1] if m > 1 else []) + [i]
            for q in group:
                if q in pending:
                    pending.remove(q)
                out[q] = len(members)
            members.append(list(group))
        else:
            pending.append(i)
            out[i] = None
    return out


def score(assign, tags, unfamiliar):
    dom = {}
    for i, d in assign.items():
        if d is not None:
            dom.setdefault(d, []).append(tags[i])
    n_assigned = sum(len(v) for v in dom.values())
    purity = sum(max(v.count(t) for t in set(v)) for v in dom.values()) / max(n_assigned, 1)
    big = {d: v for d, v in dom.items() if len(v) >= 3}
    shapes = sorted({tags[i] for i in unfamiliar})
    # fragmentation: domains needed to hold 80% of each shape's assigned episodes
    frag = []
    for s in shapes:
        counts = sorted((v.count(s) for v in dom.values() if s in v), reverse=True)
        tot, acc, k = sum(counts), 0, 0
        for c in counts:
            if acc >= 0.8 * tot:
                break
            acc, k = acc + c, k + 1
        frag.append(k)
    return {"n_domains": len(dom), "n_domains_ge3": len(big), "purity": purity,
            "coverage": n_assigned / max(len(unfamiliar), 1), "frag_mean": float(np.mean(frag)) if frag else float("nan"),
            "composition": {str(d): {t: v.count(t) for t in set(v)} for d, v in sorted(big.items(), key=lambda kv: -len(kv[1]))}}


# ------------------------------------------------------------------ supervised upper bound
def classify(G, tr, te, ytr, kind, seed=0):
    classes = sorted(set(ytr))
    yi = np.array([classes.index(y) for y in ytr])
    if kind == "centroid":
        sc = np.stack([G[np.ix_(te, [tr[j] for j in np.where(yi == k)[0]])].sum(1)
                       / math.sqrt(G[np.ix_([tr[j] for j in np.where(yi == k)[0]], [tr[j] for j in np.where(yi == k)[0]])].sum())
                       for k in range(len(classes))], 1)
        pred = sc.argmax(1)
    elif kind == "1nn":
        pred = yi[G[np.ix_(te, tr)].argmax(1)]
    else:  # kernel map: an episode is described by its similarity to every training episode
        torch.manual_seed(seed)
        xtr, xte = torch.tensor(G[np.ix_(tr, tr)], dtype=torch.float32), torch.tensor(G[np.ix_(te, tr)], dtype=torch.float32)
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        y = torch.tensor(yi)
        d = xtr.shape[1]
        net = (torch.nn.Linear(d, len(classes)) if kind == "logreg" else
               torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, len(classes))))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
        for _ in range(500):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xtr), y).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(xte).argmax(1).numpy()
    return [classes[int(k)] for k in pred]


def gram(vectors, dev):
    X = torch.stack(vectors).to(dev)
    G = (X @ X.T).cpu().numpy().astype(np.float64)
    del X
    torch.cuda.empty_cache()
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=["S7", "S2"])
    ap.add_argument("--run", default="w0", help="arm whose deltas/frames are replayed")
    ap.add_argument("--ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
    ap.add_argument("--epoch", default="3")
    ap.add_argument("--val-data", default="data/pushobj_bias_16k/val")
    ap.add_argument("--val-n", type=int, default=100, help="validation windows per training shape")
    ap.add_argument("--out", default="reports/cont_S7/route_screen")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre, params = load_world(ROOT / a.ckpt, a.epoch, dev)
    fs = 5

    cal, enc_mean = calibration(wm, pre, ROOT / a.val_data, a.val_n, fs)
    thr = {q: float(np.percentile(cal, q)) for q in (90, 95, 99)}
    L = ["# 无标签路由的离线筛选", "",
         f"θ0 在训练分布验证集上（{len(cal)} 个窗口，T/L/Z/+）的前 3 步误差：中位数 {np.median(cal):.4f}；"
         + "，".join(f"{q} 分位 {v:.4f}" for q, v in thr.items()) + "。熟悉度门槛取这些分位数，不用任何流上的标签。", ""]
    results = {"calibration": {"median": float(np.median(cal)), **{f"p{q}": v for q, v in thr.items()}}, "streams": {}}

    for stream in a.streams:
        blocks, eps = load_stream(ROOT / f"eval_outputs/cont_{stream}" / a.run, stream, pre, dev)
        visit, cnt = [], {}
        for b in blocks:
            visit.append(cnt.get(b["shape"], 0) + 1)
            cnt[b["shape"]] = visit[-1]
        tags = [e["tag"] for e in eps]
        vis = [visit[e["block"]] for e in eps]
        fam, sig = [], {k: [] for k in ("delta", "grad", "resid", "resid3", "enc")}
        for e in eps:
            with torch.no_grad():
                z = latents(wm, e["obs"], e["act"])
                fam.append(new3(wm, z))
                r = residuals(wm, z)
                sig["resid"].append(unit(r.mean(0)).cpu())
                sig["resid3"].append(unit(residuals(wm, z, upto=3).mean(0)).cpu())
                sig["enc"].append(unit(z[0, :, 0, :VIS].mean(0).cpu() - enc_mean))
            sig["grad"].append(unit(grad_signature(wm, params, e["obs"], e["act"])).cpu())
            sig["delta"].append(unit(e["delta"]).cpu())
        fam = np.array(fam)
        seen = np.array([t in SEEN for t in tags])
        S = {k: gram(v, dev) for k, v in sig.items()}  # Gram matrices of unit signatures
        S["resid3+enc"] = (S["resid3"] + S["enc"]) / 2  # concatenated unit blocks, renormalised
        S["grad+resid+enc"] = (S["grad"] + S["resid"] + S["enc"]) / 3
        del sig
        np.savez(out / f"cache_{stream}.npz", fam=fam, tags=np.array(tags), visit=np.array(vis),
                 blocks=np.array([e["block"] for e in eps]), cal=cal, **{f"G_{k}": v for k, v in S.items()})

        L += [f"## 流 {stream}（{len(eps)} 个 episode）", "", "### 熟悉度门（θ0 前 3 步误差 > 门槛 → 不熟悉）", "",
              "| 门槛 | 没见过形状判为不熟悉 | 见过形状判为不熟悉（误报） | 各形状判为不熟悉的比例 |", "|---|---:|---:|---|"]
        sr = {"familiarity": {}, "separability": {}, "clustering": {}}
        for q, v in thr.items():
            unf = fam > v
            per = "，".join(f"{s} {np.mean(unf[np.array(tags) == s]) * 100:.0f}%" for s in dict.fromkeys(tags))
            L.append(f"| {q} 分位（{v:.4f}） | {np.mean(unf[~seen]) * 100:.0f}% | {np.mean(unf[seen]) * 100:.0f}% | {per} |")
            sr["familiarity"][q] = {"tpr_unseen": float(np.mean(unf[~seen])), "fpr_seen": float(np.mean(unf[seen]))}

        # separability with true labels among truly unseen episodes: train on first visits, test on returns
        un_idx = [i for i in range(len(eps)) if not seen[i]]
        tr = [i for i in un_idx if vis[i] == 1]
        te = [i for i in un_idx if vis[i] >= 2]
        L += ["", "### 用真实标签时各指纹能把没见过的形状分得多准（首访训练、回访测试；只是上限诊断）", "",
              "| 指纹 | 最近质心 | 1-NN | 逻辑回归 | MLP（64 隐层） |", "|---|---:|---:|---:|---:|"]
        for name, G in S.items():
            accs = {}
            for kind in ("centroid", "1nn", "logreg", "mlp"):
                if not tr or not te:
                    continue
                pred = classify(G, tr, te, [tags[i] for i in tr], kind)
                accs[kind] = float(np.mean([p == tags[i] for p, i in zip(pred, te)]))
            sr["separability"][name] = accs
            if accs:
                L.append(f"| {name} | " + " | ".join(f"{accs[k] * 100:.0f}%" for k in ("centroid", "1nn", "logreg", "mlp")) + " |")

        # write rule replayed in stream order on the episodes the familiarity gate flags
        L += ["", "### 写入规则回放（只对判为不熟悉的 episode 分领域）", "",
              "每行：熟悉度门槛 / 指纹 / tau_join / tau_new / m（m=1 即现有规则）。纯度按被写入的 episode 算；覆盖率 = 被写入的不熟悉 episode 比例；"
              "碎片 = 每个形状平均要几个领域才装下它 80% 的 episode（理想为 1）。", "",
              "| 门槛 | 指纹 | tau_join | tau_new | m | 领域数（≥3 个） | 纯度 | 覆盖率 | 碎片 |", "|---|---|---:|---:|---:|---|---:|---:|---:|"]
        rows = []
        for q in (95, 99):
            unf_idx = [i for i in range(len(eps)) if fam[i] > thr[q]]
            for name in ("delta", "grad", "resid", "enc", "grad+resid+enc"):
                X = S[name]
                pc = X[np.ix_(unf_idx, unf_idx)].copy()
                np.fill_diagonal(pc, np.nan)
                grid = sorted({round(float(np.nanpercentile(pc, p)), 3) for p in (50, 70, 80, 90, 95)})
                for tj, tn, m in itertools.product(grid, grid, (1, 2, 3)):
                    if m == 1 and tn != grid[0]:
                        continue
                    sc = score(cluster(X, unf_idx, tj, tn, m), tags, unf_idx)
                    rows.append({"q": q, "sig": name, "tau_join": tj, "tau_new": tn, "m": m, **sc})
        sr["clustering"] = rows
        # per (gate, signature): the current rule at its best tau, then the best settings by purity with few domains
        for q in (95, 99):
            for name in ("delta", "grad", "resid", "enc", "grad+resid+enc"):
                cand = [r for r in rows if r["q"] == q and r["sig"] == name]
                cur = max((r for r in cand if r["m"] == 1), key=lambda r: (r["purity"] - 0.02 * r["n_domains"]))
                best = sorted((r for r in cand if r["coverage"] >= 0.8), key=lambda r: (-(r["purity"] - 0.01 * r["n_domains_ge3"]), r["n_domains"]))[:2]
                for r in [cur] + best:
                    L.append(f"| {q} | {name} | {r['tau_join']:.3f} | {r['tau_new']:.3f} | {r['m']} | {r['n_domains']}（{r['n_domains_ge3']}） | "
                             f"{r['purity']:.2f} | {r['coverage']:.2f} | {r['frag_mean']:.1f} |")
        best_all = max((r for r in rows if r["coverage"] >= 0.8), key=lambda r: (r["purity"] - 0.01 * r["n_domains_ge3"]))
        L += ["", f"最佳设置的领域构成（{best_all['q']} 分位门槛，{best_all['sig']}，tau_join {best_all['tau_join']}，tau_new {best_all['tau_new']}，m {best_all['m']}）：",
              "", "```", json.dumps(best_all["composition"], ensure_ascii=False), "```", ""]
        results["streams"][stream] = sr
        print(f"{stream} done", flush=True)

    (out / "screen.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=float))
    (out / "screen.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
