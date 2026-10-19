"""Can theta0's encoder latents tell the shapes apart? (advisor's suggestion: KNN on encoder latents)

Features of each frame from the frozen theta0 encoder (SmallResNetGeM):
  proj  - the 384-d latent the world model uses (after the projection head)
  gem   - the 512-d GeM-pooled feature before the projection head
  pool  - mean and max pooling of the last 7x7x512 feature map (1024-d)

Data: the experiment-0 bank (100 dataset segments x 6 frames per shape, data/exp0_bank) and the
episodes the planner executed on a stream (w0 run, episode_frames.npz).

  A  supervised KNN inside the bank, split by segment (an upper bound: labelled references);
  B  the bank as labelled references, the stream episodes as queries (first frame / first 3
     frames / whole episode, majority vote over frames) -- "each dataset gives some latents";
  C  label-free: cluster the stream episodes (k-means, k = number of shapes) and score purity.
Every variant is run with cosine and Euclidean distance, raw and z-scored features (statistics
of the reference set), k in {1, 5, 15}; 7-way (all shapes), 4-way (trained) and 3-way (unseen).

    python scripts/cont_knn_latent.py --stream S7 --out reports/cont_S7/knn_latent
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cont_offline import load_world  # noqa: E402

SHAPES = ["I", "L", "square", "Z", "small_tee", "+", "T"]
SEEN = {"T", "L", "Z", "+"}


class Features:
    def __init__(self, wm, pre, dev):
        self.enc, self.pre, self.dev = wm.encoder, pre, dev
        self.buf = {}
        self.enc.gem.register_forward_hook(lambda m, i, o: self.buf.__setitem__("gem", o.flatten(1)))
        self.enc.rb5.register_forward_hook(lambda m, i, o: self.buf.__setitem__("map", o))

    @torch.no_grad()
    def __call__(self, frames, bs=128):
        """frames: uint8 (N, H, W, 3) -> dict of (N, d) float arrays."""
        out = {"proj": [], "gem": [], "pool": []}
        for s in range(0, len(frames), bs):
            x = self.pre.transform_obs_visual(frames[None, s : s + bs])[0].to(self.dev)  # expects (b, t, h, w, c)
            proj = self.enc(x).squeeze(1)
            m = self.buf["map"]
            out["proj"].append(proj.cpu())
            out["gem"].append(self.buf["gem"].cpu())
            out["pool"].append(torch.cat([m.mean((2, 3)), m.amax((2, 3))], 1).cpu())
        return {k: torch.cat(v).numpy().astype(np.float64) for k, v in out.items()}


def prep(ref, qry, metric, z):
    if z:
        mu, sd = ref.mean(0), ref.std(0) + 1e-6
        ref, qry = (ref - mu) / sd, (qry - mu) / sd
    if metric == "cos":
        ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
        qry = qry / (np.linalg.norm(qry, axis=1, keepdims=True) + 1e-12)
        return ref, qry, lambda q, r: -(q @ r.T)
    return ref, qry, lambda q, r: (q ** 2).sum(1)[:, None] + (r ** 2).sum(1)[None] - 2 * q @ r.T


def knn(ref, ylab, qry, k, metric, z):
    ref, qry, dist = prep(ref, qry, metric, z)
    d = dist(qry, ref)
    nn = np.argsort(d, axis=1)[:, :k]
    votes = ylab[nn]
    return np.array([np.bincount(v, minlength=ylab.max() + 1).argmax() for v in votes])


def kmeans(x, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
    return a


def purity(assign, y):
    return sum(np.bincount(y[assign == c]).max() for c in np.unique(assign)) / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S7")
    ap.add_argument("--out", default="reports/cont_S7/knn_latent")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre, _ = load_world(ROOT / "checkpoints/pushobj_bias_16k_e3_seed0", "3", dev)
    F = Features(wm, pre, dev)

    # bank: 100 segments x 6 frames per shape
    bank = {}
    for s in SHAPES:
        v = np.load(ROOT / "data/exp0_bank" / f"{s}.npz")["visual"]  # (100, 6, H, W, 3)
        f = F(v.reshape(-1, *v.shape[2:]))
        bank[s] = {k: x.reshape(v.shape[0], v.shape[1], -1) for k, x in f.items()}
    # stream episodes of the reference run
    blocks = json.loads((ROOT / "data/cont_streams" / a.stream / "manifest.json").read_text())["blocks"]
    eps = []
    for b in blocks:
        fr = np.load(ROOT / f"eval_outputs/cont_{a.stream}/w0" / b["chunk"] / "episode_frames.npz")
        for key in sorted((k for k in fr.files if k.endswith("_visual")), key=lambda k: int(k[2:].split("_")[0])):
            eps.append({"shape": b["shape"], "f": F(fr[key])})

    L = [f"# 编码器 latent 能否区分形状（KNN），流 {a.stream}", "",
         "θ0 编码器（冻结）的三种特征：proj = 世界模型用的 384 维 latent；gem = 投影头之前的 512 维 GeM 池化特征；"
         "pool = 最后 7×7×512 特征图的均值+最大值池化（1024 维）。z = 按参照集逐维标准化。数值为分类准确率。", ""]
    res = {}
    groups = {"7 类（全部）": SHAPES, "4 类（见过）": [s for s in SHAPES if s in SEEN], "3 类（没见过）": [s for s in SHAPES if s not in SEEN]}

    # A: supervised KNN inside the bank, split by segment (70 / 30)
    L += ["## A. 数据集内部（每形状 70 段做参照、30 段做测试；逐帧分类）", "",
          "| 类别 | 特征 | 度量 | 标准化 | k=1 | k=5 | k=15 |", "|---|---|---|---|---:|---:|---:|"]
    best_A = {}
    for gname, shapes in groups.items():
        for feat in ("proj", "gem", "pool"):
            for metric in ("cos", "euc"):
                for z in (False, True):
                    ref = np.concatenate([bank[s][feat][:70].reshape(-1, bank[s][feat].shape[-1]) for s in shapes])
                    yr = np.concatenate([[i] * (70 * bank[s][feat].shape[1]) for i, s in enumerate(shapes)])
                    q = np.concatenate([bank[s][feat][70:].reshape(-1, bank[s][feat].shape[-1]) for s in shapes])
                    yq = np.concatenate([[i] * (30 * bank[s][feat].shape[1]) for i, s in enumerate(shapes)])
                    accs = [float((knn(ref, yr, q, k, metric, z) == yq).mean()) for k in (1, 5, 15)]
                    res[f"A|{gname}|{feat}|{metric}|{z}"] = accs
                    best_A[gname] = max(best_A.get(gname, (0, "")), (max(accs), f"{feat}/{metric}/{'z' if z else 'raw'}"))
                    L.append(f"| {gname} | {feat} | {metric} | {'是' if z else '否'} | " + " | ".join(f"{x * 100:.0f}%" for x in accs) + " |")

    # B: bank as labelled references, stream episodes as queries
    L += ["", f"## B. 用数据集 latent 做带标签参照，给流 {a.stream} 上规划器实际执行的 episode 分类", "",
          "查询方式：首帧；前 3 帧逐帧分类后多数投票；整个 episode 逐帧多数投票。", "",
          "| 类别 | 特征 | 度量 | 标准化 | k | 首帧 | 前 3 帧 | 整个 episode |", "|---|---|---|---|---:|---:|---:|---:|"]
    for gname, shapes in groups.items():
        sub = [e for e in eps if e["shape"] in shapes]
        y = np.array([shapes.index(e["shape"]) for e in sub])
        for feat in ("proj", "gem", "pool"):
            ref = np.concatenate([bank[s][feat].reshape(-1, bank[s][feat].shape[-1]) for s in shapes])
            yr = np.concatenate([[i] * bank[s][feat].shape[0] * bank[s][feat].shape[1] for i, s in enumerate(shapes)])
            for metric in ("cos", "euc"):
                for z in (False, True):
                    for k in (5,):
                        accs = []
                        for nfr in (1, 3, None):
                            pred = []
                            for e in sub:
                                q = e["f"][feat][:nfr] if nfr else e["f"][feat]
                                p = knn(ref, yr, q, k, metric, z)
                                pred.append(np.bincount(p, minlength=len(shapes)).argmax())
                            accs.append(float((np.array(pred) == y).mean()))
                        res[f"B|{gname}|{feat}|{metric}|{z}|{k}"] = accs
                        L.append(f"| {gname} | {feat} | {metric} | {'是' if z else '否'} | {k} | " + " | ".join(f"{x * 100:.0f}%" for x in accs) + " |")

    # C: label-free clustering of the stream episodes (episode mean of the first 3 frames)
    L += ["", f"## C. 无标签：对流 {a.stream} 的 episode 做 k-means（k = 形状数；每个 episode 用前 3 帧的平均 latent）", "",
          "| 类别 | 特征 | 标准化 | 纯度 | 随机基线（最大类占比） |", "|---|---|---|---:|---:|"]
    for gname, shapes in groups.items():
        sub = [e for e in eps if e["shape"] in shapes]
        y = np.array([shapes.index(e["shape"]) for e in sub])
        for feat in ("proj", "gem", "pool"):
            for z in (False, True):
                x = np.stack([e["f"][feat][:3].mean(0) for e in sub])
                if z:
                    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
                pur = float(np.mean([purity(kmeans(x, len(shapes), seed=s), y) for s in range(5)]))
                res[f"C|{gname}|{feat}|{z}"] = pur
                L.append(f"| {gname} | {feat} | {'是' if z else '否'} | {pur * 100:.0f}% | {np.bincount(y).max() / len(y) * 100:.0f}% |")
    L += ["", "## 各设置的最好结果（A）", ""] + [f"- {g}：{v[0] * 100:.0f}%（{v[1]}）" for g, v in best_A.items()]
    (out / "knn.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / "knn.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
