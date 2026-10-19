"""Offline replay of slow-memory write/read rules on a recorded stream (no planning).

A w=0 (AdaJEPA) stream run with save_deltas/save_frames gives, per episode, the
fast delta AdaJEPA learned from theta0 and the executed frames. A rule decides,
after each episode, which slow memory receives what part of the delta, and,
before each episode, which memory (or none) the episode starts from. Without
re-planning we then measure how well the episode-start weights predict the
episode's own transitions, relative to theta0:

  one    1-step teacher-forced latent error over all the episode's windows
  roll   open-loop error of 5-step rollouts from single frames (as the planner uses)
  new3   newest-frame error over replans 1-3 (the quantity the online runs logged)

Ratios < 1 mean the slow memory already helps on that episode. Returning blocks
show benefit; a new domain (T after I/square) shows interference. At the end, each
rule's memories are also probed on held-out windows (data/exp0_bank) of all
seven shapes.

Approximation: deltas come from a run without consolidation, whereas with slow
weights in place the fast deltas would differ. The shared fixed-w rules can be
checked against the real S2 runs (new3).

    python scripts/cont_offline.py --stream S2 --run eval_outputs/cont_S2/w0
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
H = 5


def load_world(ckpt_dir, epoch, device):
    import hydra
    import plan
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD, STATE_MEAN, STATE_STD
    from planning.adajepa import AdaJEPATrainer
    from preprocessor import Preprocessor

    cfg = OmegaConf.load(Path(ckpt_dir) / "hydra.yaml")
    wm = plan.load_model(Path(ckpt_dir) / "checkpoints" / f"model_{epoch}.pth", cfg, cfg.num_action_repeat, device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    tcfg = cfg.env.dataset.get("transform", None)
    pre = Preprocessor(ACTION_MEAN, ACTION_STD, STATE_MEAN[:5], STATE_STD[:5], PROPRIO_MEAN[:4], PROPRIO_STD[:4],
                       hydra.utils.instantiate(tcfg) if tcfg is not None else default_transform())
    tr = AdaJEPATrainer(wm=wm, lr=5e-4, finetune_encoder=True, last_layer_only=True, encoder_lr=1e-5,
                        encoder_last_layer_only=True)
    params = list(tr._ada_predictor_params) + list(tr._ada_encoder_params)
    return wm, pre, params


class Weights:
    def __init__(self, params):
        self.params = params
        self.theta0 = torch.cat([p.detach().flatten() for p in params]).clone()

    @torch.no_grad()
    def set(self, slow=None):
        v = self.theta0 if slow is None else self.theta0 + slow
        o = 0
        for p in self.params:
            n = p.numel()
            p.copy_(v[o : o + n].view_as(p))
            o += n


@torch.no_grad()
def measure(wm, ep, first_only=False):
    """Latent prediction errors of the current weights on one episode's transitions."""
    drop = wm.action_dim
    obs, act = ep["obs"], ep["act"]
    k = act.shape[1]
    z = wm.encode(obs, torch.cat([act, torch.zeros_like(act[:, :1])], dim=1))
    mse = lambda a, b: float(torch.mean((a[..., :-drop] - b[..., :-drop]) ** 2))
    if first_only:  # frame 1 from frame 0
        return mse(wm.predict(z[:, 0:1])[:, -1:], z[:, 1:2])
    w = min(wm.num_hist, k)
    one = np.mean([mse(wm.predict(z[:, t : t + w]), z[:, t + 1 : t + 1 + w]) for t in range(k - w + 1)])
    new3 = np.mean([mse(wm.predict(z[:, max(0, t - wm.num_hist) : t])[:, -1:], z[:, t : t + 1])
                    for t in range(1, min(3, k) + 1)])
    h = min(H, k)
    starts = list(range(k - h + 1))
    obs0 = {key: torch.stack([v[0, t : t + 1] for t in starts]) for key, v in obs.items()}
    acts = torch.stack([act[0, t : t + h] for t in starts])
    z_roll, _ = wm.rollout(obs_0=obs0, act=acts)
    true = wm.encode_obs(obs)
    roll = float(np.mean([
        torch.mean((z_roll["visual"][i, 1:] - true["visual"][0, t + 1 : t + 1 + h]) ** 2).item()
        + torch.mean((z_roll["proprio"][i, 1:] - true["proprio"][0, t + 1 : t + 1 + h]) ** 2).item()
        for i, t in enumerate(starts)]))
    return {"one": float(one), "new3": float(new3), "roll": roll}


# ------------------------------------------------------------------ rules
class Rule:
    """read(ep) -> memory key or None (theta0); write(ep, delta)."""
    name = "rule"

    def __init__(self):
        self.mem = {}  # key -> {"sum": tensor, "n": int, "mean": tensor}

    def slow(self, key):
        return None if key is None or key not in self.mem else self.value(self.mem[key])

    def value(self, m):
        raise NotImplementedError

    def add(self, key, delta):
        m = self.mem.setdefault(key, {"sum": torch.zeros_like(delta), "n": 0})
        m["sum"] += delta
        m["n"] += 1


class NoSlow(Rule):
    name = "w0 (AdaJEPA)"

    def read(self, ep, wm, W):
        return None

    def write(self, ep, delta):
        pass


class Fixed(Rule):
    """slow <- slow + w * delta, one shared memory or one per true shape (oracle)."""

    def __init__(self, w, oracle=False):
        super().__init__()
        self.w, self.oracle = w, oracle
        self.name = f"{'oracle' if oracle else 'shared'} fixed w={w:g}"

    def value(self, m):
        return self.w * m["sum"]

    def read(self, ep, wm, W):
        return ep["tag"] if self.oracle else "shared"

    def write(self, ep, delta):
        self.add(ep["tag"] if self.oracle else "shared", delta)


class Mean(Rule):
    """slow = alpha * mean of the deltas a memory received (the consistent part survives averaging).

    write: 'shared' | 'oracle' (true shape) | 'delta' (nearest memory by cosine to its mean,
    new memory below tau). read: 'same' as write (oracle/shared), or 'pred' (the memory, or
    theta0, with the lowest error on the episode's first transition).
    """

    def __init__(self, alpha, write="oracle", read="same", tau=0.03, min_n=1):
        super().__init__()
        self.alpha, self.write_mode, self.read_mode, self.tau, self.min_n = alpha, write, read, tau, min_n
        self.name = f"mean a={alpha:g} write={write} read={read}"
        self.routes = []  # (true tag, read key, write key)

    def value(self, m):
        return self.alpha * m["sum"] / m["n"] if m["n"] >= self.min_n else None

    def read(self, ep, wm, W):
        if self.read_mode == "same":
            key = "shared" if self.write_mode == "shared" else (ep["tag"] if self.write_mode == "oracle" else None)
        else:
            cands = [None] + [k for k, m in self.mem.items() if m["n"] >= self.min_n]
            errs = []
            for key in cands:
                W.set(self.slow(key))
                errs.append(measure(wm, ep, first_only=True))
            key = cands[int(np.argmin(errs))]
        self._read_key = key
        return key

    def write(self, ep, delta):
        if self.write_mode == "shared":
            key = "shared"
        elif self.write_mode == "oracle":
            key = ep["tag"]
        else:
            best, key = -1.0, None
            for k, m in self.mem.items():
                c = float(torch.nn.functional.cosine_similarity(delta, m["sum"] / m["n"], dim=0))
                if c > best:
                    best, key = c, k
            if key is None or best < self.tau:
                key = f"m{len(self.mem)}"
        self.routes.append((ep["tag"], self._read_key, key))
        self.add(key, delta)


# ------------------------------------------------------------------ data
def load_stream(run_dir, stream, pre, device):
    blocks = json.loads((ROOT / "data/cont_streams" / stream / "manifest.json").read_text())["blocks"]
    eps = []
    for k, b in enumerate(blocks):
        d = run_dir / b["chunk"]
        deltas = torch.load(d / "deltas.pt")
        frames = np.load(d / "episode_frames.npz")
        for e, delta in zip(deltas["episodes"], deltas["deltas"]):
            act = torch.as_tensor(frames[f"ep{e}_act"])[None].to(device)
            if act.shape[1] < 1:
                continue
            obs = pre.transform_obs({"visual": frames[f"ep{e}_visual"][None], "proprio": frames[f"ep{e}_proprio"][None]})
            eps.append({"e": e, "block": k, "tag": b["shape"], "delta": delta.float().to(device),
                        "obs": {key: v.to(device) for key, v in obs.items()}, "act": act})
    return blocks, eps


def load_probes(pre, shapes, n, device, fs=5):
    out = {}
    for shape in shapes:
        d = np.load(ROOT / "data/exp0_bank" / f"{shape}.npz")
        act = pre.normalize_actions(torch.as_tensor(d["env_actions"][:n], dtype=torch.float32))
        act = act.reshape(act.shape[0], -1, fs * act.shape[-1])
        out[shape] = []
        for j in range(min(n, len(act))):
            obs = pre.transform_obs({"visual": d["visual"][j : j + 1], "proprio": d["proprio"][j : j + 1]})
            out[shape].append({"tag": shape, "obs": {k: v.to(device) for k, v in obs.items()},
                               "act": act[j : j + 1].to(device)})
    return out


def gmean(xs):
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S2")
    ap.add_argument("--run", default="eval_outputs/cont_S2/w0")
    ap.add_argument("--ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
    ap.add_argument("--epoch", default="3")
    ap.add_argument("--probe-shapes", nargs="+", default=["I", "square", "T", "L", "Z", "+", "small_tee"])
    ap.add_argument("--probe-n", type=int, default=30)
    ap.add_argument("--out", default="reports/cont_S2/offline")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre, params = load_world(ROOT / a.ckpt, a.epoch, device)
    W = Weights(params)
    blocks, eps = load_stream(ROOT / a.run, a.stream, pre, device)
    probes = load_probes(pre, a.probe_shapes, a.probe_n, device)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    W.set(None)
    base = [measure(wm, ep) for ep in eps]
    base_probe = {s: [measure(wm, p) for p in ps] for s, ps in probes.items()}

    rules = [NoSlow(), Fixed(0.003), Fixed(0.01), Fixed(0.03), Fixed(0.03, oracle=True),
             Mean(1.0, write="shared"), Mean(1.0, write="oracle"), Mean(2.0, write="oracle"),
             Mean(1.0, write="oracle", read="pred"),
             Mean(1.0, write="delta", read="pred", tau=0.02, min_n=3),
             Mean(2.0, write="delta", read="pred", tau=0.02, min_n=3)]
    results = {}
    for rule in rules:
        per_ep = []
        for ep, b0 in zip(eps, base):
            key = rule.read(ep, wm, W)
            W.set(rule.slow(key))
            m = measure(wm, ep)
            per_ep.append({"block": ep["block"], "tag": ep["tag"], "read": key,
                           **{k: m[k] / b0[k] for k in m}})
            rule.write(ep, ep["delta"])
        # end of stream: held-out windows of every shape, read the way the rule would
        probe = {}
        for s, ps in probes.items():
            ratios = []
            for p, b0 in zip(ps, base_probe[s]):
                key = rule.read(p, wm, W)
                W.set(rule.slow(key))
                ratios.append(measure(wm, p)["roll"] / b0["roll"])
            probe[s] = gmean(ratios)
        blk = {}
        for k in range(len(blocks)):
            rows = [r for r in per_ep if r["block"] == k]
            blk[k] = {m: gmean([r[m] for r in rows]) for m in ("one", "new3", "roll")}
        res = {"blocks": blk, "probe_roll": probe}
        if isinstance(rule, Mean) and rule.routes:
            tags = [t for t, _, _ in rule.routes]
            wkeys = [w for _, _, w in rule.routes]
            # purity of write clusters w.r.t. true shapes
            purity = sum(max(sum(1 for t, w in zip(tags, wkeys) if w == c and t == s) for s in set(tags))
                         for c in set(wkeys)) / len(tags)
            res["write_clusters"] = {c: dict(zip(*np.unique([t for t, w in zip(tags, wkeys) if w == c], return_counts=True)))
                                     for c in sorted(set(wkeys))}
            res["write_purity"] = purity
            if rule.read_mode == "pred":
                res["read_keys"] = [r for _, r, _ in rule.routes]
        results[rule.name] = res
        print(rule.name, {k: round(v["roll"], 3) for k, v in blk.items()}, {s: round(v, 3) for s, v in probe.items()}, flush=True)

    real = real_new3(ROOT / a.run.replace("/w0", ""), blocks)
    summary = {"blocks": [(b["chunk"], b["shape"]) for b in blocks], "results": results, "real_new3": real}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else str(o)))
    write_md(summary, out / "summary.md")
    print((out / "summary.md").read_text())


def real_new3(root, blocks):
    """The online runs' own start-of-episode ratio (newest frame, replans 1-3), per arm and block."""
    out = {}
    for arm in sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("w")):
        vals = []
        for b in blocks:
            f = root / arm / b["chunk"] / "per_iter_logs.json"
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
            rows = [r for r in rows if r.get("step", 99) <= 3 and "loss_new_start" in r]
            vals.append(gmean([r["loss_new_start"] / r["loss_new_init"] for r in rows if r["loss_new_init"] > 0]))
        out[arm] = vals
    return out


def write_md(s, path):
    blocks = s["blocks"]
    hdr = " | ".join(f"{k}:{sh}" for k, (_, sh) in enumerate(blocks))
    L = ["# 离线写入规则筛选（S2，不重跑规划）", "",
         "数值 = 用 episode 开始时的权重（θ0 + 读取的慢记忆）预测该 episode 自己的转移，相对 θ0 的误差比（几何平均）；<1 表示慢记忆已经有帮助，>1 表示干扰。",
         "roll = 从单帧出发 5 步开环预测（与规划器的用法相同）；new3 = 前 3 次重规划的最新帧误差（在线实验记录的同一指标）。", "",
         "## 模拟器校验：共享固定 w 的 new3，模拟 vs 在线实测", "", f"| 规则 | {hdr} |", "|---|" + "---:|" * len(blocks)]
    for w in ("0.003", "0.01", "0.03"):
        sim = s["results"].get(f"shared fixed w={w}")
        if sim:
            L.append(f"| 模拟 w={w} | " + " | ".join(f"{sim['blocks'][k]['new3']:.2f}" if isinstance(sim['blocks'], dict) and k in sim['blocks'] else f"{sim['blocks'][str(k)]['new3']:.2f}" for k in range(len(blocks))) + " |")
        if f"w{w}" in s["real_new3"]:
            L.append(f"| 实测 w={w} | " + " | ".join(f"{v:.2f}" for v in s["real_new3"][f"w{w}"]) + " |")
    L += ["", "## 各规则：开环 5 步预测误差比（每段几何平均）", "", f"| 规则 | {hdr} |", "|---|" + "---:|" * len(blocks)]
    for name, r in s["results"].items():
        b = r["blocks"]
        L.append(f"| {name} | " + " | ".join(f"{(b[k] if k in b else b[str(k)])['roll']:.3f}" for k in range(len(blocks))) + " |")
    shapes = list(next(iter(s["results"].values()))["probe_roll"])
    L += ["", "## 流结束后，各形状保留数据上的开环误差比（按规则的读取方式选择记忆）", "",
          "| 规则 | " + " | ".join(shapes) + " |", "|---|" + "---:|" * len(shapes)]
    for name, r in s["results"].items():
        L.append(f"| {name} | " + " | ".join(f"{r['probe_roll'][sh]:.3f}" for sh in shapes) + " |")
    L += ["", "## 无标签路由的质量", ""]
    for name, r in s["results"].items():
        if "write_purity" in r:
            L.append(f"- {name}：写入聚类纯度 {r['write_purity']:.2f}，聚类构成 {r['write_clusters']}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
