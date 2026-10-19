"""Offline study: what can a slow learner extract from past episodes of a domain?

Trains candidate slow learners on the executed transitions of the first two S2
blocks (I x20, square x20; saved by a w=0 stream run) and measures, relative to
theta0, the 5-step open-loop latent error (as the planner uses the model) on
  * future episodes of the same domains (blocks 3: I, 4: square)  -> what transfers
  * the T block and held-out windows of T, L, Z, +                -> interference
  * held-out small_tee windows                                    -> transfer to an unseen shape
Candidates differ in which parameters learn (predictor last block / whole
predictor / encoder head / whole encoder / combinations), their learning rates,
the loss (AdaJEPA's 1-step windows vs 5-step rollouts) and whether one learner
serves both domains or each domain has its own. Delta-averaging memories (what
the online memory arms consolidate) are measured on the same footing.

    python scripts/cont_slow_learner.py
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cont_offline import H, gmean, load_probes, load_stream, load_world, measure  # noqa: E402


def groups(wm):
    pred_last = list(wm.predictor.transformer.layers[-1].parameters()) + list(wm.predictor.transformer.norm.parameters())
    return {
        "pred_last": pred_last,
        "pred_all": list(wm.predictor.parameters()),
        "enc_head": list(wm.encoder.projection.parameters()),
        "enc_all": list(wm.encoder.parameters()),
    }


def loss_one(wm, ep, rng, max_len=6):
    """AdaJEPA's sliding-window 1-step loss on a random sub-sequence (targets stop-gradient)."""
    obs, act = ep["obs"], ep["act"]
    k = act.shape[1]
    L = min(max_len - 1, k)
    t0 = rng.randint(0, k - L)
    o = {key: v[:, t0 : t0 + L + 1] for key, v in obs.items()}
    a = torch.cat([act[:, t0 : t0 + L], torch.zeros_like(act[:, :1])], dim=1)
    z = wm.encode(o, a)
    w = min(wm.num_hist, L)
    drop = wm.action_dim
    losses = []
    for t in range(L - w + 1):
        pred = wm.predict(z[:, t : t + w])
        tgt = z[:, t + 1 : t + 1 + w].detach()
        losses.append(torch.mean((pred[..., :-drop] - tgt[..., :-drop]) ** 2))
    return torch.stack(losses).mean()


def loss_roll(wm, ep, rng):
    """5-step open-loop rollout from a single frame, as the planner uses the model."""
    obs, act = ep["obs"], ep["act"]
    k = act.shape[1]
    h = min(H, k)
    t = rng.randint(0, k - h)
    z_roll, _ = wm.rollout(obs_0={key: v[:, t : t + 1] for key, v in obs.items()}, act=act[:, t : t + h])
    with torch.no_grad():
        true = wm.encode_obs({key: v[:, t + 1 : t + 1 + h] for key, v in obs.items()})
    return (torch.mean((z_roll["visual"][:, 1:] - true["visual"]) ** 2)
            + torch.mean((z_roll["proprio"][:, 1:] - true["proprio"]) ** 2))


def anchor_loss(wm, enc0, ep, rng, n=6):
    """|E(o) - sg(E0(o))|^2 on a few frames: keeps the latent space where the predictor expects it."""
    v = ep["obs"]["visual"][0]
    idx = rng.sample(range(v.shape[0]), min(n, v.shape[0]))
    x = wm.encoder_transform(v[idx])
    with torch.no_grad():
        z0 = enc0(x)
    return torch.mean((wm.encoder(x) - z0) ** 2)


def train(wm, spec, train_eps, steps, batch, seed):
    """spec: {"groups": {name: lr}, "loss": "one"|"roll", "anchor": beta}. Trains in place; returns the final loss."""
    g = groups(wm)
    beta = float(spec.get("anchor", 0.0))
    enc0 = copy.deepcopy(wm.encoder).eval() if beta > 0 else None
    for p in wm.parameters():
        p.requires_grad_(False)
    param_groups, seen = [], set()
    for name, lr in spec["groups"].items():
        ps = [p for p in g[name] if id(p) not in seen]
        seen.update(id(p) for p in ps)
        for p in ps:
            p.requires_grad_(True)
        param_groups.append({"params": ps, "lr": lr})
    opt = torch.optim.Adam(param_groups)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    wm.train()
    last = []
    for step in range(steps):
        opt.zero_grad()
        batch_eps = rng.sample(train_eps, min(batch, len(train_eps)))
        loss = torch.stack([
            (loss_one(wm, ep, rng) if spec["loss"] == "one" else loss_roll(wm, ep, rng)) for ep in batch_eps]).mean()
        if beta > 0:
            loss = loss + beta * torch.stack([anchor_loss(wm, enc0, ep, rng) for ep in batch_eps]).mean()
        loss.backward()
        opt.step()
        last.append(loss.item())
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return float(np.mean(last[-20:]))


@torch.no_grad()
def latent_var(wm, eps):
    """Mean per-dimension variance of visual latents over the set's frames (collapse check)."""
    z = torch.cat([wm.encode_obs({k: v[:, :6] for k, v in ep["obs"].items()})["visual"].flatten(0, 2) for ep in eps])
    return float(z.var(0).mean())


def evaluate(wm, sets, base, var0=None):
    res = {name: gmean([measure(wm, ep)["roll"] / b["roll"] for ep, b in zip(eps, base[name])])
           for name, eps in sets.items()}
    if var0 is not None:  # < 1: latents shrank, so a lower error may partly be collapse
        res["latent_var_ratio"] = latent_var(wm, sets["I_future"] + sets["bank_T"]) / var0
    return res


def export(wm, ckpt_dir, epoch, out_dir, meta):
    """Write theta0's checkpoint with the trained encoder/predictor, so plan.py can load it."""
    import shutil
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(ckpt_dir) / "hydra.yaml", out_dir / "hydra.yaml")
    payload = torch.load(Path(ckpt_dir) / "checkpoints" / f"model_{epoch}.pth", map_location="cpu")
    payload["encoder"] = copy.deepcopy(wm.encoder).cpu()
    payload["predictor"] = copy.deepcopy(wm.predictor).cpu()
    torch.save(payload, out_dir / "checkpoints" / f"model_{epoch}.pth")
    (out_dir / "slow_learner.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="eval_outputs/cont_S2/w0")
    ap.add_argument("--stream", default="S2")
    ap.add_argument("--ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--probe-n", type=int, default=30)
    ap.add_argument("--out", default="reports/cont_S2/slow_learner")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    ap.add_argument("--export", default=None, help='JSON spec, e.g. {"groups": {"pred_last": 3e-4}, "loss": "one"}: '
                    "train it on --export-domains and write a loadable checkpoint dir to --export-dir instead of the grid")
    ap.add_argument("--export-domains", nargs="+", default=["I", "square"])
    ap.add_argument("--export-dir", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre, params = load_world(ROOT / a.ckpt, "3", device)
    blocks, eps = load_stream(ROOT / a.run, a.stream, pre, device)
    probes = load_probes(pre, ["T", "L", "Z", "+", "small_tee"], a.probe_n, device)
    by_block = {k: [e for e in eps if e["block"] == k] for k in range(len(blocks))}
    train_sets = {"I": by_block[0], "square": by_block[1]}
    test = {"I_future": by_block[3], "square_future": by_block[4], "T_block": by_block[2],
            **{f"bank_{s}": ps for s, ps in probes.items()}}
    if a.export:
        spec = json.loads(a.export)
        tl = train(wm, spec, sum((train_sets[d] for d in a.export_domains), []), a.steps, a.batch, seed=0)
        export(wm, ROOT / a.ckpt, "3", ROOT / a.export_dir,
               {"spec": spec, "domains": a.export_domains, "steps": a.steps, "batch": a.batch, "train_loss": tl,
                "source_run": a.run, "blocks": [blocks[0]["chunk"], blocks[1]["chunk"]]})
        print(f"exported to {a.export_dir} (train loss {tl:.4f})")
        return
    theta0 = copy.deepcopy(wm.state_dict())
    base = {name: [measure(wm, ep) for ep in es] for name, es in test.items()}
    var0 = latent_var(wm, test["I_future"] + test["bank_T"])
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(name, regime, res, train_loss=float("nan")):
        row = {"name": name, "regime": regime, "train_loss": train_loss, **res}
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    # --- delta-averaging memories (what the online memory arms consolidate), oracle domain routing
    flat = lambda: torch.cat([p.detach().flatten() for p in params])
    def set_flat(v):
        o = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.copy_(v[o : o + n].view_as(p))
                o += n
    th0 = flat()
    means = {d: torch.stack([e["delta"] for e in es]).mean(0) for d, es in train_sets.items()}
    for alpha in (1.0, 2.0):
        res = {}
        for d, key in (("I", "I_future"), ("square", "square_future")):
            set_flat(th0 + alpha * means[d])
            res[key] = evaluate(wm, {key: test[key]}, base)[key]
        set_flat(th0 + alpha * (means["I"] + means["square"]) / 2)
        res.update({k: v for k, v in evaluate(wm, {k: test[k] for k in test if not k.endswith("future")}, base).items()})
        set_flat(th0)
        record(f"delta-mean a={alpha:g}", "per-domain (future) / shared (interference)", res)

    # --- trained slow learners
    grid = [  # (label, {group: lr}, anchor beta)
        ("pred_last", {"pred_last": 1e-4}, 0), ("pred_last", {"pred_last": 3e-4}, 0), ("pred_last", {"pred_last": 1e-3}, 0),
        ("pred_all", {"pred_all": 3e-5}, 0), ("pred_all", {"pred_all": 1e-4}, 0), ("pred_all", {"pred_all": 3e-4}, 0),
        ("enc_head", {"enc_head": 1e-5}, 0), ("enc_head", {"enc_head": 1e-4}, 0),
        ("enc_head", {"enc_head": 1e-5}, 10), ("enc_head", {"enc_head": 1e-4}, 10),
        ("enc_all", {"enc_all": 1e-5}, 0), ("enc_all", {"enc_all": 3e-5}, 0),
        ("enc_all", {"enc_all": 1e-5}, 10), ("enc_all", {"enc_all": 3e-5}, 10),
        ("pred_last+enc_head", {"pred_last": 3e-4, "enc_head": 1e-4}, 0),
        ("pred_last+enc_head", {"pred_last": 3e-4, "enc_head": 1e-4}, 10),
    ]
    roll_grid = [g for g in grid if (g[0], g[2]) in (("pred_last", 0), ("pred_all", 0), ("pred_last+enc_head", 10))
                 and list(g[1].values())[0] in (3e-4, 1e-4)]
    if a.quick:
        grid, roll_grid = grid[1:2] + grid[9:10], grid[1:2]
    for loss, configs in (("one", grid), ("roll", roll_grid)):
        for gname, lrs, beta in configs:
            spec = {"groups": lrs, "loss": loss, "anchor": beta}
            label = (f"{gname} " + " ".join(f"{k}={v:g}" for k, v in lrs.items()) + f" loss={loss}"
                     + (f" anchor={beta:g}" if beta else ""))
            # shared: one learner on both domains' past episodes
            tl = train(wm, spec, train_sets["I"] + train_sets["square"], a.steps, a.batch, seed=0)
            record(label, "shared", evaluate(wm, test, base, var0), tl)
            wm.load_state_dict(theta0)
            # per-domain: one learner per domain, evaluated on its own future; interference averaged
            res, inter = {}, {}
            for d, key in (("I", "I_future"), ("square", "square_future")):
                train(wm, spec, train_sets[d], a.steps, a.batch, seed=0)
                r = evaluate(wm, test, base, var0)
                res[key] = r[key]
                for k, v in r.items():
                    if not k.endswith("future"):
                        inter.setdefault(k, []).append(v)
                wm.load_state_dict(theta0)
            res.update({k: gmean(v) for k, v in inter.items()})
            record(label, "per-domain", res)
    (out / "results.json").write_text(json.dumps({"steps": a.steps, "batch": a.batch, "rows": rows}, indent=1))
    write_md(rows, out / "summary.md", a)
    print((out / "summary.md").read_text())


def write_md(rows, path, a):
    cols = ["I_future", "square_future", "T_block", "bank_T", "bank_L", "bank_Z", "bank_+", "bank_small_tee", "latent_var_ratio"]
    L = ["# 离线慢学习器：从过去的 episode 里能学到多少可迁移的东西", "",
         f"训练数据：S2 前两段（I×20、square×20）执行过的转移；每个学习器 {a.steps} 步、batch {a.batch}。",
         "数值 = 5 步开环预测误差（规划器的用法）相对 θ0 的比值，几何平均；<1 更准，>1 变差。",
         "I_future / square_future = S2 第 4、5 段（之后的同领域 episode）；T_block、bank_T/L/Z/+ = 干扰；bank_small_tee = 未出现形状。",
         "latent_var_ratio = 隐表示方差 / θ0 的方差（明显 <1 说明 encoder 在塌缩，误差变小可能是假象）。", "",
         "| 学习器 | 方式 | " + " | ".join(cols) + " |", "|---|---|" + "---:|" * len(cols)]
    for r in rows:
        L.append(f"| {r['name']} | {r['regime']} | " + " | ".join(f"{r.get(c, float('nan')):.3f}" for c in cols) + " |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
