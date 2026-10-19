"""Offline AdaJEPA adaptivity test (CPU): does one adaptation step per replan make
the world model predict the *upcoming* frames better?

Replays the AdaJEPA loop without a planner: along recorded trajectories, at each
model step t the model is adapted exactly as planning does (AdaJEPATrainer, one
Adam step, recent-5 buffer merged into one sequence), then predicts frames
t+1..t+H open-loop from the true actions. Errors are latent MSEs in each model's
own space (action dims excluded, as in the adaptation loss) and are reported as
ratios to the same model's frozen error, so models with different latent scales
are comparable.

    python scripts/diagnose_adaptation.py --out reports/.../diagnostics/adaptivity.json
"""
import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

FRAMESKIP, HIST, HORIZON, BUFFER = 5, 3, 5, 5


def build_wm(model_dir, ckpt_name):
    import plan  # noqa: E402  (reuses the planner's loader)
    cfg = OmegaConf.load(Path(model_dir) / "hydra.yaml")
    wm = plan.load_model(Path(model_dir) / "checkpoints" / ckpt_name, cfg, cfg.num_action_repeat, torch.device("cpu"))
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False  # as BasePlanner does
    return wm


def our_val_trajectories(data_dir, shape, n):
    from datasets.pusht_dset import PushTDataset
    from datasets.img_transforms import default_transform
    ds = PushTDataset(n_rollout=None, transform=default_transform(224), data_path=str(Path(data_dir) / "val"),
                      normalize_action=True, with_velocity=True)
    idx = [i for i in range(len(ds)) if ds.shapes[i] == shape][:n]
    out = []
    for i in idx:
        T = int(ds.get_seq_length(i))
        m = (T - 1) // FRAMESKIP  # model steps with a full action chunk
        frames = list(range(0, m * FRAMESKIP + 1, FRAMESKIP))
        obs, _, _, _ = ds.get_frames(i, frames)
        act = ds.actions[i, : m * FRAMESKIP]  # already normalized
        out.append({"visual": obs["visual"], "proprio": obs["proprio"],
                    "act": act.reshape(m, FRAMESKIP * act.shape[-1])})
    return out


def author_segments(shape, n, seed):
    """Author test segments replayed in the env (26 env frames -> 6 model frames)."""
    import gym
    import env  # noqa: F401
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD
    segs = pickle.load(open(ROOT / f"data/pushobj_eval/val_{shape}/plan_targets.pkl", "rb"))["segments"]
    rs = np.random.RandomState(seed)
    e = gym.make("pushobj", with_velocity=True, with_target=True).unwrapped
    e.update_env({"shape": shape})
    tf = default_transform(224)
    out = []
    for k in rs.choice(len(segs), n, replace=False):
        s = segs[k]
        acts = np.asarray(s["actions"], dtype=np.float64)
        obs, _ = e.prepare(0, np.r_[np.asarray(s["states"][0], dtype=np.float64)[:5], 0.0, 0.0])
        vis, prop = [obs["visual"]], [obs["proprio"]]
        for a in acts:
            obs, _, _, _ = e.step(a / 100.0)
            vis.append(obs["visual"]); prop.append(obs["proprio"])
        vis, prop = np.stack(vis)[::FRAMESKIP], np.stack(prop)[::FRAMESKIP]
        m = len(acts) // FRAMESKIP
        visual = tf(torch.tensor(vis).permute(0, 3, 1, 2).float() / 255.0)
        proprio = (torch.tensor(prop, dtype=torch.float32) - PROPRIO_MEAN[:4]) / PROPRIO_STD[:4]
        a = (torch.tensor(acts[: m * FRAMESKIP], dtype=torch.float32) / 100.0 - ACTION_MEAN) / ACTION_STD
        out.append({"visual": visual[: m + 1], "proprio": proprio[: m + 1], "act": a.reshape(m, -1)})
    e.close()
    return out


def seg(traj, a, b):
    """Frames a..b (inclusive) with the actions a..b-1, batched, as AdaJEPA buffers them."""
    return {"visual": traj["visual"][a:b + 1][None], "proprio": traj["proprio"][a:b + 1][None]}, traj["act"][a:b][None]


@torch.no_grad()
def future_error(wm, traj, t, H):
    """Open-loop latent error for frames t+1..t+H from history t-2..t and the true actions."""
    drop = wm.action_dim
    act = traj["act"][t - HIST + 1: t + H][None]              # actions at frames t-2 .. t+H-1
    obs0 = {"visual": traj["visual"][t - HIST + 1: t + 1][None], "proprio": traj["proprio"][t - HIST + 1: t + 1][None]}
    _, z = wm.rollout(obs0, act)
    pred = z[:, HIST: HIST + H, :, :-drop]
    tgt_obs = {"visual": traj["visual"][t + 1: t + H + 1][None], "proprio": traj["proprio"][t + 1: t + H + 1][None]}
    tgt = wm.encode(tgt_obs, torch.zeros(1, H, act.shape[-1]))[..., :-drop]
    err = (pred - tgt).pow(2).mean(dim=(0, 2, 3))              # per horizon step
    return err.numpy()


@torch.no_grad()
def buffer_loss(wm, trainer, buf_o, buf_a):
    mo, ma = trainer._merge_segments(buf_o, buf_a) if len(buf_o) > 1 else (buf_o, buf_a)
    return float(trainer._prediction_loss(wm.encode(*trainer._prepare_segment(mo[0], ma[0]))))


def run(wm, trajs, lr, enc_lr):
    from planning.adajepa import AdaJEPATrainer
    trainer = AdaJEPATrainer(wm, lr=lr, steps=1, optimizer_name="adam", finetune_encoder=True,
                             last_layer_only=True, encoder_lr=enc_lr, encoder_last_layer_only=True)
    rows = []
    for j, tr in enumerate(trajs):
        m = tr["act"].shape[0]
        steps = list(range(HIST - 1, m))
        trainer.reset()
        frozen = {t: future_error(wm, tr, t, min(HORIZON, m - t)) for t in steps}
        buf_o, buf_a = [], []
        for t in range(1, m):
            o, a = seg(tr, t - 1, t)                           # the segment executed at replan t
            buf_o.append(o); buf_a.append(a)
            del buf_o[:-BUFFER], buf_a[:-BUFFER]
            before = buffer_loss(wm, trainer, buf_o, buf_a)   # eval mode, like `after`
            trainer.finetune(buf_o, buf_a, merge=True)
            after = buffer_loss(wm, trainer, buf_o, buf_a)
            if t in frozen:
                ad = future_error(wm, tr, t, min(HORIZON, m - t))
                fz = frozen[t]
                rows.append({"traj": j, "t": t, "buffer_before": before, "buffer_after": after,
                             "next1_ratio": float(ad[0] / fz[0]),
                             "rollout_ratio": float(ad.mean() / fz.mean()),
                             "frozen_next1": float(fz[0]), "frozen_rollout": float(fz.mean())})
        trainer.reset()
    return rows


def summarize(rows):
    r = lambda k: np.array([x[k] for x in rows])
    gm = lambda v: float(np.exp(np.mean(np.log(v))))
    buf = r("buffer_after") / r("buffer_before")
    late = [x for x in rows if x["t"] >= 6]
    return {"n": len(rows),
            "buffer_fit_ratio_gmean": gm(buf),
            "next1_ratio_gmean": gm(r("next1_ratio")), "next1_improved_frac": float((r("next1_ratio") < 1).mean()),
            "rollout_ratio_gmean": gm(r("rollout_ratio")), "rollout_improved_frac": float((r("rollout_ratio") < 1).mean()),
            "rollout_ratio_gmean_t>=6": gm(np.array([x["rollout_ratio"] for x in late])) if late else None,
            "frozen_rollout_median": float(np.median(r("frozen_rollout"))),
            "frozen_next1_median": float(np.median(r("frozen_next1")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/pushobj_bias_16k/adaptivity_default.json")
    ap.add_argument("--shape", default="T")
    ap.add_argument("--n-ours", type=int, default=50)
    ap.add_argument("--n-author", type=int, default=40)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--models", default=None,
                    help="comma list of name=model_dir:ckpt_name (default: released, bias16k e3)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    # The raw-replay models (ours e3/e5 in reports/pushobj_replay_20k_e5) were deleted to free disk;
    # their numbers stay in that report. data/pushobj_replay_20k/val is kept as the fixed reference set.
    models = {"released (e3)": ("checkpoints/pushobj_shape_shift", "model_latest.pth"),
              "bias16k e3": ("checkpoints/pushobj_bias_16k_e3_seed0", "model_3.pth")}
    if a.models:
        models = {}
        for item in a.models.split(","):
            name, spec = item.split("=")
            d, ck = spec.split(":")
            models[name] = (d, ck)
    data = {"ours val T (held-out pusht_noise replays)": our_val_trajectories(ROOT / "data/pushobj_replay_20k", a.shape, a.n_ours),
            "author test T segments (replayed)": author_segments(a.shape, a.n_author, seed=0)}
    print({k: (len(v), int(sum(x["act"].shape[0] for x in v))) for k, v in data.items()}, flush=True)
    result = {"config": {"lr": 5e-4, "encoder_lr": 1e-5, "buffer": BUFFER, "horizon": HORIZON, "frameskip": FRAMESKIP},
              "results": {}}
    for mname, (d, ck) in models.items():
        wm = build_wm(ROOT / d, ck)
        for dname, trajs in data.items():
            rows = run(wm, trajs, 5e-4, 1e-5)
            result["results"].setdefault(mname, {})[dname] = summarize(rows)
            print(mname, "|", dname, "|", json.dumps(result["results"][mname][dname]), flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
