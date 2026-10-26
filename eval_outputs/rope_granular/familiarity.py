"""theta0's familiarity score on rope and granular data windows, for the rope DINO-WM.

The score is the one CLSMPCPlanner._familiarity uses to route an episode: theta0's newest-frame latent error over the
first 3 transitions (geometric mean, action channels excluded), here on 4-frame windows of DINO-WM's deformable data
(validation split, frameskip 1 as the model; actions normalized with the model's own rope statistics, as planning
does). It calibrates planner.continual.familiar_thr (90th percentile on rope, theta0's training domain) and shows,
before any stream runs, whether the familiarity gate separates granular from rope. Writes familiarity.json.

    /data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_granular/familiarity.py [--n 200]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
CKPT = Path("/data/zjc/adaJEPA/CL-JEPA/checkpoints/rope_dino_h1_fs1_e100_seed0")
DATA = "/data/zjc/adaJEPA/CL-JEPA/data/deformable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="windows per object")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import plan  # noqa: E402  (repo import, after sys.path)
    from datasets.deformable_env_dset import load_deformable_dset_slice_train_val
    from datasets.img_transforms import default_transform

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(CKPT / "hydra.yaml")
    wm = plan.load_model(CKPT / "checkpoints" / "model_latest.pth", cfg, cfg.num_action_repeat, dev)
    wm.eval()  # VWorldModel.eval does not return the model
    drop = wm.action_dim
    transform = default_transform(img_size=cfg.img_size)

    def val_split(obj):
        _, traj = load_deformable_dset_slice_train_val(transform, n_rollout=None, data_path=DATA, object_name=obj,
                                                       normalize_action=True, split_ratio=cfg.env.dataset.split_ratio,
                                                       num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
        return traj["valid"]

    rope_val = val_split("rope")
    rope_ds = rope_val.dataset
    rope_mean, rope_std = rope_ds.action_mean, rope_ds.action_std

    @torch.no_grad()
    def scores(val, own_mean, own_std):
        rng = random.Random(a.seed)
        ds, idx = val.dataset, list(val.indices)
        out = []
        for i in rng.sample(idx, min(a.n, len(idx))):
            T = int(ds.get_seq_length(i))
            t0 = rng.randrange(0, T - 3)
            obs, act, _, _ = ds.get_frames(i, list(range(t0, t0 + 4)))
            act = (act * own_std + own_mean - rope_mean) / rope_std  # the model's own (rope) normalization
            act = torch.cat([act[:3], torch.zeros_like(act[:1])])  # 3 transitions + the dummy last action
            o = {"visual": obs["visual"][None].to(dev), "proprio": obs["proprio"][None].to(dev)}
            z = wm.encode(o, act[None].to(dev))
            errs = []
            for t in range(1, 4):
                src = z[:, max(0, t - wm.num_hist) : t]
                pred = rearrange(wm.predictor(rearrange(src, "b t p d -> b (t p) d")), "b (t p) d -> b t p d", t=src.shape[1])
                errs.append(float(torch.mean((pred[:, -1:, ..., :-drop] - z[:, t : t + 1, ..., :-drop]) ** 2)))
            out.append(float(np.exp(np.mean(np.log(np.maximum(errs, 1e-12))))))
        return np.array(out)

    s_rope = scores(rope_val, rope_mean, rope_std)
    gran_val = val_split("granular")
    s_gran = scores(gran_val, gran_val.dataset.action_mean, gran_val.dataset.action_std)
    thr = float(np.percentile(s_rope, 90))
    auc = float(np.mean([(g > r) + 0.5 * (g == r) for g in s_gran for r in s_rope]))
    res = {"n_rope": len(s_rope), "n_granular": len(s_gran),
           "rope_percentiles": {q: float(np.percentile(s_rope, q)) for q in (10, 50, 80, 90, 95, 99)},
           "granular_percentiles": {q: float(np.percentile(s_gran, q)) for q in (1, 5, 10, 50, 90)},
           "familiar_thr_rope_p90": thr, "granular_above_thr": float(np.mean(s_gran > thr)),
           "granular_above_rope_max": float(np.mean(s_gran > s_rope.max())), "auc_granular_vs_rope": auc}
    (Path(__file__).resolve().parent / "familiarity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
