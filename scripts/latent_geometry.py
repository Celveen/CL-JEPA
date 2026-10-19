"""Latent geometry of trained world models on one shared set of validation windows (CPU).

For every model: the curvature its straightening loss penalizes (1 - cos between
consecutive latent displacements of the visual features), the per-dimension latent
variance, and the 1-step prediction MSE of the visual features, also divided by
that variance so models with different latent scales are comparable.

    python scripts/latent_geometry.py --data data/pushobj_T16k_bias \
        --models "released visual=checkpoints/pusht_visual_shift:model_latest.pth,ours 1e-2=checkpoints/pushobj_T16k_bias_e3_seed0:model_3.pth" \
        --out reports/pushobj_T16k_bias/latent_geometry.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def build_wm(model_dir, ckpt_name):
    import plan
    cfg = OmegaConf.load(Path(model_dir) / "hydra.yaml")
    wm = plan.load_model(Path(model_dir) / "checkpoints" / ckpt_name, cfg, cfg.num_action_repeat, torch.device("cpu"))
    wm.eval()  # VWorldModel.eval() does not return the module
    return wm, cfg


@torch.no_grad()
def measure(wm, batches):
    curv, var_feats, mse_vis, mse_all = [], [], [], []
    for obs, act in batches:
        z = wm.encode(obs, act)
        feats = wm.visual_only(z)                       # (b, t, p, d) visual features only
        curv.append(float(wm.total_curvature(feats, mode="cos")))
        var_feats.append(feats.reshape(-1, feats.shape[-1]))
        z_pred = wm.predict(z[:, : wm.num_hist])
        tgt = z[:, wm.num_pred:]
        vis_pred, vis_tgt = wm.visual_only(z_pred), wm.visual_only(tgt)
        mse_vis.append(float(((vis_pred - vis_tgt) ** 2).mean()))
    v = torch.cat(var_feats).var(0).mean().item()
    m = float(np.mean(mse_vis))
    return {"one_minus_cos": float(np.mean(curv)), "latent_var_per_dim": v, "pred_mse_visual": m,
            "pred_mse_visual_over_var": m / v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pushobj_T16k_bias")
    ap.add_argument("--models", required=True, help="comma list of name=model_dir:ckpt_name")
    ap.add_argument("--windows", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import load_pusht_slice_train_val
    slices, _ = load_pusht_slice_train_val(default_transform(224), n_rollout=None, data_path=str(ROOT / a.data),
                                           normalize_action=True, num_hist=3, num_pred=1, frameskip=5, with_velocity=True)
    val = slices["valid"]
    idx = np.random.RandomState(0).choice(len(val), a.windows, replace=False)
    items = [val[int(i)] for i in idx]
    batches = []
    for k in range(0, len(items), a.batch):
        chunk = items[k:k + a.batch]
        obs = {key: torch.stack([c[0][key] for c in chunk]) for key in chunk[0][0]}
        act = torch.stack([c[1] for c in chunk])
        batches.append((obs, act))
    print(f"{len(items)} validation windows from {a.data}/val", flush=True)
    result = {"data": a.data, "windows": len(items), "models": {}}
    for item in a.models.split(","):
        name, spec = item.split("=")
        d, ck = spec.split(":")
        wm, cfg = build_wm(ROOT / d, ck)
        r = measure(wm, batches)
        r["straighten"] = cfg.training.get("straighten")
        result["models"][name] = r
        print(name, json.dumps(r), flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
