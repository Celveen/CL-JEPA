"""Check that dataset frames are the env's default rendering of the stored states.

Training frames are decoded from lossy mp4 (imageio/libx264, yuv420p) while planning
observes raw env renders. For a sample of episodes this re-renders each stored state
and compares it with the decoded frame (PSNR, mean |diff|, error on the blue agent).

    python scripts/check_visual_fidelity.py --data data/pushobj_T16k_bias \
        --out reports/pushobj_T16k_bias/visual_fidelity.json
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--episodes", type=int, default=16, help="random episodes per split")
    ap.add_argument("--repeats", type=int, default=8, help="extra train episodes from second draws (--max-draws)")
    ap.add_argument("--frames", type=int, default=5, help="frames per episode, evenly spaced")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import gym
    import env  # noqa: F401
    from decord import VideoReader
    e = gym.make("pushobj", with_velocity=True, with_target=True).unwrapped
    rs = np.random.RandomState(0)
    result = {}
    for split in ("train", "val"):
        root = ROOT / a.data / split
        prov = json.loads((root / "provenance.json").read_text())
        lens = pickle.load(open(root / "seq_lengths.pkl", "rb"))
        shapes = pickle.load(open(root / "shapes.pkl", "rb"))
        S = torch.load(root / "states.pth", weights_only=True).numpy()
        eps = list(rs.choice(len(prov), a.episodes, replace=False))
        rep = [p["episode"] for p in prov if p.get("draw", 0) > 0]
        if split == "train" and rep:
            eps += list(rs.choice(rep, min(a.repeats, len(rep)), replace=False))
        psnr, mad, agent = [], [], []
        for i in eps:
            e.update_env({"shape": shapes[i]})
            vr = VideoReader(str(root / "obses" / f"episode_{i:03d}.mp4"), num_threads=1)
            assert len(vr) == lens[i], (split, i, len(vr), lens[i])
            for t in np.linspace(0, lens[i] - 1, a.frames).astype(int):
                dec = vr[int(t)].asnumpy().astype(np.float64)
                obs, _ = e.prepare(0, np.r_[S[i, t, :5], 0.0, 0.0])
                ref = obs["visual"].astype(np.float64)
                mse = ((dec - ref) ** 2).mean()
                psnr.append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
                mad.append(np.abs(dec - ref).mean())
                m = (ref[..., 2] > 150) & (ref[..., 0] < 120)  # blue agent pixels
                if m.sum():
                    agent.append(np.abs(dec[m] - ref[m]).mean())
        result[split] = {"episodes": len(eps), "frames": len(psnr), "repeat_draw_episodes": len(eps) - a.episodes,
                         "psnr_median_db": float(np.median(psnr)), "psnr_min_db": float(np.min(psnr)),
                         "mean_abs_diff": float(np.mean(mad)), "agent_mean_abs_diff": float(np.mean(agent)),
                         "frame_size": list(ref.shape)}
        print(split, json.dumps(result[split]), flush=True)
    e.close()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
