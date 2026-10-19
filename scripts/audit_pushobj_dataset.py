"""Validate the generated dataset and write reproducible dataset statistics."""
import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader


def check_video(item):
    path, length = item
    reader = VideoReader(str(path), num_threads=1)
    assert len(reader) == length, (path, len(reader), length)
    frames = reader.get_batch([0, length // 2, length - 1]).asnumpy()
    assert frames.shape[0] == 3 and frames.shape[-1] == 3
    return tuple(frames.shape[1:])


_ENV = None


def _init_env():
    global _ENV
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import gym
    import env  # noqa: F401  # registers pushobj
    _ENV = gym.make("pushobj", with_velocity=True, with_target=True).unwrapped


def start_state_check(item):
    """Recompute, from the stored first state, the generator's start-state gate."""
    shape, state0 = item
    _ENV.update_env({"shape": shape})
    _ENV.prepare(0, np.concatenate([state0, [0.0, 0.0]]))
    gap = min(s.point_query(tuple(_ENV.agent.position)).distance for s in _ENV.block.shapes)
    verts = np.array([tuple(_ENV.block.local_to_world(v)) for s in _ENV.block.shapes for v in s.get_vertices()])
    return float(gap), float(verts.min()), float(verts.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    config = json.loads((args.data / "generation_config.json").read_text())
    report, fingerprints, sources = {}, {}, {}
    for split in ("train", "val"):
        root = args.data / split
        states = torch.load(root / "states.pth", weights_only=True)
        actions = torch.load(root / "rel_actions.pth", weights_only=True) / 100
        velocities = torch.load(root / "velocities.pth", weights_only=True)
        lengths = [int(t) for t in pickle.loads((root / "seq_lengths.pkl").read_bytes())]
        shapes = pickle.loads((root / "shapes.pkl").read_bytes())
        provenance = json.loads((root / "provenance.json").read_text())
        n, t_max = len(lengths), max(lengths)
        # Episodes have variable length (pusht_noise sources); tensors are zero-padded.
        assert states.shape == (n, t_max, 5)
        assert actions.shape == velocities.shape == (n, t_max, 2)
        assert len(shapes) == len(provenance) == n
        assert all(torch.isfinite(x).all() for x in (states, actions, velocities))
        assert len(list((root / "obses").glob("*.mp4"))) == n
        with ThreadPoolExecutor(max_workers=8) as pool:
            sizes = list(pool.map(check_video, [(root / "obses" / f"episode_{i:03d}.mp4", t)
                                               for i, t in enumerate(lengths)]))
        valid = torch.arange(t_max)[None] < torch.tensor(lengths)[:, None]
        agent = states[..., :2][valid]
        # The kinematic agent passes through walls; the generator must keep it on screen.
        agent_out = int(((agent < 0) | (agent > 512)).any(-1).sum())
        assert agent_out == 0, f"{split}: {agent_out} frames with the agent outside the frame"
        fingerprints[split] = {hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in states}
        # repeated sources (--max-draws) must still give distinct trajectories
        assert len(fingerprints[split]) == n, f"{split}: {n - len(fingerprints[split])} duplicate trajectories"
        sources[split] = {p["source_index"] for p in provenance}
        last = torch.tensor(lengths) - 1
        movement = torch.linalg.vector_norm(states[torch.arange(n), last, 2:4] - states[:, 0, 2:4], dim=-1)
        proprio = torch.cat([states[..., :2], velocities], -1)[valid]
        acts = actions[valid]
        # Start-state gate, recomputed independently of the generator's bookkeeping.
        with Pool(32, initializer=_init_env) as pool:
            checks = np.array(pool.map(start_state_check, [(sh, states[i, 0].double().numpy())
                                                            for i, sh in enumerate(shapes)], chunksize=64))
        max_pen = config.get("max_penetration", 3.0)
        bad_start = int(((checks[:, 0] < 15.0 - max_pen - 0.5) | (checks[:, 1] < 7.0 - 0.5)
                         | (checks[:, 2] > 504.0 + 0.5)).sum())  # 0.5 px slack: prepare() steps physics once
        assert bad_start == 0, f"{split}: {bad_start} episodes start with the agent in the block or the block in a wall"
        steps = torch.linalg.vector_norm(states[:, 1:, 2:4] - states[:, :-1, 2:4], dim=-1)[valid[:, 1:]]
        max_step = float(steps.max())
        assert max_step <= config.get("max_block_step", 80.0) + 1e-3, f"{split}: block step {max_step:.1f}px"
        contact = np.array([p["contact_steps"] for p in provenance])
        report[split] = {
            "episodes": n, "frames": sum(lengths),
            "length_quantiles": np.quantile(lengths, [0, .25, .5, .75, 1]).tolist(),
            "shapes": dict(Counter(shapes)), "unique_trajectories": len(fingerprints[split]),
            "unique_sources": len(sources[split]),
            "video_sizes": [list(s) for s in sorted(set(sizes))], "videos_checked": n,
            "slices": sum(max(0, t - 20 + 1) for t in lengths),
            "agent_out_of_frame_frames": agent_out,
            "invalid_start_states": bad_start,
            "min_start_gap_agent_to_block_px": float(checks[:, 0].min()),
            "block_step_px_q99_max": [float(torch.quantile(steps[torch.randperm(len(steps))[:1_000_000]], .99)), max_step],
            "action_mean": acts.mean(0).tolist(), "action_std": acts.std(0).tolist(),
            "proprio_mean": proprio.mean(0).tolist(), "proprio_std": proprio.std(0).tolist(),
            "episodes_with_contact": float((contact > 0).mean()),
            "contact_steps_per_episode_by_shape": {
                sh: float(np.mean([p["contact_steps"] for p in provenance if p["shape"] == sh]))
                for sh in sorted(set(shapes))},
            "block_moved_gt5px_fraction": float((movement > 5).float().mean()),
            "block_displacement_quantiles": np.quantile(movement.numpy(), [0, .25, .5, .75, 1]).tolist(),
        }
    report["duplicate_trajectories_across_splits"] = len(fingerprints["train"] & fingerprints["val"])
    report["shared_source_trajectories_across_splits"] = len(sources["train"] & sources["val"])
    assert report["duplicate_trajectories_across_splits"] == 0
    assert report["shared_source_trajectories_across_splits"] == 0
    n_shapes = len(config["shapes"].split(","))
    assert report["train"]["episodes"] == config["train_per_shape"] * n_shapes
    assert report["val"]["episodes"] == config["val_per_shape"] * n_shapes
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
