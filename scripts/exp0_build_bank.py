"""Pre-render other-episode windows for experiment 0's other_data arm.

ProbeMPCPlanner(probe.mode=other_data) adapts on a window of transitions from
episodes it is not solving. This script builds that pool per shape: it takes
plan_targets.pkl segments that no evaluation draw uses (plan.py picks
random.Random(seed).sample(segments, n_evals) for each eval seed), also drops
segments cut from the same source trajectory (ep_idx) as any evaluation target,
replays the rest in the env and stores the frames at the model stride.

    python scripts/exp0_build_bank.py --shapes T I square --n 100
"""
import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100, help="segments per shape")
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--n-evals", type=int, default=50)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--data", default="data/pushobj_eval")
    ap.add_argument("--out", default="data/exp0_bank")
    a = ap.parse_args()

    import gym
    import env  # noqa: F401  (registers pushobj)

    sim = gym.make("pushobj", with_velocity=True, with_target=True).unwrapped
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for shape in a.shapes:
        segs = pickle.load(open(ROOT / a.data / f"val_{shape}" / "plan_targets.pkl", "rb"))["segments"]
        used = set()
        for s in a.eval_seeds:
            idx = random.Random(s).sample(range(len(segs)), min(a.n_evals, len(segs)))
            drawn = random.Random(s).sample(segs, min(a.n_evals, len(segs)))
            assert all(x is segs[i] for x, i in zip(drawn, idx)), "index draw no longer matches plan.py"
            used.update(idx)
        used_sources = {segs[i]["ep_idx"] for i in used}
        free = [i for i in range(len(segs)) if i not in used and segs[i]["ep_idx"] not in used_sources]
        random.Random(0).shuffle(free)
        pick = sorted(free[: a.n])
        if len(pick) < a.n:
            raise SystemExit(f"{shape}: only {len(pick)} unused segments")

        visual, proprio, actions, states = [], [], [], []
        for i in pick:
            seg = segs[i]
            init = np.zeros(7, dtype=np.float64)
            init[: len(seg["states"][0])] = seg["states"][0]  # zero velocities, as plan.py pads
            env_actions = np.asarray(seg["actions"], dtype=np.float32) / 100.0  # pixels -> env units
            sim.update_env({"shape": seg["shape"]})
            obs, st = sim.rollout(0, init, env_actions)
            assert obs["visual"].shape[0] == env_actions.shape[0] + 1
            visual.append(obs["visual"][:: a.frameskip])
            proprio.append(obs["proprio"][:: a.frameskip])
            actions.append(env_actions)
            states.append(st)
        visual = np.stack(visual).astype(np.uint8)
        np.savez_compressed(
            out / f"{shape}.npz",
            visual=visual,
            proprio=np.stack(proprio).astype(np.float32),
            env_actions=np.stack(actions),
            states=np.stack(states).astype(np.float32),
            indices=np.asarray(pick),
            ep_idx=np.asarray([segs[i]["ep_idx"] for i in pick]),
        )
        print(f"{shape}: {len(pick)} segments, frames {visual.shape}, excluded {len(used)} eval draws "
              f"and {len(used_sources)} source trajectories", flush=True)


if __name__ == "__main__":
    main()
