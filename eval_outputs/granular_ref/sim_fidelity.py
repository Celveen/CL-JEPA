"""Does our PyFleX build reproduce the dynamics of DINO-WM's deformable data? Replays dataset actions from dataset start
states in our env and compares each replayed state with the recorded one (Chamfer distance, the env's own metric).

For trajectory i: prepare(init_state=states[i, 0]), then push actions[i, 0..K-1] one at a time; after push k compare
the replayed particles with states[i, k]. For scale, also the CD between consecutive recorded states (how far one push
moves the object) and between the recorded state and the start.

    source /data/zjc/adaJEPA/PyFleX/env.sh
    CUDA_VISIBLE_DEVICES=1 /data/zjc/envs/cljepa/bin/python eval_outputs/granular_ref/sim_fidelity.py rope 10
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from env.deformable_env.FlexEnvWrapper import FlexEnvWrapper, chamfer_distance  # noqa: E402

obj, n = sys.argv[1], int(sys.argv[2])
K = 5
data = Path("/data/zjc/adaJEPA/CL-JEPA/data/deformable") / obj
states = torch.load(data / "states.pth")  # (N, T, P, 4)
actions = torch.load(data / "actions.pth").float()  # (N, T, 4), raw (action_scale 1)
idx = list(range(len(states) - n, len(states)))  # from the validation tail (split_ratio 0.9)

env = FlexEnvWrapper(obj)
cd = lambda a, b: float(chamfer_distance(torch.tensor(np.asarray(a))[None].float(), torch.tensor(np.asarray(b))[None].float()))
rows = []
for i in idx:
    _, st = env.prepare(0, states[i, 0].reshape(-1).numpy())
    rep = [cd(st["state"], states[i, 0])]
    for k in range(K):
        _, _, _, info = env.step_multiple(actions[i, k : k + 1].numpy())
        rep.append(cd(info["state"][-1], states[i, k + 1]))
    step = [cd(states[i, k + 1], states[i, k]) for k in range(K)]
    total = cd(states[i, K], states[i, 0])
    rows.append({"traj": i, "replay_vs_recorded": rep, "recorded_step": step, "recorded_start_to_K": total})
    print(json.dumps(rows[-1]), flush=True)

r = np.array([x["replay_vs_recorded"] for x in rows])
s = np.array([x["recorded_step"] for x in rows])
t = np.array([x["recorded_start_to_K"] for x in rows])
print(f"{obj}: replay vs recorded CD after 0..{K} pushes (mean over {len(rows)}): {np.round(r.mean(0), 3).tolist()}")
print(f"{obj}: recorded CD moved per push {s.mean():.3f}; start -> {K} pushes {t.mean():.3f}")
