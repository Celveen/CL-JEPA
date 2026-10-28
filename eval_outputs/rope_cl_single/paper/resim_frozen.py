"""Real-simulation frames of the frozen theta0 arm, for the qualitative figures. The batched frozen planner saves no
frames, so one chunk's run is replayed in FleX exactly as plan.py executed it: its executed actions
(final_eval_results.npz, normalized) are denormalized with the training set's action statistics
(datasets/deformable_env_dset.py, normalize_action=true) and simulated with each sample's eval seed
(plan.py: seed * n + 1), in the run's order -- a fresh env, then for every MPC iteration (5 pushes,
planner.resim_from_start=false) one rollout per sample, each restarting the simulator from the particle positions that
sample's previous rollout ended at. The FleX env carries state from one call to the next, so replaying a single
sample on its own, or the whole episode in one rollout, does not reproduce the run; this order does, and every
sample's replayed final CD is printed next to the CD the run logged.

    source /data/zjc/adaJEPA/PyFleX/env.sh
    CUDA_VISIBLE_DEVICES=3 /data/zjc/envs/cljepa/bin/python eval_outputs/rope_cl_single/paper/resim_frozen.py \
        <dino_ours|granular_ref> <chunk>

Output: paper/frames/<model>_frozen_ep<e>.npz for the chunk's 10 episodes (visual (16, 224, 224, 3) uint8, cd,
logged_cd).
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CL = HERE.parent
REPO = CL.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CL))
from analyze import chamfer_distance, final_cd  # noqa: E402
from env.deformable_env.FlexEnvWrapper import FlexEnvWrapper  # noqa: E402

DATA = Path("/data/zjc/adaJEPA/CL-JEPA/data/deformable")
OBJ = {"dino_ours": "rope", "granular_ref": "granular"}
SEEDS = {"dino_ours": (99, 100, 101, 102, 103), "granular_ref": (301, 302, 303, 304, 305)}
N_TAKEN = 5  # pushes per MPC iteration (conf/cont_plan_cem_rope.yaml)


def action_stats(obj):
    a = torch.load(DATA / obj / "actions.pth").float()
    a = a.reshape(-1, a.shape[-1])
    return a.mean(0).numpy(), (a.std(0) + 1e-6).numpy()


def main():
    model, k = sys.argv[1], int(sys.argv[2])
    env = FlexEnvWrapper(OBJ[model])
    mean, std = action_stats(OBJ[model])
    r = np.load(CL / model / "frozen" / f"chunk_{k}" / "final_eval_results.npz")
    acts = r["actions"] * std + mean  # (10, 15, 4); action_len is inf: every planned action was executed
    n = len(acts)
    seeds = [SEEDS[model][k] * i + 1 for i in range(n)]
    states = [r["state_0"][i] for i in range(n)]
    vis = [[] for _ in range(n)]
    for j in range(0, acts.shape[1], N_TAKEN):  # MPC iteration-major, as the batched planner ran it
        for i in range(n):
            obses, st = env.rollout(seeds[i], states[i], acts[i, j : j + N_TAKEN])
            vis[i].append(obses["visual"] if j == 0 else obses["visual"][1:])
            states[i] = st[-1]
        print(f"MPC iteration {j // N_TAKEN} replayed", flush=True)
    logged = final_cd(model, "frozen", k)
    out = HERE / "frames"
    out.mkdir(exist_ok=True)
    for i in range(n):
        g = torch.tensor(np.asarray(r["state_g"][i], dtype=np.float64))[None]
        cd = float(chamfer_distance(torch.tensor(np.asarray(states[i], dtype=np.float64).reshape(1, -1, 4)), g))
        e = 10 * k + i
        np.savez_compressed(out / f"{model}_frozen_ep{e}.npz", visual=np.asarray(np.concatenate(vis[i]), np.uint8),
                            cd=cd, logged_cd=logged[i])
        print(f"frozen ep{e}: replay CD {cd:.4f} vs logged {logged[i]:.4f}", flush=True)


if __name__ == "__main__":
    main()
