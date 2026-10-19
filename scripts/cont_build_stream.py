"""Build an ordered deployment stream for the continual (fast/slow) experiments.

A stream is a list of blocks (shape, n episodes); each block becomes one chunk
file of plan_targets.pkl format that plan.py runs in file order
(segments_order=file). Targets come from data/pushobj_eval/val_<shape>/ and
exclude every evaluation draw of the given eval seeds (random.Random(seed).sample,
as plan.py draws them), the source trajectories (ep_idx) of those draws, and the
experiment-0 bank, so probes on the eval targets stay held out. Within a shape,
the stream never reuses a segment or a source trajectory.

    python scripts/cont_build_stream.py --name S2 --blocks I:20 square:20 T:20 I:20 square:20
    python scripts/cont_build_stream.py --name S7b --seed 1 --exclude-streams S2 S7 --blocks I:20 L:20 ...
"""
import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--blocks", nargs="+", required=True, help="shape:n_episodes, in stream order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--n-evals", type=int, default=50)
    ap.add_argument("--data", default="data/pushobj_eval")
    ap.add_argument("--bank", default="data/exp0_bank")
    ap.add_argument("--out", default="data/cont_streams")
    ap.add_argument("--exclude-streams", nargs="*", default=[], help="also exclude the source trajectories of these streams")
    a = ap.parse_args()
    excluded = {}  # shape -> source trajectories (ep_idx) used by earlier streams
    for name in a.exclude_streams:
        for b in json.loads((ROOT / a.out / name / "manifest.json").read_text())["blocks"]:
            excluded.setdefault(b["shape"], set()).update(b["ep_idx"])

    blocks = [(b.split(":")[0], int(b.split(":")[1])) for b in a.blocks]
    rng = random.Random(a.seed)
    pools, segs_by_shape = {}, {}
    for shape in dict.fromkeys(s for s, _ in blocks):
        segs = pickle.load(open(ROOT / a.data / f"val_{shape}" / "plan_targets.pkl", "rb"))["segments"]
        used = set()
        for s in a.eval_seeds:
            used.update(random.Random(s).sample(range(len(segs)), min(a.n_evals, len(segs))))
        used_sources = {segs[i]["ep_idx"] for i in used}
        bank = ROOT / a.bank / f"{shape}.npz"
        bank_idx = set(np.load(bank)["indices"].tolist()) if bank.exists() else set()
        free = [i for i in range(len(segs))
                if i not in used and i not in bank_idx and segs[i]["ep_idx"] not in used_sources
                and segs[i]["ep_idx"] not in excluded.get(shape, set())]
        rng.shuffle(free)
        # one segment per source trajectory, so the stream never repeats a trajectory
        seen, pool = set(), []
        for i in free:
            if segs[i]["ep_idx"] not in seen:
                seen.add(segs[i]["ep_idx"])
                pool.append(i)
        pools[shape], segs_by_shape[shape] = pool, segs

    out = ROOT / a.out / a.name
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"name": a.name, "blocks": [], "seed": a.seed, "eval_seeds": a.eval_seeds}
    for k, (shape, n) in enumerate(blocks):
        if len(pools[shape]) < n:
            sys.exit(f"{shape}: only {len(pools[shape])} unused source trajectories left")
        idx, pools[shape] = pools[shape][:n], pools[shape][n:]
        segs = segs_by_shape[shape]
        tag = shape.replace("+", "plus")  # safe in file names and Hydra overrides
        chunk = f"chunk_{k:02d}_{tag}"
        with open(out / f"{chunk}.pkl", "wb") as f:
            pickle.dump({"segments": [segs[i] for i in idx], "traj_len": 25, "shape": shape,
                         "n_samples": n, "seed": a.seed, "indices": idx}, f)
        manifest["blocks"].append({"chunk": chunk, "shape": shape, "tag": tag, "n": n, "indices": idx,
                                   "ep_idx": [segs[i]["ep_idx"] for i in idx]})
        print(f"{chunk}: {n} episodes", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
