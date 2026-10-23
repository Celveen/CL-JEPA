"""Build a continual stream whose rounds are the standard evaluation draws of different test seeds.

Round r uses test seed seeds[r]: for every shape (in the given order) its chunk holds exactly the
n targets plan.py draws for that seed (random.Random(seed).sample(segments, n) on
data/pushobj_eval/val_<shape>/plan_targets.pkl), in that order, and the chunk is planned with
seed=seeds[r] (episode env seeds seed*n+1, as in the standard evaluation). So round 1 of shape s
is the usual "seed 100, val_s, 50 targets" evaluation, round 2 the seed-200 one, and so on.

    python scripts/cont_build_seed_stream.py --name S7t3 --seeds 100 200 300 --n 50

--exclude draws only from targets none of the listed streams used (a held-out evaluation set); without it the
draw is exactly plan.py's.

    python scripts/cont_build_seed_stream.py --name H7t3 --seeds 2000 2100 2200 --exclude S7t5 S7t5s400 S7t5s700
"""
import argparse
import json
import pickle
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--shapes", nargs="+", default=["I", "L", "square", "Z", "small_tee", "+", "T"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--data", default="data/pushobj_eval")
    ap.add_argument("--out", default="data/cont_streams")
    ap.add_argument("--exclude", nargs="*", default=[], help="streams whose targets this draw leaves out")
    a = ap.parse_args()
    used = {}  # shape -> segment indices used by the excluded streams
    for name in a.exclude:
        for b in json.loads((ROOT / a.out / name / "manifest.json").read_text())["blocks"]:
            used.setdefault(b["shape"], set()).update(b["indices"])
    out = ROOT / a.out / a.name
    out.mkdir(parents=True, exist_ok=True)
    blocks, k = [], 0
    for seed in a.seeds:
        for shape in a.shapes:
            segs = pickle.load(open(ROOT / a.data / f"val_{shape}" / "plan_targets.pkl", "rb"))["segments"]
            pool = [i for i in range(len(segs)) if i not in used.get(shape, ())]
            idx = [pool[j] for j in random.Random(seed).sample(range(len(pool)), min(a.n, len(pool)))]  # plan.py's draw if no --exclude
            tag = shape.replace("+", "plus")
            chunk = f"chunk_{k:02d}_{tag}"
            with open(out / f"{chunk}.pkl", "wb") as f:
                pickle.dump({"segments": [segs[i] for i in idx], "traj_len": 25, "shape": shape,
                             "n_samples": len(idx), "seed": seed, "indices": idx}, f)
            blocks.append({"chunk": chunk, "shape": shape, "tag": tag, "n": len(idx), "seed": seed,
                           "indices": idx, "ep_idx": [segs[i]["ep_idx"] for i in idx]})
            print(f"{chunk}: {len(idx)} targets of test seed {seed}", flush=True)
            k += 1
    (out / "manifest.json").write_text(json.dumps({"name": a.name, "blocks": blocks, "seeds": a.seeds}, indent=1))


if __name__ == "__main__":
    main()
