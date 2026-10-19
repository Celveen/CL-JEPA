"""Split the standard evaluation draw of each shape into chunks for scripts/escape_run.py.

plan.py draws its targets as random.Random(seed).sample(segments, n_evals) from
data/pushobj_eval/val_<shape>/plan_targets.pkl (the AdaJEPA protocol: seed 100, 50 targets per
shape). This script takes the same draw and writes it as ordered chunks (segments_order=file), so
the chunks run in parallel and every arm sees the same targets in the same order. Chunking changes
only the env seeds plan.py hands out (seed * i + 1 within a chunk), which do not affect a target
given by its initial state and actions.

    python scripts/escape_build_eval.py --name EVAL50 --shapes I L square Z small_tee + T --seed 100 --n 50 --chunk 10
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
    ap.add_argument("--shapes", nargs="+", default=["I", "L", "square", "Z", "small_tee", "+", "T"])
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--data", default="data/pushobj_eval")
    ap.add_argument("--out", default="data/cont_streams")
    a = ap.parse_args()
    out = ROOT / a.out / a.name
    out.mkdir(parents=True, exist_ok=True)
    manifest, k = {"name": a.name, "seed": a.seed, "n": a.n, "blocks": []}, 0
    for shape in a.shapes:
        data = pickle.load(open(ROOT / a.data / f"val_{shape}" / "plan_targets.pkl", "rb"))
        chosen = random.Random(a.seed).sample(data["segments"], min(a.n, len(data["segments"])))
        tag = shape.replace("+", "plus")
        for j in range(0, len(chosen), a.chunk):
            chunk = f"chunk_{k:02d}_{tag}"
            segs = chosen[j : j + a.chunk]
            with open(out / f"{chunk}.pkl", "wb") as f:
                pickle.dump({"segments": segs, "traj_len": data["traj_len"], "shape": shape, "n_samples": len(segs),
                             "seed": a.seed, "draw_index": list(range(j, j + len(segs)))}, f)
            manifest["blocks"].append({"chunk": chunk, "shape": shape, "tag": tag, "n": len(segs),
                                       "draw_index": list(range(j, j + len(segs)))})
            k += 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{k} chunks -> {out}")


if __name__ == "__main__":
    main()
