"""Run one chunk of a stream for one arm exactly as scripts/cont_run_stream.py would, plus extra overrides.
Used to check the CUDA-graph GD path on single chunks (README_cudagraph.md).

    python scripts/cont_run_chunk.py cls7_free S7t3 chunk_00_I 5 eval_outputs/cg_check/E2E_graph 7 null \
        planner.sub_planner.cuda_graph=true
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cont_run_stream import NAMED_ARMS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("stream")
ap.add_argument("chunk")
ap.add_argument("n", type=int, help="episodes (n_evals)")
ap.add_argument("out")
ap.add_argument("gpu")
ap.add_argument("state_in", nargs="?", default="null", help="continual_state.pt to start from, or null")
ap.add_argument("extra", nargs="*", help="extra Hydra overrides")
ap.add_argument("--ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
ap.add_argument("--epoch", default="3")
a = ap.parse_args()

blocks = json.loads((ROOT / "data/cont_streams" / a.stream / "manifest.json").read_text())["blocks"]
block = next(b for b in blocks if b["chunk"] == a.chunk)
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, str(ROOT / "plan.py"), "--config-name", "cont_plan_gd_pushobj",
       f"ckpt_base_path={(ROOT / a.ckpt).resolve()}", f"model_epoch={a.epoch}",
       f"eval_data_path={ROOT / 'data/cont_streams' / a.stream / (a.chunk + '.pkl')}", f"n_evals={a.n}",
       "+wandb_logging=false", f"hydra.run.dir={out}",
       *(NAMED_ARMS[a.arm] if a.arm in NAMED_ARMS else [f"planner.continual.w={float(a.arm[1:]):g}"]),  # w0, w0.01, ...
       f"planner.continual.tag={block.get('tag', block['shape'])}", f"planner.continual.state_in={a.state_in}",
       "planner.continual.save_deltas=false", *a.extra]
if "seed" in block:
    cmd.append(f"seed={block['seed']}")
env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpu, "WANDB_MODE": "offline", "OMP_NUM_THREADS": "2",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2", "SDL_VIDEODRIVER": "dummy"}
with open(out / "console.log", "w") as fh:
    rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
print(f"{a.arm} {a.chunk} n={a.n} rc={rc} -> {out}")
sys.exit(rc)
