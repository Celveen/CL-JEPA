"""Evaluate the released frozen model as an additional data-distribution reference."""
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval_outputs/pushobj_released_reference"
SHAPES = ("T", "L", "Z", "+", "I", "small_tee", "square")


def worker(gpu, jobs):
    while True:
        try:
            shape = jobs.get_nowait()
        except queue.Empty:
            return
        out = OUTPUT / shape
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "plan.py"), "--config-name", "adajepa_plan_gd_pushobj",
               f"ckpt_base_path={ROOT / 'checkpoints/pushobj_shape_shift'}",
               f"eval_data_path={ROOT / 'data/pushobj_eval' / ('val_' + shape) / 'plan_targets.pkl'}",
               "n_evals=50", "seed=100", "+wandb_logging=false", "decode_for_viz=false",
               "planner._target_=planning.mpc.MPCPlanner", "~planner.adapt", f"hydra.run.dir={out}"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2", "SDL_VIDEODRIVER": "dummy"}
        print(f"START reference {shape} GPU={gpu}", flush=True)
        start = time.monotonic()
        with (out / "console.log").open("w") as f:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        (out / "run_metadata.json").write_text(json.dumps({"command": cmd, "wall_seconds": time.monotonic()-start,
                                                           "returncode": result.returncode}, indent=2))
        print(f"DONE reference {shape} exit={result.returncode}", flush=True)
        jobs.task_done()
        if result.returncode:
            raise RuntimeError(f"Reference failed: {out / 'console.log'}")


if __name__ == "__main__":
    jobs = queue.Queue()
    for shape in SHAPES:
        jobs.put(shape)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu, jobs) for gpu in ("5", "7")]
        for future in futures:
            future.result()
