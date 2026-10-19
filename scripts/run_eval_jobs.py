"""Run a list of PushObj plan.py evaluations over a pool of GPU slots.

Each line of the jobs file (tab-separated, '#' comments allowed):

    out_dir  arm  ckpt_base_path  model_epoch  shape  seed  [extra overrides]

arm is 'frozen' or 'adajepa'; model_epoch '-' keeps the plan config default
(model_latest.pth). The optional 7th column holds space-separated Hydra
overrides, e.g. 'planner.max_iter=30'. Jobs run in file order, so put the long AdaJEPA jobs first.
A GPU id may repeat in --gpus to run several jobs per card (AdaJEPA plans one
target at a time and leaves most of the card idle). Finished outputs are skipped.

    python scripts/run_eval_jobs.py --jobs jobs.tsv --gpus 4,4,5,5,6,6,7,7
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lock = threading.Lock()


def finished(out):
    logs = out / "logs.json"
    return (out / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()


def command(arm, ckpt, epoch, shape, seed, out, extra=""):
    cmd = [sys.executable, str(ROOT / "plan.py"), "--config-name", "adajepa_plan_gd_pushobj",
           f"ckpt_base_path={(ROOT / ckpt).resolve()}",
           f"eval_data_path={ROOT / 'data/pushobj_eval' / ('val_' + shape) / 'plan_targets.pkl'}",
           "n_evals=50", f"seed={seed}", "+wandb_logging=false", "decode_for_viz=false", f"hydra.run.dir={out}"]
    if epoch != "-":
        cmd.insert(5, f"model_epoch={epoch}")
    if arm == "frozen":
        cmd += ["planner._target_=planning.mpc.MPCPlanner", "~planner.adapt"]
    return cmd + extra.split()


def worker(gpu, jobs, results):
    while True:
        try:
            out, arm, ckpt, epoch, shape, seed, extra = jobs.get_nowait()
        except queue.Empty:
            return
        out = (ROOT / out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "WANDB_MODE": "offline", "OMP_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2", "SDL_VIDEODRIVER": "dummy"}
        with lock:
            print(f"START {out.relative_to(ROOT)} GPU={gpu}", flush=True)
        start = time.monotonic()
        with (out / "console.log").open("w") as log:
            rc = subprocess.run(command(arm, ckpt, epoch, shape, seed, out, extra), cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        elapsed = time.monotonic() - start
        ok = rc == 0 and finished(out)
        if ok:
            (out / "run_metadata.json").write_text(json.dumps({"wall_seconds": elapsed, "gpu": gpu}, indent=2))
        with lock:
            results.append(ok)
            print(f"{'DONE' if ok else 'FAIL'} {out.relative_to(ROOT)} seconds={elapsed:.0f}" + ("" if ok else f" exit={rc}"), flush=True)
        jobs.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", default="4,5,6,7", help="one worker per entry; repeat an id for several jobs per card")
    a = ap.parse_args()
    jobs, skipped = queue.Queue(), 0
    for line in Path(a.jobs).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        job = line.split("\t")
        assert len(job) in (6, 7) and job[1] in ("frozen", "adajepa"), line
        job = job + [""] * (7 - len(job))
        if finished((ROOT / job[0]).resolve()):
            skipped += 1
        else:
            jobs.put(tuple(job))
    print(f"{jobs.qsize()} jobs queued, {skipped} already finished", flush=True)
    results = []
    threads = [threading.Thread(target=worker, args=(g, jobs, results)) for g in a.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"finished: {sum(results)} ok, {len(results) - sum(results)} failed", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
