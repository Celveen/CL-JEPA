"""Wait for the final training epoch, evaluate all paired conditions, then build the report."""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import psutil
import report_pushobj_experiment as rep
from report_pushobj_experiment import ROOT, SHAPES, build_report, rows

lock = threading.Lock()
state = {"stage": "waiting_for_training", "jobs": {}}


def update():
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path = rep.REPORT / "pipeline_status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(path)


def training_alive():
    for p in psutil.process_iter(["cmdline"]):
        args = p.info["cmdline"] or []
        if len(args) >= 3 and args[-3:-1] == ["train.py", "--config-name"] and args[-1].startswith("train_pushobj"):
            return True
    return False


def worker(gpu, jobs):
    while True:
        try:
            shape, arm = jobs.get_nowait()
        except queue.Empty:
            return
        key = f"{shape}/{arm}"
        output = rep.EVAL / shape / arm
        output.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "plan.py"), "--config-name", "adajepa_plan_gd_pushobj",
               f"ckpt_base_path={rep.MODEL}", f"model_epoch={rep.EPOCHS}",
               f"eval_data_path={ROOT / 'data/pushobj_eval' / ('val_' + shape) / 'plan_targets.pkl'}",
               "n_evals=50", "seed=100", "+wandb_logging=false", "decode_for_viz=false",
               f"hydra.run.dir={output}"]
        if arm == "frozen":
            cmd += ["planner._target_=planning.mpc.MPCPlanner", "~planner.adapt"]
        with lock:
            state["jobs"][key] = {"status": "running", "gpu": gpu, "command": cmd}
            update()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "offline",
               "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2",
               "SDL_VIDEODRIVER": "dummy"}
        print(f"START {key} GPU={gpu}", flush=True)
        start = time.monotonic()
        try:
            with (output / "console.log").open("w") as log:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
            elapsed = time.monotonic() - start
            if result.returncode:
                raise RuntimeError(f"exit {result.returncode}: {output / 'console.log'}")
            if not any("final_eval/success_rate" in row for row in rows(output / "logs.json")):
                raise RuntimeError("Evaluation exited without a final success rate")
            (output / "run_metadata.json").write_text(json.dumps({"wall_seconds": elapsed, "gpu": gpu}, indent=2))
            with lock:
                state["jobs"][key].update(status="complete", wall_seconds=elapsed)
                update()
                build_report()
            print(f"DONE {key} seconds={elapsed:.1f}", flush=True)
        except Exception as exc:
            with lock:
                state["jobs"][key].update(status="failed", error=str(exc))
                update()
            print(f"FAIL {key}: {exc}", flush=True)
        finally:
            jobs.task_done()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="4,5,7")
    p.add_argument("--arms", default="frozen,adajepa", help="planners to run, e.g. 'frozen' for a quick check")
    rep.add_args(p)
    args = p.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    assert set(arms) <= {"frozen", "adajepa"}, arms
    rep.configure_from_args(args)
    rep.REPORT.mkdir(parents=True, exist_ok=True)
    state["pid"] = os.getpid()
    update()
    last_epoch = -1
    while True:
        metrics = rows(rep.MODEL / "metrics.jsonl")
        epoch = max((r["epoch"] for r in metrics), default=0)
        if epoch != last_epoch:
            build_report()
            print(f"TRAIN {epoch}/{rep.EPOCHS} epochs complete", flush=True)
            last_epoch = epoch
        alive = training_alive()
        if epoch == rep.EPOCHS and (rep.MODEL / f"checkpoints/model_{rep.EPOCHS}.pth").exists() and not alive:
            break
        if not alive:
            state.update(stage="training_failed", error=f"Training exited after epoch {epoch}; inspect train.log")
            update()
            raise RuntimeError(state["error"])
        time.sleep(30)
    state["stage"] = "evaluating"
    update()
    jobs = queue.Queue()
    # Run frozen arms first so inexpensive baseline results are available early.
    for arm in [a for a in ("frozen", "adajepa") if a in arms]:
        for shape in SHAPES:
            output = rep.EVAL / shape / arm
            if (output / "final_eval_results.npz").exists() and any("final_eval/success_rate" in r for r in rows(output / "logs.json")):
                state["jobs"][f"{shape}/{arm}"] = {"status": "complete", "reused": True}
            else:
                jobs.put((shape, arm))
    gpus = args.gpus.split(",")
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs) for gpu in gpus]
        for future in futures:
            future.result()
    result = build_report()
    ok = all(j.get("status") == "complete" for j in state["jobs"].values())
    state["stage"] = "complete" if ok else "evaluation_failed"
    update()
    print(json.dumps(result, indent=2), flush=True)
    if not ok:
        raise RuntimeError("Some evaluations failed; see pipeline_status.json")


if __name__ == "__main__":
    main()
