"""Run the experiment-0 grid (arm x shape x eval seed) over a pool of GPU slots.

Every arm plans the same 50 targets per (shape, seed) with the same checkpoint,
so outcomes are paired episode by episode. Arms (see planning/adajepa_probe.py):

    F_batch    frozen, all targets in one batch (the usual frozen baseline)
    F_seq      frozen, planned one target at a time like the adapting arms
    A          AdaJEPA
    A_asc      the AdaJEPA step negated (anti-informative, same size)
    R_tgt      AdaJEPA step towards a random latent shift (natural, size-matched, no information)
    R_tgt_fix  the same with one shift per episode (coherent drift)
    R_sign     AdaJEPA's step with random per-element signs (parameter-space noise)
    W_dom      AdaJEPA step on a window from other episodes of the same shape
    P_restart  frozen weights, GD restarted from random actions at every replan

Outputs go to eval_outputs/exp0_shape/s<seed>/<shape>/<arm>; finished runs are
skipped, so the script can be re-run to resume. A GPU id may repeat in --gpus to
run several jobs per card. Each run forks one env worker per target (~9 GB for 50),
so a job only starts once MemAvailable exceeds --min-free-gb.

    python scripts/exp0_run.py --seeds 100 --shapes T I square --gpus 0,0,1,1,2,2,3,3
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
CKPT = "checkpoints/pushobj_bias_16k_e3_seed0"
BANK = ROOT / "data/exp0_bank"
ARMS = {
    "F_seq": ["planner.probe.mode=none"],
    "A": ["planner.probe.mode=grad"],
    "A_asc": ["planner.probe.mode=ascent"],
    "R_tgt": ["planner.probe.mode=random_target"],
    "R_tgt_fix": ["planner.probe.mode=random_target_fixed"],
    "R_sign": ["planner.probe.mode=random_sign"],
    "W_dom": ["planner.probe.mode=other_data", f"planner.probe.bank_dir={BANK}",
              "planner.probe.bank_shapes=['{shape}']"],
    "P_restart": ["planner.probe.mode=none", "planner.sub_planner.sample_type=randn", "planner.reuse_actions=false"],
    "F_batch": ["planner._target_=planning.mpc.MPCPlanner", "~planner.adapt", "~planner.probe"],
}
lock = threading.Lock()  # printing / results
start_lock = threading.Lock()  # job starts


def rel(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def finished(out):
    logs = out / "logs.json"
    return (out / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()


def command(arm, shape, seed, out, a):
    cmd = [sys.executable, str(ROOT / "plan.py"), "--config-name", "exp0_plan_gd_pushobj",
           f"ckpt_base_path={(ROOT / a.ckpt).resolve()}", f"model_epoch={a.epoch}",
           f"eval_data_path={ROOT / 'data/pushobj_eval' / ('val_' + shape) / 'plan_targets.pkl'}",
           f"n_evals={a.n_evals}", f"seed={seed}", "+wandb_logging=false", f"hydra.run.dir={out}"]
    return cmd + [o.format(shape=shape) for o in ARMS[arm]] + a.extra


def mem_available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


def worker(gpu, jobs, results, a):
    while True:
        try:
            arm, shape, seed, out = jobs.get_nowait()
        except queue.Empty:
            return
        out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "WANDB_MODE": "offline", "OMP_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2", "SDL_VIDEODRIVER": "dummy"}
        with start_lock:  # one start per --stagger seconds, and only with enough memory left
            while mem_available_gb() < a.min_free_gb:
                time.sleep(30)
            with lock:
                print(f"START {rel(out)} GPU={gpu}", flush=True)
            log = (out / "console.log").open("w")
            proc = subprocess.Popen(command(arm, shape, seed, out, a), cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            time.sleep(a.stagger)
        start = time.monotonic()
        rc = proc.wait()
        log.close()
        elapsed = time.monotonic() - start + a.stagger
        ok = rc == 0 and finished(out)
        if ok:
            (out / "run_metadata.json").write_text(json.dumps(
                {"wall_seconds": elapsed, "gpu": gpu, "arm": arm, "shape": shape, "seed": seed,
                 "ckpt": a.ckpt, "epoch": a.epoch, "extra": a.extra}, indent=2))
        with lock:
            results.append(ok)
            print(f"{'DONE' if ok else 'FAIL'} {rel(out)} seconds={elapsed:.0f}"
                  + ("" if ok else f" exit={rc}"), flush=True)
        jobs.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--shapes", nargs="+", default=["T", "I", "square"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--epoch", default="3")
    ap.add_argument("--n-evals", type=int, default=50)
    ap.add_argument("--root", default="eval_outputs/exp0_shape")
    ap.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every run")
    ap.add_argument("--min-free-gb", type=float, default=40.0, help="start a job only above this MemAvailable")
    ap.add_argument("--stagger", type=float, default=20.0, help="seconds between job starts")
    a = ap.parse_args()
    unknown = set(a.arms) - set(ARMS)
    assert not unknown, f"unknown arms {unknown}"

    jobs, skipped = queue.Queue(), 0
    # Long sequential arms first; the batched frozen baseline takes minutes.
    for arm in sorted(a.arms, key=lambda x: x == "F_batch"):
        for seed in a.seeds:
            for shape in a.shapes:
                out = (ROOT / a.root / f"s{seed}" / shape / arm).resolve()
                if finished(out):
                    skipped += 1
                else:
                    jobs.put((arm, shape, seed, out))
    print(f"{jobs.qsize()} jobs queued, {skipped} already finished", flush=True)
    results = []
    threads = [threading.Thread(target=worker, args=(g, jobs, results, a)) for g in a.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"finished: {sum(results)} ok, {len(results) - sum(results)} failed", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
