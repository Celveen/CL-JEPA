"""Run stall-escape arms (planning/escape.py) on the chunks of a continual stream.

AdaJEPA resets every episode, so chunks are independent and run in parallel (unlike
the chained continual arms). Each run uses the chunk's targets in stored order, so
episodes pair with eval_outputs/cont_<stream>/w0 (AdaJEPA) one by one.

    eval_outputs/escape_<stream>/<arm>/<chunk>/

Finished runs are skipped, so the script can be re-run to resume.

    python scripts/escape_run.py --stream S7 --arms none K2 K2P K3 --gpus 0,1,2,3,4,5,6,7 --jobs 24
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
ARMS = {
    "none": ["planner.escape.mode=none"],
    "K1_5": ["planner.escape.mode=kick_random", "planner.escape.kick_px=5"],
    "K1_10": ["planner.escape.mode=kick_random", "planner.escape.kick_px=10"],
    "K2": ["planner.escape.mode=kick_goal", "planner.escape.kick_px=10"],
    "K2P": ["planner.escape.mode=kick_goal", "planner.escape.kick_px=10", "planner.escape.proprio_boost=10"],
    "K3": ["planner.escape.mode=restart", "planner.escape.restarts=8"],
    # object first (proprio term off), then K2P at later stalls
    "KS": ["planner.escape.mode=staged", "planner.escape.kick_px=10", "planner.escape.proprio_boost=10",
           "planner.escape.object_scale=0"],
    # K2P with up to 7 escapes, 3 replans apart (fits 7 into replans 10-29)
    "K2P7": ["planner.escape.mode=kick_goal", "planner.escape.kick_px=10", "planner.escape.proprio_boost=10",
             "planner.escape.max_n=7", "planner.escape.gap=3"],
    # raised limits (for 40-replan runs: --extra planner.max_iter=40): up to 7 escapes 4 replans apart,
    # offsets up to 15 px per env step; KS_7x15 keeps the object phase for its first two escapes
    "K2P_7x15": ["planner.escape.mode=kick_goal", "planner.escape.kick_px=15", "planner.escape.proprio_boost=10",
                 "planner.escape.max_n=7"],
    # K2P_7x15 whose proprio boost lasts only for the kick replan and the 2 replans after it
    "K2P_short2": ["planner.escape.mode=kick_goal", "planner.escape.kick_px=15", "planner.escape.proprio_boost=10",
                   "planner.escape.max_n=7", "planner.escape.boost_replans=2"],
    # the same with the boost on the kick replan and only 1 replan after it
    "K2P_short1": ["planner.escape.mode=kick_goal", "planner.escape.kick_px=15", "planner.escape.proprio_boost=10",
                   "planner.escape.max_n=7", "planner.escape.boost_replans=1"],
    # K2P_short2 + workspace clip + action magnitude cap of 60 px per env step
    "K2P_short2_ws_cap60": ["planner.escape.mode=kick_goal", "planner.escape.kick_px=15", "planner.escape.proprio_boost=10",
                            "planner.escape.max_n=7", "planner.escape.boost_replans=2",
                            "planner.escape.workspace_clip=true", "planner.escape.action_cap_px=60"],
    # the same with a 70 px cap
    "K2P_short2_ws_cap70": ["planner.escape.mode=kick_goal", "planner.escape.kick_px=15", "planner.escape.proprio_boost=10",
                            "planner.escape.max_n=7", "planner.escape.boost_replans=2",
                            "planner.escape.workspace_clip=true", "planner.escape.action_cap_px=70"],
    # K2P_short2 + workspace clip, no action cap
    "K2P_short2_ws": ["planner.escape.mode=kick_goal", "planner.escape.kick_px=15", "planner.escape.proprio_boost=10",
                      "planner.escape.max_n=7", "planner.escape.boost_replans=2", "planner.escape.workspace_clip=true"],
    # AdaJEPA with the pusher kept inside the arena (no escapes)
    "none_ws": ["planner.escape.mode=none", "planner.escape.workspace_clip=true"],
    "KS_7x15": ["planner.escape.mode=staged", "planner.escape.kick_px=15", "planner.escape.proprio_boost=10",
                "planner.escape.object_scale=0", "planner.escape.object_escapes=2", "planner.escape.max_n=7"],
}
lock = threading.Lock()


def log(msg):
    with lock:
        print(msg, flush=True)


def rel(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def finished(out):
    logs = out / "logs.json"
    return (out / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()


def mem_available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


def worker(gpu, jobs, a, results):
    while True:
        try:
            out, cmd = jobs.get_nowait()
        except queue.Empty:
            return
        with lock:  # one start at a time, memory permitting
            while mem_available_gb() < a.min_free_gb:
                time.sleep(30)
            out.mkdir(parents=True, exist_ok=True)
            print(f"START {rel(out)} GPU={gpu}", flush=True)
            fh = (out / "console.log").open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "WANDB_MODE": "offline", "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "SDL_VIDEODRIVER": "dummy"}
            proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            time.sleep(a.stagger)
        t0 = time.monotonic()
        rc = proc.wait()
        fh.close()
        ok = rc == 0 and finished(out)
        results.append(ok)
        log(f"{'DONE' if ok else 'FAIL'} {rel(out)} seconds={time.monotonic() - t0:.0f}" + ("" if ok else f" exit={rc}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S7")
    ap.add_argument("--arms", nargs="+", default=["none", "K2", "K2P", "K3"])
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--jobs", type=int, default=14, help="concurrent runs (spread over --gpus)")
    ap.add_argument("--ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
    ap.add_argument("--epoch", default="3")
    ap.add_argument("--max-episodes", type=int, default=0, help="cap episodes per chunk (smoke tests)")
    ap.add_argument("--chunks", nargs="*", default=None, help="only these chunk names")
    ap.add_argument("--root", default=None, help="default eval_outputs/escape_<stream>")
    ap.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every run")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--stagger", type=float, default=10.0)
    a = ap.parse_args()

    blocks = json.loads((ROOT / "data/cont_streams" / a.stream / "manifest.json").read_text())["blocks"]
    if a.chunks:
        blocks = [b for b in blocks if b["chunk"] in a.chunks]
    root = ROOT / (a.root or f"eval_outputs/escape_{a.stream}")
    jobs = queue.Queue()
    for b in blocks:  # chunk-major, so every arm progresses through the stream together
        for arm in a.arms:
            out = root / arm / b["chunk"]
            if finished(out):
                continue
            n = min(b["n"], a.max_episodes) if a.max_episodes else b["n"]
            cmd = [sys.executable, ROOT / "plan.py", "--config-name", "escape_plan_gd_pushobj",
                   f"ckpt_base_path={(ROOT / a.ckpt).resolve()}", f"model_epoch={a.epoch}",
                   f"eval_data_path={ROOT / 'data/cont_streams' / a.stream / (b['chunk'] + '.pkl')}",
                   f"n_evals={n}", "+wandb_logging=false", f"hydra.run.dir={out}"] + ARMS[arm] + a.extra
            jobs.put((out, cmd))
    gpus = a.gpus.split(",")
    log(f"{jobs.qsize()} runs: {len(blocks)} chunks x {a.arms}; {a.jobs} at a time on GPUs {gpus}")
    results = []
    threads = [threading.Thread(target=worker, args=(gpus[i % len(gpus)], jobs, a, results)) for i in range(a.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"finished: {sum(results)} ok, {len(results) - sum(results)} failed")
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()
