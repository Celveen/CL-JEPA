"""Run a continual (fast/slow) stream for several consolidation weights w, plus frozen probes.

Each w is one chain: the stream's chunks (scripts/cont_build_stream.py) run in
order, every chunk loading the slow state the previous one saved. Every arm sees
the same episodes in the same order, so arms are paired episode by episode.
When a chunk finishes, frozen probes of theta0 + slow at that point are queued:
batched planning on the held-out eval targets of each probe shape (seed
--probe-seed draw). Probes of theta0 itself are run once, shared by all arms.

    eval_outputs/cont_<stream>/w<w>/chunk_<k>_<shape>/       stream chunks
    eval_outputs/cont_<stream>/w<w>/probe_after_<k>/<shape>/ probes after chunk k
    eval_outputs/cont_<stream>/probe_base/<shape>/           probes of theta0

Finished runs are skipped, so the script can be re-run to resume.

    python scripts/cont_run_stream.py --stream S7t3 --ws 0 --independent --no-base-probes \
        --chain-gpus 0,1,4,5,6,7 --probe-gpus 0
    python scripts/cont_run_stream.py --stream S7t3 --arms cls6a_free --no-base-probes --chain-gpus 0 --probe-gpus 0
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
NAMED_ARMS = {  # per-domain slow predictors trained on replay; no frozen probes
    # v5: per-domain learners (CLSMPCPlanner) with maturity/write gates, familiar episodes never written, new
    # domains from 3 aligned pending episodes, + validated, symmetric merging (planning/continual_merge.py)
    "cls5_free": ["planner._target_=planning.continual_merge.CLSMergeMPCPlanner", "planner.continual.write=delta",
                  "planner.continual.tau=0.10", "planner.continual.mature_k=3", "planner.continual.mature_ratio=0.9",
                  "planner.continual.reject_ratio=1.2", "planner.continual.familiar_thr=0.0146",
                  "planner.continual.pending_m=3", "planner.continual.tau_new=0.05", "planner.continual.fp_unit=true",
                  "planner.continual.merge_eps=0.3", "planner.continual.merge_k=3"],
    # v7 = v5 (classmate's settings) + writes prefer existing domains (learner writes unless rejected, functional
    # join <= 0.9), new domains need 5 pending episodes, periodic absorption of immature domains (<= 0.8),
    # familiar data reads theta0, merge_k 2 / merge_eps 0.5; familiar episodes never written (planning/continual_v7.py)
    "cls7_free": ["planner._target_=planning.continual_v7.CLSv7MPCPlanner", "planner.continual.write=delta",
                  "planner.continual.tau=0.10", "planner.continual.mature_k=3", "planner.continual.mature_ratio=0.9",
                  "planner.continual.reject_ratio=1.2", "planner.continual.familiar_thr=0.0146",
                  "planner.continual.pending_m=5", "planner.continual.tau_new=0.05", "planner.continual.fp_unit=true",
                  "planner.continual.merge_eps=0.5", "planner.continual.merge_k=2",
                  "planner.continual.familiar_read=true", "planner.continual.join_ratio=0.9",
                  "planner.continual.absorb_ratio=0.8", "planner.continual.absorb_every=25"],
    # v7c = v7 with v5's founding (3) and merge (k 3, eps 0.3), absorption limited to fragments (< 6 episodes)
    "cls7c_free": ["planner._target_=planning.continual_v7.CLSv7MPCPlanner", "planner.continual.write=delta",
                   "planner.continual.tau=0.10", "planner.continual.mature_k=3", "planner.continual.mature_ratio=0.9",
                   "planner.continual.reject_ratio=1.2", "planner.continual.familiar_thr=0.0146",
                   "planner.continual.pending_m=3", "planner.continual.tau_new=0.05", "planner.continual.fp_unit=true",
                   "planner.continual.merge_eps=0.3", "planner.continual.merge_k=3",
                   "planner.continual.familiar_read=true", "planner.continual.join_ratio=0.9",
                   "planner.continual.absorb_ratio=0.8", "planner.continual.absorb_every=25",
                   "planner.continual.absorb_frag=6"],
    # v5a = v5's writing and merging unchanged + familiar data reads theta0 + fragment absorption (no pending joins)
    "cls5a_free": ["planner._target_=planning.continual_v7.CLSv7MPCPlanner", "planner.continual.write=delta",
                   "planner.continual.tau=0.10", "planner.continual.mature_k=3", "planner.continual.mature_ratio=0.9",
                   "planner.continual.reject_ratio=1.2", "planner.continual.familiar_thr=0.0146",
                   "planner.continual.pending_m=3", "planner.continual.tau_new=0.05", "planner.continual.fp_unit=true",
                   "planner.continual.merge_eps=0.3", "planner.continual.merge_k=3",
                   "planner.continual.familiar_read=true", "planner.continual.join_ratio=0",
                   "planner.continual.learner_win=true", "planner.continual.absorb_ratio=0.8",
                   "planner.continual.absorb_every=25", "planner.continual.absorb_frag=6",
                   "planner.continual.absorb_pending=false"],
    # v6a = v5 + relaxed learner writes and relaxed merge (merge_k 2, merge_eps 0.5) + familiar data reads theta0
    # + fragment absorption (no joins); familiar episodes never written, no shape labels
    "cls6a_free": ["planner._target_=planning.continual_v7.CLSv7MPCPlanner", "planner.continual.write=delta",
                   "planner.continual.tau=0.10", "planner.continual.mature_k=3", "planner.continual.mature_ratio=0.9",
                   "planner.continual.reject_ratio=1.2", "planner.continual.familiar_thr=0.0146",
                   "planner.continual.pending_m=3", "planner.continual.tau_new=0.05", "planner.continual.fp_unit=true",
                   "planner.continual.merge_eps=0.5", "planner.continual.merge_k=2",
                   "planner.continual.familiar_read=true", "planner.continual.join_ratio=0",
                   "planner.continual.learner_win=false", "planner.continual.absorb_ratio=0.8",
                   "planner.continual.absorb_every=25", "planner.continual.absorb_frag=6",
                   "planner.continual.absorb_pending=false"],
}
print_lock = threading.Lock()
start_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def rel(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def mem_available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


def finished(out, need_state=False):
    logs = out / "logs.json"
    ok = (out / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()
    return ok and (not need_state or (out / "continual_state.pt").exists())


def arm_name(w):
    return f"w{float(w):g}"


def run(cmd, out, gpu, a):
    """Start one plan.py run (one start per --stagger seconds, memory permitting) and wait for it."""
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "WANDB_MODE": "offline", "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2", "SDL_VIDEODRIVER": "dummy"}
    with start_lock:
        while mem_available_gb() < a.min_free_gb:
            time.sleep(30)
        log(f"START {rel(out)} GPU={gpu}")
        fh = (out / "console.log").open("w")
        proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        time.sleep(a.stagger)
    t0 = time.monotonic()
    rc = proc.wait()
    fh.close()
    return rc, time.monotonic() - t0 + a.stagger


def base_cmd(out, data, n, a):
    return [sys.executable, ROOT / "plan.py", "--config-name", a.config,
            f"ckpt_base_path={(ROOT / a.ckpt).resolve()}", f"model_epoch={a.epoch}",
            f"eval_data_path={data}", f"n_evals={n}", "+wandb_logging=false", f"hydra.run.dir={out}"]


def probe_job(out, shape, state, a):
    data = ROOT / "data/pushobj_eval" / f"val_{shape}" / "plan_targets.pkl"
    return out, base_cmd(out, data, a.probe_n, a) + [
        "planner._target_=planning.continual.SlowFrozenMPCPlanner", "segments_order=random",
        f"seed={a.probe_seed}", f"planner.continual.state_in={state if state else 'null'}"] + a.extra


def chain(name, overrides, w, gpu, blocks, root, a, probes, results):
    """w: the fixed consolidation weight (its arms get frozen probes), or None for a named arm."""
    arm = root / name
    # resume after the last finished chunk that still has its state (--keep-states prunes earlier ones)
    start = 0
    for k, b in enumerate(blocks):
        if not finished(arm / b["chunk"]):
            break
        if (arm / b["chunk"] / "continual_state.pt").exists():
            start = k + 1
    prev_state = arm / blocks[start - 1]["chunk"] / "continual_state.pt" if start else None
    for k, b in enumerate(blocks):
        out = arm / b["chunk"]
        if k < start:
            if w is not None and float(w) > 0 and (out / "continual_state.pt").exists():
                for shape in a.probe_shapes:
                    probes.put(probe_job(arm / f"probe_after_{k:02d}" / shape, shape, out / "continual_state.pt", a))
            continue
        if not finished(out, need_state=True):
            data = ROOT / "data/cont_streams" / a.stream / f"{b['chunk']}.pkl"
            n = min(b["n"], a.max_episodes) if a.max_episodes else b["n"]
            cmd = base_cmd(out, data, n, a) + overrides + [
                f"planner.continual.tag={b.get('tag', b['shape'])}",
                f"planner.continual.state_in={prev_state if prev_state else 'null'}"] + a.extra
            if "seed" in b:  # streams built from test-seed draws plan each chunk with its own seed
                cmd.append(f"seed={b['seed']}")
            rc, sec = run(cmd, out, gpu, a)
            ok = rc == 0 and finished(out, need_state=True)
            (out / "run_metadata.json").write_text(json.dumps({"seconds": sec, "gpu": gpu, "arm": name,
                                                               "overrides": overrides, "rc": rc}))
            log(f"{'DONE' if ok else 'FAIL'} {rel(out)} seconds={sec:.0f}" + ("" if ok else f" exit={rc}"))
            results.append(ok)
            if not ok:
                return  # later chunks need this state
        if a.keep_states and prev_state is not None and prev_state.exists() and (k - 1) not in a.keep_states:
            prev_state.unlink()  # only the next chunk needed it
        prev_state = out / "continual_state.pt"
        if w is not None and float(w) > 0:  # w = 0 never changes theta0: its probes are probe_base
            for shape in a.probe_shapes:
                probes.put(probe_job(arm / f"probe_after_{k:02d}" / shape, shape, prev_state, a))


def independent_jobs(name, overrides, blocks, root, a, jobs):
    """w = 0 (AdaJEPA) keeps no state between episodes, so its chunks can run in parallel."""
    for b in blocks:
        out = root / name / b["chunk"]
        if finished(out):
            continue
        data = ROOT / "data/cont_streams" / a.stream / f"{b['chunk']}.pkl"
        n = min(b["n"], a.max_episodes) if a.max_episodes else b["n"]
        cmd = base_cmd(out, data, n, a) + overrides + [f"planner.continual.tag={b.get('tag', b['shape'])}",
                                                        "planner.continual.state_in=null"] + a.extra
        if "seed" in b:
            cmd.append(f"seed={b['seed']}")
        jobs.put((out, cmd))


def job_worker(gpu, jobs, a, results):
    while True:
        try:
            out, cmd = jobs.get_nowait()
        except queue.Empty:
            return
        rc, sec = run(cmd, out, gpu, a)
        ok = rc == 0 and finished(out)
        log(f"{'DONE' if ok else 'FAIL'} {rel(out)} seconds={sec:.0f}" + ("" if ok else f" exit={rc}"))
        results.append(ok)


def probe_worker(gpu, probes, chains_done, a, results):
    while True:
        try:
            out, cmd = probes.get(timeout=20)
        except queue.Empty:
            if chains_done.is_set() and probes.empty():
                return
            continue
        if not finished(out):
            rc, sec = run(cmd, out, gpu, a)
            ok = rc == 0 and finished(out)
            log(f"{'DONE' if ok else 'FAIL'} {rel(out)} seconds={sec:.0f}" + ("" if ok else f" exit={rc}"))
            results.append(ok)
        probes.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S7t3")
    ap.add_argument("--ws", nargs="*", default=[], help="fixed consolidation weights, e.g. 0 0.003 0.01 0.03 1")
    ap.add_argument("--arms", nargs="*", default=[], help=f"named arms: {list(NAMED_ARMS)}")
    ap.add_argument("--no-base-probes", action="store_true")
    ap.add_argument("--chain-gpus", default="0,1,2,3,4", help="one entry per arm (cycled)")
    ap.add_argument("--probe-gpus", default="5,6,7,5", help="one probe worker per entry")
    ap.add_argument("--probe-shapes", nargs="+", default=["I", "square", "T", "L", "small_tee"])
    ap.add_argument("--probe-seed", type=int, default=100)
    ap.add_argument("--probe-n", type=int, default=50)
    ap.add_argument("--config", default="cont_plan_gd_pushobj", help="plan config: cont_plan_gd_pushobj or cont_plan_cem_pushobj")
    ap.add_argument("--ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
    ap.add_argument("--epoch", default="3")
    ap.add_argument("--root", default=None, help="default eval_outputs/cont_<stream>")
    ap.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every run (chunks and probes)")
    ap.add_argument("--max-episodes", type=int, default=0, help="cap episodes per chunk (smoke tests)")
    ap.add_argument("--keep-states", type=int, nargs="*", default=None,
                    help="if given, delete each chunk's slow state once the next chunk has used it, except these "
                         "chunk indices (and the last chunk); default keeps every state")
    ap.add_argument("--independent", action="store_true",
                    help="w = 0 arms only: run all chunks in parallel (one job per --chain-gpus entry at a time)")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--stagger", type=float, default=15.0)
    a = ap.parse_args()

    blocks = json.loads((ROOT / "data/cont_streams" / a.stream / "manifest.json").read_text())["blocks"]
    root = ROOT / (a.root or f"eval_outputs/cont_{a.stream}")
    probes, results, chains_done = queue.Queue(), [], threading.Event()
    if not a.no_base_probes:
        for shape in a.probe_shapes:
            probes.put(probe_job(root / "probe_base" / shape, shape, None, a))
    arms = [(arm_name(w), [f"planner.continual.w={w}"], w) for w in a.ws]
    arms += [(n, NAMED_ARMS[n], None) for n in a.arms]
    assert arms, "give --ws and/or --arms"
    gpus = a.chain_gpus.split(",")
    if a.independent:
        assert all(w is not None and float(w) == 0 for _, _, w in arms), "--independent is for --ws 0 only"
        jobs = queue.Queue()
        for n, ov, _ in arms:
            independent_jobs(n, ov, blocks, root, a, jobs)
        log(f"stream {a.stream}: {jobs.qsize()} independent chunk runs on GPUs {gpus}")
        workers = [threading.Thread(target=job_worker, args=(g, jobs, a, results)) for g in gpus]
        for t in workers:
            t.start()
        for t in workers:
            t.join()
        log(f"finished: {sum(results)} ok, {len(results) - sum(results)} failed")
        sys.exit(0 if all(results) else 1)
    chains = [threading.Thread(target=chain, args=(n, ov, w, gpus[i % len(gpus)], blocks, root, a, probes, results))
              for i, (n, ov, w) in enumerate(arms)]
    workers = [threading.Thread(target=probe_worker, args=(g, probes, chains_done, a, results))
               for g in a.probe_gpus.split(",")]
    log(f"stream {a.stream}: {len(blocks)} chunks x {len(arms)} arms ({[n for n, _, _ in arms]}); probes on {a.probe_shapes}")
    for t in chains + workers:
        t.start()
    for t in chains:
        t.join()
    chains_done.set()
    for t in workers:
        t.join()
    log(f"finished: {sum(results)} ok, {len(results) - sum(results)} failed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
