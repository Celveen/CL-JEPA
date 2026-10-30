"""Collect every number and frame the shape-shift figures need from the archived runs (CPU only).

Run from the CL-JEPA-cudagraph checkout after compute_frozen_prediction_loss.py:
    python reports/shape_shift_figs/scripts/collect_data.py

Sources (all under eval_outputs/ of this checkout unless noted):
  H7t3 held-out comparison, 7 shapes x 50 episodes x test seeds 2000/2100/2200, identical targets for all methods
    Frozen   official shape-shift weights, no adaptation   cmp_{gd,cem}_frozen_official/w0
    AdaJEPA  official shape-shift weights                   cmp_{gd,cem}_official/w0
    Ours     round-5 continual state of each of the planner's three chains, read-only
             GD:  cmp_gd_v6a_s100, cmp_gd_v6a_s400, cmp_gd_v6a_s700
             CEM: cmp_cem_v6a_s100, cmp_cem_v6a_s100b, cmp_cem_v6a_s700
  Ours five rounds: the same chains on their training streams, arm cls6a_free
             GD:  ../CL-JEPA/eval_outputs/cont_S7t3, cont_S7t3s400, cont_S7t3s700
             CEM: cont_S7t3_cem, cont_S7t3b_cem, cont_S7t3s700_cem
  CEM's S7t3s400 chain is left out: its round-5 learner lost maturity on small_tee and the state planned with the
  base model alone. S7t3b replaces it: the S7t3 stream (same targets and seeds) run again on a second machine
  (results branch of the repository; its round-5 state downloaded into cont_S7t3b_cem/cls6a_free/chunk_34_T).

Writes data/*.json|csv and replay/<planner>/<shape>/<method>/{step_XX,goal}.png next to this folder.
"""
import csv
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
EV = REPO / "eval_outputs"
EV_MAIN = REPO.parent / "CL-JEPA" / "eval_outputs"
sys.path.insert(0, str(REPO))

PLANNERS = ["GD", "CEM"]
SHAPES = ["I", "L", "square", "Z", "small_tee", "plus", "T"]
SEEN = {"T", "L", "Z", "plus"}
SEEDS = [2000, 2100, 2200]
METHODS = ["frozen", "adajepa", "ours"]
CHAINS = {"GD": ["s100", "s400", "s700"], "CEM": ["s100", "s100b", "s700"]}
STREAM_OF = {"s100": "S7t3", "s100b": "S7t3b", "s400": "S7t3s400", "s700": "S7t3s700"}
STREAM5_OF = {"s100": "S7t5", "s100b": "S7t5", "s400": "S7t5s400", "s700": "S7t5s700"}  # 5-round manifests
MAX_STEPS = 30


def h7_chunk(shape, seed):
    return f"chunk_{SEEDS.index(seed) * 7 + SHAPES.index(shape):02d}_{shape}"


def h7_dirs(planner, method):
    p = planner.lower()
    if method == "frozen":
        return {"official": EV / f"cmp_{p}_frozen_official/w0"}
    if method == "adajepa":
        return {"official": EV / f"cmp_{p}_official/w0"}
    return {c: EV / f"cmp_{p}_v6a_{c}/cls6a_free" for c in CHAINS[planner]}


def stream_root(planner, chain):
    s = STREAM_OF[chain]
    if planner == "GD":
        return (EV_MAIN if chain == "s100" else EV) / f"cont_{s}" / "cls6a_free"
    return EV / f"cont_{s}_cem" / "cls6a_free"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_iter(path):
    return [json.loads(l) for l in (path / "per_iter_logs.json").read_text().splitlines()]


def finished(d):
    logs = d / "logs.json"
    return (d / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()


# ---------------------------------------------------------------- H7t3 episodes
episodes, sources, checks = [], [], {"paired_targets": 0, "per_iter_success_step_matches": 0}
ref_targets = {}
for planner in PLANNERS:
    for method in METHODS:
        for unit, root in h7_dirs(planner, method).items():
            for seed in SEEDS:
                for shape in SHAPES:
                    d = root / h7_chunk(shape, seed)
                    assert finished(d), f"unfinished run: {d}"
                    r = np.load(d / "final_eval_results.npz")
                    key = (shape, seed)
                    tgt = (r["state_0"], r["state_g"])
                    if key not in ref_targets:
                        ref_targets[key] = tgt
                    assert np.array_equal(ref_targets[key][0], tgt[0]) and np.array_equal(ref_targets[key][1], tgt[1]), d
                    checks["paired_targets"] += 1
                    succ = r["successes"].astype(bool)
                    al = r["action_len"]
                    assert len(succ) == 50
                    assert np.all(np.isfinite(al) == succ), d
                    if method != "frozen":  # sequential runs log every sample's every replan
                        rows = per_iter(d)
                        for i in range(50):
                            rr = [x for x in rows if x["sample_idx"] == i]
                            first = next((x["step"] for x in rr if x["success"]), None)
                            assert (first == al[i]) if succ[i] else (first is None and len(rr) == MAX_STEPS), (d, i)
                            checks["per_iter_success_step_matches"] += 1
                    for i in range(50):
                        episodes.append({"planner": planner, "method": method, "unit": unit, "seed": seed,
                                         "shape": shape, "sample_idx": i, "success": int(succ[i]),
                                         "success_step": int(al[i]) if succ[i] else "",
                                         "mpc_steps": int(al[i]) if succ[i] else MAX_STEPS})
                    sources.append({"planner": planner, "method": method, "unit": unit, "seed": seed, "shape": shape,
                                    "path": str(d.relative_to(REPO.parent)),
                                    "final_eval_results_sha256": sha(d / "final_eval_results.npz")})
print("episodes", len(episodes), checks, flush=True)


def select(**kw):
    return [e for e in episodes if all(e[k] == v for k, v in kw.items())]


def block(planner, method, seed, shapes):
    """One seed block of one method: Ours pools its three chains (150 episodes per shape), baselines 50."""
    return [e for e in episodes if e["planner"] == planner and e["method"] == method and e["seed"] == seed
            and e["shape"] in shapes]


def stats(values):
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "units": v.tolist()}


GROUPS = {"all": SHAPES, "seen": [s for s in SHAPES if s in SEEN], "unseen": [s for s in SHAPES if s not in SEEN]}
summary = {"unit": "test-seed block (2000, 2100, 2200); Ours pools its three chains inside each block",
           "groups": {k: v for k, v in GROUPS.items()}, "rates": {}, "by_shape": {}, "curves": {}, "per_chain": {}}
for planner in PLANNERS:
    summary["rates"][planner], summary["by_shape"][planner], summary["curves"][planner] = {}, {}, {}
    for method in METHODS:
        summary["rates"][planner][method] = {}
        for g, shapes in GROUPS.items():
            rates = [100 * np.mean([e["success"] for e in block(planner, method, s, shapes)]) for s in SEEDS]
            eps = [e for s in SEEDS for e in block(planner, method, s, shapes)]
            summary["rates"][planner][method][g] = {**stats(rates), "episodes": len(eps),
                                                    "successes": int(sum(e["success"] for e in eps))}
    for shape in SHAPES:
        summary["by_shape"][planner][shape], summary["curves"][planner][shape] = {}, {}
        for method in METHODS:
            rates = [100 * np.mean([e["success"] for e in block(planner, method, s, [shape])]) for s in SEEDS]
            eps = [e for s in SEEDS for e in block(planner, method, s, [shape])]
            summary["by_shape"][planner][shape][method] = {**stats(rates), "episodes": len(eps),
                                                           "successes": int(sum(e["success"] for e in eps))}
            curves = []
            for s in SEEDS:
                b = block(planner, method, s, [shape])
                curves.append([100 * np.mean([1 if e["success"] and e["success_step"] <= k else 0 for e in b])
                               for k in range(1, MAX_STEPS + 1)])
            c = np.array(curves)
            summary["curves"][planner][shape][method] = {"mean": c.mean(0).tolist(), "sd": c.std(0, ddof=1).tolist()}
            assert abs(c.mean(0)[-1] - summary["by_shape"][planner][shape][method]["mean"]) < 1e-9
    summary["per_chain"][planner] = {
        c: {g: 100 * np.mean([e["success"] for e in select(planner=planner, method="ours", unit=c) if e["shape"] in sh])
            for g, sh in GROUPS.items()} for c in CHAINS[planner]}

# ---------------------------------------------------------------- MPC steps (equal episodes per method)
mpc = {}
for planner in PLANNERS:
    mpc[planner] = {}
    for method in METHODS:
        per_seed_all = [np.mean([e["mpc_steps"] for e in block(planner, method, s, SHAPES)]) for s in SEEDS]
        per_seed_succ = [np.mean([e["mpc_steps"] for e in block(planner, method, s, SHAPES) if e["success"]])
                         for s in SEEDS]
        eps = [e for s in SEEDS for e in block(planner, method, s, SHAPES)]
        mpc[planner][method] = {"episodes": len(eps), "success_rate": summary["rates"][planner][method]["all"],
                                "steps_all": stats(per_seed_all), "steps_success_only": stats(per_seed_succ),
                                "pooled_steps_all": float(np.mean([e["mpc_steps"] for e in eps])),
                                "pooled_steps_success_only": float(np.mean([e["mpc_steps"] for e in eps if e["success"]]))}

# ---------------------------------------------------------------- Ours five rounds on the training streams
rounds = {}
for planner in PLANNERS:
    rounds[planner] = {}
    per_chain = {}
    for chain in CHAINS[planner]:
        root = stream_root(planner, chain)
        blocks = json.loads((REPO / "data/cont_streams" / STREAM5_OF[chain] / "manifest.json").read_text())["blocks"]
        assert len(blocks) == 35
        per_chain[chain] = {}
        for r in range(5):
            res = {g: [] for g in GROUPS}
            for b in blocks[7 * r: 7 * r + 7]:
                d = root / b["chunk"]
                assert finished(d), d
                s = np.load(d / "final_eval_results.npz")["successes"].astype(int)
                shape = "plus" if b["shape"] == "+" else b["shape"]
                for g, sh in GROUPS.items():
                    if shape in sh:
                        res[g].extend(s.tolist())
            per_chain[chain][r + 1] = {g: {"rate": 100 * float(np.mean(v)), "episodes": len(v)} for g, v in res.items()}
    for r in range(1, 6):
        rounds[planner][r] = {g: {**stats([per_chain[c][r][g]["rate"] for c in CHAINS[planner]]),
                                  "episodes": sum(per_chain[c][r][g]["episodes"] for c in CHAINS[planner])}
                              for g in GROUPS}
    rounds[planner]["per_chain"] = per_chain

# ---------------------------------------------------------------- trajectories and logged prediction loss
frozen = json.loads((OUT / "data/frozen_prediction_losses.json").read_text())
frozen_map = {(c["planner"], c["shape"]): c for c in frozen["cases"]}
assert len(frozen_map) == 14 and frozen["model_state_unchanged"]
TRAJ_DIRS = {"adajepa": lambda p, s: h7_dirs(p, "adajepa")["official"] / h7_chunk(s, 2000),
             "ours": lambda p, s: h7_dirs(p, "ours")["s100"] / h7_chunk(s, 2000)}
traj_cases = []
for planner in PLANNERS:
    for shape in SHAPES:
        case = {"planner": planner, "shape": shape, "seed": 2000, "sample_idx": 0, "methods": {}}
        goal = None
        for method in METHODS:
            dst = OUT / "replay" / planner / shape / method
            dst.mkdir(parents=True, exist_ok=True)
            if method == "frozen":
                fc = frozen_map[(planner, shape)]
                tr = np.load(dst / "trajectory.npz")
                frames = tr["visual"]
                d = REPO / fc["source"]
                n = len(fc["steps"])
                item = {"source": fc["source"], "steps": fc["steps"], "loss": fc["loss"],
                        "eval_mode_loss": fc["eval_mode_loss"], "success": fc["success"], "n_steps": n,
                        "loss_origin": "computed: archived official weights + replayed original actions",
                        "frames_origin": "simulator replay of the run's executed actions"}
            else:
                d = TRAJ_DIRS[method](planner, shape)
                rows = [x for x in per_iter(d) if x["sample_idx"] == 0]
                n = len(rows)
                assert [x["step"] for x in rows] == list(range(1, n + 1))
                ep = [json.loads(l) for l in (d / "continual_episodes.jsonl").read_text().splitlines()][0]
                assert ep["sample_idx"] == 0 and ep["steps"] == n
                with np.load(d / "episode_frames.npz") as f:  # Ours numbers episodes on from its loaded state
                    frames = f[f"ep{ep['episode']}_visual"]
                assert len(frames) == n + 1
                raw_csv = (d / "adajepa_losses.csv").read_text()
                crow = [x for x in csv.DictReader(io.StringIO(raw_csv)) if int(x["sample_idx"]) == 0]
                assert len(crow) == n and all(int(x["adajepa_step"]) == 1 for x in crow)
                assert all(int(c["mpc_iter"]) == x["step"] and float(c["loss"]) == x["adajepa/pred_loss"]
                           for c, x in zip(crow, rows))
                item = {"source": str(d.relative_to(REPO.parent)), "per_iter_logs_sha256": sha(d / "per_iter_logs.json"),
                        "steps": [x["step"] for x in rows], "loss": [x["adajepa/pred_loss"] for x in rows],
                        "logged_success": [x["success"] for x in rows], "success": bool(rows[-1]["success"]),
                        "n_steps": n, "csv_matches_per_iter_exactly": True,
                        "loss_origin": "logged adaptation loss (adajepa_losses.csv = per_iter_logs.json)",
                        "frames_origin": "frames saved during the run (episode_frames.npz)"}
                if method == "ours":
                    assert "s100" in str(d)
            with open(d / "plan_targets.pkl", "rb") as f:
                g = pickle.load(f)["obs_g"]["visual"][0, 0]
            if goal is None:
                goal = g
            assert np.array_equal(goal, g), (planner, shape, method)
            for k, fr in enumerate(frames):
                Image.fromarray(np.asarray(fr, np.uint8)).save(dst / f"step_{k:02d}.png")
            Image.fromarray(np.asarray(goal, np.uint8)).save(dst / "goal.png")
            case["methods"][method] = item
        traj_cases.append(case)
        print("trajectory", planner, shape, {m: (v["n_steps"], v["success"]) for m, v in case["methods"].items()},
              flush=True)

thumb = OUT / "thumbnails"
thumb.mkdir(exist_ok=True)
for shape in SHAPES:
    Image.open(OUT / "replay/GD" / shape / "adajepa/step_00.png").save(thumb / f"{shape}.png")

# ---------------------------------------------------------------- write
D = OUT / "data"
with (D / "shape7_episodes.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(episodes[0]))
    w.writeheader()
    w.writerows(episodes)
(D / "shape7_summary.json").write_text(json.dumps(summary, indent=2))
(D / "ours_rounds.json").write_text(json.dumps(rounds, indent=2))
(D / "mpc_steps.json").write_text(json.dumps(mpc, indent=2))
(D / "trajectories.json").write_text(json.dumps({"cases": traj_cases}, indent=2))
(D / "sources.json").write_text(json.dumps({"h7t3_runs": sources, "checks": checks}, indent=2))
print("DONE", len(episodes), "episodes", flush=True)
