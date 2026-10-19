"""Build the interactive HTML report for a PushObj run.

Collects every number the page shows from saved outputs (metrics, paired eval
results, per-replan logs, dataset audit) into one JSON blob and inlines it,
with the example-frame image, into scripts/report_page_template.html.

    python scripts/build_report_page.py            # defaults: the 20k / 5-epoch run

The contact-bias run adds the diagnosis inputs and uses the second template:

    python scripts/build_report_page.py --report reports/pushobj_bias_16k \
        --model checkpoints/pushobj_bias_16k_e3_seed0 --eval-dir eval_outputs/pushobj_bias_16k_e3 \
        --data data/pushobj_bias_16k --train-config train_pushobj_bias16k --prev-report reports/pushobj_replay_20k_e5 \
        --prev-model checkpoints/pushobj_replay_20k_e5_seed0 --prev-label "原样回放 2 万条" \
        --prev-runs randomwalk=reports/pushobj_2000_e20 replay2k=reports/pushobj_replay_e10 \
                    replay20k=reports/pushobj_replay_20k_e5 --run-key bias16k --epoch-diag \
        --prev-epoch-diag eval_outputs/pushobj_replay_20k_e3 \
        --distribution reports/pushobj_bias_16k/distribution_vs_author.json \
        --calibration reports/pushobj_replay_20k_e5/diagnostics/contact_bias_calibration.json \
        --calibration-fine reports/pushobj_replay_20k_e5/diagnostics/contact_bias_calibration_fine.json \
        --adaptivity reports/pushobj_replay_20k_e5/diagnostics/adaptivity.json reports/pushobj_bias_16k/adaptivity.json \
        --action-match reports/pushobj_replay_20k_e5/diagnostics/action_source_match.json \
        --seed-summary reports/pushobj_bias_16k/seed_eval_summary.json --prev-data data/pushobj_replay_20k \
        --it30 eval_outputs/pushobj_bias_16k_e3_seed200_it30 eval_outputs/pushobj_bias_16k_e3_seed300_it30 \
        --paper-fig2 reports/pushobj_bias_16k/paper_fig2_readout.json --dump-json reports/pushobj_bias_16k/report_data.json \
        --template scripts/report_page_template_v2.html
"""
import argparse
import base64
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ("T", "L", "Z", "+", "I", "small_tee", "square")
SEEN = SHAPES[:4]
LABELS = {"T": "T", "L": "L", "Z": "Z", "+": "+", "I": "I", "small_tee": "small T", "square": "square"}
# AdaJEPA paper (arXiv:2606.32026): data-scale study K=4, N=4k (App. B.3) and the hyper-parameter
# ablation on T / square (App. B.2, at most 20 replans). The main experiment (Figure 2, PushObj-TLZ+-4k,
# the setting reproduced here) uses at most 30 replans; its values come from --paper-fig2.
PAPER = [
    {"label": "已见形状", "key": "seen", "frozen": [0.51, 0.51], "adajepa": [0.81, 0.81],
     "note": "数据规模实验（附录 B.3）K=4、N=4k；7 组训练形状、3 个种子平均；重规划上限未单独说明"},
    {"label": "未见形状", "key": "unseen", "frozen": [0.36, 0.36], "adajepa": [0.52, 0.52], "note": "同上"},
    {"label": "T", "key": "T", "frozen": [0.50, 0.50], "adajepa": [0.81, 0.88],
     "note": "超参数消融（附录 B.2），最多 20 次重规划；AdaJEPA 为不同超参与回放设置的范围"},
    {"label": "square", "key": "square", "frozen": [0.20, 0.20], "adajepa": [0.35, 0.51], "note": "同上"},
]


RUN_LABELS = {"randomwalk": "随机游走 · 2000 条 · 20 轮", "replay2k": "原样回放 · 2000 条 · 10 轮",
              "replay20k": "原样回放 · 2 万条 · 5 轮", "bias16k": "动作偏置回放 · 2 万条 · 3 轮"}
# offline adaptivity (scripts/diagnose_adaptation.py): model key -> (label, run key, epoch)
ADAPT_MODELS = {"released (e3)": ("发布检查点 · 第 3 轮", "released", 3), "ours e3": ("原样回放 · 第 3 轮", "replay20k", 3),
                "ours e5": ("原样回放 · 第 5 轮", "replay20k", 5), "bias16k e2": ("动作偏置 · 第 2 轮", "bias16k", 2),
                "bias16k e3": ("动作偏置 · 第 3 轮", "bias16k", 3)}


def rows(path):
    out = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def outcome(d):
    p = Path(d) / "final_eval_results.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {"success": z["successes"].reshape(-1).astype(int).tolist(), "state_0": z["state_0"]}


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - r, c + r]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def shape_polygons():
    """Block outlines in body-local coordinates, centred on the centre of gravity."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str(ROOT))
    import gym
    import env  # noqa: F401
    e = gym.make("pushobj", with_velocity=True, with_target=True).unwrapped
    out = {}
    for s in SHAPES:
        e.update_env({"shape": s})
        e.prepare(0, np.array([100.0, 100.0, 256.0, 256.0, 0.0, 0.0, 0.0]))
        cog = e.block.center_of_gravity
        out[s] = [[[round(v.x - cog.x, 1), round(v.y - cog.y, 1)] for v in sh.get_vertices()] for sh in e.block.shapes]
    e.close()
    return out


def cumulative_success(per_iter_path, max_iter, n=50):
    """Share of targets reached by replan k, from the per-sample AdaJEPA log."""
    per = rows(per_iter_path)
    if not per:
        return None
    first = {}
    for r in per:
        if r.get("success") and r["sample_idx"] not in first:
            first[r["sample_idx"]] = r["step"]
    return [sum(1 for s in first.values() if s <= k) / n for k in range(1, max_iter + 1)]


def mpc_curves(eval_dir, shape, max_iter, released_adajepa_dir):
    """Cumulative success rate after each replan: frozen (logged), AdaJEPA and released AdaJEPA (per sample)."""
    frozen = [r["mpc/success_rate"] for r in rows(Path(eval_dir) / shape / "frozen" / "logs.json")
              if "mpc/success_rate" in r]
    return {"frozen": frozen[:max_iter],
            "adajepa": cumulative_success(Path(eval_dir) / shape / "adajepa" / "per_iter_logs.json", max_iter),
            "released_adajepa": cumulative_success(Path(released_adajepa_dir) / shape / "per_iter_logs.json", max_iter)}


def adaptation_curve(eval_dir, shape):
    by = defaultdict(list)
    p = Path(eval_dir) / shape / "adajepa" / "adajepa_losses.csv"
    if p.exists():
        with p.open() as f:
            for r in csv.DictReader(f):
                by[int(r["mpc_iter"])].append(float(r["loss"]))
    steps = sorted(by)
    return {"step": steps, "mean": [float(np.mean(by[k])) for k in steps], "n": [len(by[k]) for k in steps]}


def run_counts(report_dir):
    out = {}
    p = Path(report_dir) / "evaluation.csv"
    if p.exists():
        with p.open() as f:
            for r in csv.DictReader(f):
                out.setdefault(r["planner"], {})[r["shape"]] = int(r["successes"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/pushobj_replay_20k_e5")
    ap.add_argument("--model", default="checkpoints/pushobj_replay_20k_e5_seed0")
    ap.add_argument("--eval-dir", default="eval_outputs/pushobj_replay_20k_e5")
    ap.add_argument("--data", default="data/pushobj_replay_20k")
    ap.add_argument("--prev-report", default="reports/pushobj_replay_e10")
    ap.add_argument("--prev-model", default="checkpoints/pushobj_replay_2000_e10_seed0")
    ap.add_argument("--first-report", default="reports/pushobj_2000_e20")
    ap.add_argument("--released", default="eval_outputs/pushobj_released_reference")
    ap.add_argument("--released-adajepa", default="eval_outputs/pushobj_released_reference_adajepa")
    ap.add_argument("--epoch-diag", nargs="*", default=["eval_outputs/pushobj_replay_20k_e3"],
                    help="eval dirs of other epochs of the same model (<dir>/<shape>/<frozen|adajepa>)")
    ap.add_argument("--prev-label", default="上一轮（2000 条回放数据）", help="legend for the previous run's training curve")
    ap.add_argument("--prev-runs", nargs="*", default=["randomwalk=reports/pushobj_2000_e20", "replay2k=reports/pushobj_replay_e10"],
                    help="earlier runs on the same targets, key=report_dir (labels in RUN_LABELS)")
    ap.add_argument("--run-key", default="replay20k", help="RUN_LABELS key of this run")
    ap.add_argument("--train-config", default="train_pushobj_20k", help="hydra config name, for the reproduce command")
    ap.add_argument("--distribution", default=None, help="compare_data_distribution-style json: author/new/old per shape")
    ap.add_argument("--calibration", default=None, help="calibrate_contact_bias.py output")
    ap.add_argument("--calibration-fine", default=None, help="finer calibrate_contact_bias.py grid around the optimum")
    ap.add_argument("--adaptivity", nargs="*", default=[], help="diagnose_adaptation.py outputs (merged)")
    ap.add_argument("--action-match", default=None, help="nearest-source action match json")
    ap.add_argument("--seed-summary", default=None, help="summarize_seed_evals.py output (frozen over test seeds)")
    ap.add_argument("--prev-data", default=None, help="data dir of the previous run, for the change table")
    ap.add_argument("--prev-epoch-diag", nargs="*", default=[], help="eval dirs of other epochs of the previous run's model")
    ap.add_argument("--it30", nargs="*", default=[], help="eval dirs run with planner.max_iter=30, one per test seed")
    ap.add_argument("--paper-fig2", default=None, help="values read from the paper's Figure 2 (json)")
    ap.add_argument("--dump-json", default=None, help="also write the page data to this json (for the offline pack)")
    ap.add_argument("--template", default="scripts/report_page_template.html")
    ap.add_argument("--out", default=None, help="default: <report>/report.html")
    a = ap.parse_args()
    R = lambda p: (ROOT / p).resolve()
    report, model, eval_dir, data = R(a.report), R(a.model), R(a.eval_dir), R(a.data)

    cfg = yaml.safe_load((model / "hydra.yaml").read_text())
    gen = json.loads((data / "generation_config.json").read_text())
    audit = json.loads((report / "dataset_audit.json").read_text())
    plan_cfg = yaml.safe_load((eval_dir / "T" / "adajepa" / ".hydra" / "config.yaml").read_text())
    max_iter = int(plan_cfg["planner"]["max_iter"])

    # ---- per-episode outcomes on the shared targets ----
    outcomes, counts = {}, {"frozen": {}, "adajepa": {}}
    released, released_ada = {}, {}
    for s in SHAPES:
        f, d = outcome(eval_dir / s / "frozen"), outcome(eval_dir / s / "adajepa")
        rel, rel_a = outcome(R(a.released) / s), outcome(R(a.released_adajepa) / s)
        for other in (d, rel, rel_a):
            if other is not None:
                assert np.array_equal(other["state_0"], f["state_0"]), f"targets differ for {s}"
        outcomes[s] = {"frozen": f["success"], "adajepa": d["success"],
                       "released": rel["success"] if rel else None,
                       "released_adajepa": rel_a["success"] if rel_a else None}
        counts["frozen"][s], counts["adajepa"][s] = sum(f["success"]), sum(d["success"])
        if rel:
            released[s] = sum(rel["success"])
        if rel_a:
            released_ada[s] = sum(rel_a["success"])

    def paired(subset, x="frozen", y="adajepa"):
        b = sum(1 for s in subset for i in range(50) if outcomes[s][y][i] and not outcomes[s][x][i])
        c = sum(1 for s in subset for i in range(50) if outcomes[s][x][i] and not outcomes[s][y][i])
        return {"y_only": b, "x_only": c, "p": mcnemar(b, c)}

    groups = {"seen": list(SEEN), "unseen": list(SHAPES[4:]), "all": list(SHAPES)}
    summary = {}
    for g, sub in groups.items():
        n = 50 * len(sub)
        kf, ka = sum(counts["frozen"][s] for s in sub), sum(counts["adajepa"][s] for s in sub)
        summary[g] = {"n": n, "frozen": kf / n, "adajepa": ka / n, "frozen_ci": wilson(kf, n), "adajepa_ci": wilson(ka, n),
                      "paired": paired(sub),
                      "released": (sum(released[s] for s in sub) / n) if all(s in released for s in sub) else None,
                      "vs_released": paired(sub, "released", "frozen") if all(s in released for s in sub) else None}
        if all(s in released_ada for s in sub):
            summary[g].update(released_adajepa=sum(released_ada[s] for s in sub) / n,
                              released_adajepa_gain=paired(sub, "released", "released_adajepa"),
                              vs_released_adajepa=paired(sub, "released_adajepa", "adajepa"))

    per_shape = []
    for s in SHAPES:
        kf, ka = counts["frozen"][s], counts["adajepa"][s]
        o = outcomes[s]
        per_shape.append({
            "id": s, "label": LABELS[s], "seen": s in SEEN,
            "frozen": kf / 50, "adajepa": ka / 50, "frozen_k": kf, "adajepa_k": ka,
            "frozen_ci": wilson(kf, 50), "adajepa_ci": wilson(ka, 50),
            "released": released[s] / 50 if s in released else None,
            "released_adajepa": released_ada[s] / 50 if s in released_ada else None,
            "paired": {"both": sum(1 for i in range(50) if o["frozen"][i] and o["adajepa"][i]),
                       "adajepa_only": sum(1 for i in range(50) if o["adajepa"][i] and not o["frozen"][i]),
                       "frozen_only": sum(1 for i in range(50) if o["frozen"][i] and not o["adajepa"][i]),
                       "neither": sum(1 for i in range(50) if not o["frozen"][i] and not o["adajepa"][i])},
            "mpc": mpc_curves(eval_dir, s, max_iter, R(a.released_adajepa)),
            "adapt": adaptation_curve(eval_dir, s),
        })
    released_ada_paired = {}
    for s in released_ada:
        released_ada_paired[s] = paired([s], "released", "released_adajepa")

    # ---- runs on the same targets ----
    runs = []
    for spec in a.prev_runs:
        key, path = spec.split("=", 1)
        c = run_counts(R(path))
        if c:
            runs.append({"key": key, "label": RUN_LABELS[key],
                         "frozen": sum(c["frozen"].values()) / 350, "adajepa": sum(c["adajepa"].values()) / 350,
                         "per_shape": c})
    label = RUN_LABELS.get(a.run_key) or f"回放数据 · {audit['train']['episodes'] + audit['val']['episodes']:,} 条 · {cfg['training']['epochs']} 轮"
    runs.append({"key": a.run_key, "label": label, "current": True,
                 "frozen": summary["all"]["frozen"], "adajepa": summary["all"]["adajepa"], "per_shape": counts})

    # ---- training curves ----
    batch = int(cfg.get("effective_batch_size", 64))
    ipe = math.ceil(audit["train"]["slices"] / batch)
    epochs = [{"epoch": r["epoch"], "train_z": r["train_z_loss"], "val_z": r["val_z_loss"],
               "train_vis": r["train_z_visual_loss"], "val_vis": r["val_z_visual_loss"],
               "train_prop": r["train_z_proprio_loss"], "val_prop": r["val_z_proprio_loss"],
               "val_curv": r["val_curvature_loss_used_for_training"], "minutes": r["epoch_seconds"] / 60,
               "samples_m": r["epoch"] * ipe * batch / 1e6}
              for r in sorted({r["epoch"]: r for r in rows(model / "metrics.jsonl")}.values(), key=lambda r: r["epoch"])]
    iters = [{"epoch": r["epoch"], "iter": r["iter"], "x": (r["epoch"] - 1) + r["iter"] / ipe,
              "samples_m": ((r["epoch"] - 1) * ipe + r["iter"]) * batch / 1e6,
              "train_z": r["train_z_loss"], "train_vis": r["train_z_visual_loss"], "train_prop": r["train_z_proprio_loss"]}
             for r in rows(model / "metrics_iter.jsonl")]
    prev_epochs = []
    prev_audit = R(a.prev_report) / "dataset_audit.json"
    if prev_audit.exists() and (R(a.prev_model) / "metrics.jsonl").exists():
        p_ipe = math.ceil(json.loads(prev_audit.read_text())["train"]["slices"] / 64)
        prev_epochs = [{"epoch": r["epoch"], "val_z": r["val_z_loss"], "train_z": r["train_z_loss"],
                        "samples_m": r["epoch"] * p_ipe * 64 / 1e6}
                       for r in sorted({r["epoch"]: r for r in rows(R(a.prev_model) / "metrics.jsonl")}.values(),
                                       key=lambda r: r["epoch"])]

    def other_epochs(dirs):
        out = []
        for d in dirs:
            for s in SHAPES:
                base = R(d) / s
                f, ad = outcome(base / "frozen"), outcome(base / "adajepa")
                if f is None or ad is None:
                    continue
                assert np.array_equal(f["state_0"], outcome(eval_dir / s / "frozen")["state_0"]), f"targets differ: {base}"
                ep = next((int(l.split()[3].rstrip(":")) for l in (base / "adajepa" / "console.log").read_text(errors="ignore").splitlines()
                           if l.startswith("Resuming from epoch")), None)
                fs, as_ = f["success"], ad["success"]
                out.append({"shape": s, "epoch": ep, "frozen": sum(fs) / 50, "adajepa": sum(as_) / 50,
                            "adajepa_only": sum(1 for i in range(50) if as_[i] and not fs[i]),
                            "frozen_only": sum(1 for i in range(50) if fs[i] and not as_[i]),
                            "p": mcnemar(sum(1 for i in range(50) if as_[i] and not fs[i]),
                                         sum(1 for i in range(50) if fs[i] and not as_[i])),
                            "curve": cumulative_success(base / "adajepa" / "per_iter_logs.json", max_iter)})
        return out

    epoch_diag = other_epochs(a.epoch_diag)
    prev_epoch_diag = other_epochs(a.prev_epoch_diag)

    released_epoch = None
    rel_console = R(a.released_adajepa) / "T" / "console.log"
    if rel_console.exists():
        for line in rel_console.read_text(errors="ignore").splitlines():
            if line.startswith("Resuming from epoch"):
                released_epoch = int(line.split()[3].rstrip(":"))
                break

    # AdaJEPA of this model against the released checkpoint's AdaJEPA, where both ran
    for ps in per_shape:
        o = outcomes[ps["id"]]
        if o["released_adajepa"] is not None:
            b = sum(1 for i in range(50) if o["adajepa"][i] and not o["released_adajepa"][i])
            c = sum(1 for i in range(50) if o["released_adajepa"][i] and not o["adajepa"][i])
            ps["vs_released_adajepa"] = {"ours_only": b, "released_only": c, "p": mcnemar(b, c)}
        if o["released"] is not None:
            b = sum(1 for i in range(50) if o["frozen"][i] and not o["released"][i])
            c = sum(1 for i in range(50) if o["released"][i] and not o["frozen"][i])
            ps["vs_released"] = {"ours_only": b, "released_only": c, "p": mcnemar(b, c)}

    diag = {}
    if a.distribution:
        diag["distribution"] = json.loads(R(a.distribution).read_text())
    if a.calibration:
        cal = json.loads(R(a.calibration).read_text())
        diag["calibration"] = {"targets": cal["targets"],
                               "grid": [{"bias": r["bias"], "noise": r["noise"], "placement": r["placement"], "score": r["score"],
                                         "shapes": {k: {m: v[m] for m in ("score", "cos_to_block", "action_mag", "contact_frac", "mean_gap")}
                                                    for k, v in r["shapes"].items()}}
                                        for r in cal["results"]],
                               "chosen": {"bias": gen.get("action_bias_px"), "noise": gen.get("action_noise_px"),
                                          "placement": gen.get("placement")}}
        if a.calibration_fine:
            diag["calibration"]["fine"] = [{k: r[k] for k in ("bias", "noise", "placement", "score")}
                                           for r in json.loads(R(a.calibration_fine).read_text())["results"]]
    if a.adaptivity:
        merged = {}
        for p in a.adaptivity:
            for k, v in json.loads(R(p).read_text())["results"].items():
                lab, run, ep = ADAPT_MODELS.get(k, (k, None, None))
                merged[k] = {"label": lab, "run": run, "epoch": ep,
                             "ours": next(x for n, x in v.items() if n.startswith("ours")),
                             "author": next(x for n, x in v.items() if n.startswith("author"))}
        diag["adaptivity"] = [dict(key=k, **merged[k]) for k in ADAPT_MODELS if k in merged]
    if a.action_match:
        diag["action_match"] = json.loads(R(a.action_match).read_text())
    if a.seed_summary:
        diag["seeds"] = json.loads(R(a.seed_summary).read_text())
    prev_data = None
    if a.prev_data:
        pg = json.loads((R(a.prev_data) / "generation_config.json").read_text())
        prev_data = {"train": pg["summary"]["train"]["episodes"], "val": pg["summary"]["val"]["episodes"],
                     "per_shape_train": pg["train_per_shape"], "per_shape_val": pg["val_per_shape"],
                     "episodes_with_contact": pg["summary"]["train"]["episodes_with_contact"],
                     "mean_contact_steps": pg["summary"]["train"]["mean_contact_steps"],
                     "max_penetration": pg.get("max_penetration"), "max_block_step": pg.get("max_block_step"),
                     "epochs": len(prev_epochs) or None}

    it30 = None
    if a.it30:
        per_seed, raw = {}, {}
        for d in a.it30:
            pc = yaml.safe_load((R(d) / "T" / "adajepa" / ".hydra" / "config.yaml").read_text())
            seed, n_it = int(pc["seed"]), int(pc["planner"]["max_iter"])
            got = {}
            for sh in SHAPES:
                f, ad = outcome(R(d) / sh / "frozen"), outcome(R(d) / sh / "adajepa")
                if f is None or ad is None:
                    break
                assert np.array_equal(f["state_0"], ad["state_0"]), f"targets differ: {d}/{sh}"
                fc = [r["mpc/success_rate"] for r in rows(R(d) / sh / "frozen" / "logs.json") if "mpc/success_rate" in r][:n_it]
                ac = cumulative_success(R(d) / sh / "adajepa" / "per_iter_logs.json", n_it)
                got[sh] = {"frozen": f["success"], "adajepa": ad["success"], "frozen_curve": fc, "adajepa_curve": ac}
            if len(got) < len(SHAPES):
                continue  # this seed is still running
            raw[seed] = got
            per_seed[seed] = {sh: {"frozen": float(np.mean(g["frozen"])), "adajepa": float(np.mean(g["adajepa"])),
                                   "frozen_20": g["frozen_curve"][19], "adajepa_20": g["adajepa_curve"][19]}
                              for sh, g in got.items()}
        if raw:
            seeds = sorted(raw)

            def pooled(sub):
                fz = [x for sd in seeds for sh in sub for x in raw[sd][sh]["frozen"]]
                ad = [x for sd in seeds for sh in sub for x in raw[sd][sh]["adajepa"]]
                b = sum(1 for x, y in zip(fz, ad) if y and not x)
                c = sum(1 for x, y in zip(fz, ad) if x and not y)
                a20 = float(np.mean([raw[sd][sh]["adajepa_curve"][19] for sd in seeds for sh in sub]))
                f20 = float(np.mean([raw[sd][sh]["frozen_curve"][19] for sd in seeds for sh in sub]))
                return {"n": len(fz), "frozen": float(np.mean(fz)), "adajepa": float(np.mean(ad)),
                        "frozen_ci": wilson(sum(fz), len(fz)), "adajepa_ci": wilson(sum(ad), len(ad)),
                        "adajepa_only": b, "frozen_only": c, "p": mcnemar(b, c), "frozen_20": f20, "adajepa_20": a20}
            it30 = {"seeds": seeds, "max_iter": n_it, "per_seed": per_seed,
                    "shapes": {sh: dict(pooled([sh]),
                                        frozen_curve=np.mean([raw[sd][sh]["frozen_curve"] for sd in seeds], 0).tolist(),
                                        adajepa_curve=np.mean([raw[sd][sh]["adajepa_curve"] for sd in seeds], 0).tolist())
                               for sh in SHAPES},
                    "groups": {g: pooled(sub) for g, sub in groups.items()}}
    paper_fig2 = json.loads(R(a.paper_fig2).read_text()) if a.paper_fig2 else None

    s_train, s_val = gen["summary"]["train"], gen["summary"]["val"]
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "shapes": per_shape, "summary": summary, "runs": runs, "paper": PAPER,
        "polygons": shape_polygons(),
        "outcomes": outcomes,
        "released_adajepa_paired": released_ada_paired, "released_epoch": released_epoch, "epoch_diag": epoch_diag,
        "prev_epoch_diag": prev_epoch_diag,
        "training": {"epochs": epochs, "iters": iters, "iters_per_epoch": ipe, "batch": batch,
                     "gpu_batch": int(cfg.get("gpu_batch_size", batch)), "prev_epochs": prev_epochs, "prev_label": a.prev_label,
                     "decoder_start_epoch": cfg["training"].get("decoder_start_epoch"),
                     "save_every": cfg["training"].get("save_every_x_iterations"),
                     "lr": {"encoder": cfg["training"]["encoder_lr"], "predictor": cfg["training"]["predictor_lr"]},
                     "has_decoder": bool(cfg.get("has_decoder", True))},
        "eval": {"n": 50, "seed": 100, "max_iter": max_iter, "horizon": plan_cfg["planner"]["sub_planner"]["horizon"],
                 "n_taken": plan_cfg["planner"]["n_taken_actions"], "opt_steps": plan_cfg["planner"]["sub_planner"]["opt_steps"],
                 "adapt": plan_cfg["planner"]["adapt"]},
        "data": {"train": audit["train"], "val": audit["val"], "per_shape_train": gen["train_per_shape"],
                 "per_shape_val": gen["val_per_shape"], "source_episodes": gen["source_episodes"],
                 "excluded_off_frame": gen["source_excluded_agent_leaves_frame"],
                 "max_penetration": gen.get("max_penetration"), "max_block_step": gen.get("max_block_step"),
                 "action_bias_px": gen.get("action_bias_px"), "action_noise_px": gen.get("action_noise_px"),
                 "placement": gen.get("placement"), "train_pool": gen.get("train_pool"),
                 "shared_sources": audit.get("shared_source_trajectories_across_splits"),
                 "mean_contact_steps": s_train.get("mean_contact_steps"),
                 "gate": {sp: {"candidates": gen["summary"][sp]["candidates"],
                               "dropped": gen["summary"][sp]["candidates_dropped_by_quality_gate"],
                               "fixed": gen["summary"][sp]["kept_with_rejected_placement"],
                               "episodes": gen["summary"][sp]["episodes"]} for sp in ("train", "val")}},
        "diag": diag, "run_key": a.run_key, "prev_data": prev_data, "it30": it30, "paper_fig2": paper_fig2,
        "paths": {"report": a.report, "model": a.model, "eval": a.eval_dir, "data": a.data,
                  "train_config": a.train_config},
    }
    html = (ROOT / a.template).read_text()
    img = base64.b64encode((report / "dataset_examples.png").read_bytes()).decode()
    html = html.replace("__REPORT_DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__EXAMPLES_PNG__", "data:image/png;base64," + img)
    if a.dump_json:
        Path(a.dump_json).write_text(json.dumps({k: v for k, v in payload.items() if k != "polygons"}, ensure_ascii=False, indent=1))
    out = Path(a.out) if a.out else report / "report.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
