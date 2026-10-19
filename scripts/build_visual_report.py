"""Build the visual-shift report page (AdaJEPA Figure 2, bottom row) for a T-only model.

Reads the per-condition evaluations (<eval-dir>/<condition>/<frozen|adajepa>), the
paper readout, the training logs, the dataset audit and the released checkpoint's
config, renders one start frame per condition, and inlines everything into
scripts/report_page_visual.html.

    python scripts/build_visual_report.py --dump-json reports/pushobj_T16k_bias/report_data.json
"""
import argparse
import base64
import csv
import io
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# id, label, kind, description, overrides (as passed to plan.py)
CONDITIONS = [
    ("default", "默认", "default", "训练时的渲染", {}),
    ("blur", "blur", "corruption", "高斯模糊 σ=2", {"ood_corruption": "blur"}),
    ("snp", "snp", "corruption", "椒盐噪声 1%", {"ood_corruption": "snp1"}),
    ("dark", "dark", "corruption", "亮度 ×0.9", {"ood_corruption": "dark", "ood_level": 0.9}),
    ("redAgent", "红色推杆", "color", "推杆 蓝 → 红", {"agent_color": "Red"}),
    ("redBlock", "红色物体", "color", "被推的 T 灰 → 红", {"color": "Red"}),
    ("redAnchor", "红色锚点", "color", "固定的目标 T 绿 → 红", {"goal_color": "Red"}),
]


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


def cumulative_adajepa(path, n_iter, n=50):
    first = {}
    for r in rows(path):
        if r.get("success") and r["sample_idx"] not in first:
            first[r["sample_idx"]] = r["step"]
    return [sum(1 for s in first.values() if s <= k) / n for k in range(1, n_iter + 1)]


def adaptation_curve(path):
    by = defaultdict(list)
    if Path(path).exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                by[int(r["mpc_iter"])].append(float(r["loss"]))
    steps = sorted(by)
    return {"step": steps, "mean": [float(np.mean(by[k])) for k in steps], "n": [len(by[k]) for k in steps]}


def thumbnails(seed, targets_path):
    """Start frame of the first evaluated target under every condition, as the paper's Figure 2 shows."""
    import gym
    import env  # noqa: F401
    import pickle
    from PIL import Image
    from planning.image_corruption import corrupt_frames
    segs = pickle.load(open(targets_path, "rb"))["segments"]
    chosen = random.Random(seed).sample(segs, 50)[0]  # same draw as plan.py (goal_source=segments)
    init = np.zeros(7)
    init[:5] = np.asarray(chosen["states"][0], dtype=np.float64)[:5]
    out = {}
    for cid, _, _, _, ov in CONDITIONS:
        e = gym.make("pushobj", with_velocity=True, with_target=True).unwrapped
        info = {"shape": chosen.get("shape", "T")}
        info.update({k: v for k, v in ov.items() if k in ("color", "agent_color", "goal_color")})
        e.update_env(info)
        obs, _ = e.prepare(1, init)  # plan.py's first eval seed: seed * 0 + 1
        frame = obs["visual"]
        if "ood_corruption" in ov:
            frame = corrupt_frames(frame, ov["ood_corruption"], ov.get("ood_level"), seed=seed + 1)
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="PNG")
        out[cid] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        e.close()
    return out


def cfg_rows(ours, rel_visual, rel_shape):
    spec = [("环境", ("env", "name")), ("训练轮数（配置）", ("training", "epochs")), ("曲率正则 straighten", ("training", "straighten")),
            ("有 decoder / 训练 decoder", None), ("有效 batch", ("effective_batch_size",)), ("每卡 batch", ("gpu_batch_size",)),
            ("encoder lr", ("training", "encoder_lr")), ("predictor lr", ("training", "predictor_lr")),
            ("action/proprio encoder lr", ("training", "action_encoder_lr")), ("历史帧 / frameskip", None),
            ("编码器", ("encoder", "_target_")), ("预测器层数 / 头数", None), ("混合精度", ("training", "mixed_precision")),
            ("训练/验证切分", ("env", "dataset", "split_ratio"))]
    out = []
    for name, keys in spec:
        vals = []
        for c in (ours, rel_visual, rel_shape):
            if c is None:
                vals.append(None)
            elif name.startswith("有 decoder"):
                vals.append(f"{c.get('has_decoder')} / {c.get('model', {}).get('train_decoder')}")
            elif name.startswith("历史帧"):
                vals.append(f"{c.get('num_hist')} / {c.get('frameskip')}")
            elif name.startswith("预测器"):
                vals.append(f"{c['predictor'].get('depth')} / {c['predictor'].get('heads')}")
            else:
                v = _get(c, keys)
                vals.append(v.split(".")[-1] if isinstance(v, str) and v.startswith("models.") else v)
        out.append({"name": name, "ours": vals[0], "released_visual": vals[1], "released_shape": vals[2],
                    "differs": vals[1] is not None and str(vals[0]) != str(vals[1])})
    return out


def _get(c, keys):
    for k in keys:
        if not isinstance(c, dict) or k not in c:
            return None
        c = c[k]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="eval_outputs/pushobj_T16k_bias_e3")
    ap.add_argument("--model", default="checkpoints/pushobj_T16k_bias_e3_seed0")
    ap.add_argument("--data", default="data/pushobj_T16k_bias")
    ap.add_argument("--report", default="reports/pushobj_T16k_bias")
    ap.add_argument("--paper", default="reports/pushobj_T16k_bias/paper_fig2_visual_readout.json")
    ap.add_argument("--released-hydra", default="reports/pushobj_T16k_bias/released_pusht_visual_shift_hydra.yaml")
    ap.add_argument("--shape-hydra", default="checkpoints/pushobj_shape_shift/hydra.yaml")
    ap.add_argument("--train-config", default="train_pushobj_T16k")
    ap.add_argument("--seed-dirs", nargs="*", default=[],
                    help="eval dirs of the same model on further test seeds (<dir>/<condition>/<arm>); pooled with --eval-dir")
    ap.add_argument("--ablation", default=None,
                    help="second model evaluated on the same targets, as label:eval_dir:model_dir (e.g. the cos1e-3 run)")
    ap.add_argument("--released-facts", default="reports/pushobj_T16k_bias/released_visual_ckpt_facts.json")
    ap.add_argument("--latent-geometry", default="reports/pushobj_T16k_bias/latent_geometry.json")
    ap.add_argument("--probe", default="reports/pushobj_T16k_bias/adaptivity_visual.json",
                    help="diagnose_adaptation.py output with the released visual model")
    ap.add_argument("--probe-reference", nargs="*", default=["reports/pushobj_replay_20k_e5/diagnostics/adaptivity.json",
                                                             "reports/pushobj_bias_16k/adaptivity.json"])
    ap.add_argument("--template", default="scripts/report_page_visual.html")
    ap.add_argument("--out", default=None, help="default: <report>/report.html")
    ap.add_argument("--dump-json", default=None)
    a = ap.parse_args()
    R = lambda p: (ROOT / p).resolve()
    ev, model, data, report = R(a.eval_dir), R(a.model), R(a.data), R(a.report)
    cfg = yaml.safe_load((model / "hydra.yaml").read_text())
    gen = json.loads((data / "generation_config.json").read_text())
    audit = json.loads((report / "dataset_audit.json").read_text())
    paper = json.loads(R(a.paper).read_text())
    plan_cfg = yaml.safe_load((ev / "default" / "adajepa" / ".hydra" / "config.yaml").read_text())
    n_iter = int(plan_cfg["planner"]["max_iter"])
    seed = int(plan_cfg["seed"])

    conds, ref_state = [], None
    for cid, label, kind, desc, ov in CONDITIONS:
        f, d = outcome(ev / cid / "frozen"), outcome(ev / cid / "adajepa")
        if f is None or d is None:
            continue
        ref_state = f["state_0"] if ref_state is None else ref_state
        for o in (f, d):
            assert np.array_equal(o["state_0"], ref_state), f"targets differ: {cid}"
        fs, ds = f["success"], d["success"]
        b = sum(1 for i in range(50) if ds[i] and not fs[i])
        c = sum(1 for i in range(50) if fs[i] and not ds[i])
        fc = [r["mpc/success_rate"] for r in rows(ev / cid / "frozen" / "logs.json") if "mpc/success_rate" in r][:n_iter]
        ac = cumulative_adajepa(ev / cid / "adajepa" / "per_iter_logs.json", n_iter)
        meta = lambda arm: json.loads((ev / cid / arm / "run_metadata.json").read_text()) if (ev / cid / arm / "run_metadata.json").exists() else {}
        conds.append({"id": cid, "label": label, "kind": kind, "desc": desc,
                      "frozen": sum(fs) / 50, "adajepa": sum(ds) / 50, "frozen_k": sum(fs), "adajepa_k": sum(ds),
                      "frozen_ci": wilson(sum(fs), 50), "adajepa_ci": wilson(sum(ds), 50),
                      "paired": {"both": sum(1 for i in range(50) if fs[i] and ds[i]), "adajepa_only": b, "frozen_only": c,
                                 "neither": sum(1 for i in range(50) if not fs[i] and not ds[i]), "p": mcnemar(b, c)},
                      "frozen_curve": fc, "adajepa_curve": ac, "frozen_20": fc[19], "adajepa_20": ac[19],
                      "outcomes": {"frozen": fs, "adajepa": ds}, "adapt": adaptation_curve(ev / cid / "adajepa" / "adajepa_losses.csv"),
                      "minutes": {"frozen": meta("frozen").get("wall_seconds", 0) / 60, "adajepa": meta("adajepa").get("wall_seconds", 0) / 60},
                      "paper": {"gd": paper["gd"].get(cid), "cem": paper["cem"].get(cid)}})

    def group(ids):
        sub = [c for c in conds if c["id"] in ids]
        if not sub:
            return None
        fs = [x for c in sub for x in c["outcomes"]["frozen"]]
        ds = [x for c in sub for x in c["outcomes"]["adajepa"]]
        b = sum(1 for x, y in zip(fs, ds) if y and not x)
        c_ = sum(1 for x, y in zip(fs, ds) if x and not y)
        pf = [c["paper"]["gd"] for c in sub if c["paper"]["gd"]]
        return {"n": len(fs), "conditions": len(sub), "frozen": float(np.mean(fs)), "adajepa": float(np.mean(ds)),
                "frozen_20": float(np.mean([c["frozen_20"] for c in sub])), "adajepa_20": float(np.mean([c["adajepa_20"] for c in sub])),
                "adajepa_only": b, "frozen_only": c_, "p": mcnemar(b, c_),
                "paper": {"frozen_20": np.mean([p[0] for p in pf]) / 100, "adajepa_20": np.mean([p[1] for p in pf]) / 100,
                          "frozen": np.mean([p[2] for p in pf]) / 100, "adajepa": np.mean([p[3] for p in pf]) / 100} if pf else None}

    groups = {"shifts": group([c[0] for c in CONDITIONS if c[0] != "default"]), "all": group([c[0] for c in CONDITIONS]),
              "corruption": group([c[0] for c in CONDITIONS if c[2] == "corruption"]), "color": group([c[0] for c in CONDITIONS if c[2] == "color"])}

    multi = None
    if a.seed_dirs:
        raw = {}
        for d in [ev] + [R(x) for x in a.seed_dirs]:
            sd = int(yaml.safe_load((d / "default" / "adajepa" / ".hydra" / "config.yaml").read_text())["seed"])
            got = {}
            for c in conds:
                f, dd = outcome(d / c["id"] / "frozen"), outcome(d / c["id"] / "adajepa")
                if f is None or dd is None:
                    break
                assert np.array_equal(f["state_0"], dd["state_0"]), f"targets differ: {d}/{c['id']}"
                got[c["id"]] = {"frozen": f["success"], "adajepa": dd["success"],
                                "fc": [r["mpc/success_rate"] for r in rows(d / c["id"] / "frozen" / "logs.json") if "mpc/success_rate" in r][:n_iter],
                                "ac": cumulative_adajepa(d / c["id"] / "adajepa" / "per_iter_logs.json", n_iter)}
            if len(got) == len(conds):
                raw[sd] = got
        seeds = sorted(raw)
        if len(seeds) > 1:
            def pooled(ids):
                fs = [x for sd in seeds for i in ids for x in raw[sd][i]["frozen"]]
                ds = [x for sd in seeds for i in ids for x in raw[sd][i]["adajepa"]]
                b = sum(1 for x, y in zip(fs, ds) if y and not x)
                c_ = sum(1 for x, y in zip(fs, ds) if x and not y)
                return {"n": len(fs), "frozen": float(np.mean(fs)), "adajepa": float(np.mean(ds)),
                        "frozen_ci": wilson(sum(fs), len(fs)), "adajepa_ci": wilson(sum(ds), len(ds)),
                        "adajepa_only": b, "frozen_only": c_, "p": mcnemar(b, c_),
                        "frozen_20": float(np.mean([raw[sd][i]["fc"][19] for sd in seeds for i in ids])),
                        "adajepa_20": float(np.mean([raw[sd][i]["ac"][19] for sd in seeds for i in ids])),
                        "per_seed": {sd: {"frozen": float(np.mean([np.mean(raw[sd][i]["frozen"]) for i in ids])),
                                          "adajepa": float(np.mean([np.mean(raw[sd][i]["adajepa"]) for i in ids]))} for sd in seeds}}
            ids_all = [c["id"] for c in conds]
            mconds = {i: dict(pooled([i]), frozen_curve=np.mean([raw[sd][i]["fc"] for sd in seeds], 0).tolist(),
                              adajepa_curve=np.mean([raw[sd][i]["ac"] for sd in seeds], 0).tolist()) for i in ids_all}
            sel = {"shifts": [i for i in ids_all if i != "default"], "all": ids_all,
                   "corruption": [c[0] for c in CONDITIONS if c[2] == "corruption"], "color": [c[0] for c in CONDITIONS if c[2] == "color"]}
            mgroups = {g: dict(pooled(ids), paper=groups[g]["paper"] if groups.get(g) else None) for g, ids in sel.items()}
            multi = {"seeds": seeds, "conditions": mconds, "groups": mgroups}

    # training
    batch = int(cfg.get("effective_batch_size", 64))
    ipe = math.ceil(audit["train"]["slices"] / batch)
    epochs = [{"epoch": r["epoch"], "train_z": r["train_z_loss"], "val_z": r["val_z_loss"], "train_vis": r["train_z_visual_loss"],
               "val_vis": r["val_z_visual_loss"], "train_prop": r["train_z_proprio_loss"], "val_prop": r["val_z_proprio_loss"],
               "val_curv": r.get("val_curvature_loss_used_for_training"), "minutes": r["epoch_seconds"] / 60}
              for r in sorted({r["epoch"]: r for r in rows(model / "metrics.jsonl")}.values(), key=lambda r: r["epoch"])]
    iters = [{"epoch": r["epoch"], "iter": r["iter"], "x": (r["epoch"] - 1) + r["iter"] / ipe, "train_z": r["train_z_loss"],
              "train_vis": r["train_z_visual_loss"], "train_prop": r["train_z_proprio_loss"]} for r in rows(model / "metrics_iter.jsonl")]

    ablation = None
    if a.ablation:
        label, adir, mdir = a.ablation.split(":", 2)
        adir, mdir = R(adir), R(mdir)
        acfg = yaml.safe_load((mdir / "hydra.yaml").read_text())

        def cmp(x, y):  # x: main model, y: ablation, same targets in the same order
            b = sum(1 for u, v in zip(x, y) if v and not u)
            c_ = sum(1 for u, v in zip(x, y) if u and not v)
            return {"ablation_only": b, "main_only": c_, "p": mcnemar(b, c_)}

        arows = []
        for c in conds:
            f, d = outcome(adir / c["id"] / "frozen"), outcome(adir / c["id"] / "adajepa")
            if f is None or d is None:
                continue
            for o in (f, d):
                assert np.array_equal(o["state_0"], ref_state), f"targets differ: ablation {c['id']}"
            fs, ds = f["success"], d["success"]
            b = sum(1 for i in range(50) if ds[i] and not fs[i])
            c_ = sum(1 for i in range(50) if fs[i] and not ds[i])
            arows.append({"id": c["id"], "frozen": sum(fs) / 50, "adajepa": sum(ds) / 50, "frozen_k": sum(fs), "adajepa_k": sum(ds),
                          "frozen_ci": wilson(sum(fs), 50), "adajepa_ci": wilson(sum(ds), 50),
                          "paired": {"adajepa_only": b, "frozen_only": c_, "p": mcnemar(b, c_)},
                          "vs_main_frozen": cmp(c["outcomes"]["frozen"], fs), "vs_main_adajepa": cmp(c["outcomes"]["adajepa"], ds),
                          "frozen_curve": [r["mpc/success_rate"] for r in rows(adir / c["id"] / "frozen" / "logs.json") if "mpc/success_rate" in r][:n_iter],
                          "adajepa_curve": cumulative_adajepa(adir / c["id"] / "adajepa" / "per_iter_logs.json", n_iter),
                          "outcomes": {"frozen": fs, "adajepa": ds}})

        def agroup(ids):
            sub = [r for r in arows if r["id"] in ids]
            if not sub:
                return None
            main = {c["id"]: c for c in conds}
            fs = [x for r in sub for x in r["outcomes"]["frozen"]]
            ds = [x for r in sub for x in r["outcomes"]["adajepa"]]
            mf = [x for r in sub for x in main[r["id"]]["outcomes"]["frozen"]]
            md = [x for r in sub for x in main[r["id"]]["outcomes"]["adajepa"]]
            b = sum(1 for x, y in zip(fs, ds) if y and not x)
            c_ = sum(1 for x, y in zip(fs, ds) if x and not y)
            return {"n": len(fs), "conditions": len(sub), "frozen": float(np.mean(fs)), "adajepa": float(np.mean(ds)),
                    "adajepa_only": b, "frozen_only": c_, "p": mcnemar(b, c_),
                    "vs_main_frozen": cmp(mf, fs), "vs_main_adajepa": cmp(md, ds)}

        ablation = {"label": label, "straighten": acfg["training"].get("straighten"),
                    "paths": {"eval": str(adir.relative_to(ROOT)), "model": str(mdir.relative_to(ROOT))},
                    "conditions": arows,
                    "groups": {"shifts": agroup([c[0] for c in CONDITIONS if c[0] != "default"]), "all": agroup([c[0] for c in CONDITIONS])},
                    "epochs": [{"epoch": r["epoch"], "train_z": r["train_z_loss"], "val_z": r["val_z_loss"],
                                "val_curv": r.get("val_curvature_loss_used_for_training"), "minutes": r["epoch_seconds"] / 60}
                               for r in sorted({r["epoch"]: r for r in rows(mdir / "metrics.jsonl")}.values(), key=lambda r: r["epoch"])]}

    def probe_rows(path, tag):
        out = []
        for name, v in json.loads(R(path).read_text())["results"].items():
            o = next(x for k, x in v.items() if k.startswith("ours"))
            au = next(x for k, x in v.items() if k.startswith("author"))
            out.append({"model": name, "tag": tag, "unbiased": o["frozen_rollout_median"], "author": au["frozen_rollout_median"],
                        "ratio": au["frozen_rollout_median"] / o["frozen_rollout_median"],
                        "next1_unbiased": o["next1_ratio_gmean"], "next1_author": au["next1_ratio_gmean"]})
        return out
    released_checks = {
        "facts": json.loads(R(a.released_facts).read_text()) if a.released_facts and R(a.released_facts).exists() else None,
        "geometry": json.loads(R(a.latent_geometry).read_text()) if a.latent_geometry and R(a.latent_geometry).exists() else None,
        "probe": (probe_rows(a.probe, "visual") + [r for f in a.probe_reference if R(f).exists() for r in probe_rows(f, "reference")])
                 if a.probe and R(a.probe).exists() else None,
    }

    rel_visual = yaml.safe_load(R(a.released_hydra).read_text()) if a.released_hydra and R(a.released_hydra).exists() else None
    rel_shape = yaml.safe_load(R(a.shape_hydra).read_text()) if a.shape_hydra and R(a.shape_hydra).exists() else None
    dist = report / "distribution_vs_author.json"
    fid = report / "visual_fidelity.json"
    targets = ROOT / "data/pushobj_eval/val_T/plan_targets.pkl"
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "conditions": conds, "groups": groups, "paper_source": paper["source"],
        "thumbnails": thumbnails(seed, targets),
        "eval": {"n": 50, "seed": seed, "max_iter": n_iter, "horizon": plan_cfg["planner"]["sub_planner"]["horizon"],
                 "n_taken": plan_cfg["planner"]["n_taken_actions"], "opt_steps": plan_cfg["planner"]["sub_planner"]["opt_steps"],
                 "adapt": plan_cfg["planner"]["adapt"]},
        "training": {"epochs": epochs, "iters": iters, "iters_per_epoch": ipe, "batch": batch,
                     "gpu_batch": int(cfg.get("gpu_batch_size", batch)), "straighten": cfg["training"].get("straighten"),
                     "has_decoder": bool(cfg.get("has_decoder", True)),
                     "lr": {"encoder": cfg["training"]["encoder_lr"], "predictor": cfg["training"]["predictor_lr"]}},
        "config_compare": cfg_rows(cfg, rel_visual, rel_shape), "ablation": ablation, "released_checks": released_checks, "multi": multi,
        "data": {"train": audit["train"], "val": audit["val"], "shared_sources": audit.get("shared_source_trajectories_across_splits"),
                 "duplicates": audit.get("duplicate_trajectories_across_splits"),
                 "gen": {k: gen.get(k) for k in ("action_bias_px", "action_noise_px", "placement", "train_pool", "max_draws",
                                                 "train_per_shape", "val_per_shape", "source_episodes", "source_excluded_agent_leaves_frame")},
                 "summary": gen["summary"],
                 "distribution": json.loads(dist.read_text()) if dist.exists() else None,
                 "fidelity": json.loads(fid.read_text()) if fid.exists() else None},
        "paths": {"eval": a.eval_dir, "model": a.model, "data": a.data, "report": a.report, "train_config": a.train_config},
    }
    html = R(a.template).read_text()
    html = html.replace("__REPORT_DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=float))
    paper_png = report / "paper_figure2_visual.png"
    html = html.replace("__PAPER_FIG__", "data:image/png;base64," + base64.b64encode(paper_png.read_bytes()).decode() if paper_png.exists() else "")
    if a.dump_json:
        Path(a.dump_json).write_text(json.dumps({k: v for k, v in payload.items() if k != "thumbnails"}, ensure_ascii=False, indent=1, default=float))
    out = Path(a.out) if a.out else report / "report.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB); conditions: {[c['id'] for c in conds]}")


if __name__ == "__main__":
    main()
