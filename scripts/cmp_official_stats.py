"""Statistics of the comparison: official AdaJEPA shape weights vs v6a frozen at round 5, on the held-out stream H7t3.

An experiment = (model, planner, test seed): 7 shapes x 50 episodes = 350 episodes. Every model runs the same
episodes, so comparisons are paired. v6a has several round-5 states per planner (GD: 2, CEM: 3); its numbers are
averaged over states x seeds, the official numbers over seeds. Paired differences come with a 95% bootstrap
interval stratified by seed and a McNemar test per v6a state; also success within a replan budget and the share of
episodes in which frozen v6a read a learner. Works on partial runs (chunks every run of a planner has finished).

The runs (README_CL-JEPA.md) write to eval_outputs/cmp_<planner>_<model>/, e.g.
    python scripts/cont_run_stream.py --stream H7t3 --root eval_outputs/cmp_gd_official --ws 0 --independent \
        --ckpt checkpoints/pushobj_shape_shift --epoch latest --no-base-probes --extra planner.sub_planner.cuda_graph=true
    python scripts/cont_run_stream.py --stream H7t3 --root eval_outputs/cmp_gd_v6a_s400 --arms cls6a_free --independent \
        --state-in eval_outputs/cont_S7t3s400/cls6a_free/chunk_34_T/continual_state.pt --no-base-probes ...

    python scripts/cmp_official_stats.py            # -> reports/cmp/report_data.json (read by scripts/fig_cmp_official.py)
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = ROOT  # output roots below are relative to the repository
SEEN = {"T", "L", "Z", "+"}
SHAPES = ["I", "L", "square", "Z", "small_tee", "+", "T"]
STREAM = "H7t3"
RUNS = {  # planner -> model -> list of (run id, label, output dir of the arm)
    "gd": {"official": [("gd_official", "官方", W / "eval_outputs/cmp_gd_official/w0")],
           "v6a": [("gd_v6a_s400", "v6a · S7t3s400", W / "eval_outputs/cmp_gd_v6a_s400/cls6a_free"),
                   ("gd_v6a_s700", "v6a · S7t3s700", W / "eval_outputs/cmp_gd_v6a_s700/cls6a_free")]},
    "cem": {"official": [("cem_official", "官方", W / "eval_outputs/cmp_cem_official/w0")],
            "v6a": [("cem_v6a_s100", "v6a · S7t3", W / "eval_outputs/cmp_cem_v6a_s100/cls6a_free"),
                    ("cem_v6a_s400", "v6a · S7t3s400", W / "eval_outputs/cmp_cem_v6a_s400/cls6a_free"),
                    ("cem_v6a_s700", "v6a · S7t3s700", W / "eval_outputs/cmp_cem_v6a_s700/cls6a_free")]},
}
BUDGETS = [5, 10, 15, 20, 25, 30]
blocks = json.loads((W / "data/cont_streams" / STREAM / "manifest.json").read_text())["blocks"]
SEEDS = sorted({b["seed"] for b in blocks})


def mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(d):
    """Per chunk: successes, replans used (inf = failed) and, for v6a, what each episode read."""
    out = {}
    for b in blocks:
        c = d / b["chunk"]
        logs = c / "logs.json"
        if not ((c / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()):
            continue
        z = np.load(c / "final_eval_results.npz")
        reads = None
        if (c / "continual_episodes.jsonl").exists():
            reads = [json.loads(l).get("read") for l in open(c / "continual_episodes.jsonl") if l.strip()]
        out[b["chunk"]] = {"succ": z["successes"].astype(int), "len": z["action_len"].astype(float), "reads": reads}
    return out


bmeta = {b["chunk"]: b for b in blocks}


def sel(run, chunks, seed=None, shape=None, group=None):
    ks = [k for k in chunks if (seed is None or bmeta[k]["seed"] == seed) and (shape is None or bmeta[k]["shape"] == shape)
          and (group is None or (group == "seen") == (bmeta[k]["shape"] in SEEN))]
    if not ks:
        return None, None
    return np.concatenate([run[k]["succ"] for k in ks]), np.concatenate([run[k]["len"] for k in ks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/cmp/report_data.json")
    out_path = ROOT / ap.parse_args().out
    data = {p: {m: [(rid, lab, load(d)) for rid, lab, d in runs] for m, runs in ms.items()} for p, ms in RUNS.items()}
    # chunks finished by every run of a planner: all statistics of that planner use exactly these episodes
    common = {p: [b["chunk"] for b in blocks if all(b["chunk"] in r[2] for runs in ms.values() for r in runs)]
              for p, ms in data.items()}
    GROUPS = {"all": None, "seen": "seen", "unseen": "unseen"}
    out = {"generated": dt.datetime.now().strftime("%F %T"), "seeds": SEEDS, "shapes": [{"id": s, "seen": s in SEEN} for s in SHAPES],
           "planners": {}}
    rng = np.random.default_rng(0)

    for p, ms in data.items():
        ch = common[p]
        P = {"n_chunks": len(ch), "runs": {}, "models": {}, "diffs": {}, "budget": {}, "reads": {}}
        # experiments: (run, seed) rates, per group and per shape
        for m, runs in ms.items():
            for rid, lab, run in runs:
                ex = {}
                for s in SEEDS:
                    e = {}
                    for g, gg in GROUPS.items():
                        x, _ = sel(run, ch, seed=s, group=gg)
                        e[g] = None if x is None else round(100 * x.mean(), 1)
                    for sh in SHAPES:
                        x, _ = sel(run, ch, seed=s, shape=sh)
                        e[sh] = None if x is None else round(100 * x.mean(), 1)
                    x, _ = sel(run, ch, seed=s)
                    e["n"] = 0 if x is None else int(len(x))
                    ex[str(s)] = e
                P["runs"][rid] = {"model": m, "label": lab, "experiments": ex}
        # model level: mean over (run, seed) experiments, with min/max for the spread
        for m, runs in ms.items():
            agg = {}
            for key in list(GROUPS) + SHAPES:
                vals = [P["runs"][rid]["experiments"][str(s)][key] for rid, _, _ in runs for s in SEEDS]
                vals = [v for v in vals if v is not None]
                agg[key] = {"mean": round(float(np.mean(vals)), 1) if vals else None,
                            "min": min(vals) if vals else None, "max": max(vals) if vals else None, "n_exp": len(vals)}
            P["models"][m] = agg
        # paired differences on identical episodes; for v6a the difference is averaged over its states
        pairs = [("v6a", "official")]
        for a, b in pairs:
            res = {}
            for g, gg in list(GROUPS.items()) + [(sh, None) for sh in SHAPES]:
                shape = g if g in SHAPES else None
                ys = [sel(r[2], ch, shape=shape, group=gg)[0] for r in ms[b]]
                xs = [sel(r[2], ch, shape=shape, group=gg)[0] for r in ms[a]]
                if xs[0] is None or ys[0] is None:
                    res[g] = None
                    continue
                y = np.mean(ys, axis=0)  # the official model has one run
                d = np.mean([x - y for x in xs], axis=0)  # per episode, averaged over v6a states
                # bootstrap over episodes, stratified by seed (same episodes for every model)
                seeds = np.concatenate([[bmeta[k]["seed"]] * bmeta[k]["n"] for k in ch
                                        if (shape is None or bmeta[k]["shape"] == shape) and (gg is None or (gg == "seen") == (bmeta[k]["shape"] in SEEN))])
                idx_by = [np.where(seeds == s)[0] for s in SEEDS if (seeds == s).any()]
                boots = np.zeros(10000)
                for ix in idx_by:
                    boots += d[ix][rng.integers(len(ix), size=(10000, len(ix)))].sum(1)
                boots = boots / len(d) * 100
                per_run = []
                for (rid, lab, run), x in zip(ms[a], xs):
                    bb, cc = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
                    per_run.append({"run": rid, "label": lab, "diff": round(100 * (x.mean() - y.mean()), 1), "only_a": bb, "only_b": cc,
                                    "p": mcnemar_p(bb, cc)})
                res[g] = {"diff": round(100 * float(d.mean()), 1), "ci": [round(float(np.percentile(boots, 2.5)), 1), round(float(np.percentile(boots, 97.5)), 1)],
                          "n": int(len(d)), "per_run": per_run, "p_max": max(r["p"] for r in per_run)}
            P["diffs"][f"{a}-{b}"] = res
        # success within a replan budget (seen / unseen), averaged over runs of the model
        for m, runs in ms.items():
            P["budget"][m] = {}
            for g, gg in GROUPS.items():
                curves = []
                for _, _, run in runs:
                    x, ln = sel(run, ch, group=gg)
                    if x is not None:
                        curves.append([float(np.mean((x == 1) & (ln <= k)) * 100) for k in BUDGETS])
                P["budget"][m][g] = [round(v, 1) for v in np.mean(curves, axis=0)] if curves else None
            P["budget"]["ks"] = BUDGETS
        # what frozen v6a read, per shape (share of episodes that planned with a learner), and success by read
        for rid, lab, run in ms["v6a"]:
            r = {}
            for sh in SHAPES:
                reads = [x for k in ch if bmeta[k]["shape"] == sh for x in (run[k]["reads"] or [])]
                r[sh] = round(100 * float(np.mean([x is not None for x in reads])), 1) if reads else None
            P["reads"][rid] = r
        out["planners"][p] = P

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False))
    print(out_path, f"{out_path.stat().st_size / 1024:.0f} KB")
    for p, P in out["planners"].items():
        M = P["models"]
        print(f"== {p}: {P['n_chunks']}/21 chunks common to all runs")
        for m in ("official", "v6a"):
            print(f"   {m:8s} all {M[m]['all']['mean']}  seen {M[m]['seen']['mean']}  unseen {M[m]['unseen']['mean']}  (experiments {M[m]['all']['n_exp']})")
        for k, v in P["diffs"].items():
            print(f"   {k:16s}", {g: (v[g]['diff'], v[g]['ci']) for g in ('all', 'seen', 'unseen') if v.get(g)})


if __name__ == "__main__":
    main()
