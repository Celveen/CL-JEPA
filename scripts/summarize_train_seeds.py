"""Planning success across training seeds x test seeds for the contact-bias models.

Training seed 0 lives in the original eval dirs; seeds >= 1 in
eval_outputs/pushobj_bias_16k_e3_train<ts>/test<seed>_<it20|it30>/<shape>/<arm>.
Test seed 100 runs at most 20 replans (paired with the released checkpoint); test
seeds 200/300 run at most 30 replans as in the paper's Figure 2. A test seed picks the
same 50 targets per shape for every model, so all comparisons are paired.

    python scripts/summarize_train_seeds.py --out reports/pushobj_bias_16k/train_seed_summary.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ("T", "L", "Z", "+", "I", "small_tee", "square")
GROUPS = {"seen": SHAPES[:4], "unseen": SHAPES[4:], "all": SHAPES}
TESTS = ((100, "it20"), (200, "it30"), (300, "it30"))


def eval_dir(ts, test_seed, tag, shape, arm):
    if ts == 0:
        base = {100: "eval_outputs/pushobj_bias_16k_e3", 200: "eval_outputs/pushobj_bias_16k_e3_seed200_it30",
                300: "eval_outputs/pushobj_bias_16k_e3_seed300_it30"}[test_seed]
        return ROOT / base / shape / arm
    return ROOT / f"eval_outputs/pushobj_bias_16k_e3_train{ts}/test{test_seed}_{tag}/{shape}/{arm}"


def released_dir(test_seed, shape, arm):
    if arm == "adajepa":
        return ROOT / "eval_outputs/pushobj_released_reference_adajepa" / shape if test_seed == 100 else None
    suffix = "" if test_seed == 100 else f"_seed{test_seed}"
    return ROOT / f"eval_outputs/pushobj_released_reference{suffix}" / shape


def load(d):
    if d is None or not (d / "final_eval_results.npz").exists():
        return None
    with np.load(d / "final_eval_results.npz") as z:
        return z["successes"].reshape(-1).astype(bool), z["state_0"]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def paired(x, y):
    """x, y: boolean arrays on identical targets -> rates, discordant counts, exact McNemar p."""
    b, c = int((y & ~x).sum()), int((x & ~y).sum())
    return {"x": float(x.mean()), "y": float(y.mean()), "y_only": b, "x_only": c, "p": mcnemar(b, c), "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", default="0,1,2")
    ap.add_argument("--paper", default="reports/pushobj_bias_16k/paper_fig2_readout.json")
    ap.add_argument("--out", default="reports/pushobj_bias_16k/train_seed_summary.json")
    a = ap.parse_args()
    paper = json.loads((ROOT / a.paper).read_text())["gd"]
    res, targets = {}, {}
    for ts in [int(s) for s in a.train_seeds.split(",")]:
        for test_seed, tag in TESTS:
            for s in SHAPES:
                for arm in ("frozen", "adajepa"):
                    r = load(eval_dir(ts, test_seed, tag, s, arm))
                    if r is None:
                        continue
                    key = (test_seed, s)
                    if key in targets:
                        assert np.array_equal(targets[key], r[1]), f"targets differ: ts{ts} {key}"
                    targets[key] = r[1]
                    res[ts, test_seed, s, arm] = r[0]
    seeds = sorted({k[0] for k in res})
    complete = [ts for ts in seeds if all((ts, t, s, arm) in res for t, _ in TESTS for s in SHAPES for arm in ("frozen", "adajepa"))]
    cat = lambda ts, tests, sub, arm: np.concatenate([res[ts, t, s, arm] for t in tests for s in sub])
    out = {"complete_train_seeds": complete, "incomplete": [s for s in seeds if s not in complete], "per_seed": {}, "across": {}}
    for ts in complete:
        d = out["per_seed"][ts] = {}
        for label, tests in (("test100_it20", (100,)), ("test200_300_it30", (200, 300))):
            d[label] = {g: paired(cat(ts, tests, sub, "frozen"), cat(ts, tests, sub, "adajepa")) for g, sub in GROUPS.items()}
            d[label]["shapes"] = {s: paired(cat(ts, tests, (s,), "frozen"), cat(ts, tests, (s,), "adajepa")) for s in SHAPES}
    # released checkpoint on the same targets: frozen at 100/200/300 (20 replans), AdaJEPA at 100
    rel = {}
    for t, _ in TESTS:
        for s in SHAPES:
            for arm in ("frozen", "adajepa"):
                r = load(released_dir(t, s, arm))
                if r is not None:
                    assert np.array_equal(targets[t, s], r[1]), f"released targets differ: {t} {s}"
                    rel[t, s, arm] = r[0]
    for ts in complete:
        d = out["per_seed"][ts]
        d["vs_released"] = {
            "frozen_test100_200_300": {g: paired(np.concatenate([rel[t, s, "frozen"] for t, _ in TESTS for s in sub]),
                                                 cat(ts, (100, 200, 300), sub, "frozen")) for g, sub in GROUPS.items()},
            "adajepa_test100": {g: paired(np.concatenate([rel[100, s, "adajepa"] for s in sub]), cat(ts, (100,), sub, "adajepa"))
                                for g, sub in GROUPS.items()}}
    # across training seeds: mean and spread of the per-seed rates
    for label in ("test100_it20", "test200_300_it30"):
        A = out["across"][label] = {}
        for g in list(GROUPS) + list(SHAPES):
            pick = lambda ts: out["per_seed"][ts][label][g] if g in GROUPS else out["per_seed"][ts][label]["shapes"][g]
            fz = [pick(ts)["x"] for ts in complete]
            ad = [pick(ts)["y"] for ts in complete]
            A[g] = {"frozen": fz, "adajepa": ad, "frozen_mean": float(np.mean(fz)), "adajepa_mean": float(np.mean(ad)),
                    "frozen_sd": float(np.std(fz, ddof=1)) if len(fz) > 1 else 0.0,
                    "adajepa_sd": float(np.std(ad, ddof=1)) if len(ad) > 1 else 0.0}
            if label == "test200_300_it30":
                sub = GROUPS.get(g, (g,))
                A[g]["paper_frozen"] = float(np.mean([paper[s][2] for s in sub])) / 100
                A[g]["paper_adajepa"] = float(np.mean([paper[s][3] for s in sub])) / 100
    out["released"] = {g: {"frozen_test100_200_300": float(np.mean(np.concatenate([rel[t, s, "frozen"] for t, _ in TESTS for s in sub]))),
                           "adajepa_test100": float(np.mean(np.concatenate([rel[100, s, "adajepa"] for s in sub])))}
                       for g, sub in GROUPS.items()}
    (ROOT / a.out).write_text(json.dumps({**out, "per_seed": {str(k): v for k, v in out["per_seed"].items()}}, indent=1))

    pc = lambda v: f"{v * 100:5.1f}%"
    print(f"complete training seeds: {complete}; incomplete: {out['incomplete']}")
    print("\n== Paper protocol: test seeds 200+300 pooled (100 targets/shape), up to 30 replans: frozen -> AdaJEPA")
    print(f"{'train seed':>10} | {'seen':>17} | {'unseen':>17} | {'all':>17}")
    for ts in complete:
        d = out["per_seed"][ts]["test200_300_it30"]
        print(f"{ts:>10} | " + " | ".join(f"{pc(d[g]['x'])} -> {pc(d[g]['y'])}" for g in GROUPS))
    A = out["across"]["test200_300_it30"]
    print(f"{'mean±sd':>10} | " + " | ".join(f"{pc(A[g]['frozen_mean'])} -> {pc(A[g]['adajepa_mean'])} ±{A[g]['adajepa_sd'] * 100:.1f}" for g in GROUPS))
    print(f"{'paper':>10} | " + " | ".join(f"{pc(A[g]['paper_frozen'])} -> {pc(A[g]['paper_adajepa'])}" for g in GROUPS))
    print("\n   per shape, AdaJEPA (30 replans) per training seed | mean | paper")
    for s in SHAPES:
        x = A[s]
        print(f"   {s:>9}: " + " ".join(f"{v * 100:3.0f}" for v in x["adajepa"]) + f" | {x['adajepa_mean'] * 100:5.1f} | {x['paper_adajepa'] * 100:3.0f}"
              + f"   (frozen " + " ".join(f"{v * 100:3.0f}" for v in x["frozen"]) + f" | paper {x['paper_frozen'] * 100:3.0f})")
    print("\n== Test seed 100, up to 20 replans: frozen -> AdaJEPA, and the released checkpoint on the same targets")
    for ts in complete:
        d = out["per_seed"][ts]["test100_it20"]
        v = out["per_seed"][ts]["vs_released"]["adajepa_test100"]
        print(f"{ts:>10} | " + " | ".join(f"{g} {pc(d[g]['x'])} -> {pc(d[g]['y'])}" for g in GROUPS)
              + f" | AdaJEPA vs released (ours-only/released-only): seen {v['seen']['y_only']}/{v['seen']['x_only']} p={v['seen']['p']:.2g},"
              f" unseen {v['unseen']['y_only']}/{v['unseen']['x_only']} p={v['unseen']['p']:.2g}")
    R = out["released"]
    print(f"{'released':>10} | AdaJEPA seed100: seen {pc(R['seen']['adajepa_test100'])}, unseen {pc(R['unseen']['adajepa_test100'])};"
          f" frozen over 3 test seeds: seen {pc(R['seen']['frozen_test100_200_300'])}, unseen {pc(R['unseen']['frozen_test100_200_300'])}")
    print("\n== Frozen vs released over test seeds 100/200/300 (same targets)")
    for ts in complete:
        v = out["per_seed"][ts]["vs_released"]["frozen_test100_200_300"]
        print(f"{ts:>10} | " + " | ".join(f"{g} {pc(v[g]['y'])} vs {pc(v[g]['x'])} ({v[g]['y_only']}/{v[g]['x_only']}, p={v[g]['p']:.1g})" for g in GROUPS))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
