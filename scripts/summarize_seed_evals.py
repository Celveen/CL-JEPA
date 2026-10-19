"""Paired comparison of our model and the released checkpoint across test seeds.

Each test seed draws a different 50 of the 1,000 author segments per shape. For
every seed, both models are compared target by target (exact McNemar); seeds are
then pooled, as the paper averages 3 test seeds x 50.

    python scripts/summarize_seed_evals.py --out reports/pushobj_bias_16k/seed_eval_summary.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ("T", "L", "Z", "+", "I", "small_tee", "square")
SEEN = SHAPES[:4]


def load(d):
    p = ROOT / d / "final_eval_results.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["successes"].reshape(-1).astype(bool), z["state_0"]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - r, c + r]


def compare(pairs):
    """pairs: list of (ours, released) boolean arrays on identical targets."""
    o = np.concatenate([p[0] for p in pairs])
    r = np.concatenate([p[1] for p in pairs])
    b, c = int((o & ~r).sum()), int((r & ~o).sum())
    return {"n": len(o), "ours": float(o.mean()), "released": float(r.mean()), "ours_ci": wilson(int(o.sum()), len(o)),
            "released_ci": wilson(int(r.sum()), len(r)), "ours_only": b, "released_only": c, "p": mcnemar(b, c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100,200,300")
    ap.add_argument("--out", default="reports/pushobj_bias_16k/seed_eval_summary.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    ours_dir = lambda seed, s, arm: (f"eval_outputs/pushobj_bias_16k_e3/{s}/{arm}" if seed == 100
                                     else f"eval_outputs/pushobj_bias_16k_e3_seed{seed}/{s}/{arm}")
    rel_dir = lambda seed, s: (f"eval_outputs/pushobj_released_reference/{s}" if seed == 100
                               else f"eval_outputs/pushobj_released_reference_seed{seed}/{s}")
    frozen = {}  # (seed, shape) -> (ours, released)
    for seed in seeds:
        for s in SHAPES:
            o, r = load(ours_dir(seed, s, "frozen")), load(rel_dir(seed, s))
            if o is None or r is None:
                continue
            assert np.array_equal(o[1], r[1]), f"targets differ: seed {seed} {s}"
            frozen[seed, s] = (o[0], r[0])
    groups = {"seen": SEEN, "unseen": SHAPES[4:], "all": SHAPES}
    out = {"seeds": seeds, "frozen": {"per_seed": {}, "per_shape_pooled": {}, "pooled": {}}}
    for seed in seeds:
        if all((seed, s) in frozen for s in SHAPES):
            out["frozen"]["per_seed"][seed] = {g: compare([frozen[seed, s] for s in sub]) for g, sub in groups.items()}
            out["frozen"]["per_seed"][seed]["shapes"] = {s: compare([frozen[seed, s]]) for s in SHAPES}
    done = [seed for seed in seeds if seed in out["frozen"]["per_seed"]]
    for s in SHAPES:
        out["frozen"]["per_shape_pooled"][s] = compare([frozen[seed, s] for seed in done])
    for g, sub in groups.items():
        out["frozen"]["pooled"][g] = compare([frozen[seed, s] for seed in done for s in sub])
    out["frozen"]["pooled_seeds"] = done

    # AdaJEPA at seed 100: released vs ours, and each model's gain over its own frozen run
    ada = {}
    for s in SHAPES:
        oa, ra = load(ours_dir(100, s, "adajepa")), load(f"eval_outputs/pushobj_released_reference_adajepa/{s}")
        if oa is None or ra is None:
            continue
        assert np.array_equal(oa[1], ra[1]), f"targets differ: adajepa {s}"
        of, rf = frozen[100, s]
        ada[s] = {"vs_released": compare([(oa[0], ra[0])]),
                  "ours_gain": compare([(oa[0], of)]), "released_gain": compare([(ra[0], rf)])}
    out["adajepa_seed100"] = {"shapes": ada}
    if len(ada) == len(SHAPES):
        for g, sub in groups.items():
            pick = lambda f: [f(s) for s in sub]
            oa = [load(ours_dir(100, s, "adajepa"))[0] for s in sub]
            ra = [load(f"eval_outputs/pushobj_released_reference_adajepa/{s}")[0] for s in sub]
            out["adajepa_seed100"][g] = {"vs_released": compare(list(zip(oa, ra))),
                                         "ours_gain": compare(list(zip(oa, pick(lambda s: frozen[100, s][0])))),
                                         "released_gain": compare(list(zip(ra, pick(lambda s: frozen[100, s][1]))))}
    path = ROOT / a.out
    path.write_text(json.dumps(out, indent=2))

    f = out["frozen"]
    print("FROZEN  (ours vs released, same targets)")
    print(f"{'seed':>6} | {'seen':>15} | {'unseen':>15} | {'all':>15} | ours-only / released-only, p (all)")
    for seed in done:
        r = f["per_seed"][seed]
        cell = lambda g: f"{r[g]['ours']:.1%} / {r[g]['released']:.1%}"
        print(f"{seed:>6} | {cell('seen'):>15} | {cell('unseen'):>15} | {cell('all'):>15} | "
              f"{r['all']['ours_only']} / {r['all']['released_only']}, p={r['all']['p']:.2g}")
    r = f["pooled"]
    print(f"{'pooled':>6} | {r['seen']['ours']:.1%} / {r['seen']['released']:.1%} | {r['unseen']['ours']:.1%} / {r['unseen']['released']:.1%} | "
          f"{r['all']['ours']:.1%} / {r['all']['released']:.1%} | {r['all']['ours_only']} / {r['all']['released_only']}, p={r['all']['p']:.2g}")
    print("\nper shape, pooled over seeds", done)
    for s in SHAPES:
        x = f["per_shape_pooled"][s]
        per = "  ".join(f"{int(round(f['per_seed'][seed]['shapes'][s]['ours'] * 50)):2d}/{int(round(f['per_seed'][seed]['shapes'][s]['released'] * 50)):2d}" for seed in done)
        print(f"{s:>10}: ours {x['ours']:.1%}  released {x['released']:.1%}  ({x['ours_only']}/{x['released_only']}, p={x['p']:.2g})   per seed ours/released: {per}")
    if ada:
        print("\nADAJEPA seed 100: frozen -> AdaJEPA")
        for s, x in ada.items():
            print(f"{s:>10}: ours {x['ours_gain']['released']:.0%} -> {x['ours_gain']['ours']:.0%} (p={x['ours_gain']['p']:.2g})   "
                  f"released {x['released_gain']['released']:.0%} -> {x['released_gain']['ours']:.0%} (p={x['released_gain']['p']:.2g})   "
                  f"ours vs released AdaJEPA {x['vs_released']['ours_only']}/{x['vs_released']['released_only']} p={x['vs_released']['p']:.2g}")
        for g in ("seen", "unseen", "all"):
            if g in out["adajepa_seed100"]:
                x = out["adajepa_seed100"][g]
                print(f"{g:>10}: ours {x['ours_gain']['released']:.1%} -> {x['ours_gain']['ours']:.1%} (p={x['ours_gain']['p']:.2g})   "
                      f"released {x['released_gain']['released']:.1%} -> {x['released_gain']['ours']:.1%} (p={x['released_gain']['p']:.2g})   "
                      f"AdaJEPA ours vs released {x['vs_released']['ours_only']}/{x['vs_released']['released_only']} p={x['vs_released']['p']:.2g}")
    print("wrote", path)


if __name__ == "__main__":
    main()
