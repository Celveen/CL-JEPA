"""Collect every number the five-round report shows into reports/gen5/report_data.json.

v6a ran all five rounds; the AdaJEPA baselines (w0) were stopped during rounds 4-5, so:
  * paired v6a - w0 statistics use only rounds that are complete for both arms in that run;
  * v6a's own trajectory (success per round, per-shape visits, domains, slow-learner effect) uses all its rounds.
cont_rounds.py is run twice per experiment: with w0 (paired reads over the paired prefix) and v6a alone (gates,
slow-learner effect over every finished chunk). Partial runs are fine.

    python reports/gen5/build_data.py
"""
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / "scripts"))
from cont_rounds import SEEN  # noqa: E402

OUT = W / "reports/gen5"
RUNS = [  # id, label, planner, 5-round stream, root
    # GD · S7t3 lives in the main checkout: rounds 1-3 are the first S7t3 batch (eager GD), rounds 4-5 resumed from it
    ("gd_s100", "GD · S7t3", "gd", "S7t5", W.parent / "CL-JEPA/eval_outputs/cont_S7t3"),
    ("gd_s400", "GD · S7t3s400", "gd", "S7t5s400", W / "eval_outputs/cont_S7t3s400"),
    ("gd_s700", "GD · S7t3s700", "gd", "S7t5s700", W / "eval_outputs/cont_S7t3s700"),
    ("cem_s100", "CEM · S7t3", "cem", "S7t5", W / "eval_outputs/cont_S7t3_cem"),
    ("cem_s400", "CEM · S7t3s400", "cem", "S7t5s400", W / "eval_outputs/cont_S7t3s400_cem"),
    ("cem_s700", "CEM · S7t3s700", "cem", "S7t5s700", W / "eval_outputs/cont_S7t3s700_cem"),
]
SHAPES = ["I", "L", "square", "Z", "small_tee", "+", "T"]


def mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def strat_boot(diffs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = [d for d in diffs if len(d)]
    tot = sum(len(d) for d in diffs)
    acc = np.zeros(n)
    for d in diffs:
        acc += d[rng.integers(len(d), size=(n, len(d)))].sum(1)
    acc = acc / tot * 100
    return [round(float(np.percentile(acc, 2.5)), 1), round(float(np.percentile(acc, 97.5)), 1)]


def compare(xs, ys):
    xs, ys = [x for x in xs if len(x)], [y for y in ys if len(y)]
    if not xs:
        return None
    X, Y = np.concatenate(xs), np.concatenate(ys)
    b, c = int(((X == 1) & (Y == 0)).sum()), int(((X == 0) & (Y == 1)).sum())
    return {"v6a": round(100 * X.mean(), 1), "w0": round(100 * Y.mean(), 1), "diff": round(100 * (X.mean() - Y.mean()), 1),
            "ci": strat_boot([x - y for x, y in zip(xs, ys)]), "only_v6a": b, "only_w0": c, "p": mcnemar_p(b, c), "n": int(len(X))}


def chunk(d):
    logs = d / "logs.json"
    if not ((d / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()):
        return None
    return np.load(d / "final_eval_results.npz")["successes"].astype(int)


# ---- load per chunk ------------------------------------------------------------------------------------
R = {}
for rid, label, planner, stream, root in RUNS:
    blocks = json.loads((W / "data/cont_streams" / stream / "manifest.json").read_text())["blocks"]
    vis, seen_before = [], {}
    for b in blocks:
        vis.append(seen_before.get(b["shape"], 0) + 1)
        seen_before[b["shape"]] = vis[-1]
    v6 = [chunk(root / "cls6a_free" / b["chunk"]) for b in blocks]
    w0 = [chunk(root / "w0" / b["chunk"]) for b in blocks]
    R[rid] = {"label": label, "planner": planner, "stream": stream, "root": root, "blocks": blocks, "visit": vis,
              "v6": v6, "w0": w0}


def round_ok(arr, r):
    return all(arr[k] is not None for k in range(7 * (r - 1), 7 * r))


def paired_rounds(d):
    return [r for r in range(1, 6) if round_ok(d["v6"], r) and round_ok(d["w0"], r)]


def sel(d, arm, ks, filt=None):
    """Concatenate episodes of chunks ks (optionally only chunks whose block passes filt)."""
    parts = [d[arm][k] for k in ks if (filt is None or filt(d["blocks"][k], d["visit"][k]))]
    return np.concatenate(parts) if parts else np.array([], int)


GFILT = {"all": None, "seen": lambda b, v: b["shape"] in SEEN, "unseen": lambda b, v: b["shape"] not in SEEN,
         "unseen_revisit": lambda b, v: b["shape"] not in SEEN and v >= 2,
         "unseen_visit1": lambda b, v: b["shape"] not in SEEN and v == 1}
chunks_of = lambda rs: [k for r in rs for k in range(7 * (r - 1), 7 * r)]

out = {"generated": dt.datetime.now().strftime("%F %T"),
       "runs": [{"id": r, "label": d["label"], "planner": d["planner"], "stream": d["stream"],
                 "seeds": [d["blocks"][7 * i]["seed"] for i in range(5)],
                 "v6a_rounds": [r for r in range(1, 6) if round_ok(d["v6"], r)],
                 "w0_rounds": [r for r in range(1, 6) if round_ok(d["w0"], r)],
                 "paired_rounds": paired_rounds(d)} for r, d in R.items()]}

# ---- per round: v6a absolute (all its rounds) and paired diff (rounds complete in both) -------------------
out["per_round"], out["v6a_round"], out["w0_round"] = {}, {}, {}
for rid, d in R.items():
    for g in ("all", "seen", "unseen"):
        pr, va, wa = [], [], []
        for r in range(1, 6):
            ks = chunks_of([r])
            pr.append(compare([sel(d, "v6", ks, GFILT[g])], [sel(d, "w0", ks, GFILT[g])]) if r in paired_rounds(d) else None)
            va.append(round(100 * sel(d, "v6", ks, GFILT[g]).mean(), 1) if round_ok(d["v6"], r) else None)
            wa.append(round(100 * sel(d, "w0", ks, GFILT[g]).mean(), 1) if round_ok(d["w0"], r) else None)
        out["per_round"].setdefault(rid, {})[g] = pr
        out["v6a_round"].setdefault(rid, {})[g] = va
        out["w0_round"].setdefault(rid, {})[g] = wa

out["per_round_pooled"], out["v6a_round_pooled"] = {}, {}
for p in ("gd", "cem"):
    rs = [d for d in R.values() if d["planner"] == p]
    for g in ("all", "seen", "unseen"):
        pr, va = [], []
        for r in range(1, 6):
            ks = chunks_of([r])
            pr.append(compare([sel(d, "v6", ks, GFILT[g]) for d in rs], [sel(d, "w0", ks, GFILT[g]) for d in rs])
                      if all(r in paired_rounds(d) for d in rs) else None)
            va.append(round(100 * np.concatenate([sel(d, "v6", ks, GFILT[g]) for d in rs]).mean(), 1)
                      if all(round_ok(d["v6"], r) for d in rs) else None)
        out["per_round_pooled"].setdefault(p, {})[g] = pr
        out["v6a_round_pooled"].setdefault(p, {})[g] = va

# ---- overall paired: every round complete in both arms of that run --------------------------------------
out["overall"], out["overall_pooled"] = {}, {}
for g in ("all", "seen", "unseen_revisit", "unseen_visit1"):
    for rid, d in R.items():
        ks = chunks_of(paired_rounds(d))
        out["overall"].setdefault(rid, {})[g] = compare([sel(d, "v6", ks, GFILT[g])], [sel(d, "w0", ks, GFILT[g])])
    for p in ("gd", "cem"):
        rs = [d for d in R.values() if d["planner"] == p]
        out["overall_pooled"].setdefault(p, {})[g] = compare([sel(d, "v6", chunks_of(paired_rounds(d)), GFILT[g]) for d in rs],
                                                             [sel(d, "w0", chunks_of(paired_rounds(d)), GFILT[g]) for d in rs])

# ---- cumulative paired gain over the contiguous both-finished prefix (every 10 episodes) ---------------
out["cumulative"] = {}
for rid, d in R.items():
    kk = 0
    while kk < 35 and d["v6"][kk] is not None and d["w0"][kk] is not None:
        kk += 1
    diff = np.concatenate([d["v6"][k] - d["w0"][k] for k in range(kk)]) if kk else np.array([], int)
    c = np.cumsum(diff)
    out["cumulative"][rid] = [int(c[i]) for i in range(9, len(c), 10)]
spans, s = [], 0
for k, b in enumerate(R["gd_s400"]["blocks"]):
    spans.append({"shape": b["shape"], "start": s, "end": s + b["n"], "seen": b["shape"] in SEEN, "round": k // 7 + 1})
    s += b["n"]
out["chunks"] = spans

# ---- per shape x visit (pooled per planner): v6a over all its visits, w0 where every run has it -----------
out["shape_visit"] = {}
for p in ("gd", "cem"):
    rs = [d for d in R.values() if d["planner"] == p]
    for sh in SHAPES:
        row = {"v6a": [], "w0": [], "n": []}
        for v in range(1, 6):
            ks = {id(d): [k for k, b in enumerate(d["blocks"]) if b["shape"] == sh and d["visit"][k] == v] for d in rs}
            for arm, key in (("v6", "v6a"), ("w0", "w0")):
                if all(all(d[arm][k] is not None for k in ks[id(d)]) for d in rs):
                    row[key].append(round(100 * np.concatenate([d[arm][k] for d in rs for k in ks[id(d)]]).mean(), 1))
                else:
                    row[key].append(None)
            row["n"].append(50 * len(rs))
        out["shape_visit"].setdefault(p, {})[sh] = row

# ---- cont_rounds: paired (reads vs w0) and v6a-only (gates, slow-learner effect, all rounds) -------------
out["gates"], out["reads"], out["slow"] = {}, {}, {}
for rid, d in R.items():
    for tag, arms in (("paired", ["w0", "cls6a_free"]), ("v6a", ["cls6a_free"])):
        o = OUT / "rounds" / f"{rid}_{tag}"
        o.mkdir(parents=True, exist_ok=True)
        (o / "rounds.json").unlink(missing_ok=True)
        # without w0, cont_rounds writes rounds.json and then fails in its markdown step (no vs_ref); the JSON is all we use
        subprocess.run([sys.executable, str(W / "scripts/cont_rounds.py"), "--stream", d["stream"], "--arms", *arms,
                        "--frozen", "", "--root", str(d["root"]), "--out", str(o)], check=(tag == "paired"), capture_output=True)
        assert (o / "rounds.json").exists(), f"cont_rounds wrote no rounds.json for {rid} {tag}"
    Rp = json.loads((OUT / "rounds" / f"{rid}_paired" / "rounds.json").read_text())
    Rv = json.loads((OUT / "rounds" / f"{rid}_v6a" / "rounds.json").read_text())
    g = Rv["gates"]["cls6a_free"]
    doms = sorted(g["domains"].items(), key=lambda kv: -kv[1]["n"])
    out["gates"][rid] = {"n_written": g["n_written"], "n_domains": g["n_domains"], "n_domains_ge3": g["n_domains_ge3"],
                         "purity": round(g["write_purity"], 2), "n_rejections": len(g["rejections"]),
                         "top": [{"id": k, "n": v["n"], "tags": v["tags"], "mature_now": v.get("mature_now")} for k, v in doms[:3]],
                         "per_chunk_reads": [{"chunk": c["chunk"], "shape": c["shape"], "visit": c["visit"], **c["reads"]} for c in g["per_chunk"]]}
    out["reads"][rid] = Rp["gates"]["cls6a_free"]["read_success"]
    out["slow"][rid] = [round(x, 3) if x is not None else None for x in Rv["slow_effect"]["cls6a_free"]]

out["reads_pooled"] = {}
for p in ("gd", "cem"):
    for v in range(1, 6):
        for kind in ("theta0", "own", "other"):
            parts = [out["reads"][rid][f"visit{v}|{kind}"] for rid, d in R.items()
                     if d["planner"] == p and f"visit{v}|{kind}" in out["reads"][rid]]
            n = sum(x["n"] for x in parts)
            if n:
                oa, orf = sum(x["only_arm"] for x in parts), sum(x["only_ref"] for x in parts)
                out["reads_pooled"].setdefault(p, {}).setdefault(kind, []).append(
                    {"visit": v, "n": n, "v6a": round(sum(x["n"] * x["arm"] for x in parts) / n, 1),
                     "w0": round(sum(x["n"] * x["ref"] for x in parts) / n, 1), "p": mcnemar_p(oa, orf)})

# ---- runtime ----------------------------------------------------------------------------------------------
L = W / "eval_outputs/gen_logs"


def wall(path):
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    st, en = [l for l in lines if l.startswith("start ")], [l for l in lines if l.startswith("EXIT=")]
    if not st or not en or not en[-1].startswith("EXIT=0"):
        return None
    f = lambda l: dt.datetime.strptime(" ".join(l.split()[1:3]), "%Y-%m-%d %H:%M:%S")
    return round((f(en[-1]) - f(st[-1])).total_seconds() / 3600, 2)


R45_LOG = {"gd_s100": "gd3_v6a_s100_r45.log"}  # the S7t3 GD rounds 4-5 were logged under their own name
# GD · S7t3 rounds 1-3 ran eager in the first batch (no log here): the page shows them as the eager reference row
out["runtime"] = {rid: {"v6a_r13": wall(L / f"v6a_{rid}.log"), "v6a_r45": wall(L / R45_LOG.get(rid, f"v6a_{rid}_r45.log"))}
                  for rid in R}

OUT.joinpath("report_data.json").write_text(json.dumps(out, ensure_ascii=False))
print(OUT / "report_data.json", f"{OUT.joinpath('report_data.json').stat().st_size / 1024:.0f} KB")
for r in out["runs"]:
    print(f"  {r['label']:16s} v6a rounds {r['v6a_rounds']}  w0 rounds {r['w0_rounds']}  paired {r['paired_rounds']}")
