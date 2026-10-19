"""Compare our PushObj training data with the authors' data (as seen through their
released test segments) on 25-step windows, per training shape.

The authors' eval segments (data/pushobj_eval/val_<shape>/plan_targets.pkl) are
25 env steps cut from their held-out trajectories and filtered to contain at
least one contact. Our windows are drawn at random from the training split; the
"contact" subset applies the same filter (block moves at least once).

    python scripts/compare_data_distribution.py
"""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHAPES = ("T", "L", "Z", "+")
AGENT_R, WALL_LO, WALL_HI, L = 15.0, 7.0, 504.0, 25
MOVE_PX, MOVE_RAD = 0.2, 0.002  # a step "moves the block" above these

_POLY = {}


def local_polygons():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gym
    import env  # noqa: F401
    e = gym.make("pushobj", with_velocity=True, with_target=True).unwrapped
    out = {}
    for s in SHAPES:
        e.update_env({"shape": s})
        e.prepare(0, np.array([100.0, 100.0, 256.0, 256.0, 0.0, 0.0, 0.0]))
        out[s] = [[(v.x, v.y) for v in sh.get_vertices()] for sh in e.block.shapes]
    e.close()
    return out


def _init(polys):
    global _POLY
    import warnings
    warnings.filterwarnings("ignore")  # shapely union of edge-sharing parts
    _POLY = polys


def world_poly(shape, x, y, a):
    c, s = np.cos(a), np.sin(a)
    return unary_union([Polygon([(x + c * u - s * v, y + s * u + c * v) for u, v in p]) for p in _POLY[shape]])


def gap(shape, st):
    poly = world_poly(shape, *st[2:5])
    pt = Point(st[0], st[1])
    d = -poly.boundary.distance(pt) if poly.contains(pt) else poly.distance(pt)
    return d - AGENT_R, poly


def window_stats(item):
    shape, states, actions = item  # states (26, 5), actions (25, 2) pixel deltas
    gaps, wall = [], None
    for k, st in enumerate(states):
        g, poly = gap(shape, st)
        gaps.append(g)
        if k == 0:
            x0, y0, x1, y1 = poly.bounds
            wall = float(min(x0 - WALL_LO, WALL_HI - x1, y0 - WALL_LO, WALL_HI - y1))
    blk = states[:, 2:4]
    dpos = np.linalg.norm(np.diff(blk, axis=0), axis=1)
    dang = np.abs((np.diff(states[:, 4]) + np.pi) % (2 * np.pi) - np.pi)
    moving = (dpos > MOVE_PX) | (dang > MOVE_RAD)
    return {"start_gap": float(gaps[0]), "mean_gap": float(np.mean(gaps)), "contact_frac": float(moving.mean()),
            "block_disp": float(np.linalg.norm(blk[-1] - blk[0])), "block_rot": float(dang.sum()),
            "agent_speed": float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).mean()),
            "action_mag": float(np.linalg.norm(actions, axis=1).mean()), "wall_clearance": wall}


def author_windows(shape):
    segs = pickle.load(open(ROOT / f"data/pushobj_eval/val_{shape}/plan_targets.pkl", "rb"))["segments"]
    return [(shape, np.asarray(s["states"], dtype=np.float64)[:, :5], np.asarray(s["actions"], dtype=np.float64)) for s in segs]


def our_windows(data_dir, shape, n, rs):
    root = Path(data_dir) / "train"
    S = torch.load(root / "states.pth", weights_only=True).double().numpy()
    A = torch.load(root / "rel_actions.pth", weights_only=True).double().numpy()
    lens = pickle.load(open(root / "seq_lengths.pkl", "rb"))
    shapes = pickle.load(open(root / "shapes.pkl", "rb"))
    idx = [i for i, s in enumerate(shapes) if s == shape and lens[i] > L]
    out = []
    for _ in range(n):
        i = idx[rs.randint(len(idx))]
        o = rs.randint(0, lens[i] - L)
        out.append((shape, S[i, o:o + L + 1], A[i, o:o + L]))
    return out


def describe(rows):
    keys = rows[0].keys()
    return {k: {"median": float(np.median([r[k] for r in rows])), "q10": float(np.quantile([r[k] for r in rows], .1)),
                "q90": float(np.quantile([r[k] for r in rows], .9)), "mean": float(np.mean([r[k] for r in rows]))} for k in keys} | \
        {"n": len(rows), "start_gap_lt_5px": float(np.mean([r["start_gap"] < 5 for r in rows])),
         "start_overlap": float(np.mean([r["start_gap"] < 0 for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pushobj_bias_16k")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="reports/pushobj_bias_16k/data_distribution.json")
    a = ap.parse_args()
    rs = np.random.RandomState(0)
    polys = local_polygons()
    result = {}
    with Pool(32, initializer=_init, initargs=(polys,)) as pool:
        for s in SHAPES:
            au = pool.map(window_stats, author_windows(s), chunksize=16)
            ours = pool.map(window_stats, our_windows(ROOT / a.data, s, a.n, rs), chunksize=16)
            ours_c = [r for r in ours if r["contact_frac"] > 0]
            result[s] = {"author": describe(au), "ours_all": describe(ours), "ours_contact": describe(ours_c),
                         "author_has_contact_frac": float(np.mean([r["contact_frac"] > 0 for r in au]))}
            print(s, "author n", len(au), "ours", len(ours), "ours contact", len(ours_c), flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    keys = ["start_gap", "mean_gap", "contact_frac", "block_disp", "block_rot", "agent_speed", "action_mag", "wall_clearance"]
    for s in SHAPES:
        print(f"\n== {s}  (median [q10, q90]);  author windows with contact: {result[s]['author_has_contact_frac']:.0%}")
        print(f"{'stat':15s} | {'author':>22s} | {'ours (contact)':>22s} | {'ours (all)':>22s}")
        for k in keys:
            cell = lambda d: f"{d[k]['median']:7.2f} [{d[k]['q10']:6.1f},{d[k]['q90']:6.1f}]"
            print(f"{k:15s} | {cell(result[s]['author']):>22s} | {cell(result[s]['ours_contact']):>22s} | {cell(result[s]['ours_all']):>22s}")
        for k in ["start_gap_lt_5px", "start_overlap"]:
            print(f"{k:15s} | {result[s]['author'][k]:22.1%} | {result[s]['ours_contact'][k]:22.1%} | {result[s]['ours_all'][k]:22.1%}")


if __name__ == "__main__":
    main()
