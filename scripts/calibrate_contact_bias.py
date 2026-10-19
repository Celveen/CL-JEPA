"""Calibrate the action-level contact bias of gen_pushobj_dataset.py against the
authors' released test segments (CPU only, physics without rendering).

For each (bias b, noise sigma, placement) setting, the same source trajectories
are replayed; 25-step windows are drawn, re-weighted to the authors' block-
displacement distribution (their segments are selected for motion), and compared
on statistics the selection does not create: action magnitude, agent speed, the
distance to the nearest pusht_noise action window and the direction of that
residual relative to the block, plus contact fraction and agent-block gap.

    python scripts/calibrate_contact_bias.py --out reports/.../diagnostics/contact_bias_calibration.json

With --compare, generated datasets are scored on the same statistics instead of
running the grid (1500 random training windows per shape and dataset):

    python scripts/calibrate_contact_bias.py --compare new=data/pushobj_bias_16k \
        --shapes T,L,Z,+ --out reports/pushobj_bias_16k/distribution_vs_author.json
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import gen_pushobj_dataset as G  # noqa: E402
import compare_data_distribution as C  # noqa: E402

W = 25
BINS = np.array([0, 20, 40, 60, 80, 100, 130, 160, 200, 1e9])
KEYS = ["action_mag", "agent_speed", "nn_rms", "cos_to_block", "toward_frac", "contact_frac", "mean_gap"]


class NearestSource:
    """Nearest 25-step pusht_noise action window (L2) for query windows."""

    def __init__(self, A, lengths):
        wins = [sliding_window_view(A[e, :lengths[e]], (W, 2))[:, 0].reshape(-1, 2 * W)
                for e in range(len(lengths)) if lengths[e] >= W]
        self.X = torch.from_numpy(np.concatenate(wins).astype(np.float32))
        self.Xn = (self.X ** 2).sum(1)

    def residual(self, q_act, q_states):
        Q = torch.from_numpy(q_act.reshape(len(q_act), -1).astype(np.float32))
        idx = []
        for b in range(0, len(Q), 256):
            q = Q[b:b + 256]
            d = (q ** 2).sum(1, keepdim=True) + self.Xn[None] - 2 * q @ self.X.T
            idx.append(d.argmin(1))
        src = self.X[torch.cat(idx)].numpy().reshape(-1, W, 2)
        r = q_act - src
        d = q_states[:, :W, 2:4] - q_states[:, :W, :2]
        cos = (r * d).sum(-1) / (np.linalg.norm(r, axis=-1) * np.linalg.norm(d, axis=-1) + 1e-6)
        return np.sqrt((r ** 2).sum(-1).mean(1)), cos.mean(1), (cos > 0).mean(1)


def window_table(items, nn):
    """items: list of (shape, states(26,5), actions(25,2)) -> dict of per-window arrays."""
    with Pool(32, initializer=C._init, initargs=(POLYS,)) as pool:
        rows = pool.map(C.window_stats, items, chunksize=16)
    tab = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    rms, cos, tow = nn.residual(np.stack([it[2] for it in items]), np.stack([it[1] for it in items]))
    tab.update(nn_rms=rms, cos_to_block=cos, toward_frac=tow)
    return tab


def weighted(tab, ref_disp):
    ha = np.histogram(ref_disp, BINS)[0] / len(ref_disp)
    idx = np.digitize(tab["block_disp"], BINS) - 1
    ho = np.bincount(idx, minlength=len(BINS) - 1)[: len(BINS) - 1] / len(idx)
    w = np.array([ha[i] / ho[i] if ho[i] > 0 else 0.0 for i in idx])
    out = {k: float(np.sum(w * tab[k]) / w.sum()) for k in KEYS if k != "nn_rms"}
    out["nn_rms"] = float(np.median(tab["nn_rms"]))  # median, as reported for the authors
    return out


def simulate(jobs, n_workers):
    with Pool(n_workers, initializer=G._init_worker, initargs=(False,)) as pool:
        return pool.map(G._run, jobs, chunksize=2)


def compare(datasets, shapes, targets, nn, n=1500):
    """Score generated datasets against the authors' statistics, as the grid does."""
    out = {s: {"author": {k: v for k, v in targets[s].items() if not k.startswith("_")}} for s in shapes}
    for tag, data in datasets:
        rs = np.random.RandomState(3)
        for s in shapes:
            m = weighted(window_table(C.our_windows(ROOT / data, s, n, rs), nn), targets[s]["_disp"])
            m["score"] = float(sum(((m[k] - targets[s][k]) / (abs(targets[s][k]) + 1e-6)) ** 2 for k in KEYS))
            out[s][tag] = m
            print(tag, s, {k: round(v, 3) for k, v in m.items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="T,+")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--windows-per-episode", type=int, default=6)
    ap.add_argument("--bias", default="0,3,5,7,9")
    ap.add_argument("--noise", default="0,5,8,10,12")
    ap.add_argument("--placements", default="cog,source")
    ap.add_argument("--n-workers", type=int, default=64)
    ap.add_argument("--compare", default=None, help="tag=data_dir,... : score datasets instead of running the grid")
    ap.add_argument("--out", default="reports/pushobj_replay_20k_e5/diagnostics/contact_bias_calibration.json")
    a = ap.parse_args()
    global POLYS
    POLYS = C.local_polygons()
    S, A, V, L = G.load_source(ROOT / "data/pusht_noise_meta")
    leaves = ((((S[..., :2] < 0) | (S[..., :2] > G.WINDOW)).any(-1)) & (np.arange(S.shape[1])[None] < L[:, None])).any(1)
    pool_idx = np.where((np.arange(len(L)) < 16000) & ~leaves & (L > W + 5))[0]
    nn = NearestSource(A.astype(np.float32), L)
    shapes = a.shapes.split(",")
    rs = np.random.RandomState(0)
    src = {s: rs.choice(pool_idx, a.episodes, replace=False) for s in shapes}

    # authors' targets
    targets = {}
    for s in shapes:
        au = C.author_windows(s)
        tab = window_table(au, nn)
        targets[s] = {k: (float(np.median(tab[k])) if k == "nn_rms" else float(np.mean(tab[k]))) for k in KEYS}
        targets[s]["_disp"] = tab["block_disp"]
        print("author", s, {k: round(v, 3) for k, v in targets[s].items() if not k.startswith("_")}, flush=True)

    if a.compare:
        result = compare([item.split("=", 1) for item in a.compare.split(",")], shapes, targets, nn)
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        return

    grid = list(itertools.product([float(x) for x in a.bias.split(",")], [float(x) for x in a.noise.split(",")],
                                  a.placements.split(",")))
    results = []
    for b, sig, place in grid:
        per_shape = {}
        for s in shapes:
            if s == "T" and place != a.placements.split(",")[0]:
                per_shape[s] = next(r["shapes"]["T"] for r in results if r["bias"] == b and r["noise"] == sig)
                continue
            jobs = [(np.concatenate([S[e, 0], V[e, 0]]), A[e, :L[e]], s, int(e * 7 + 1), 4, 20.0, 0.3, 3.0, 80.0,
                     None, 10, b, sig, place) for e in src[s]]
            res = simulate(jobs, a.n_workers)
            items, wrs = [], np.random.RandomState(1)
            for r in res:
                if not r["valid"]:
                    continue
                st, ac = r["states"].astype(np.float64), r["actions"].astype(np.float64)
                T = len(st)
                for o in wrs.randint(0, T - W, size=min(a.windows_per_episode, T - W)):
                    items.append((s, st[o:o + W + 1], ac[o:o + W]))
            tab = window_table(items, nn)
            m = weighted(tab, targets[s]["_disp"])
            m["valid_frac"] = float(np.mean([r["valid"] for r in res]))
            m["score"] = float(sum(((m[k] - targets[s][k]) / (abs(targets[s][k]) + 1e-6)) ** 2 for k in KEYS))
            per_shape[s] = m
        score = float(np.mean([per_shape[s]["score"] for s in shapes]))
        results.append({"bias": b, "noise": sig, "placement": place, "score": score, "shapes": per_shape})
        print(f"b={b:4.1f} sigma={sig:4.1f} {place:6s} score={score:.3f} " +
              " ".join(f"{s}:[mag {per_shape[s]['action_mag']:.1f} spd {per_shape[s]['agent_speed']:.1f} nn {per_shape[s]['nn_rms']:.1f} "
                       f"cos {per_shape[s]['cos_to_block']:.2f} cf {per_shape[s]['contact_frac']:.2f} gap {per_shape[s]['mean_gap']:.1f}]"
                       for s in shapes), flush=True)
    results.sort(key=lambda r: r["score"])
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"targets": {s: {k: v for k, v in t.items() if not k.startswith("_")} for s, t in targets.items()},
                               "results": results}, indent=2))
    print("\nbest:", json.dumps(results[:3], indent=1)[:3000])


if __name__ == "__main__":
    main()
