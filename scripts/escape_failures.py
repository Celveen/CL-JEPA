"""Failure cases of stall-escape arms: what the failed episodes look like and why.

For every failed episode (privileged env state, analysis only) it records the final errors
(object position, angle, pusher), a category (which part of the success test fails), whether the
episode stalled, whether the pusher is touching the object at the end, whether the object or its
goal lies against a wall, how far the object had to turn initially, and whether the episode got
close and then moved away. It writes a summary per arm and failure galleries: per episode the
goal pose (green outline), start pose (grey dashed), final pose (red), the pusher's path (blue,
goal x), and the escapes along it (purple: object phase, orange: kicks).

    python scripts/escape_failures.py --stream EVAL50 --arms none K2P_7x15 KS_7x15 --gallery K2P_7x15 KS_7x15
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cont_analyze import finished, jsonl  # noqa: E402

ANG = math.pi / 9
WALL_LO, WALL_HI = 5.0, 506.0  # arena walls (env/pusht/pusht_env.py)
PUSHER_R = 15.0
CATS = ("pusher-only", "angle-only", "position-only", "position+angle")
CAT_CN = {"pusher-only": "只差推杆", "angle-only": "只差角度", "position-only": "只差物体位置", "position+angle": "物体位置和角度都差"}
_ENVS = {}


def polygons(shape, state):
    """World polygons of the object at env state (x_agent, y_agent, x_obj, y_obj, angle, ...)."""
    if shape not in _ENVS:
        from env.pusht.pusht_env import PushTEnv
        env = PushTEnv(shape=shape)
        env.reset()
        _ENVS[shape] = env
    body = _ENVS[shape].block
    body.angle = float(state[4])
    body.position = (float(state[2]), float(state[3]))
    return [np.array([tuple(body.local_to_world(v)) for v in s.get_vertices()]) for s in body.shapes]


def point_poly_dist(p, polys):
    """Distance from point p to the union of convex polygons (0 inside one)."""
    best = np.inf
    for P in polys:
        crosses = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            ab, ap = b - a, p - a
            t = np.clip(ap @ ab / (ab @ ab), 0, 1)
            best = min(best, float(np.linalg.norm(ap - t * ab)))
            crosses.append(ab[0] * ap[1] - ab[1] * ap[0])
        if all(c >= 0 for c in crosses) or all(c <= 0 for c in crosses):
            return 0.0
    return best


def centroid(polys):
    """Area centroid of a union of disjoint polygons."""
    A = cx = cy = 0.0
    for P in polys:
        x, y = P[:, 0], P[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        c = x * y1 - x1 * y
        A += c.sum() / 2
        cx += ((x + x1) * c).sum() / 6
        cy += ((y + y1) * c).sum() / 6
    return np.array([cx / A, cy / A])


def wall_gap(polys):
    v = np.concatenate(polys)
    return float(min(v.min() - WALL_LO, WALL_HI - v.max()))


def ang_err(g, s):
    return abs((g[4] - s[4] + math.pi) % (2 * math.pi) - math.pi)


def load(root, arm, blocks):
    eps = []
    for b in blocks:
        d = root / arm / b["chunk"]
        if not finished(d):
            return None
        z = np.load(d / "final_eval_results.npz")
        it = jsonl(d / "per_iter_logs.json")
        for i in range(b["n"]):
            eps.append({"chunk": b["chunk"], "i": i, "shape": b["shape"], "succ": bool(z["successes"][i]),
                        "len": float(z["action_len"][i]), "s0": z["state_0"][i], "g": z["state_g"][i],
                        "rows": [r for r in it if r["sample_idx"] == i]})
    return eps


def features(e):
    rows, g = e["rows"], e["g"]
    S = np.array([r["state"] for r in rows])
    s = S[-1]
    miss = np.array([max(r["dist4"] / 20, r["ang"] / ANG) for r in rows])
    obj, ang, pus = rows[-1]["dist_obj"], math.degrees(rows[-1]["ang"]), rows[-1]["dist_agent"]
    cat = ("pusher-only" if ang < 20 else "angle-only") if obj < 20 else ("position-only" if ang < 20 else "position+angle")
    polys = polygons(e["shape"], s)
    c = centroid(polys)
    d = centroid(polygons(e["shape"], g)) - c  # where the object still has to go
    r = s[:2] - c  # where the pusher is, seen from the object
    side = float(r @ d / (np.linalg.norm(r) * np.linalg.norm(d))) if np.linalg.norm(d) > 10 and np.linalg.norm(r) > 1 else float("nan")
    w = min(10, len(S) - 1)
    esc = [(j, r["escape"]) for j, r in enumerate(rows) if "escape" in r]
    f = {"chunk": e["chunk"], "i": e["i"], "shape": e["shape"], "category": cat,
         "obj_err": obj, "ang_err": ang, "pusher_err": pus, "miss": float(miss[-1]),
         "near": bool(obj < 40 and ang < 40),
         "stalled": bool(len(miss) > 10 and miss[-11] - miss[-1] < 0.1),
         "object_moved_last10": bool(np.linalg.norm(S[-1, 2:4] - S[-1 - w, 2:4]) > 2 or abs(S[-1, 4] - S[-1 - w, 4]) > math.radians(2)),
         "pusher_object_gap": point_poly_dist(s[:2], polys) - PUSHER_R,
         "object_wall_gap": wall_gap(polys), "goal_wall_gap": wall_gap(polygons(e["shape"], g)),
         "start_obj_err": float(np.linalg.norm(g[2:4] - e["s0"][2:4])), "start_ang_err": math.degrees(ang_err(g, e["s0"])),
         "best_miss": float(miss.min()), "best_step": int(miss.argmin()) + 1,
         "regressed": bool(miss.min() < 1.5 and miss[-1] - miss.min() > 0.5),
         "n_escapes": len(esc),
         "pusher_side_cos": side,  # > 0.3: the pusher sits on the side the object must move toward
         "left_arena": bool(((S[:, :2] < 0) | (S[:, :2] > 512)).any())}
    obj_phase = [j for j, x in esc if x.get("phase") == "object"]
    if obj_phase:
        j0 = obj_phase[0]
        j1 = next((j for j, x in esc if x.get("phase") == "pusher"), len(rows))
        f["object_phase_obj_err_change"] = rows[max(j1 - 1, j0)]["dist_obj"] - (rows[j0 - 1]["dist_obj"] if j0 > 0 else rows[0]["dist_obj"])
        f["object_phase_ang_change"] = math.degrees(rows[max(j1 - 1, j0)]["ang"] - (rows[j0 - 1]["ang"] if j0 > 0 else rows[0]["ang"]))
    return f


def gallery(eps, feats, title, path, per_page=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    order = sorted(range(len(feats)), key=lambda k: (CATS.index(feats[k]["category"]), feats[k]["shape"], feats[k]["obj_err"]))
    pages = [order[i : i + per_page] for i in range(0, len(order), per_page)]
    out = []
    for pi, page in enumerate(pages):
        cols = 4
        rows_n = math.ceil(len(page) / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4.2 * rows_n))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")
        for ax, k in zip(axes.flat, page):
            e, f = eps[k], feats[k]
            S = np.array([r["state"] for r in e["rows"]])
            for P in polygons(e["shape"], e["g"]):
                ax.add_patch(Polygon(P, closed=True, fill=False, ec="tab:green", lw=2))
            for P in polygons(e["shape"], e["s0"]):
                ax.add_patch(Polygon(P, closed=True, fill=False, ec="grey", lw=1, ls="--"))
            for P in polygons(e["shape"], S[-1]):
                ax.add_patch(Polygon(P, closed=True, fc="tab:red", alpha=0.35, ec="tab:red"))
            ax.plot(S[:, 0], S[:, 1], "-", color="tab:blue", lw=0.8, alpha=0.7)
            ax.plot(*e["s0"][:2], "o", color="tab:blue", mfc="none", ms=6)
            ax.plot(*S[-1, :2], "o", color="tab:blue", ms=7)
            ax.plot(*e["g"][:2], "x", color="tab:blue", ms=9, mew=2)
            for j, r in enumerate(e["rows"]):
                if "escape" in r:
                    c = "purple" if r["escape"].get("phase") == "object" else "orange"
                    ax.plot(*S[j, :2], "^", color=c, ms=6)
            ax.set_xlim(0, 512)
            ax.set_ylim(512, 0)
            ax.set_aspect("equal")
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{e['shape']} {e['chunk'][-12:]}#{e['i']} | {f['category']}\n"
                         f"obj {f['obj_err']:.0f}px ang {f['ang_err']:.0f}° pusher {f['pusher_err']:.0f}px"
                         f"{' | stalled' if f['stalled'] else ''}{' | wall' if f['object_wall_gap'] < 15 else ''}"
                         f"{' | goal side' if f['pusher_side_cos'] > 0.3 else ''}{' | LEFT ARENA' if f['left_arena'] else ''}", fontsize=8)
        fig.suptitle(f"{title} — failures {pi * per_page + 1}–{pi * per_page + len(page)} of {len(order)}  "
                     "(green: goal, grey dashed: start, red: final; blue: pusher path, o start, ● end, x goal; "
                     "▲ purple: object phase, ▲ orange: kick)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = path.parent / f"{path.stem}_{pi + 1}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="EVAL50")
    ap.add_argument("--arms", nargs="+", default=["none", "K2P_7x15", "KS_7x15"])
    ap.add_argument("--gallery", nargs="*", default=["K2P_7x15", "KS_7x15"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = ROOT / (a.root or f"eval_outputs/escape_{a.stream}")
    out = ROOT / (a.out or f"reports/escape_{a.stream}/failures")
    out.mkdir(parents=True, exist_ok=True)
    blocks = json.loads((ROOT / "data/cont_streams" / a.stream / "manifest.json").read_text())["blocks"]
    E = {arm: load(root, arm, blocks) for arm in a.arms}
    E = {k: v for k, v in E.items() if v is not None}
    F = {arm: [features(e) if not e["succ"] else None for e in eps] for arm, eps in E.items()}
    L = [f"# 失败案例剖析（{a.stream}）", "",
         "按结束时的真实状态归类（只用于分析）。成功判定：(推杆 xy, 物体 xy) 的联合误差 < 20 px 且角度 < 20°。", "",
         "| 组 | 失败数 | " + " | ".join(CAT_CN[c] for c in CATS) + " | 差一点（物体 < 40 px 且角度 < 40°） | 最后 10 次停滞 | 结束时推杆贴着物体（< 2 px） | 推杆在物体朝目标的一侧（需绕行） | 推杆跑出场地 | 物体靠墙（< 15 px） | 目标靠墙（< 15 px） | 曾接近（miss < 1.5）后变差 | 初始需转 > 90° |",
         "|---|---:|" + "---:|" * (len(CATS) + 9)]
    summary = {}
    for arm, fs in F.items():
        fl = [f for f in fs if f is not None]
        n = len(fl)
        frac = lambda cond: sum(cond(f) for f in fl) / max(n, 1) * 100
        row = {"n_fail": n, **{c: sum(f["category"] == c for f in fl) for c in CATS},
               "near": frac(lambda f: f["near"]), "stalled": frac(lambda f: f["stalled"]),
               "contact": frac(lambda f: f["pusher_object_gap"] < 2),
               "goal_side": 100 * float(np.mean([f["pusher_side_cos"] > 0.3 for f in fl if not np.isnan(f["pusher_side_cos"])] or [np.nan])),
               "left_arena": frac(lambda f: f["left_arena"]), "object_wall": frac(lambda f: f["object_wall_gap"] < 15),
               "goal_wall": frac(lambda f: f["goal_wall_gap"] < 15), "regressed": frac(lambda f: f["regressed"]),
               "big_turn": frac(lambda f: f["start_ang_err"] > 90)}
        summary[arm] = row
        L.append(f"| {arm} | {n} | " + " | ".join(str(row[c]) for c in CATS) +
                 f" | {row['near']:.0f}% | {row['stalled']:.0f}% | {row['contact']:.0f}% | {row['goal_side']:.0f}% | {row['left_arena']:.0f}% | {row['object_wall']:.0f}% | "
                 f"{row['goal_wall']:.0f}% | {row['regressed']:.0f}% | {row['big_turn']:.0f}% |")
    # failure categories by shape
    shapes = list(dict.fromkeys(b["shape"] for b in blocks))
    L += ["", "## 各形状的失败构成（只差推杆 / 只差角度 / 只差位置 / 都差）", "", "| 组 | " + " | ".join(shapes) + " |", "|---|" + "---|" * len(shapes)]
    for arm, fs in F.items():
        cells = []
        for sh in shapes:
            fl = [f for f in fs if f is not None and f["shape"] == sh]
            cells.append("/".join(str(sum(f["category"] == c for f in fl)) for c in CATS))
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    # medians of the main failure type
    L += ["", "## 失败时的典型数值（中位数）", "",
          "| 组 | 物体误差 | 角度误差 | 推杆误差 | 推杆到物体的间隙 | 初始物体误差 | 初始角度误差 | 最接近时的 miss | 干预次数 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, fs in F.items():
        fl = [f for f in fs if f is not None]
        med = lambda k: float(np.median([f[k] for f in fl])) if fl else float("nan")
        L.append(f"| {arm} | {med('obj_err'):.0f} px | {med('ang_err'):.0f}° | {med('pusher_err'):.0f} px | {med('pusher_object_gap'):.0f} px | "
                 f"{med('start_obj_err'):.0f} px | {med('start_ang_err'):.0f}° | {med('best_miss'):.2f} | {med('n_escapes'):.0f} |")
    # shared vs unique failures
    if "none" in F:
        base_fail = {(f["chunk"], f["i"]) for f in F["none"] if f is not None}
        L += ["", "## 与 AdaJEPA（none）失败集合的关系", ""]
        for arm, fs in F.items():
            if arm == "none":
                continue
            mine = {(f["chunk"], f["i"]) for f in fs if f is not None}
            L.append(f"- {arm}：失败 {len(mine)} 个，其中 {len(mine & base_fail)} 个 none 也失败，{len(mine - base_fail)} 个是新增失败；救回 none 的失败 {len(base_fail - mine)} 个。")
    # object phase effect for staged arms
    for arm, fs in F.items():
        ch = [f["object_phase_obj_err_change"] for f in fs if f is not None and "object_phase_obj_err_change" in f]
        if ch:
            ch = np.array(ch)
            L += ["", f"- {arm} 失败 episode 的物体阶段：物体误差变化中位数 {np.median(ch):+.1f} px，"
                      f"改善 > 5 px 的占 {np.mean(ch < -5) * 100:.0f}%，变差 > 5 px 的占 {np.mean(ch > 5) * 100:.0f}%（n={len(ch)}）。"]
    figs = []
    for arm in a.gallery:
        if arm in E:
            idx = [k for k, f in enumerate(F[arm]) if f is not None]
            figs += gallery([E[arm][k] for k in idx], [F[arm][k] for k in idx], arm, out / f"gallery_{arm}")
    if figs:
        L += ["", "## 失败图集", ""] + [f"- [{p.name}]({p.name})" for p in figs]
    (out / "failures.md").write_text("\n".join(L) + "\n")
    (out / "failures.json").write_text(json.dumps({"summary": summary, "cases": {arm: [f for f in fs if f is not None] for arm, fs in F.items()}},
                                                   indent=1, ensure_ascii=False, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
