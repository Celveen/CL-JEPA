"""Two checks of per-shape slow learners on held-out return visits (no planning).

Learners: the CLS state saved at the end of round 1 of a stream (oracle writing, so one
pure learner per shape). Episodes: the second visits of the reference (w0) run, which no
learner has trained on.

  * contact vs free: learner / theta0 1-step error on transitions where the object moved
    (> 1 px or > 0.01 rad between model steps) and where it did not;
  * reads: does the error-based read rule (lowest error over the first 3 transitions among
    theta0 and the learners) pick theta0 for trained shapes and the own learner for unseen
    shapes, with learners for unseen shapes only or for all shapes, with or without the
    familiarity gate on reads.

    python scripts/cont_check_learners.py --stream S7 --state eval_outputs/cont_S7/cls3_oracle/chunk_06_T/continual_state.pt
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cont_offline import load_stream, load_world  # noqa: E402

UNSEEN = ("I", "square", "small_tee")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S7")
    ap.add_argument("--state", default="eval_outputs/cont_S7/cls3_oracle/chunk_06_T/continual_state.pt")
    ap.add_argument("--first-test-block", type=int, default=7)
    ap.add_argument("--familiar-thr", type=float, default=0.0220)
    ap.add_argument("--ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre, _ = load_world(ROOT / a.ckpt, "3", dev)
    st = torch.load(ROOT / a.state, map_location="cpu")
    learners = {}
    for k, d in st["domains"].items():
        if d["learner"] is not None:
            m = copy.deepcopy(wm.predictor)
            m.load_state_dict(d["learner"])
            learners[k] = m.to(dev).eval()
    drop = wm.action_dim

    def pred(mod, z):
        T = z.shape[1]
        return rearrange(mod(rearrange(z, "b t p d -> b (t p) d")), "b (t p) d -> b t p d", t=T)

    def err(mod, z, t):  # predict frame t from frames < t (history up to num_hist)
        return float(torch.mean((pred(mod, z[:, max(0, t - wm.num_hist) : t])[:, -1:, ..., :-drop] - z[:, t : t + 1, ..., :-drop]) ** 2))

    run = ROOT / f"eval_outputs/cont_{a.stream}/w0"
    blocks, eps = load_stream(run, a.stream, pre, dev)
    rows, reads = [], {}
    for e in eps:
        if e["block"] < a.first_test_block:
            continue
        b = blocks[e["block"]]
        tag = b["shape"].replace("+", "plus")
        d = run / b["chunk"]
        it = [json.loads(l) for l in open(d / "per_iter_logs.json") if l.strip()]
        first_ep = min(int(k.split("_")[0][2:]) for k in np.load(d / "episode_frames.npz").files)
        i = e["e"] - first_ep
        states = [np.load(d / "final_eval_results.npz")["state_0"][i]] + [
            np.array(r["state"]) for r in sorted((r for r in it if r["sample_idx"] == i), key=lambda r: r["step"])]
        with torch.no_grad():
            z = wm.encode(e["obs"], torch.cat([e["act"], torch.zeros_like(e["act"][:, :1])], dim=1))
            k = z.shape[1] - 1
            if tag in learners and len(states) >= k + 1:
                for t in range(2, k + 1):
                    dobj = float(np.linalg.norm(states[t][2:4] - states[t - 1][2:4]))
                    dang = abs((states[t][4] - states[t - 1][4] + math.pi) % (2 * math.pi) - math.pi)
                    rows.append((b["shape"], dobj > 1.0 or dang > 0.01, err(wm.predictor, z, t), err(learners[tag], z, t)))
            f3 = lambda mod: float(np.mean([err(mod, z, t) for t in range(1, min(3, k) + 1)]))
            e0 = f3(wm.predictor)
            le = {key: f3(m) for key, m in learners.items()}
        for cname, cands in (("unseen-shape learners", [c for c in UNSEEN if c in le]), ("all learners", list(le))):
            for gate in (False, True):
                choice = None if (gate and e0 <= a.familiar_thr) else min({None: e0, **{c: le[c] for c in cands}}.items(), key=lambda kv: kv[1])[0]
                grp = "unseen" if tag in UNSEEN else "seen"
                kind = "θ0" if choice is None else ("own" if choice == tag else "other")
                r = reads.setdefault((cname, gate, grp), {})
                r[kind] = r.get(kind, 0) + 1

    gm = lambda x: float(np.exp(np.mean(np.log(x))))
    print("== learner / theta0 1-step error on held-out return visits")
    print("shape       contact share   learner/theta0 on contact   on free motion")
    for s in dict.fromkeys(r[0] for r in rows):
        R = [r for r in rows if r[0] == s]
        C = [r for r in R if r[1]]
        F = [r for r in R if not r[1]]
        print(f"{s:10s}  {len(C) / len(R) * 100:12.0f}%   {gm([r[3] / r[2] for r in C]):10.2f} (n={len(C):3d})"
              f"   {gm([r[3] / r[2] for r in F]):10.2f} (n={len(F):3d})")
    print("\n== error-based reads over the first 3 transitions")
    for (cname, gate, grp), r in sorted(reads.items()):
        n = sum(r.values())
        right = r.get("θ0", 0) if grp == "seen" else r.get("own", 0)
        print(f"{cname:22s} familiarity gate on reads: {'on ' if gate else 'off'}  {grp:6s}  right {right}/{n} = {right / n * 100:.0f}%  {r}")


if __name__ == "__main__":
    main()
