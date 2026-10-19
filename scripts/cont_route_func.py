"""Functional offline evaluation of routing rules (no planning).

Classification purity is not the goal: what matters for planning is that an episode -- above
all a returning unseen-shape episode -- is planned with a learner that predicts it well. So this
replay scores every rule by the read learner's 5-step open-loop latent error over theta0's on
the episode it is read for (the learner has not trained on that episode yet); < 1 means the
read helps (online, episodes whose ratio fell below ~0.6 were the ones that turned into wins).

Recorded w0 (AdaJEPA) episodes are replayed in stream order; per-domain learners are trained as
online (whole predictor, theta0 latents, 15 updates per written episode, catch-up at creation);
reads use the online rule (lowest first-3-transition error among theta0 and mature learners).

  oracle  write by true shape; read the own-shape learner when mature (the online oracle)
  v4      online v4 (familiarity gate first; learner rule needs a 1-step win over theta0)
  pool    no domains among the unfamiliar: every episode above the 90th-percentile familiarity
          threshold is written to one shared learner (pool80: 80th percentile); reads as online.
  v5      v4 writing + v5's symmetric merge (each learner within 30% of the other's own learner on its
          last 3 newly written episodes, and better than theta0, both ways)
  v5B     v4 writing + merge by validated pooling (candidate learner on both replays, 2 held-out per side), checked at block ends
  v5M     v4 writing + two-tier merging at block ends: fragments (< 6 episodes, no validation history)
          are absorbed by the mature domain whose learner predicts their stored episodes best (median
          ratio <= 0.8); two mature domains pool only if a candidate trained on both is no worse than
          either own learner on 4 held-out validation episodes per side (and better than theta0)
  --knn   additionally records, before each episode, which learner a KNN on the start+goal encoder
          features would read first (references: earlier episodes labelled by the domain they were
          written to, or theta0 when familiar), as CLSKnnReadMPCPlanner does online

    python scripts/cont_route_func.py --stream S7b3 --run eval_outputs/cont_S7b/w0 --variant v5M --knn
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cont_merge_screen import Slow  # noqa: E402
from cont_offline import load_stream, load_world  # noqa: E402
from cont_route_screen import latents, new3  # noqa: E402

SEEN = {"T", "L", "Z", "+"}


class R:
    def __init__(self, v, S, theta0, G, fam, thr80, thr90, Z, tags, feats=None):
        self.v, self.S, self.theta0, self.G, self.fam, self.thr80, self.thr90, self.Z, self.tags = \
            v, S, theta0, G, fam, thr80, thr90, Z, tags
        self.feats, self.knn_refs = feats, []  # KNN early read: start+goal features, refs labelled by where episodes went
        self.dom, self.pending, self.merges, self.block, self.next_id = {}, [], [], 0, 0
        self.diag = []  # merge-rule diagnostics

    # ------------------------------------------------------------------ learners
    def ratio(self, D, i, m=None):
        m = D["L"]["m"] if m is None else m
        return self.S.rollout_err(m, self.Z[i]) / max(self.S.rollout_err(self.theta0, self.Z[i]), 1e-12)

    def mature(self, D):
        v = D["val"][-3:]
        return D["L"] is not None and len(v) >= 3 and float(np.mean(v)) <= 0.9

    def first3(self, m, i):
        z = self.Z[i]
        k = min(3, z.shape[1] - 1)
        return float(np.mean([float(torch.mean((self.S.predict(m, z[:, max(0, t - 3) : t])[:, -1:, ..., :-self.S.drop]
                                                 - z[:, t : t + 1, ..., :-self.S.drop]) ** 2)) for t in range(1, k + 1)]))

    def new_domain(self, kind="free", key=None):
        d = self.next_id if key is None else key
        if key is None:
            self.next_id += 1
        self.dom[d] = {"members": [], "fp": [], "L": None, "val": [], "valrec": [], "kind": kind,
                       "created": self.block, "reads": 0, "writes": 0, "absorbed": []}
        return d

    def write(self, d, i, fp):
        D = self.dom[d]
        D["writes"] += 1
        if D["L"] is not None:
            r = self.ratio(D, i)
            D["val"].append(r)
            D["valrec"].append((i, r))
        D["members"].append(i)
        if fp:
            D["fp"].append(i)
        if len(D["members"]) >= 3:
            if D["L"] is None:
                D["L"] = self.S.new_learner()
                self.S.train(D["L"], [self.Z[j] for j in D["members"]], self.S.steps * len(D["members"]))
            else:
                self.S.train(D["L"], [self.Z[j] for j in D["members"]], self.S.steps)

    def ccos(self, A, B):
        return float(self.G[np.ix_(A, B)].sum() / math.sqrt(max(self.G[np.ix_(A, A)].sum() * self.G[np.ix_(B, B)].sum(), 1e-12)))

    def centroid_cos(self, i, D):
        return self.ccos([i], D["fp"]) if D["fp"] else -1.0

    # ------------------------------------------------------------------ one episode
    def step(self, i):
        tag = self.tags[i]
        rec = {"i": i, "tag": tag, "read": None, "write": None, "rule": None}
        if self.feats is not None:
            rec["knn"] = self.knn_choice(i)
            rec["knn_scores"] = self.knn_scores(i)
        if self.v == "oracle":
            D = self.dom.get(tag)
            rec["read"] = tag if D is not None and self.mature(D) else None
        else:
            mats = {dk: D for dk, D in self.dom.items() if self.mature(D)}
            if mats:
                e0 = self.first3(self.theta0, i)
                errs = {dk: self.first3(D["L"]["m"], i) for dk, D in mats.items()}
                best = min(errs, key=errs.get)
                if errs[best] < e0:
                    rec["read"] = best
        rec["read_ratio"] = 1.0 if rec["read"] is None else self.ratio(self.dom[rec["read"]], i)
        if rec["read"] is not None:
            self.dom[rec["read"]]["reads"] += 1
        # ---- write
        if self.v in ("pool", "pool80"):  # one learner for everything theta0 does not already know
            if self.fam[i] > (self.thr90 if self.v == "pool" else self.thr80):
                if "U" not in self.dom:
                    self.new_domain("free", key="U")
                self.write("U", i, fp=False)
                rec.update(write="U", rule="pool")
            else:
                rec["rule"] = "familiar"
            return rec
        if self.v == "oracle":
            if tag not in self.dom:
                self.new_domain("free", key=tag)
            self.write(tag, i, fp=True)
            rec.update(write=tag, rule="oracle")
            return rec
        fam = self.fam[i]
        rd = rec["read"]
        if rd is not None and self.dom[rd]["kind"] == "free":
            if True:  # v4 writing (also v5, v5B, v5M)
                ok = fam > self.thr90 and rec["read_ratio"] <= 1.2 and \
                    float(self.S.loss(self.dom[rd]["L"]["m"], self.Z[i])) < float(self.S.loss(self.theta0, self.Z[i]))
            if ok:
                self.write(rd, i, fp=False)
                rec.update(write=rd, rule="learner")
                return rec
        if self.v in ("v4", "v5", "v5B", "v5M"):
            if fam <= self.thr90:
                rec["rule"] = "familiar"
                return rec
        cands = sorted(((self.centroid_cos(i, D), dk) for dk, D in self.dom.items() if D["kind"] == "free" and D["fp"]), reverse=True)
        for c, dk in cands:
            if c < 0.10:
                break
            D = self.dom[dk]
            if D["L"] is not None and self.ratio(D, i) > 1.2:
                continue
            self.write(dk, i, fp=True)
            rec.update(write=dk, rule="direction")
            return rec
        nb = [p for p in self.pending if self.G[i, p] >= 0.05]
        if len(nb) + 1 >= 3:
            group = nb[:2] + [i]
            self.pending = [p for p in self.pending if p not in group]
            d = self.new_domain()
            for j in group:
                self.write(d, j, fp=True)
            rec.update(write=d, rule="new_from_pending")
        else:
            self.pending = (self.pending + [i])[-40:]
            rec["rule"] = "pending"
        return rec

    # ------------------------------------------------------------------ KNN early read (evaluation)
    def knn_choice(self, i, k=5, vote=0.6, min_ref=3):
        ok = {dk for dk, D in self.dom.items() if self.mature(D)}
        refs = [(f, lab) for f, lab in self.knn_refs if lab == "theta0" or lab in ok]
        if len(refs) < k:
            return None
        X = np.stack([f for f, _ in refs])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - mu) / sd
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
        q = (self.feats[i] - mu) / sd
        q /= np.linalg.norm(q) + 1e-12
        votes = [refs[j][1] for j in np.argsort(-(Xn @ q))[:k]]
        best = max(set(votes), key=votes.count)
        if best == "theta0" or votes.count(best) < vote * k or sum(1 for _, lab in refs if lab == best) < min_ref:
            return None
        return best

    def knn_scores(self, i, m=3):
        """Class-balanced KNN: per class (theta0, each mature domain) the mean of its m most similar references."""
        ok = {dk for dk, D in self.dom.items() if self.mature(D)}
        refs = [(f, lab) for f, lab in self.knn_refs if lab == "theta0" or lab in ok]
        if not refs:
            return {}
        X = np.stack([f for f, _ in refs])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - mu) / sd
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
        q = (self.feats[i] - mu) / sd
        q /= np.linalg.norm(q) + 1e-12
        sim = Xn @ q
        labs = np.array([str(lab) for _, lab in refs])
        out = {}
        for lab in set(labs):
            v = np.sort(sim[labs == lab])[::-1]
            if lab == "theta0" or len(v) >= 3:
                out[lab] = float(v[:m].mean())
        return out

    def add_ref(self, rec):
        if self.feats is None:
            return
        if rec["write"] is not None:
            self.knn_refs.append((self.feats[rec["i"]], rec["write"]))
        elif rec["rule"] == "familiar":
            self.knn_refs.append((self.feats[rec["i"]], "theta0"))

    def v5_validate_all(self, rec):
        """v5: every existing learner validated on a newly written unfamiliar episode (before training on it)."""
        i, d = rec["i"], rec["write"]
        if d is None or self.fam[i] <= self.thr90:
            return
        ratios = {dk: self.ratio(D, i) for dk, D in self.dom.items() if D["L"] is not None and dk != d}
        own = self.dom[d]["valrec"][-1][1] if self.dom[d]["valrec"] and self.dom[d]["valrec"][-1][0] == i else None
        if own is not None:
            ratios[d] = own
        self.dom[d].setdefault("vals", []).append((i, ratios))

    def _merge_v5(self, eps=0.3, k=3):
        gm = lambda xs: math.exp(float(np.mean(np.log(np.maximum(xs, 1e-12)))))
        changed = True
        while changed:
            changed = False
            ids = sorted(dk for dk, D in self.dom.items() if D["L"] is not None and D["kind"] == "free")
            for x, a in enumerate(ids):
                for b in ids[x + 1 :]:
                    A, B = self.dom[a], self.dom[b]
                    on_a = [(r[b], r[a]) for _, r in A.get("vals", []) if a in r and b in r][-k:]
                    on_b = [(r[a], r[b]) for _, r in B.get("vals", []) if a in r and b in r][-k:]
                    if len(on_a) < k or len(on_b) < k:
                        continue
                    if (gm([p for p, _ in on_a]) <= min(1.0, (1 + eps) * gm([q for _, q in on_a]))
                            and gm([p for p, _ in on_b]) <= min(1.0, (1 + eps) * gm([q for _, q in on_b]))):
                        keep, gone = (a, b) if len(A["members"]) >= len(B["members"]) else (b, a)
                        n = len(self.dom[gone]["members"])
                        vals_gone = self.dom[gone].get("vals", [])
                        self._merge(keep, gone, "v5_symmetric")
                        self.dom[keep]["vals"] = self.dom[keep].get("vals", []) + vals_gone
                        self.S.train(self.dom[keep]["L"], [self.Z[j] for j in self.dom[keep]["members"]], self.S.steps * n)
                        changed = True
                        break
                if changed:
                    break

    # ------------------------------------------------------------------ merging
    def _merge(self, keep, gone, how, **info):
        K, G_ = self.dom[keep], self.dom.pop(gone)
        K["absorbed"].append({"block": self.block, "from": gone, "created": G_["created"], "n": len(G_["members"]),
                              "tags": {t: [self.tags[j] for j in G_["members"]].count(t) for t in set(self.tags[j] for j in G_["members"])},
                              "how": how, **info})
        K["members"] += G_["members"]
        K["fp"] += G_["fp"]
        K["valrec"] += G_["valrec"]
        self.merges.append(K["absorbed"][-1] | {"into": keep})

    def _merge_two_tier(self, absorb_ratio=0.8, small=6, hold=4):
        gm = lambda xs: math.exp(float(np.mean(np.log(np.maximum(xs, 1e-12)))))
        # tier 1: fragments too small to have a validation history are absorbed by the mature domain
        # whose learner already predicts their stored episodes best (they are unseen by that learner)
        changed = True
        while changed:
            changed = False
            bigs = [dk for dk, D in self.dom.items() if self.mature(D) and D["kind"] == "free"]
            for s_ in [dk for dk, D in self.dom.items() if D["kind"] == "free" and len(D["members"]) < small]:
                best, br = None, absorb_ratio
                for b in bigs:
                    if b == s_:
                        continue
                    r = float(np.median([self.ratio(self.dom[b], j) for j in self.dom[s_]["members"]]))
                    if r <= br:
                        best, br = b, r
                if best is not None:
                    n = len(self.dom[s_]["members"])
                    self._merge(best, s_, "absorb_fragment", ratio=round(br, 3))
                    self.S.train(self.dom[best]["L"], [self.Z[j] for j in self.dom[best]["members"]], self.S.steps * n)
                    changed = True
                    break
        # tier 2: two mature domains pool only if a learner trained on both is no worse on either side
        changed = True
        while changed:
            changed = False
            ids = sorted(dk for dk, D in self.dom.items() if self.mature(D) and D["kind"] == "free" and len(D["valrec"]) >= hold)
            for x, a in enumerate(ids):
                for b in ids[x + 1 :]:
                    A, B = self.dom[a], self.dom[b]
                    ha, hb = A["valrec"][-hold:], B["valrec"][-hold:]
                    xa = gm([self.ratio(B, j) for j, _ in ha])
                    xb = gm([self.ratio(A, j) for j, _ in hb])
                    if xa >= 1.0 or xb >= 1.0:
                        continue
                    keep, gone = (a, b) if len(A["members"]) >= len(B["members"]) else (b, a)
                    K, Gd = self.dom[keep], self.dom[gone]
                    held = {j for j, _ in ha + hb}
                    cand = {"m": copy.deepcopy(K["L"]["m"])}
                    for p in cand["m"].parameters():
                        p.requires_grad_(True)
                    cand["opt"] = torch.optim.Adam(cand["m"].parameters(), lr=self.S.lr)
                    self.S.train(cand, [self.Z[j] for j in K["members"] + Gd["members"] if j not in held],
                                 min(300, self.S.steps * len(Gd["members"])))
                    ca = gm([self.ratio(None, j, m=cand["m"]) for j, _ in ha])
                    cb = gm([self.ratio(None, j, m=cand["m"]) for j, _ in hb])
                    oa, ob = gm([r for _, r in ha]), gm([r for _, r in hb])
                    ok = ca < 1.0 and cb < 1.0 and ca <= oa and cb <= ob
                    self.diag.append({"block": self.block, "a": a, "b": b, "cross": [round(xa, 3), round(xb, 3)],
                                      "cand": [round(ca, 3), round(cb, 3)], "own": [round(oa, 3), round(ob, 3)], "merged": ok,
                                      "tags_a": self._maj(a), "tags_b": self._maj(b)})
                    if ok:
                        self._merge(keep, gone, "validated_strict", cand=[round(ca, 3), round(cb, 3)], own=[round(oa, 3), round(ob, 3)])
                        self.dom[keep]["L"] = cand
                        self.S.train(cand, [self.Z[j] for j in self.dom[keep]["members"]], self.S.steps * len(held))
                        changed = True
                        break
                if changed:
                    break

    def block_end(self):
        if self.v == "v5M":
            self._merge_two_tier()
        if self.v == "v5B":
            self._merge_validated()


    def _merge_validated(self, eps=0.1, hold=2, max_steps=300):
        gm = lambda xs: math.exp(float(np.mean(np.log(np.maximum(xs, 1e-12)))))
        changed = True
        while changed:
            changed = False
            ids = sorted(dk for dk, D in self.dom.items() if D["kind"] == "free" and D["L"] is not None and len(D["valrec"]) >= hold)
            for x, a in enumerate(ids):
                for b in ids[x + 1 :]:
                    A, B = self.dom[a], self.dom[b]
                    ha, hb = A["valrec"][-hold:], B["valrec"][-hold:]
                    # cheap pre-filter: each learner must transfer to the other's held-out episodes
                    xa = gm([self.ratio(B, j) for j, _ in ha])
                    xb = gm([self.ratio(A, j) for j, _ in hb])
                    if xa >= 1.0 or xb >= 1.0:
                        continue
                    keep, gone = (a, b) if len(A["members"]) >= len(B["members"]) else (b, a)
                    K, Gd = self.dom[keep], self.dom[gone]
                    held = {j for j, _ in ha + hb}
                    cand = {"m": copy.deepcopy(K["L"]["m"])}
                    for p in cand["m"].parameters():
                        p.requires_grad_(True)
                    cand["opt"] = torch.optim.Adam(cand["m"].parameters(), lr=self.S.lr)
                    union = [self.Z[j] for j in K["members"] + Gd["members"] if j not in held]
                    self.S.train(cand, union, min(max_steps, self.S.steps * len(Gd["members"])))
                    ca = gm([self.ratio(None, j, m=cand["m"]) for j, _ in ha])
                    cb = gm([self.ratio(None, j, m=cand["m"]) for j, _ in hb])
                    oa, ob = gm([r for _, r in ha]), gm([r for _, r in hb])
                    ok = ca < 1.0 and cb < 1.0 and ca <= (1 + eps) * oa and cb <= (1 + eps) * ob
                    self.diag.append({"block": self.block, "a": a, "b": b, "cross": [round(xa, 3), round(xb, 3)],
                                      "cand": [round(ca, 3), round(cb, 3)], "own": [round(oa, 3), round(ob, 3)], "merged": ok,
                                      "tags_a": self._maj(a), "tags_b": self._maj(b)})
                    if ok:
                        self._merge(keep, gone, "validated", cand=[round(ca, 3), round(cb, 3)], own=[round(oa, 3), round(ob, 3)])
                        self.dom[keep]["L"] = cand
                        self.S.train(cand, [self.Z[j] for j in self.dom[keep]["members"]], self.S.steps * len(held))
                        changed = True
                        break
                if changed:
                    break

    def _maj(self, d):
        ts = [self.tags[j] for j in self.dom[d]["members"]]
        return {t: ts.count(t) for t in sorted(set(ts), key=lambda t: -ts.count(t))[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="S7b3")
    ap.add_argument("--run", default="eval_outputs/cont_S7b/w0")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--cal", default="reports/cont_S7/route_screen/cache_S7.npz")
    ap.add_argument("--out", default="reports/cont_S7b3/route_func")
    ap.add_argument("--knn", action="store_true", help="also evaluate the KNN early read (start+goal encoder features)")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre, _ = load_world(ROOT / "checkpoints/pushobj_bias_16k_e3_seed0", "3", dev)
    blocks, eps = load_stream(ROOT / a.run, a.stream, pre, dev)
    visit, cnt = [], {}
    for b in blocks:
        visit.append(cnt.get(b["shape"], 0) + 1)
        cnt[b["shape"]] = visit[-1]
    tags = [e["tag"] for e in eps]
    vis = [visit[e["block"]] for e in eps]
    Z, fam = [], []
    with torch.no_grad():
        for e in eps:
            z = latents(wm, e["obs"], e["act"])
            Z.append(z)
            fam.append(new3(wm, z))
            e["obs"] = None
    U = torch.stack([(e["delta"] / (e["delta"].norm() + 1e-12)).half() for e in eps])
    for e in eps:
        e["delta"] = None
    G = (U.float() @ U.float().T).cpu().numpy().astype(np.float64)
    del U
    torch.cuda.empty_cache()
    cal = np.load(ROOT / a.cal)["cal"]
    feats = None
    if a.knn:
        import pickle
        from cont_knn_latent import Features
        Fx = Features(wm, pre, dev)
        feats = []
        for k, b in enumerate(blocks):
            t = pickle.load(open(ROOT / a.run / b["chunk"] / "plan_targets.pkl", "rb"))
            n = sum(1 for e in eps if e["block"] == k)
            f0 = Fx(t["obs_0"]["visual"][:n, 0])["pool"]
            fg = Fx(t["obs_g"]["visual"][:n, 0])["pool"]
            feats += [np.concatenate([x, y]) for x, y in zip(f0, fg)]
    router = R(a.variant, Slow(wm), wm.predictor, G, np.array(fam), float(np.percentile(cal, 80)), float(np.percentile(cal, 90)), Z, tags,
               feats=feats)
    recs = []
    for k, b in enumerate(blocks):
        router.block = k
        for i in [j for j, e in enumerate(eps) if e["block"] == k]:
            rec = router.step(i)
            if a.variant == "v5":
                router.v5_validate_all(rec)
                router._merge_v5()
            router.add_ref(rec)
            recs.append(rec)
        router.block_end()
    for r in recs:
        r["visit"] = vis[r["i"]]
    final_of = {j: dk for dk, D in router.dom.items() for j in D["members"]}
    summary = {"variant": a.variant, "stream": a.stream, "records": recs, "merges": router.merges, "diag": router.diag,
               "domains": {str(dk): {"kind": D["kind"], "mature": router.mature(D), "n": len(D["members"]),
                                     "tags": {t: [tags[j] for j in D["members"]].count(t) for t in set(tags[j] for j in D["members"])},
                                     "reads": D["reads"], "writes": D["writes"], "created": D["created"], "absorbed": D["absorbed"]}
                           for dk, D in router.dom.items()},
               "coverage": {s: float(np.mean([r["i"] in final_of for r in recs if r["tag"] == s])) for s in dict.fromkeys(tags)},
               "blocks": [(b["chunk"], b["shape"], visit[k]) for k, b in enumerate(blocks)]}
    (out / f"{a.variant}.json").write_text(json.dumps(summary, ensure_ascii=False, default=float))
    gm = lambda xs: math.exp(float(np.mean(np.log(np.maximum(xs, 1e-12))))) if xs else float("nan")
    for grp in ("unseen", "seen"):
        for v in (1, 2, 3):
            rr = [r["read_ratio"] for r in recs if (r["tag"] in SEEN) == (grp == "seen") and r["visit"] == v]
            if rr:
                print(f"{a.variant} {grp} visit{v}: read ratio gmean {gm(rr):.3f}, useful(<0.8) {np.mean(np.array(rr) < 0.8) * 100:.0f}%, "
                      f"harmful(>1) {np.mean(np.array(rr) > 1.0) * 100:.0f}%", flush=True)
    print(f"{a.variant}: domains {len(router.dom)}, merges {len(router.merges)}", flush=True)


if __name__ == "__main__":
    main()
