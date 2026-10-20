"""Independent extensions of v5 (CLSMergeMPCPlanner), each changing one thing.

CLSStickyReadMPCPlanner -- early read by carrying over the previous episode's read.
    v5's first plan always uses theta0. Here an episode starts on the learner the previous
    episode ended on (followed through merges), provided it is still mature; consecutive
    episodes tend to come from the same task, and no label is needed. The usual error-based
    reroute over the first 3 replans still runs, so a wrong carry-over (typically the first
    episode after a task switch) is corrected after the first executed segment. Writing is
    unchanged. Offline (reports/cont_S7b3/v5x_offline_check.md) this picked the episode's own
    domain about as often as v5's read after replans 1-3 did.

CLSKnnReadMPCPlanner -- early read by KNN on encoder features.
    v5 decides what to plan with only after the first transitions (reroute over replans 1-3),
    so the first plan always uses theta0. Here, before the first plan, the start and goal images
    are encoded (theta0 backbone, mean+max pooled last feature map) and compared by KNN with the
    same features of earlier episodes, labelled by where they went: the domain an episode was
    written to, or "theta0" for familiar episodes that were not written. If a mature domain wins
    at least knn_vote of the knn_k votes, its learner is loaded before the first plan (as the
    oracle does with true labels). The usual error-based reroute over the first 3 replans still
    runs and can switch to another learner or back to theta0. Writing is unchanged.

CLSMergeBMPCPlanner -- merging by validated pooling (offline "B").
    v5 merges two domains only after each has collected new cross-validated episodes, which
    small domains rarely do, so fragments (e.g. several small_tee domains) never merge. Here,
    every mergeb_every episodes, for two domains whose learners transfer to each other's last
    mergeb_hold validation episodes (ratio < 1 both ways), a candidate learner -- the larger
    domain's, trained on the union of both replays minus those held-out episodes -- is validated
    on them. The domains merge when the candidate predicts both sides better than theta0 and
    within (1 + mergeb_eps) of each domain's own recorded validation (taken when those episodes
    were still unseen); the merged domain keeps the candidate learner.
"""
import copy
import math

import numpy as np
import torch

from .adajepa_probe import ProbeMPCPlanner
from .continual_merge import CLSMergeMPCPlanner


class CLSStickyReadMPCPlanner(CLSMergeMPCPlanner):
    def __init__(self, continual=None, **kwargs):
        if not hasattr(self, "_last_read"):
            self._last_read = None  # the learner the previous episode ended on (None: theta0)
        self._early = None
        super().__init__(continual=continual, **kwargs)

    def _carried(self):
        """The previous episode's final read, followed through merges; None unless it may be read now."""
        key = self._last_read
        while key is not None and self.domains.get(key, {}).get("merged_into"):
            key = self.domains[key]["merged_into"]
        return key if key in self._learners() else None

    def _plan_single(self, **kw):
        # as CLSMPCPlanner._plan_single, plus the carried-over read before the first plan
        with torch.no_grad():
            for p, q in zip(self._pred_params, self._pred0):
                p.copy_(q)
            for p, q in zip(self._params, self._init):
                p.copy_(q)
        self._base, self._route_log = None, []
        self._start = [p.detach().clone() for p in self._params]
        self._early = self._carried()
        if self._early is not None:
            self._load_base(self._early, carry_fast=False)
        out = ProbeMPCPlanner._plan_single(self, **kw)
        self._end_episode()
        return out

    def _end_episode(self):
        super()._end_episode()
        rec = self._ep_records[-1]
        rec["early_read"] = self._early
        self._last_read = rec["read"]

    def _save_state(self, path):
        super()._save_state(path)
        st = torch.load(path, map_location="cpu")
        st["last_read"] = self._last_read
        torch.save(st, path)

    def _load_state(self, path):
        super()._load_state(path)
        self._last_read = torch.load(path, map_location="cpu").get("last_read")


class CLSKnnReadMPCPlanner(CLSMergeMPCPlanner):
    def __init__(self, continual=None, **kwargs):
        c = dict(continual or {})
        self.knn_k = int(c.get("knn_k", 5))
        self.knn_vote = float(c.get("knn_vote", 0.6))
        self.knn_min_ref = int(c.get("knn_min_ref", 3))
        if not hasattr(self, "knn_refs"):
            self.knn_refs = []  # [(feature (2048,), label)] label = domain key or "theta0"
        self._fmap = {}
        super().__init__(continual=continual, **kwargs)
        self.wm.encoder.rb5.register_forward_hook(lambda m, i, o: self._fmap.__setitem__("map", o))

    @torch.no_grad()
    def _pool(self, obs):
        x = self.preprocessor.transform_obs_visual(obs["visual"])
        x = x.reshape(-1, *x.shape[-3:])[:1].to(self._params[0].device)
        self.wm.encoder(x)  # the backbone is never adapted, so these are theta0 features
        m = self._fmap["map"]
        return torch.cat([m.mean((2, 3)), m.amax((2, 3))], 1)[0].float().cpu().numpy()

    def _knn(self, feat):
        ok = {dk for dk, d in self.domains.items() if d["learner"] is not None and self._mature(d) and not d.get("merged_into")}
        refs = [(f, lab) for f, lab in self.knn_refs if lab == "theta0" or lab in ok]
        if len(refs) < self.knn_k:
            return None, {}
        X = np.stack([f for f, _ in refs])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - mu) / sd
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
        q = (feat - mu) / sd
        q /= np.linalg.norm(q) + 1e-12
        nn = np.argsort(-(Xn @ q))[: self.knn_k]
        votes = [refs[j][1] for j in nn]
        tally = {lab: votes.count(lab) for lab in set(votes)}
        best = max(tally, key=tally.get)
        n_best = sum(1 for _, lab in refs if lab == best)
        if best == "theta0" or tally[best] < self.knn_vote * self.knn_k or n_best < self.knn_min_ref:
            return None, tally
        return best, tally

    def _plan_single(self, **kw):
        # as CLSMPCPlanner._plan_single, plus the KNN read before the first plan
        self._cur_feat = np.concatenate([self._pool(kw["obs_0_i"]), self._pool(kw["obs_g_i"])])
        with torch.no_grad():
            for p, q in zip(self._pred_params, self._pred0):
                p.copy_(q)
            for p, q in zip(self._params, self._init):
                p.copy_(q)
        self._base, self._route_log = None, []
        self._start = [p.detach().clone() for p in self._params]
        choice, tally = self._knn(self._cur_feat)
        if choice is not None:
            self._load_base(choice, carry_fast=False)
        self._knn_log = {"choice": choice, "votes": {str(k): v for k, v in tally.items()}}
        out = ProbeMPCPlanner._plan_single(self, **kw)
        self._end_episode()
        return out

    def _end_episode(self):
        super()._end_episode()
        rec = self._ep_records[-1]
        rec["knn"] = self._knn_log
        if rec["write"] is not None:
            self.knn_refs.append((self._cur_feat, rec["write"]))
        elif rec.get("write_rule") == "familiar":
            self.knn_refs.append((self._cur_feat, "theta0"))
        for m in rec.get("merges", []):  # references follow merged domains
            self.knn_refs = [(f, m["keep"] if lab == m["gone"] else lab) for f, lab in self.knn_refs]

    def _save_state(self, path):
        super()._save_state(path)
        st = torch.load(path, map_location="cpu")
        st["knn_refs"] = [(f, lab) for f, lab in self.knn_refs]
        torch.save(st, path)

    def _load_state(self, path):
        super()._load_state(path)
        self.knn_refs = list(torch.load(path, map_location="cpu").get("knn_refs", []))


class CLSMergeBMPCPlanner(CLSMergeMPCPlanner):
    def __init__(self, continual=None, **kwargs):
        c = dict(continual or {})
        self.mergeb_every = int(c.get("mergeb_every", 5))
        self.mergeb_eps = float(c.get("mergeb_eps", 0.1))
        self.mergeb_hold = int(c.get("mergeb_hold", 2))
        self.mergeb_max_steps = int(c.get("mergeb_max_steps", 300))
        self._seen_eps = set()
        super().__init__(continual=continual, **kwargs)

    def _sync_valrec(self):
        """Record (replay index, validation ratio) of the episode just written, before any merge check."""
        rec = self._ep_records[-1]
        if rec["episode"] in self._seen_eps:
            return
        self._seen_eps.add(rec["episode"])
        r = rec.get("val_ratio")
        if rec["write"] is not None and r is not None and isinstance(r, float) and math.isfinite(r):
            d = self.domains[rec["write"]]
            d.setdefault("valrec", []).append((len(d["replay"]) - 1, float(r)))

    def _ratio(self, module, z):
        return self._rollout_err(module, z) / max(self._rollout_err(self._pred0_module, z), 1e-12)

    def _try_merge(self):
        # called by CLSMergeMPCPlanner after every unfamiliar episode it wrote; check every mergeb_every calls
        self._sync_valrec()
        self._mb_calls = getattr(self, "_mb_calls", 0) + 1
        if self._mb_calls % self.mergeb_every:
            return []
        gm = lambda xs: math.exp(float(np.mean(np.log(np.maximum(xs, 1e-12)))))
        ids = sorted(dk for dk, d in self.domains.items() if d["learner"] is not None and not d.get("merged_into")
                     and len(d.get("valrec", [])) >= self.mergeb_hold)
        tests = []
        for x, a in enumerate(ids):
            for b in ids[x + 1 :]:
                A, B = self.domains[a], self.domains[b]
                ha, hb = A["valrec"][-self.mergeb_hold :], B["valrec"][-self.mergeb_hold :]
                za, zb = [A["replay"][j] for j, _ in ha], [B["replay"][j] for j, _ in hb]
                xa, xb = gm([self._ratio(B["learner"], z) for z in za]), gm([self._ratio(A["learner"], z) for z in zb])
                if xa >= 1.0 or xb >= 1.0:
                    continue
                keep, gone = (a, b) if A["n"] >= B["n"] else (b, a)
                K, G = self.domains[keep], self.domains[gone]
                held_k = {j for j, _ in (ha if keep == a else hb)}
                held_g = {j for j, _ in (hb if keep == a else ha)}
                cand = copy.deepcopy(K["learner"])
                for p in cand.parameters():
                    p.requires_grad_(True)
                tmp = {"learner": cand, "opt": torch.optim.Adam(cand.parameters(), lr=self.slow_lr),
                       "replay": [z for j, z in enumerate(K["replay"]) if j not in held_k]
                       + [z for j, z in enumerate(G["replay"]) if j not in held_g]}
                self._train(tmp, min(self.mergeb_max_steps, self.slow_steps * len(G["replay"])))
                ca, cb = gm([self._ratio(cand, z) for z in za]), gm([self._ratio(cand, z) for z in zb])
                oa, ob = gm([r for _, r in ha]), gm([r for _, r in hb])
                ok = ca < 1.0 and cb < 1.0 and ca <= (1 + self.mergeb_eps) * oa and cb <= (1 + self.mergeb_eps) * ob
                tests.append({"a": a, "b": b, "cross": [round(xa, 3), round(xb, 3)], "cand": [round(ca, 3), round(cb, 3)],
                              "own": [round(oa, 3), round(ob, 3)], "merged": ok,
                              "tags_a": dict(A["tags"]), "tags_b": dict(B["tags"])})
                if ok:
                    off = len(K["replay"])
                    K["valrec"] = K.get("valrec", []) + [(j + off, r) for j, r in G.get("valrec", [])]
                    K["learner"], K["opt"] = cand, tmp["opt"]
                    self._merge(keep, gone)  # union of replays and fingerprints, catch-up on the absorbed part, tombstone
                    self._ep_records[-1].setdefault("mergeb", []).append(tests[-1])
                    return [{"keep": keep, "gone": gone, "tags_keep": dict(K["tags"]), "how": "validated_pooling"}]
        if tests:
            self._ep_records[-1].setdefault("mergeb", []).extend(tests)
        return []

    def _save_state(self, path):
        super()._save_state(path)
        st = torch.load(path, map_location="cpu")
        st["valrec"] = {dk: d.get("valrec", []) for dk, d in self.domains.items()}
        torch.save(st, path)

    def _load_state(self, path):
        super()._load_state(path)
        for dk, v in torch.load(path, map_location="cpu").get("valrec", {}).items():
            if dk in self.domains:
                self.domains[dk]["valrec"] = list(v)
