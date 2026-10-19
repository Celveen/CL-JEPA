"""v5: the v4 label-free router plus validated, symmetric domain merging.

Splitting decisions are taken online with little evidence, so the router over-splits (one
shape can found several domains). CLSMergeMPCPlanner keeps everything of CLSMPCPlanner and
adds the merge rule designed offline (scripts/cont_merge_screen.py, reports/cont_S7/merge_design.md):

  * every unfamiliar episode is validated, before any learner trains on it, by EVERY existing
    learner (5-step open-loop latent error over theta0's); the ratios are kept per domain;
  * domains A and B merge when, on the last merge_k episodes of each where both learners
    were validated, B's learner predicts A's episodes better than theta0 and within
    (1 + merge_eps) of A's own learner, and vice versa. Merging is judged by transfer, not by
    label purity: similar shapes whose knowledge transfers may merge, others never do.

A merge keeps the learner of the larger domain, which takes the union of the replays and
catches up on the absorbed part. The absorbed domain stays as an empty tombstone
(merged_into), so domain keys are never reused.
"""
import math

import numpy as np

from .continual import CLSMPCPlanner


class CLSMergeMPCPlanner(CLSMPCPlanner):
    def __init__(self, continual=None, **kwargs):
        c = dict(continual or {})
        self.merge_eps = float(c.get("merge_eps", 0.3))
        self.merge_k = int(c.get("merge_k", 3))
        super().__init__(continual=continual, **kwargs)

    def _end_episode(self):
        ratios = {}
        if self.write_mode != "oracle":
            z = self._episode_latents()
            if self.familiar_thr <= 0 or self._familiarity(z) > self.familiar_thr:
                e0 = max(self._rollout_err(self._pred0_module, z), 1e-12)
                ratios = {dk: self._rollout_err(d["learner"], z) / e0
                          for dk, d in self.domains.items() if d["learner"] is not None}
        super()._end_episode()
        rec = self._ep_records[-1]
        merges = []
        if rec["write"] is not None and ratios:
            self.domains[rec["write"]].setdefault("vals", []).append((rec["episode"], ratios))
            merges = self._try_merge()
        rec["val_all"] = ratios
        rec["merges"] = merges

    def _try_merge(self):
        gm = lambda xs: math.exp(float(np.mean(np.log(np.maximum(xs, 1e-12)))))
        merges, changed = [], True
        while changed:
            changed = False
            ids = sorted(dk for dk, d in self.domains.items() if d["learner"] is not None and not d.get("merged_into"))
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    A, B = self.domains[a], self.domains[b]
                    on_a = [(r[b], r[a]) for _, r in A.get("vals", []) if a in r and b in r][-self.merge_k :]
                    on_b = [(r[a], r[b]) for _, r in B.get("vals", []) if a in r and b in r][-self.merge_k :]
                    if len(on_a) < self.merge_k or len(on_b) < self.merge_k:
                        continue
                    ok_a = gm([x for x, _ in on_a]) <= min(1.0, (1 + self.merge_eps) * gm([y for _, y in on_a]))
                    ok_b = gm([x for x, _ in on_b]) <= min(1.0, (1 + self.merge_eps) * gm([y for _, y in on_b]))
                    if ok_a and ok_b:
                        keep, gone = (a, b) if A["n"] >= B["n"] else (b, a)
                        merges.append({"keep": keep, "gone": gone, "tags_keep": dict(self.domains[keep]["tags"]),
                                       "tags_gone": dict(self.domains[gone]["tags"])})
                        self._merge(keep, gone)
                        changed = True
                        break
                if changed:
                    break
        return merges

    def _merge(self, keep, gone):
        K, G = self.domains[keep], self.domains[gone]
        K["n"] += G["n"]
        for t, c in G["tags"].items():
            K["tags"][t] = K["tags"].get(t, 0) + c
        K["replay"] += G["replay"]
        if G["fp_sum"] is not None:
            K["fp_sum"] = G["fp_sum"] if K["fp_sum"] is None else [s + x for s, x in zip(K["fp_sum"], G["fp_sum"])]
            K["fp_n"] += G["fp_n"]
        K["vals"] = K.get("vals", []) + G.get("vals", [])
        for D in self.domains.values():  # the absorbed learner no longer exists
            D["vals"] = [(e, {o: v for o, v in r.items() if o != gone}) for e, r in D.get("vals", [])]
        self._train(K, self.slow_steps * len(G["replay"]))  # catch up on the absorbed replay
        self.domains[gone] = {"n": 0, "tags": {}, "replay": [], "fp_sum": None, "fp_n": 0, "learner": None,
                              "opt": None, "val": [], "vals": [], "merged_into": keep}

    def _save_state(self, path):
        import torch
        super()._save_state(path)
        st = torch.load(path, map_location="cpu")
        st["merge"] = {dk: {"vals": d.get("vals", []), "merged_into": d.get("merged_into")} for dk, d in self.domains.items()}
        torch.save(st, path)

    def _load_state(self, path):
        import torch
        super()._load_state(path)
        for dk, m in torch.load(path, map_location="cpu").get("merge", {}).items():
            if dk in self.domains:
                self.domains[dk]["vals"] = [(e, dict(r)) for e, r in m.get("vals", [])]
                if m.get("merged_into"):
                    self.domains[dk]["merged_into"] = m["merged_into"]
