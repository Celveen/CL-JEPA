"""v6: v5 (the classmate's settings) with slightly relaxed writing and merging.

Everything not listed is v5 (CLSMergeMPCPlanner with tau 0.10, familiar_thr 0.0146, pending_m 3,
tau_new 0.05, fp_unit, mature_k 3, mature_ratio 0.9, reject_ratio 1.2, slow learners as CLS).

1. The trained shapes are learned too. v5 never writes a familiar episode (theta0 error over its
   first 3 transitions <= familiar_thr). Their fast-change fingerprints carry no shape signal, so
   v5's routing cannot sort them (a smoke test left all of them pending). v6 sorts them with a
   seen-shape probe: a linear classifier over theta0 encoder latents trained on the base model's
   own training data, labelled with the shapes it was trained on (scripts/cont_train_seen_probe.py;
   no deployment labels). A familiar episode the probe names with confidence >= seen_conf goes to
   that training shape's domain ("seen" kind, one per shape). Like any learner, a seen domain's
   learner is read only once its validation ratios beat theta0 (maturity gate): learned first,
   used when better. Every other episode -- unfamiliar, or familiar but not confidently a training
   shape (mostly low-error square / small_tee, which v5 dropped) -- is routed label-free as in v5
   among the "novel" domains. The two kinds never mix.
2. Learner writes relaxed. An episode that read a novel learner is written there unless the write
   gate rejects it (validation ratio > reject_ratio); v5 also required the learner to predict the
   whole episode better than theta0.
3. Merging relaxed. v5's symmetric rule with merge_k 2 (was 3) and merge_eps 0.5 (was 0.3), still
   requiring both learners to beat theta0 on each other's episodes; novel domains only.
Reading is unchanged (lowest error among theta0 and mature learners over the first replans).
"""
import copy
import math

import numpy as np
import torch

from .adajepa_probe import _cos
from .continual import ContinualMPCPlanner
from .continual_merge import CLSMergeMPCPlanner


class CLSv6MPCPlanner(CLSMergeMPCPlanner):
    def __init__(self, continual=None, **kwargs):
        c = dict(continual or {})
        super().__init__(continual=continual, **kwargs)
        self.seen_conf = float(c.get("seen_conf", 0.95))
        self.probe = None
        if c.get("seen_probe"):
            pr = torch.load(c["seen_probe"], map_location="cpu")
            dev = self._params[0].device
            self.probe = {k: pr[k].to(dev) for k in ("W", "b", "mu", "sd")}
            self.probe["classes"] = list(pr["classes"])

    @torch.no_grad()
    def _seen_class(self, z):
        """Probe on the theta0 visual latents of the first 3 executed frames: (training shape, confidence)."""
        if self.probe is None:
            return None, 0.0
        v = z[0, :3, 0, : self.probe["W"].shape[1]].float()
        p = torch.softmax(((v - self.probe["mu"]) / self.probe["sd"]) @ self.probe["W"].T + self.probe["b"], dim=1).mean(0)
        return self.probe["classes"][int(p.argmax())], float(p.max())

    @staticmethod
    def _kind(d):
        return d.get("kind", "novel")

    def _end_episode(self):
        delta = [p.detach() - s for p, s in zip(self._params, self._start)]
        z = self._episode_latents()
        fam = self._familiarity(z)
        seen_cls, seen_conf = self._seen_class(z)
        kind = "seen" if fam <= self.familiar_thr and seen_conf >= self.seen_conf else "novel"
        # every existing learner validated on this (unseen) episode: merge evidence, and the write gate's ratio
        e0 = max(self._rollout_err(self._pred0_module, z), 1e-12)
        ratios = {dk: self._rollout_err(d["learner"], z) / e0 for dk, d in self.domains.items()
                  if d["learner"] is not None and not d.get("merged_into")}
        self._val_cache = dict(ratios)
        rule, key, rejected, new_members = "none", None, [], []
        base = self._base
        if kind == "seen":  # a training shape: its own domain
            rule, key = "seen_probe", f"seen_{seen_cls}"
        elif base is not None and self._kind(self.domains[base]) == "novel":
            if self._rejects(base, z):
                rejected.append(base)
                rule = "rejected"
            else:
                rule, key = "learner", base
        if key is None and rule != "rejected":
            cands = sorted(((_cos(delta, [s / d["fp_n"] for s in d["fp_sum"]]), dk) for dk, d in self.domains.items()
                            if d["fp_n"] > 0 and self._kind(d) == "novel" and not d.get("merged_into")), reverse=True)
            rule = "direction"
            for c, dk in cands:
                if c < self.tau:
                    break
                if self._rejects(dk, z):
                    rejected.append(dk)
                    continue
                key = dk
                break
            if key is None:
                rule, key, new_members = self._pend(delta, z)
        slow_loss, val_ratio = float("nan"), float("nan")
        if key is not None:
            d = self.domains.setdefault(key, {"n": 0, "tags": {}, "replay": [], "fp_sum": None, "fp_n": 0,
                                              "learner": None, "opt": None, "val": [], "kind": kind})
            if d["learner"] is not None:  # unseen by this learner: validate before training on it
                val_ratio = self._val_ratio(key, z)
                d.setdefault("val", []).append(float(val_ratio))
            dev = self._params[0].device
            members = [(m["tag"], m["z"].to(dev), self._unflat(m["fp"].float().to(dev))) for m in new_members]
            for tag, mz, mfp in members + [(self.tag, z.detach(), self._fp(delta) if base is None else None)]:
                d["n"] += 1
                d["tags"][tag] = d["tags"].get(tag, 0) + 1
                d["replay"].append(mz)
                if mfp is not None and kind == "novel":  # theta0-read episodes define the fingerprint
                    d["fp_sum"] = [x.clone() for x in mfp] if d["fp_sum"] is None else [s + x for s, x in zip(d["fp_sum"], mfp)]
                    d["fp_n"] += 1
            if d["n"] >= self.min_n:
                if d["learner"] is None:
                    d["learner"] = copy.deepcopy(self._pred0_module)
                    for p in d["learner"].parameters():
                        p.requires_grad_(True)
                    d["opt"] = torch.optim.Adam(d["learner"].parameters(), lr=self.slow_lr)
                    slow_loss = self._train(d, self.slow_steps * d["n"])  # catch up on what it already holds
                else:
                    slow_loss = self._train(d, self.slow_steps)
        ContinualMPCPlanner._end_episode(self)
        rec = self._ep_records[-1]
        merges = []
        if key is not None and ratios:
            self.domains[key].setdefault("vals", []).append((rec["episode"], ratios))
            merges = self._try_merge()
        rec.update({
            "read": base, "route": self._route_log, "write": key, "write_rule": rule, "kind": kind,
            "slow_loss": slow_loss, "val_ratio": val_ratio, "rejected_by": rejected,
            "rejected_ratios": {dk: self._val_cache[dk] for dk in rejected if dk in self._val_cache},
            "familiar_score": fam, "seen_class": seen_cls, "seen_conf": seen_conf, "pending_n": len(self.pending),
            "new_members": [m["tag"] for m in new_members], "val_all": ratios, "merges": merges,
            "domains": {dk: {"kind": self._kind(d), "n": d["n"], "tags": dict(d["tags"]), "learner": d["learner"] is not None,
                             "mature": d["learner"] is not None and self._mature(d), "val_last": d.get("val", [])[-3:]}
                        for dk, d in self.domains.items() if d["n"] >= 2},
        })

    def _try_merge(self):
        """v5's symmetric rule between novel domains (merge_k / merge_eps from the config)."""
        gm = lambda xs: math.exp(float(np.mean(np.log(np.maximum(xs, 1e-12)))))
        merges, changed = [], True
        while changed:
            changed = False
            ids = sorted(dk for dk, d in self.domains.items() if d["learner"] is not None and not d.get("merged_into"))
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    A, B = self.domains[a], self.domains[b]
                    if self._kind(A) != "novel" or self._kind(B) != "novel":
                        continue
                    on_a = [(r[b], r[a]) for _, r in A.get("vals", []) if a in r and b in r][-self.merge_k :]
                    on_b = [(r[a], r[b]) for _, r in B.get("vals", []) if a in r and b in r][-self.merge_k :]
                    if len(on_a) < self.merge_k or len(on_b) < self.merge_k:
                        continue
                    ok_a = gm([x for x, _ in on_a]) <= min(1.0, (1 + self.merge_eps) * gm([y for _, y in on_a]))
                    ok_b = gm([x for x, _ in on_b]) <= min(1.0, (1 + self.merge_eps) * gm([y for _, y in on_b]))
                    if ok_a and ok_b:
                        keep, gone = (a, b) if A["n"] >= B["n"] else (b, a)
                        merges.append({"keep": keep, "gone": gone, "kind": self._kind(A),
                                       "tags_keep": dict(self.domains[keep]["tags"]), "tags_gone": dict(self.domains[gone]["tags"])})
                        self._merge(keep, gone)
                        self.domains[gone]["kind"] = self._kind(self.domains[keep])
                        changed = True
                        break
                if changed:
                    break
        return merges

    def _save_state(self, path):
        super()._save_state(path)
        st = torch.load(path, map_location="cpu")
        st["v6"] = {"kind": {dk: self._kind(d) for dk, d in self.domains.items()}}
        torch.save(st, path)

    def _load_state(self, path):
        super()._load_state(path)
        v6 = torch.load(path, map_location="cpu").get("v6", {})
        for dk, kd in v6.get("kind", {}).items():
            if dk in self.domains:
                self.domains[dk]["kind"] = kd
