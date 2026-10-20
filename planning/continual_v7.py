"""v7 family: v5 (the classmate's settings) with fewer new domains, as three arms of one planner.

  cls7_free   items 1-5 below (writes prefer existing domains, slower founding, absorption, relaxed merge)
  cls7c_free  as cls7_free, but v5's founding (3 pending episodes), v5's merge (merge_k 3, merge_eps 0.3),
              and absorption limited to fragments (absorb_frag)
  cls5a_free  v5's writing and merging unchanged (learner_win, join_ratio 0, no pending joins), plus item 1
              and fragment absorption
  cls6a_free  v6 without its seen-shape probe: v6's relaxed learner writes (item 2 without the join) and
              relaxed merge (item 5), plus item 1 and fragment absorption (no joins, v5's founding)
Offline on S7b3 (reports/cont_S7b3/v7_offline_replay.md) easier writes moved I episodes into the
square-led domain and worsened the I learner (0.59-0.81 vs 0.52) while cls5a_free kept v5's read
quality; online, sharing one learner between I and square helped square on S7t3, so all three are kept.

Everything not listed is v5 (CLSMergeMPCPlanner with tau 0.10, familiar_thr 0.0146, tau_new 0.05, fp_unit,
mature_k 3, mature_ratio 0.9, reject_ratio 1.2, slow learners as CLS). Familiar episodes (theta0 error over
the first 3 transitions <= familiar_thr) are never written, as in v5. Nothing uses shape labels, so the
rules carry over to visual shifts unchanged.

1. Familiar data reads theta0 (familiar_read). While the read is re-decided over the first replans, an
   episode whose theta0 error on the transitions so far is within familiar_thr (the write gate's
   statistic) stays on theta0: no learner is read for data theta0 already predicts. On S7t3, v5's
   learner reads on familiar trained-shape episodes were net harmful (2/6 vs AdaJEPA).
2. Writing goes to existing domains first. An episode that read a learner is written there unless the
   write gate rejects it (v5 also required a 1-step win over theta0 on the whole episode). Otherwise an
   unfamiliar episode joins the mature domain whose learner predicts it best, if that 5-step ratio
   (already computed for every learner as merge evidence) is at most join_ratio; then, for episodes
   planned with theta0, v5's fingerprint join; only then the pending pool.
3. A new domain needs pending_m aligned pending episodes (5 in the cls7_free arm; v5: 3).
4. Every absorb_every episodes, an immature domain whose stored episodes some mature learner predicts
   within absorb_ratio of theta0 is absorbed by that domain, and pending episodes such a learner predicts
   within join_ratio join its domain. Immature domains are never read and fingerprints of hard shapes
   rarely align, so they get no new episodes and could otherwise neither mature nor merge (v6 on S7t3: 11
   such fragments; the final small_tee learner predicted 7 of them at 0.43-0.79 of theta0).
5. Merging between mature domains as v5, with merge_k / merge_eps from the config (2 / 0.5 in cls7_free).
Switches: learner_win keeps v5's learner rule (the read learner must also beat theta0's 1-step error on
the whole episode); absorb_frag > 0 limits absorption to domains with fewer episodes and fewer than
mature_k validations (never mature); absorb_pending: false leaves the pending pool alone.
Join and absorption writes do not count as validations: those episodes were chosen because the learner
predicts them well, so they would bias the maturity gate.
"""
import copy
import math

import numpy as np
import torch

from .adajepa_probe import _cos
from .continual import ContinualMPCPlanner
from .continual_merge import CLSMergeMPCPlanner


def _gmean(xs):
    return math.exp(float(np.mean(np.log(np.maximum(xs, 1e-12)))))


class CLSv7MPCPlanner(CLSMergeMPCPlanner):
    def __init__(self, continual=None, **kwargs):
        c = dict(continual or {})
        self.familiar_read = bool(c.get("familiar_read", True))
        self.join_ratio = float(c.get("join_ratio", 0.9))
        self.absorb_ratio = float(c.get("absorb_ratio", 0.8))
        self.absorb_every = int(c.get("absorb_every", 25))
        self.learner_win = bool(c.get("learner_win", False))
        self.absorb_frag = int(c.get("absorb_frag", 0))
        self.absorb_pending = bool(c.get("absorb_pending", True))
        super().__init__(continual=continual, **kwargs)
        if self.write_mode == "oracle":
            raise ValueError("CLSv7MPCPlanner routes label-free; use CLSMPCPlanner for write=oracle")

    # ------------------------------------------------------------------ read
    @torch.no_grad()
    def _route(self, e_obses, taken_actions):
        # as CLSMPCPlanner._route, plus the familiarity gate on reads
        obs_seq, act_seq = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        bufs_o = self._obs_buffer + [obs_seq]
        bufs_a = self._act_buffer + [act_seq]
        if len(bufs_o) > 1:
            o, a = self.adajepa_trainer._merge_segments(bufs_o, bufs_a)
            obs, act = o[0], a[0]
        else:
            obs, act = obs_seq, act_seq
        z = self._theta0_latents(obs, act)
        fam = self._familiarity(z) if self.familiar_read and self.familiar_thr > 0 else float("inf")
        cands = [None] + ([] if fam <= self.familiar_thr else self._learners())
        errs = [float(self._latent_loss(self._pred0_module if key is None else self.domains[key]["learner"], z))
                for key in cands]
        best = cands[int(np.argmin(errs))]
        self._route_log.append({"iter": self.iter, "errs": {str(k): e for k, e in zip(cands, errs)}, "best": best,
                                "familiar": fam})
        if best != self._base:
            self._load_base(best, carry_fast=True)

    # ------------------------------------------------------------------ write and learn
    def _end_episode(self):
        delta = [p.detach() - s for p, s in zip(self._params, self._start)]
        z = self._episode_latents()
        fam = self._familiarity(z) if self.familiar_thr > 0 else float("nan")
        familiar = fam <= self.familiar_thr
        base = self._base
        ratios = {}
        if not familiar:  # every learner validated on this (unseen) episode: merge evidence, write gate, join
            e0 = max(self._rollout_err(self._pred0_module, z), 1e-12)
            ratios = {dk: self._rollout_err(d["learner"], z) / e0 for dk, d in self.domains.items()
                      if d["learner"] is not None and not d.get("merged_into")}
        self._val_cache = dict(ratios)
        rule, key, rejected, new_members = ("familiar" if familiar else "none"), None, [], []
        if not familiar:
            lost = False  # learner_win: the read learner did not beat theta0 on the whole episode
            if base is not None:
                if self._rejects(base, z):
                    rejected.append(base)
                elif self.learner_win and not self._wins(base, z):
                    lost = True
                else:
                    rule, key = "learner", base
            if key is None:  # the mature domain whose learner predicts the episode best
                fit = sorted((r, dk) for dk, r in ratios.items() if dk not in rejected and self._mature(self.domains[dk]))
                if fit and fit[0][0] <= self.join_ratio:
                    rule, key = "join", fit[0][1]
            if key is None and base is None:  # planned with theta0: v5's fingerprint join, then the pending pool
                cands = sorted(((_cos(delta, [s / d["fp_n"] for s in d["fp_sum"]]), dk)
                                for dk, d in self.domains.items() if d["fp_n"] > 0 and not d.get("merged_into")),
                               reverse=True)
                for c, dk in cands:
                    if c < self.tau:
                        break
                    if dk in rejected or self._rejects(dk, z):
                        if dk not in rejected:
                            rejected.append(dk)
                        continue
                    rule, key = "direction", dk
                    break
                if key is None:
                    rule, key, new_members = self._pend(delta, z)
            elif key is None:
                rule = "none" if lost else "rejected"
        slow_loss, val_ratio = float("nan"), float("nan")
        if key is not None:
            d = self.domains.setdefault(key, {"n": 0, "tags": {}, "replay": [], "fp_sum": None, "fp_n": 0,
                                              "learner": None, "opt": None, "val": []})
            if d["learner"] is not None:  # unseen by this learner: validate before training on it
                val_ratio = self._val_ratio(key, z)
                if rule != "join":
                    d.setdefault("val", []).append(float(val_ratio))
            dev = self._params[0].device
            members = [(m["tag"], m["z"].to(dev), self._unflat(m["fp"].float().to(dev))) for m in new_members]
            for tag, mz, mfp in members + [(self.tag, z.detach(), self._fp(delta) if base is None else None)]:
                self._add(d, tag, mz, mfp)
            slow_loss = self._learn(d)
        ContinualMPCPlanner._end_episode(self)
        rec = self._ep_records[-1]
        merges, absorbs = [], []
        if key is not None and ratios:
            self.domains[key].setdefault("vals", []).append((rec["episode"], ratios))
            merges = self._try_merge()
        if self.absorb_every > 0 and self._episode % self.absorb_every == 0:
            absorbs = self._absorb()
        rec.update({
            "read": base, "route": self._route_log, "write": key, "write_rule": rule, "slow_loss": slow_loss,
            "val_ratio": val_ratio, "rejected_by": rejected,
            "rejected_ratios": {dk: self._val_cache[dk] for dk in rejected if dk in self._val_cache},
            "familiar_score": fam, "pending_n": len(self.pending), "new_members": [m["tag"] for m in new_members],
            "val_all": ratios, "merges": merges, "absorbs": absorbs,
            "domains": {dk: {"n": d["n"], "tags": dict(d["tags"]), "learner": d["learner"] is not None,
                             "mature": d["learner"] is not None and self._mature(d), "val_last": d.get("val", [])[-3:]}
                        for dk, d in self.domains.items() if d["n"] >= 2},
        })

    @torch.no_grad()
    def _wins(self, key, z):
        """v5's learner rule: the domain's learner has a lower 1-step error than theta0 on the whole episode."""
        return float(self._latent_loss(self.domains[key]["learner"], z)) < float(self._latent_loss(self._pred0_module, z))

    def _add(self, d, tag, z, fp):
        d["n"] += 1
        d["tags"][tag] = d["tags"].get(tag, 0) + 1
        d["replay"].append(z)
        if fp is not None:  # theta0-planned episodes define the fingerprint
            d["fp_sum"] = [x.clone() for x in fp] if d["fp_sum"] is None else [s + x for s, x in zip(d["fp_sum"], fp)]
            d["fp_n"] += 1

    def _learn(self, d):
        if d["n"] < self.min_n:
            return float("nan")
        if d["learner"] is None:
            d["learner"] = copy.deepcopy(self._pred0_module)
            for p in d["learner"].parameters():
                p.requires_grad_(True)
            d["opt"] = torch.optim.Adam(d["learner"].parameters(), lr=self.slow_lr)
            return self._train(d, self.slow_steps * d["n"])  # catch up on what it already holds
        return self._train(d, self.slow_steps)

    def _absorb(self):
        """Immature domains, then pending episodes, that a mature learner predicts well move into its domain."""
        mature = [dk for dk, d in self.domains.items()
                  if d["learner"] is not None and not d.get("merged_into") and self._mature(d)]
        if not mature:
            return []
        out = []
        with torch.no_grad():
            for dk, d in list(self.domains.items()):
                if d["n"] == 0 or d.get("merged_into") or dk in mature:
                    continue
                if self.absorb_frag > 0 and (d["n"] >= self.absorb_frag or len(d.get("val", [])) >= self.mature_k):
                    continue  # a domain with its own validation history is left to the merge rule
                e0 = [max(self._rollout_err(self._pred0_module, z), 1e-12) for z in d["replay"]]
                r = {m: _gmean([self._rollout_err(self.domains[m]["learner"], z) / e for z, e in zip(d["replay"], e0)])
                     for m in mature}
                best = min(r, key=r.get)
                if r[best] <= self.absorb_ratio:
                    out.append({"what": "domain", "gone": dk, "keep": best, "ratio": r[best], "tags": dict(d["tags"])})
        for a in out:  # merged after scoring, so every ratio refers to the learners as they were
            self._merge(a["keep"], a["gone"])
        if not self.absorb_pending:
            return out
        dev = self._params[0].device
        stay = []
        for m in self.pending:
            z = m["z"].to(dev)
            with torch.no_grad():
                e0 = max(self._rollout_err(self._pred0_module, z), 1e-12)
                r = {k: self._rollout_err(self.domains[k]["learner"], z) / e0 for k in mature}
            best = min(r, key=r.get)
            if r[best] <= self.join_ratio:
                self._add(self.domains[best], m["tag"], z, self._unflat(m["fp"].float().to(dev)))
                self._train(self.domains[best], self.slow_steps)
                out.append({"what": "pending", "keep": best, "ratio": r[best], "tags": {m["tag"]: 1}})
            else:
                stay.append(m)
        self.pending = stay
        return out
