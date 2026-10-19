"""Domain-routed fixed-weight consolidation (the phase-1 fast/slow rule under the v4 gates).

CLSFixedWMPCPlanner keeps everything of CLSMPCPlanner -- familiarity gate, pending pool,
fingerprint routing, write gate, maturity gate and error-based reading -- but a domain's
slow model is not a predictor trained on replay. It is theta0 + slow_d, where

    slow_d <- slow_d + w * delta

for every episode written to domain d (delta: the episode's fast change; pending episodes
that found the domain contribute theirs too). This is the fixed-w rule of the first phase
(one shared slow, w = 0.003 there), now kept per domain and only for unfamiliar episodes.
Only the predictor tensors AdaJEPA adapts (last transformer block and final norm) are
consolidated; the encoder head's change (lr 1e-5) is two orders of magnitude smaller and
stays per episode. Reading theta0 + slow_d is gated like a learner (validation ratios on
unseen episodes, mature_k / mature_ratio).
"""
import math

import torch

from .continual import CLSMPCPlanner


class CLSFixedWMPCPlanner(CLSMPCPlanner):
    def __init__(self, continual=None, **kwargs):
        self.slow_w = float(dict(continual or {}).get("slow_w", 0.003))
        self._last_members = []
        super().__init__(continual=continual, **kwargs)
        names = {id(p): n for n, p in self.wm.predictor.named_parameters()}
        self._ada_names = [names[id(p)] for p in self._params[: self._groups["pred"]]]

    def _train(self, d, steps):
        return float("nan")  # no replay training: the domain's model is theta0 + slow_d

    def _pend(self, delta, z):
        rule, key, members = super()._pend(delta, z)
        if rule == "pending":  # keep the raw predictor change, in case the episode later founds a domain
            self.pending[-1]["dpred"] = [x.detach().half().cpu() for x in delta[: self._groups["pred"]]]
        self._last_members = members
        return rule, key, members

    def _end_episode(self):
        k = self._groups["pred"]
        delta = [p.detach() - s for p, s in zip(self._params[:k], self._start[:k])]
        self._last_members = []
        super()._end_episode()
        rec = self._ep_records[-1]
        key = rec["write"]
        if key is None:
            return
        d = self.domains[key]
        parts = [delta] + [[x.to(delta[0].device).float() for x in m["dpred"]]
                           for m in self._last_members if m.get("dpred") is not None]
        slow = d.get("slow") or [torch.zeros_like(x) for x in delta]
        for part in parts:
            slow = [s + self.slow_w * x.to(s.dtype) for s, x in zip(slow, part)]
        d["slow"] = slow
        self._materialize(d)
        rec["slow_norm"] = math.sqrt(sum(float(torch.sum(s.float() ** 2)) for s in slow))

    @torch.no_grad()
    def _materialize(self, d):
        """The domain's model = theta0 with the adapted predictor tensors moved by slow_d."""
        if d["learner"] is None or d.get("slow") is None:
            return
        lp = dict(d["learner"].named_parameters())
        for n, p0, s in zip(self._ada_names, self._init[: self._groups["pred"]], d["slow"]):
            lp[n].copy_(p0 + s)

    def _save_state(self, path):
        super()._save_state(path)
        st = torch.load(path, map_location="cpu")
        st["fixw"] = {"w": self.slow_w,
                      "slow": {dk: [s.cpu() for s in d["slow"]] for dk, d in self.domains.items() if d.get("slow") is not None},
                      "pending_dpred": [m.get("dpred") for m in self.pending]}
        torch.save(st, path)

    def _load_state(self, path):
        super()._load_state(path)
        fw = torch.load(path, map_location="cpu").get("fixw", {})
        dev = self._params[0].device
        for dk, s in fw.get("slow", {}).items():
            if dk in self.domains:
                self.domains[dk]["slow"] = [x.to(dev) for x in s]
        for m, dp in zip(self.pending, fw.get("pending_dpred", [])):
            if dp is not None:
                m["dpred"] = dp
