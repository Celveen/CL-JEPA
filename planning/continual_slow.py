"""How the per-domain slow learner trains: two variants of CLSMPCPlanner's replay training.

Everything else (routing, gates, 15 updates per written episode, Adam lr 1e-4, batches of 8
replayed windows) is unchanged, so with write=oracle each variant isolates the learning rule.

CLSMultiStepMPCPlanner -- multi-step rollout loss. Instead of the 1-step teacher-forced loss,
each replayed window is rolled out open loop for ms_h model steps from a single frame, feeding
the learner's own predictions back with the true action embeddings (as the planner imagines
its 5-step plans, and as the maturity gate validates), and the loss is the mean latent error
over the rolled-out steps.

CLSPriorityMPCPlanner -- prioritized replay. Successful episodes are replayed pr_succ times as
often as failed ones, and within an episode a training window is chosen with weight
1 + pr_contact x (number of transitions in it where the object moved: > 1 px or > 0.01 rad
between model steps). Contact is read from the simulator state of the executed trajectory --
privileged, like the oracle labels this variant is tested with; a label-free proxy (e.g.
theta0's prediction error, higher on contact transitions) can replace it if the rule helps.
"""
import math

import numpy as np
import torch

from .continual import CLSMPCPlanner


class CLSMultiStepMPCPlanner(CLSMPCPlanner):
    def __init__(self, continual=None, **kwargs):
        self.ms_h = int(dict(continual or {}).get("ms_h", 5))
        super().__init__(continual=continual, **kwargs)

    def _latent_window_loss(self, module, z, max_len=6):
        drop = self.wm.action_dim
        k = z.shape[1] - 1
        h = min(self.ms_h, k)
        t0 = int(self._rng.integers(0, k - h + 1))
        seq, preds = z[:, t0 : t0 + 1], []
        for j in range(h):
            nxt = self._predict(module, seq[:, -self.wm.num_hist :])[:, -1:, ..., :-drop]
            preds.append(nxt)
            seq = torch.cat([seq, torch.cat([nxt, z[:, t0 + j + 1 : t0 + j + 2, ..., -drop:]], dim=-1)], dim=1)
        return torch.mean((torch.cat(preds, dim=1) - z[:, t0 + 1 : t0 + 1 + h, ..., :-drop]) ** 2)


class CLSPriorityMPCPlanner(CLSMPCPlanner):
    def __init__(self, continual=None, **kwargs):
        c = dict(continual or {})
        self.pr_succ = float(c.get("pr_succ", 2.0))
        self.pr_contact = float(c.get("pr_contact", 3.0))
        self._cur_meta = None
        super().__init__(continual=continual, **kwargs)

    def _episode_meta(self):
        fs = self.evaluator.frameskip
        n = torch.cat(self.planned_actions, dim=1).shape[1]
        s = np.asarray(self._last_eval[3][0], dtype=np.float64)[: n * fs + 1 : fs]
        dpos = np.linalg.norm(np.diff(s[:, 2:4], axis=0), axis=1)
        dang = np.abs((np.diff(s[:, 4]) + math.pi) % (2 * math.pi) - math.pi)
        return {"success": bool(self.is_success[0]), "contact": ((dpos > 1.0) | (dang > 0.01)).tolist()}

    def _end_episode(self):
        self._cur_meta = self._episode_meta()
        super()._end_episode()
        rec = self._ep_records[-1]
        rec["contact_frac"] = float(np.mean(self._cur_meta["contact"])) if self._cur_meta["contact"] else 0.0
        if rec["write"] is not None:  # replay entries written without training (n < min_n) still need their meta
            self._sync_meta(self.domains[rec["write"]])

    def _sync_meta(self, d):
        meta = d.setdefault("meta", [])
        while len(meta) < len(d["replay"]):
            meta.append(self._cur_meta or {"success": False, "contact": []})

    def _train(self, d, steps):
        self._sync_meta(d)
        m = d["learner"]
        m.train()
        w = np.array([self.pr_succ if mt["success"] else 1.0 for mt in d["meta"]])
        p = w / w.sum()
        last = []
        for _ in range(steps):
            d["opt"].zero_grad()
            idx = self._rng.choice(len(d["replay"]), size=min(self.slow_batch, len(d["replay"])), p=p)
            loss = torch.stack([self._window_loss_pr(m, d["replay"][i], d["meta"][i]["contact"]) for i in idx]).mean()
            loss.backward()
            d["opt"].step()
            last.append(loss.item())
        m.eval()
        return float(np.mean(last[-5:]))

    def _window_loss_pr(self, module, z, contact, max_len=6):
        k = z.shape[1] - 1
        L = min(max_len - 1, k)
        starts = np.arange(k - L + 1)
        c = np.zeros(k) if len(contact) < k else np.asarray(contact[:k], dtype=float)
        w = 1.0 + self.pr_contact * np.array([c[t : t + L].sum() for t in starts])
        t0 = int(self._rng.choice(starts, p=w / w.sum()))
        return self._latent_loss(module, z[:, t0 : t0 + L + 1])

    def _save_state(self, path):
        super()._save_state(path)
        st = torch.load(path, map_location="cpu")
        st["priority_meta"] = {dk: d.get("meta", []) for dk, d in self.domains.items()}
        torch.save(st, path)

    def _load_state(self, path):
        super()._load_state(path)
        for dk, meta in torch.load(path, map_location="cpu").get("priority_meta", {}).items():
            if dk in self.domains:
                self.domains[dk]["meta"] = list(meta)
