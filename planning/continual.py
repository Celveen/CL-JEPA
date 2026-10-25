"""Fast/slow continual adaptation of AdaJEPA across episodes.

The adapted tensors are split as theta = theta0 + slow + fast. Within an episode
the usual AdaJEPA loop adapts them (the fast part); at the episode end the
change it made, delta = theta_end - (theta0 + slow), is consolidated,
slow <- slow + w * delta, and the next episode starts from theta0 + slow.
w = 0 is AdaJEPA (reset every episode); w = 1 never resets.

The slow part, the episode counter and a small replay memory of recent episodes
are saved at the end of a run (continual.state_out) and loaded by the next one
(continual.state_in), so a stream runs as consecutive chunks. Per episode,
continual_episodes.jsonl records the candidate signals for selective
consolidation: the size of delta per module, its cosine to the accumulated slow
direction and to the previous delta, and how much it changes the prediction
error on recent past episodes' transitions (same vs other tag). With
save_deltas / save_frames the run also stores every delta (fp16) and the
executed frames, so other consolidation rules can be replayed offline.

SlowFrozenMPCPlanner plans frozen (batched) with theta0 + slow from a state file;
it is the probe of what has been consolidated.
"""

import contextlib
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from .adajepa import AdaJEPATrainer
from .adajepa_probe import ProbeMPCPlanner, _cos, _norm
from .mpc import MPCPlanner


class ContinualMPCPlanner(ProbeMPCPlanner):
    def __init__(self, continual=None, probe=None, **kwargs):
        probe = dict(probe or {})
        probe["mode"] = "grad"  # the within-episode update is plain AdaJEPA
        super().__init__(probe=probe, **kwargs)
        c = dict(continual or {})
        self.write_w = float(c.get("w", 0.0))
        self.tag = c.get("tag")
        self.state_out = c.get("state_out") or "continual_state.pt"
        self.read_only = bool(c.get("read_only", False))
        self.memory_size = int(c.get("memory_size", 30))
        self.save_deltas = bool(c.get("save_deltas", False))
        self.save_frames = bool(c.get("save_frames", False))
        self._slow = [torch.zeros_like(p) for p in self._params]
        self._memory = []  # recent episodes' last windows, for the transfer signal
        self._episode = 0  # index along the whole stream
        self._prev_delta = None
        self._start = None
        if c.get("state_in"):
            self._load_state(c["state_in"])
        self._ep_records, self._deltas, self._frames = [], [], {}

    # ------------------------------------------------------------------ episodes
    def _plan_single(self, **kw):
        with torch.no_grad():
            for p, q, s in zip(self._params, self._init, self._slow):
                p.copy_(q + s)
        self._start = [p.detach().clone() for p in self._params]
        out = super()._plan_single(**kw)
        self._end_episode()
        return out

    def _end_episode(self):
        delta = [p.detach() - s for p, s in zip(self._params, self._start)]
        k = self._groups["pred"]
        rec = {
            "episode": self._episode, "sample_idx": self._sample_idx, "tag": self.tag, "w": self.write_w,
            "success": bool(self.is_success[0]), "steps": int(self.iter),
            "delta_pred": _norm(delta[:k]), "delta_enc": _norm(delta[k:]),
            "slow_pred_before": _norm(self._slow[:k]), "slow_enc_before": _norm(self._slow[k:]),
            "cos_slow": _cos(delta, self._slow), "cos_slow_pred": _cos(delta[:k], self._slow[:k]),
            "cos_prev": _cos(delta, self._prev_delta) if self._prev_delta is not None else float("nan"),
        }
        rec.update(self._transfer())
        with torch.no_grad():
            for s, d in zip(self._slow, delta):
                s.add_(self.write_w * d)
        rec["slow_pred_after"], rec["slow_enc_after"] = _norm(self._slow[:k]), _norm(self._slow[k:])
        self._prev_delta = delta
        self._remember()
        if self.save_deltas:
            self._deltas.append(torch.cat([d.flatten() for d in delta]).half().cpu())
        if self.save_frames:
            self._store_frames()
        self._ep_records.append(rec)
        self._episode += 1

    def plan(self, obs_0, obs_g, actions=None):
        out = super().plan(obs_0, obs_g, actions)
        if not self.read_only:  # a frozen evaluation leaves state_in as it was
            self._save_state(self.state_out)
        with open("continual_episodes.jsonl", "w") as f:
            for rec in self._ep_records:
                f.write(json.dumps(rec) + "\n")
        if self.save_deltas and self._deltas:
            torch.save({"episodes": [r["episode"] for r in self._ep_records], "deltas": self._deltas,
                        "shapes": [tuple(p.shape) for p in self._params], "n_pred": self._groups["pred"]},
                       "deltas.pt")
        if self.save_frames and self._frames:
            np.savez_compressed("episode_frames.npz", **self._frames)
        return out

    # ------------------------------------------------------------------ signals
    @contextlib.contextmanager
    def _weights(self, values):
        current = [p.detach().clone() for p in self._params]
        self._set(values)
        try:
            yield
        finally:
            self._set(current)

    def _losses_now_and_init(self, obs, act):
        out = super()._losses_now_and_init(obs, act)
        with self._weights(self._start):  # episode start: theta0 + slow
            for key, val in self._losses(obs, act).items():
                out[f"loss_{key}_start"] = val
        return out

    @torch.no_grad()
    def _transfer(self):
        """Relative change of the prediction error on remembered past windows, end vs start of this episode."""
        if not self._memory:
            return {}
        end = [self._mem_loss(m) for m in self._memory]
        with self._weights(self._start):
            start = [self._mem_loss(m) for m in self._memory]
        gains = [(s - e) / s for s, e in zip(start, end) if s > 0]  # > 0: this episode's change helps them
        tags = [m["tag"] for m, s in zip(self._memory, start) if s > 0]
        same = [g for g, t in zip(gains, tags) if t == self.tag]
        other = [g for g, t in zip(gains, tags) if t != self.tag]
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
        return {"transfer_all": mean(gains), "transfer_same": mean(same), "transfer_other": mean(other),
                "n_mem_same": len(same), "n_mem_other": len(other)}

    def _mem_loss(self, m):
        obs = {"visual": m["visual"].float(), "proprio": m["proprio"]}
        return self._losses(obs, m["act"])["buf"]

    def _remember(self):
        obs, act = self._merged_buffer()
        self._memory.append({"tag": self.tag, "visual": obs["visual"].half().cpu(),
                             "proprio": obs["proprio"].float().cpu(), "act": act.float().cpu()})
        self._memory = self._memory[-self.memory_size :]

    def _store_frames(self):
        e_obses = self._last_eval[2]
        fs = self.evaluator.frameskip
        e = self._episode
        self._frames[f"ep{e}_visual"] = np.asarray(e_obses["visual"][0, ::fs], dtype=np.uint8)
        self._frames[f"ep{e}_proprio"] = np.asarray(e_obses["proprio"][0, ::fs], dtype=np.float32)
        self._frames[f"ep{e}_act"] = torch.cat(self.planned_actions, dim=1)[0].float().cpu().numpy()

    # ------------------------------------------------------------------ state
    def _save_state(self, path):
        torch.save({
            "slow": [s.cpu() for s in self._slow], "shapes": [tuple(p.shape) for p in self._params],
            "n_pred": self._groups["pred"], "episode": self._episode, "w": self.write_w,
            "memory": self._memory,
            "prev_delta": [d.half().cpu() for d in self._prev_delta] if self._prev_delta is not None else None,
        }, path)

    def _load_state(self, path):
        st = torch.load(path, map_location="cpu")
        _check_shapes(st["shapes"], self._params, path)
        self._slow = [s.to(p.device, p.dtype) for s, p in zip(st["slow"], self._params)]
        self._episode = int(st["episode"])
        self._memory = st.get("memory", [])
        if st.get("prev_delta") is not None:
            self._prev_delta = [d.to(p.device, p.dtype) for d, p in zip(st["prev_delta"], self._params)]


class MemoryMPCPlanner(ContinualMPCPlanner):
    """Domain-separated slow memories with prediction-gated reading.

    Each memory m keeps mu_m, the running mean of what its episodes learned. An
    episode plans its first replan with theta0; on the first observed transition
    it reads the memory whose theta0 + alpha * mu_m predicts that transition best,
    or none (theta0) if no memory beats theta0, and the usual AdaJEPA loop goes on
    from there. At the end the episode writes x = mu_read + delta (mu_read = 0 if
    it read none) into one memory: by the true tag (write=oracle) or, label-free,
    to the memory whose mean is most aligned with x (cosine >= tau, else a new
    memory). Memories are readable once they hold min_n episodes.

    With alpha = 1 a memory is the average of where its episodes ended up. For
    alpha > 1 the read is extrapolated but the write stays unextrapolated, so a
    memory that is read and written by the same domain converges to the same
    fixed point (alpha * mu = the correction the episodes keep needing) instead
    of growing without bound.
    """

    def __init__(self, continual=None, **kwargs):
        c = dict(continual or {})
        c["w"] = 0.0  # no shared slow part
        super().__init__(continual=c, **kwargs)
        self.alpha = float(c.get("alpha", 1.0))
        self.write_mode = c.get("write", "delta")
        self.tau = float(c.get("tau", 0.06))
        self.min_n = int(c.get("min_n", 3))
        if not hasattr(self, "memories"):
            self.memories = {}  # key -> {"sum": [tensors], "n": int, "tags": {tag: count}}
        self._read_key = None
        self._read_errs = None

    # ------------------------------------------------------------------ read
    def _mu(self, key):
        m = self.memories[key]
        return [s / m["n"] for s in m["sum"]]

    def _readable(self):
        return [k for k, m in self.memories.items() if m["n"] >= self.min_n]

    def _post_env_feedback(self, taken_actions, e_obses):
        if self.iter == 0:
            self._route_read(e_obses, taken_actions)
        return super()._post_env_feedback(taken_actions, e_obses)

    @torch.no_grad()
    def _route_read(self, e_obses, taken_actions):
        obs, act = self._extract_adajepa_data(e_obses, taken_actions, 0)  # the first transition
        cands = [None] + self._readable()
        errs = []
        for key in cands:
            self._set(self._init if key is None else
                      [q + self.alpha * mu for q, mu in zip(self._init, self._mu(key))])
            errs.append(self._losses(obs, act)["new"])
        best = int(np.argmin(errs))
        self._read_key = cands[best]
        self._read_errs = {str(k): float(e) for k, e in zip(cands, errs)}
        self._set(self._init if self._read_key is None else
                  [q + self.alpha * mu for q, mu in zip(self._init, self._mu(self._read_key))])
        self._start = [p.detach().clone() for p in self._params]  # delta is measured from here

    # ------------------------------------------------------------------ write
    def _end_episode(self):
        delta = [p.detach() - s for p, s in zip(self._params, self._start)]
        mu_read = self._mu(self._read_key) if self._read_key is not None else None
        x = delta if mu_read is None else [mu + d for mu, d in zip(mu_read, delta)]
        if self.write_mode == "oracle":
            key, cos = self.tag, float("nan")
        else:
            key, cos = None, -1.0
            for k in self.memories:
                c = _cos(x, self._mu(k))
                if c > cos:
                    key, cos = k, c
            if key is None or cos < self.tau:
                key = f"m{len(self.memories)}"
        m = self.memories.setdefault(key, {"sum": [torch.zeros_like(p) for p in self._params], "n": 0, "tags": {}})
        for s, v in zip(m["sum"], x):
            s.add_(v)
        m["n"] += 1
        m["tags"][self.tag] = m["tags"].get(self.tag, 0) + 1
        super()._end_episode()  # logging, replay memory, delta/frame dumps (w = 0: no shared slow)
        self._ep_records[-1].update({
            "read": self._read_key, "read_errs": self._read_errs, "write": key, "write_cos": cos,
            "n_memories": len(self.memories), "n_readable": len(self._readable()),
        })
        self._read_key, self._read_errs = None, None

    # ------------------------------------------------------------------ state
    def _save_state(self, path):
        super()._save_state(path)
        st = torch.load(path, map_location="cpu")
        st["memories"] = {k: {"sum": [s.cpu() for s in m["sum"]], "n": m["n"], "tags": m["tags"]}
                          for k, m in self.memories.items()}
        st["memory_cfg"] = {"alpha": self.alpha, "write": self.write_mode, "tau": self.tau, "min_n": self.min_n}
        torch.save(st, path)

    def _load_state(self, path):
        super()._load_state(path)
        st = torch.load(path, map_location="cpu")
        self.memories = {k: {"sum": [s.to(p.device, p.dtype) for s, p in zip(m["sum"], self._params)],
                             "n": m["n"], "tags": m["tags"]} for k, m in st.get("memories", {}).items()}


class CLSMPCPlanner(ContinualMPCPlanner):
    """Complementary fast/slow learning with per-domain slow predictors.

    Slow: each domain owns a full copy of the predictor, trained between episodes
    on that domain's replayed transitions (latents of the theta0 encoder, which
    stays frozen, so the replay holds latents rather than frames). Fast: the usual
    AdaJEPA loop on top of whichever predictor the episode uses.

    Read: an episode plans its first replan with theta0. Over its first `reroute`
    replans it scores theta0 and every learner on all transitions seen so far and
    switches to the best (keeping the fast change already made); if no learner
    beats theta0 it stays on theta0. write=oracle instead loads the learner of the
    episode's true tag from the start.

    Write (label-free): an episode that ends on a learner which still predicts the
    whole episode better than theta0 goes to that learner's domain; an episode
    that used theta0 goes to the domain whose fingerprint (mean fast change of its
    theta0 episodes) its fast change is most aligned with, or to a new domain
    (cosine < tau); anything else is not consolidated. A domain gets a learner once
    it holds min_n episodes; each new episode then adds slow_steps updates.

    Maturity gate: before a domain's learner trains on a new episode, that episode
    (unseen by it) validates the learner: 5-step open-loop latent error of the
    learner over theta0. A learner is read only while the mean of its last
    mature_k ratios is <= mature_ratio (mature_k = 0 disables the gate), so an
    undertrained or no-longer-reliable learner falls back to theta0.

    Validation-gated writing (reject_ratio > 0, label-free only): the same check
    decides membership. An episode whose candidate domain's learner predicts it
    worse than reject_ratio x theta0 does not belong there: a direction-routed
    episode moves on to the next aligned domain (or a new one), a learner-routed
    episode is not consolidated. Rejected checks do not count towards maturity.

    Two-stage writing (familiar_thr > 0, label-free only): an episode whose theta0
    error over its first 3 transitions is at most familiar_thr (a percentile of
    theta0's error on its own training-validation data) is familiar -- theta0
    already predicts it -- and is never written; only unfamiliar episodes are
    clustered. With pending_m > 1 an unfamiliar episode that joins no domain waits
    in a pending pool, and a new domain is founded only when pending_m pending
    episodes are mutually aligned (cosine >= tau_new), so isolated episodes never
    found domains. fp_unit builds fingerprints from unit-norm fast changes.
    Reading is unchanged.
    """

    def __init__(self, continual=None, **kwargs):
        c = dict(continual or {})
        c["w"] = 0.0
        super().__init__(continual=c, **kwargs)
        self.write_mode = c.get("write", "delta")
        self.tau = float(c.get("tau", 0.06))
        self.min_n = int(c.get("min_n", 3))
        self.slow_lr = float(c.get("slow_lr", 1e-4))
        self.slow_steps = int(c.get("slow_steps", 15))
        self.slow_batch = int(c.get("slow_batch", 8))
        self.reroute = int(c.get("reroute", 3))
        self.mature_ratio = float(c.get("mature_ratio", 0.9))
        self.mature_k = int(c.get("mature_k", 0))
        self.reject_ratio = float(c.get("reject_ratio", 0.0))  # 0: off
        self.familiar_thr = float(c.get("familiar_thr", 0.0))  # 0: off
        self.pending_m = int(c.get("pending_m", 1))  # 1: off (any unmatched episode founds a domain)
        self.tau_new = float(c.get("tau_new", self.tau))
        self.pending_max = int(c.get("pending_max", 40))
        self.fp_unit = bool(c.get("fp_unit", False))
        if not hasattr(self, "pending"):
            self.pending = []  # [{"fp": flat unit fast change (cpu, fp16), "z": theta0 latents (cpu), "tag"}]
        self._pred_params = list(self.wm.predictor.parameters())
        self._pred0 = [p.detach().clone() for p in self._pred_params]
        self._pred0_module = copy.deepcopy(self.wm.predictor).eval()
        for p in self._pred0_module.parameters():
            p.requires_grad_(False)
        self._rng = np.random.default_rng(0)
        if not hasattr(self, "domains"):
            self.domains = {}  # key -> {"n", "tags", "replay": [z], "fp_sum": [t] | None, "fp_n", "learner", "opt"}
        self._base = None  # domain key whose learner the episode uses (None: theta0)
        self._route_log = []
        if self.read_only and not self.supports_read_only:
            raise ValueError(f"{type(self).__name__} has no read-only mode (planner.continual.read_only)")

    supports_read_only = False  # subclasses whose _end_episode can leave the state untouched set this

    # ------------------------------------------------------------------ helpers
    def _predict(self, module, z):
        T = z.shape[1]
        y = module(rearrange(z, "b t p d -> b (t p) d"))
        return rearrange(y, "b (t p) d -> b t p d", t=T)

    def _latent_loss(self, module, z, windows_from=0, max_len=None):
        """Mean sliding-window 1-step loss (obs dims) of `module` on latents z (1, T+1, p, d)."""
        drop = self.wm.action_dim
        k = z.shape[1] - 1
        w = min(self.wm.num_hist, k)
        losses = [torch.mean((self._predict(module, z[:, t : t + w])[..., :-drop] - z[:, t + 1 : t + 1 + w][..., :-drop]) ** 2)
                  for t in range(windows_from, k - w + 1)]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _theta0_latents(self, obs, act):
        """Latents of the theta0 encoder (the fast loop may have moved the encoder head)."""
        k = self._groups["pred"]
        current = [p.detach().clone() for p in self._params[k:]]
        for p, q in zip(self._params[k:], self._init[k:]):
            p.copy_(q)
        try:
            o, a = self.adajepa_trainer._prepare_segment(obs, act)
            return self.wm.encode(o, a)
        finally:
            for p, c in zip(self._params[k:], current):
                p.copy_(c)

    @torch.no_grad()
    def _load_base(self, key, carry_fast=True):
        """Put a learner's (or theta0's) predictor in place, keeping the fast change on the adapted tensors."""
        k = self._groups["pred"]
        fast = [p.detach() - s for p, s in zip(self._params[:k], self._start[:k])] if carry_fast else None
        src = self._pred0 if key is None else [q.detach() for q in self.domains[key]["learner"].parameters()]
        for p, q in zip(self._pred_params, src):
            p.copy_(q)
        self._start = [p.detach().clone() for p in self._params[:k]] + [s.clone() for s in self._start[k:]]
        if fast is not None:
            for p, f in zip(self._params[:k], fast):
                p.add_(f)
        self._base = key

    def _learners(self):
        """Learners the episode may read: trained, and validated on unseen episodes when the gate is on."""
        return [k for k, d in self.domains.items() if d["learner"] is not None and self._mature(d)]

    def _mature(self, d):
        if self.mature_k <= 0:
            return True
        v = d.get("val", [])
        return len(v) >= self.mature_k and float(np.mean(v[-self.mature_k:])) <= self.mature_ratio

    @torch.no_grad()
    def _rollout_err(self, module, z, h=5):
        """5-step open-loop latent error from single frames, true action embeddings fed in (as the planner rolls out)."""
        drop = self.wm.action_dim
        k = z.shape[1] - 1
        h = min(h, k)
        starts = list(range(k - h + 1))
        seq = torch.cat([z[:, t : t + 1] for t in starts], dim=0)  # (B, 1, p, d)
        for j in range(h):
            nxt = self._predict(module, seq[:, -self.wm.num_hist :])[:, -1:]
            act = torch.cat([z[:, t + j + 1 : t + j + 2, ..., -drop:] for t in starts], dim=0)
            seq = torch.cat([seq, torch.cat([nxt[..., :-drop], act], dim=-1)], dim=1)
        tgt = torch.cat([z[:, t + 1 : t + 1 + h] for t in starts], dim=0)
        return float(torch.mean((seq[:, 1:][..., :-drop] - tgt[..., :-drop]) ** 2))

    @contextlib.contextmanager
    def _initial_weights(self):
        """theta0 for the whole predictor and the encoder head (the diagnostics' 'init')."""
        k = self._groups["pred"]
        pred_now = [p.detach().clone() for p in self._pred_params]
        enc_now = [p.detach().clone() for p in self._params[k:]]
        with torch.no_grad():
            for p, q in zip(self._pred_params, self._pred0):
                p.copy_(q)
            for p, q in zip(self._params[k:], self._init[k:]):
                p.copy_(q)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, q in zip(self._pred_params, pred_now):
                    p.copy_(q)
                for p, q in zip(self._params[k:], enc_now):
                    p.copy_(q)

    # ------------------------------------------------------------------ episode
    def _plan_single(self, **kw):
        # Replaces ContinualMPCPlanner._plan_single: the shared slow part is unused here, and the
        # whole predictor (not only the adapted tensors) must start from theta0 or a learner.
        with torch.no_grad():
            for p, q in zip(self._pred_params, self._pred0):
                p.copy_(q)
            for p, q in zip(self._params, self._init):
                p.copy_(q)
        self._base, self._route_log = None, []
        self._start = [p.detach().clone() for p in self._params]
        dom = self.domains.get(self.tag)
        if self.write_mode == "oracle" and dom is not None and dom["learner"] is not None and self._mature(dom):
            self._load_base(self.tag, carry_fast=False)  # the oracle knows the domain from the start
        out = ProbeMPCPlanner._plan_single(self, **kw)
        self._end_episode()
        return out

    def _post_env_feedback(self, taken_actions, e_obses):
        if self.write_mode != "oracle" and self.iter < self.reroute:
            self._route(e_obses, taken_actions)
        return super()._post_env_feedback(taken_actions, e_obses)

    @torch.no_grad()
    def _route(self, e_obses, taken_actions):
        # all transitions so far: the buffer (before this iteration's segment is appended) plus the new one
        obs_seq, act_seq = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        bufs_o = self._obs_buffer + [obs_seq]
        bufs_a = self._act_buffer + [act_seq]
        if len(bufs_o) > 1:
            o, a = self.adajepa_trainer._merge_segments(bufs_o, bufs_a)
            obs, act = o[0], a[0]
        else:
            obs, act = obs_seq, act_seq
        z = self._theta0_latents(obs, act)
        cands = [None] + self._learners()
        errs = [float(self._latent_loss(self._pred0_module if key is None else self.domains[key]["learner"], z))
                for key in cands]
        best = cands[int(np.argmin(errs))]
        self._route_log.append({"iter": self.iter, "errs": {str(k): e for k, e in zip(cands, errs)}, "best": best})
        if best != self._base:
            self._load_base(best, carry_fast=True)

    # ------------------------------------------------------------------ write and learn
    def _end_episode(self):
        k = self._groups["pred"]
        delta = [p.detach() - s for p, s in zip(self._params, self._start)]
        z = self._episode_latents()
        rule, key = "none", None
        rejected, self._val_cache = [], {}
        fam = self._familiarity(z) if self.write_mode != "oracle" and self.familiar_thr > 0 else float("nan")
        new_members = []  # pending episodes that found a new domain together with this one
        if self.write_mode == "oracle":
            rule, key = "oracle", self.tag
        elif fam <= self.familiar_thr:  # theta0 already predicts it: nothing to consolidate
            rule, key = "familiar", None
        elif self._base is not None:
            with torch.no_grad():
                l_learner = float(self._latent_loss(self.domains[self._base]["learner"], z))
                l_theta0 = float(self._latent_loss(self._pred0_module, z))
            if l_learner < l_theta0:
                rule, key = "learner", self._base
        else:
            cands = sorted(((_cos(delta, [s / d["fp_n"] for s in d["fp_sum"]]), dk)
                            for dk, d in self.domains.items() if d["fp_n"] > 0), reverse=True)
            key, rule = None, "direction"
            for c, dk in cands:
                if c < self.tau:
                    break
                if self._rejects(dk, z):
                    rejected.append(dk)
                    continue
                key = dk
                break
            if key is None and self.pending_m > 1:
                rule, key, new_members = self._pend(delta, z)
            elif key is None:
                key = f"d{len(self.domains)}"
        if rule == "learner" and self._rejects(key, z):
            rejected.append(key)
            rule, key = "none", None
        slow_loss, val_ratio = float("nan"), float("nan")
        if key is not None:
            d = self.domains.setdefault(key, {"n": 0, "tags": {}, "replay": [], "fp_sum": None, "fp_n": 0,
                                              "learner": None, "opt": None, "val": []})
            if d["learner"] is not None:  # this episode is unseen by the learner: validate before training on it
                val_ratio = self._val_ratio(key, z)
                d.setdefault("val", []).append(float(val_ratio))
            dev = self._params[0].device
            members = [(m["tag"], m["z"].to(dev), self._unflat(m["fp"].float().to(dev))) for m in new_members]
            for tag, mz, mfp in members + [(self.tag, z.detach(), self._fp(delta) if self._base is None else None)]:
                d["n"] += 1
                d["tags"][tag] = d["tags"].get(tag, 0) + 1
                d["replay"].append(mz)
                if mfp is not None and self.write_mode != "oracle":  # theta0 episodes define the fingerprint
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
        read_key = self._base
        super()._end_episode()
        self._ep_records[-1].update({
            "read": read_key, "route": self._route_log, "write": key, "write_rule": rule, "slow_loss": slow_loss,
            "val_ratio": val_ratio, "rejected_by": rejected,
            "rejected_ratios": {dk: self._val_cache[dk] for dk in rejected if dk in self._val_cache},
            "familiar_score": fam, "pending_n": len(self.pending), "new_members": [m["tag"] for m in new_members],
            "domains": {dk: {"n": d["n"], "tags": dict(d["tags"]), "learner": d["learner"] is not None,
                             "mature": d["learner"] is not None and self._mature(d),
                             "val_last": d.get("val", [])[-3:]}
                        for dk, d in self.domains.items() if d["n"] >= 2},
        })

    @torch.no_grad()
    def _familiarity(self, z):
        """theta0's newest-frame error over the first 3 transitions (geometric mean), as calibrated offline."""
        drop = self.wm.action_dim
        errs = [float(torch.mean((self._predict(self._pred0_module, z[:, max(0, t - self.wm.num_hist) : t])[:, -1:, ..., :-drop]
                                  - z[:, t : t + 1, ..., :-drop]) ** 2))
                for t in range(1, min(3, z.shape[1] - 1) + 1)]
        return float(np.exp(np.mean(np.log(np.maximum(errs, 1e-12))))) if errs else float("inf")

    def _fp(self, delta):
        """Fingerprint contribution of a fast change: as is, or unit-norm (fp_unit)."""
        if not self.fp_unit:
            return delta
        n = math.sqrt(sum(float(torch.sum(x.float() ** 2)) for x in delta)) or 1.0
        return [x / n for x in delta]

    def _unflat(self, v):
        out, o = [], 0
        for p in self._params:
            out.append(v[o : o + p.numel()].view_as(p).to(p.dtype))
            o += p.numel()
        return out

    def _pend(self, delta, z):
        """An unfamiliar episode that joined no domain: found a domain with pending_m - 1 aligned pending episodes, or wait."""
        u = torch.cat([x.flatten().float() for x in delta])
        u = (u / (u.norm() + 1e-12)).cpu()
        nb = [m for m in self.pending if float(m["fp"].float() @ u) >= self.tau_new]
        if len(nb) + 1 < self.pending_m:
            self.pending.append({"fp": u.half(), "z": z.detach().cpu(), "tag": self.tag})
            self.pending = self.pending[-self.pending_max :]
            return "pending", None, []
        members = nb[: self.pending_m - 1]
        taken = {id(m) for m in members}  # by identity: entries hold tensors, so == between dicts is ill-defined
        self.pending = [m for m in self.pending if id(m) not in taken]
        return "new_from_pending", f"d{len(self.domains)}", members

    def _val_ratio(self, key, z):
        """5-step error of the domain's learner over theta0's on an episode it has not trained on (cached per episode)."""
        if key not in self._val_cache:
            learner = self.domains[key]["learner"]
            self._val_cache[key] = (self._rollout_err(learner, z) / max(self._rollout_err(self._pred0_module, z), 1e-12)
                                    if learner is not None else float("nan"))
        return self._val_cache[key]

    def _rejects(self, key, z):
        if self.reject_ratio <= 0 or self.write_mode == "oracle" or key not in self.domains:
            return False
        if self.domains[key]["learner"] is None:
            return False
        return self._val_ratio(key, z) > self.reject_ratio

    def _train(self, d, steps):
        m = d["learner"]
        m.train()
        last = []
        for _ in range(steps):
            d["opt"].zero_grad()
            idx = self._rng.integers(len(d["replay"]), size=min(self.slow_batch, len(d["replay"])))
            loss = torch.stack([self._latent_window_loss(m, d["replay"][i]) for i in idx]).mean()
            loss.backward()
            d["opt"].step()
            last.append(loss.item())
        m.eval()
        return float(np.mean(last[-5:]))

    def _latent_window_loss(self, module, z, max_len=6):
        k = z.shape[1] - 1
        L = min(max_len - 1, k)
        t0 = int(self._rng.integers(0, k - L + 1))
        return self._latent_loss(module, z[:, t0 : t0 + L + 1])

    @torch.no_grad()
    def _episode_latents(self):
        """theta0-encoder latents of every frame the episode executed (model stride), with its actions."""
        e_obses = self._last_eval[2]
        fs = self.evaluator.frameskip
        acts = torch.cat(self.planned_actions, dim=1).cpu()
        n = acts.shape[1]
        chunk = {k: v[:, : n * fs + 1 : fs] for k, v in e_obses.items()}
        return self._theta0_latents(self.preprocessor.transform_obs(chunk), acts)

    # ------------------------------------------------------------------ state
    def _save_state(self, path):
        super()._save_state(path)
        st = torch.load(path, map_location="cpu")
        st["domains"] = {
            dk: {"n": d["n"], "tags": d["tags"], "replay": [z.cpu() for z in d["replay"]],
                 "fp_sum": [s.cpu() for s in d["fp_sum"]] if d["fp_sum"] is not None else None, "fp_n": d["fp_n"],
                 "learner": {k: v.cpu() for k, v in d["learner"].state_dict().items()} if d["learner"] is not None else None,
                 "val": d.get("val", [])}
            for dk, d in self.domains.items()}
        st["pending"] = [{"fp": m["fp"], "z": m["z"], "tag": m["tag"]} for m in self.pending]
        st["cls_cfg"] = {"write": self.write_mode, "tau": self.tau, "min_n": self.min_n, "slow_lr": self.slow_lr,
                         "slow_steps": self.slow_steps, "slow_batch": self.slow_batch, "reroute": self.reroute,
                         "mature_ratio": self.mature_ratio, "mature_k": self.mature_k, "reject_ratio": self.reject_ratio,
                         "familiar_thr": self.familiar_thr, "pending_m": self.pending_m, "tau_new": self.tau_new,
                         "fp_unit": self.fp_unit}
        torch.save(st, path)

    def _load_state(self, path):
        super()._load_state(path)
        st = torch.load(path, map_location="cpu")
        dev = self._params[0].device
        self.domains = {}
        for dk, d in st.get("domains", {}).items():
            learner, opt = None, None
            if d["learner"] is not None:
                learner = copy.deepcopy(self.wm.predictor)
                learner.load_state_dict(d["learner"])
                learner.to(dev).eval()
                for p in learner.parameters():
                    p.requires_grad_(True)
                opt = torch.optim.Adam(learner.parameters(), lr=float(st.get("cls_cfg", {}).get("slow_lr", 1e-4)))
            self.domains[dk] = {"n": d["n"], "tags": d["tags"], "replay": [z.to(dev) for z in d["replay"]],
                                "fp_sum": [s.to(dev) for s in d["fp_sum"]] if d["fp_sum"] is not None else None,
                                "fp_n": d["fp_n"], "learner": learner, "opt": opt, "val": list(d.get("val", []))}
        self.pending = list(st.get("pending", []))


class SlowFrozenMPCPlanner(MPCPlanner):
    """Frozen, batched MPC with theta0 + slow loaded from a ContinualMPCPlanner state file."""

    def __init__(self, adapt=None, continual=None, probe=None, **kwargs):
        super().__init__(**kwargs)
        adapt, c = dict(adapt or {}), dict(continual or {})
        meta = {"state_in": c.get("state_in")}
        if c.get("state_in"):
            # Same tensor selection as AdaJEPA's adaptation (the trainer is only used for that).
            tr = AdaJEPATrainer(
                wm=self.wm, lr=adapt.get("lr", 5e-4), finetune_encoder=adapt.get("finetune_encoder", True),
                last_layer_only=adapt.get("last_layer_only", True), encoder_lr=adapt.get("encoder_lr"),
                encoder_last_layer_only=adapt.get("encoder_last_layer_only", True),
            )
            params = list(tr._ada_predictor_params) + list(tr._ada_encoder_params)
            st = torch.load(c["state_in"], map_location="cpu")
            _check_shapes(st["shapes"], params, c["state_in"])
            with torch.no_grad():
                for p, s in zip(params, st["slow"]):
                    p.add_(s.to(p.device, p.dtype))
            k = st["n_pred"]
            meta.update({"episode": st["episode"], "w": st["w"],
                         "slow_pred": _norm(st["slow"][:k]), "slow_enc": _norm(st["slow"][k:])})
        Path("slow_applied.json").write_text(json.dumps(meta, indent=2))


class LearnerFrozenMPCPlanner(MPCPlanner):
    """Frozen, batched MPC with one domain's slow predictor from a CLSMPCPlanner state file in place of theta0's.

    The theta0 encoder stays: the learner was trained on its latents. continual.state_in is the state file and
    continual.tag the domain; nothing adapts. learner_applied.json records what was loaded."""

    def __init__(self, adapt=None, continual=None, probe=None, **kwargs):
        super().__init__(**kwargs)
        c = dict(continual or {})
        if not c.get("state_in") or not c.get("tag"):
            raise ValueError("LearnerFrozenMPCPlanner needs planner.continual.state_in and planner.continual.tag")
        st = torch.load(c["state_in"], map_location="cpu")
        d = st.get("domains", {}).get(c["tag"])
        if d is None or d.get("learner") is None:
            raise ValueError(f"{c['state_in']}: no trained learner for domain {c['tag']!r}")
        self.wm.predictor.load_state_dict(d["learner"])  # the sub-planner and the evaluator share this model
        Path("learner_applied.json").write_text(json.dumps({
            "state_in": c["state_in"], "domain": c["tag"], "episode": st["episode"], "n": d["n"],
            "tags": d["tags"], "val_last": d.get("val", [])[-3:]}, indent=2))


def _check_shapes(shapes, params, path):
    if [tuple(s) for s in shapes] != [tuple(p.shape) for p in params]:
        raise ValueError(f"{path}: slow state does not match the adapted tensors")
