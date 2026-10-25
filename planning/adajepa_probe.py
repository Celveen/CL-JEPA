"""Experiment 0: does AdaJEPA's planning gain come from information or perturbation?

ProbeMPCPlanner runs the AdaJEPA loop sample by sample, but `probe.mode` decides
what each replan's weight change is. Every adapting mode first takes the
ordinary AdaJEPA step on the episode's own buffer, which gives the reference
direction d_own. The trainer builds a fresh Adam for every finetune call and
takes one step, so d_own is lr * g / (|g| + eps), i.e. about +-lr per element:
the size of an AdaJEPA update is fixed and only its direction comes from data.
The mode decides what is applied instead:

    none          no change: frozen weights, planned sample by sample
    grad          d_own                     (AdaJEPA itself)
    ascent        -d_own                    (same step, anti-informative direction)
    random_target AdaJEPA step on the own buffer with the observed next latents
                  replaced by the model's own prediction + u, u ~ N(0, I) one latent
                  shift vector per replan: a natural, size-matched step that says
                  nothing about the real transition
    random_target_fixed  the same with u drawn once per episode (coherent drift)
    random_sign   xi * d_own, xi = +-1 per element, redrawn every replan
    random_fixed  xi_ep * |d_own|, xi_ep drawn once per episode
    other_data    AdaJEPA step on a same-sized window from *other* episodes
                  (pre-rendered bank, scripts/exp0_build_bank.py)

Every mode except none takes a fresh-Adam first step (or a sign flip of one), so
all of them match AdaJEPA's per-element step size and differ only in where the
direction comes from. The parameter-space signs (random_sign/random_fixed) move
the predictions far less than a gradient step of the same norm, because a
gradient points where the outputs are most sensitive; the random_target modes
keep that sensitivity and randomise only the target, so they are the
function-scale perturbation control. probe.scale multiplies the applied change
(grad with scale 1 leaves AdaJEPA's own update untouched).

Each replan also logs task progress (distance to the goal state) and, around
the update, eval-mode prediction errors under the current and the
pre-adaptation weights: on the whole buffer, and on the newest frame alone,
which no update has seen yet. After the update it logs the step and
displacement norms per module, the cosine to d_own and the induced change in
predictions. All of it lands in per_iter_logs.json next to the usual fields.
"""

import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from .adajepa import AdaJEPATrainer
from .adajepa_mpc import AdaJEPAMPCPlanner
from .mpc import _cuda_sync

MODES = ("none", "grad", "ascent", "random_target", "random_target_fixed",
         "random_sign", "random_fixed", "other_data")


class ProbeTrainer(AdaJEPATrainer):
    """AdaJEPATrainer whose prediction targets can be replaced (target_fn)."""

    target_fn = None  # callable(z_pred) -> target, or None for the observed next latents

    def _prediction_loss(self, z, detach_src=True, detach_tgt=True):
        if self.target_fn is None:
            return super()._prediction_loss(z, detach_src, detach_tgt)
        T = z.shape[1] - 1
        window = min(self.wm.num_hist, T)
        drop = self.wm.action_dim
        losses = []
        for t in range(T - window + 1):
            z_src = z[:, t : t + window]
            if detach_src:
                z_src = z_src.detach()
            z_pred = self.wm.predict(z_src)
            z_tgt = self.target_fn(z_pred).detach()
            losses.append(self.criterion(z_pred[..., :-drop], z_tgt[..., :-drop]))
        return torch.stack(losses).mean()


class ProbeMPCPlanner(AdaJEPAMPCPlanner):
    def __init__(self, probe=None, **kwargs):
        super().__init__(**kwargs)
        old = self.adajepa_trainer
        self.adajepa_trainer = ProbeTrainer(
            wm=self.wm, lr=old.lr, steps=old.steps, optimizer_name=old.optimizer_name,
            finetune_encoder=old.finetune_encoder, last_layer_only=old.last_layer_only,
            encoder_lr=old.encoder_lr, encoder_last_layer_only=old.encoder_last_layer_only,
        )
        if self.wm.concat_dim != 1:
            raise ValueError("ProbeMPCPlanner assumes concat_dim=1 (visual | proprio | action channels)")
        probe = dict(probe) if probe else {}
        self.probe_mode = probe.get("mode", "grad")
        if self.probe_mode not in MODES:
            raise ValueError(f"probe.mode must be one of {MODES}, got {self.probe_mode!r}")
        if self.replay_mode == "hard":
            raise ValueError("ProbeMPCPlanner merges the buffer into one sequence; use recent<N> or false")
        self.probe_seed = int(probe.get("seed", 0))
        self.probe_scale = float(probe.get("scale", 1.0))
        tr = self.adajepa_trainer
        self._groups = {"pred": len(tr._ada_predictor_params), "enc": len(tr._ada_encoder_params)}
        self._params = list(tr._ada_predictor_params) + list(tr._ada_encoder_params)
        self._init = [p.detach().clone() for p in self._params]
        self._drop_act = self.wm.action_dim
        self._drop_obs = self.wm.proprio_dim + self.wm.action_dim
        self._bank = None
        if self.probe_mode == "other_data":
            self._bank = self._load_bank(probe.get("bank_dir"), probe.get("bank_shapes"))
        self._gen = None
        self._xi_fixed = None
        self._u_fixed = None
        self._last_eval = None
        meta = {
            "mode": self.probe_mode, "seed": self.probe_seed, "scale": self.probe_scale,
            "bank_dir": probe.get("bank_dir"), "bank_shapes": list(probe.get("bank_shapes") or []),
            "n_tensors": self._groups,
            "n_params": {"pred": sum(p.numel() for p in self._params[: self._groups["pred"]]),
                         "enc": sum(p.numel() for p in self._params[self._groups["pred"]:])},
            "lr": tr.lr, "encoder_lr": tr.encoder_lr, "steps": tr.steps, "replay": [self.replay_mode, self.replay_size],
        }
        Path("probe_meta.json").write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------ episode setup
    def _plan_single(self, **kw):
        """Seed this episode's random stream and keep the env feedback of every replan."""
        self._gen = torch.Generator().manual_seed(self.probe_seed * 1_000_003 + int(kw["seed_i"][0]))
        self._xi_fixed = (
            [self._rademacher(p) for p in self._params] if self.probe_mode == "random_fixed" else None
        )
        self._u_fixed = self._latent_shift() if self.probe_mode == "random_target_fixed" else None
        evaluator = self.evaluator
        inner = evaluator.eval_actions

        def eval_and_keep(*args, **kwargs):
            out = inner(*args, **kwargs)
            self._last_eval = out
            return out

        evaluator.eval_actions = eval_and_keep
        try:
            return super()._plan_single(**kw)
        finally:
            del evaluator.eval_actions

    def _record_eval(self, out):
        # resim_from_start=False: eval_and_keep saw only the last iteration's actions; keep the whole episode
        self._last_eval = out

    # ------------------------------------------------------------------ MPC hook
    def _post_env_feedback(self, taken_actions, e_obses):
        obs_seq, act_seq = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        self._obs_buffer.append(obs_seq)
        self._act_buffer.append(act_seq)
        self._segment_scores = self._trim_buffer(self._obs_buffer, self._act_buffer, self._segment_scores)
        rec = {
            "sample_idx": self._sample_idx,
            "step": self.iter + 1,
            "success": bool(self.is_success[0]),
            "t_plan_s": self.t_plan_s,
            "buf_n": len(self._obs_buffer),
            "act": [round(float(x), 4) for x in taken_actions[0].flatten().tolist()],
        }
        rec.update(self._task_progress())
        extra_logs = None
        if (self.iter + 1) % self.adajepa_finetune_every == 0:
            _cuda_sync()
            t0 = time.perf_counter()
            obs, act = self._merged_buffer()
            rec.update(self._losses_now_and_init(obs, act))
            step_losses, applied, d_own = self._update(obs, act)
            rec.update(self._update_stats(obs, act, applied, d_own))
            _cuda_sync()
            rec["t_adajepa_s"] = time.perf_counter() - t0
            pred_loss = step_losses[-1] if step_losses else float("nan")
            rec["adajepa/pred_loss"] = pred_loss
            self._record_adajepa_losses(self._sample_idx, self.iter + 1, step_losses)
            extra_logs = {"adajepa/pred_loss": pred_loss}
        self._records.append(rec)
        return extra_logs

    def _merged_buffer(self):
        if len(self._obs_buffer) == 1:
            return self._obs_buffer[0], self._act_buffer[0]
        obs, act = self.adajepa_trainer._merge_segments(self._obs_buffer, self._act_buffer)
        return obs[0], act[0]

    # ------------------------------------------------------------------ updates
    def _update(self, obs, act):
        """Apply this replan's weight change. Returns (own-data step losses, applied, d_own)."""
        if self.probe_mode == "none":
            return [], None, None
        tr = self.adajepa_trainer
        before = [p.detach().clone() for p in self._params]
        step_losses = tr.finetune([obs], [act], merge=False)
        d_own = [p.detach() - b for p, b in zip(self._params, before)]
        mode = self.probe_mode
        if mode == "grad":
            applied = d_own
        elif mode == "ascent":
            applied = [-d for d in d_own]
        elif mode == "random_sign":
            applied = [self._rademacher(d) * d for d in d_own]
        elif mode == "random_fixed":
            applied = [xi * d.abs() for xi, d in zip(self._xi_fixed, d_own)]
        elif mode in ("random_target", "random_target_fixed"):
            self._set(before)
            u = self._u_fixed if mode == "random_target_fixed" else self._latent_shift()
            tr.target_fn = lambda z_pred: z_pred + u
            try:
                tr.finetune([obs], [act], merge=False)
            finally:
                tr.target_fn = None
            applied = [p.detach() - b for p, b in zip(self._params, before)]
        else:  # other_data
            self._set(before)
            n = obs["visual"].shape[1] - 1
            o2, a2 = self._bank_window(n)
            tr.finetune([o2], [a2], merge=False)
            applied = [p.detach() - b for p, b in zip(self._params, before)]
        if mode != "grad" or self.probe_scale != 1.0:
            applied = [self.probe_scale * d for d in applied]
            self._set([b + d for b, d in zip(before, applied)])
        return step_losses, applied, d_own

    def _rademacher(self, like):
        signs = torch.randint(0, 2, like.shape, generator=self._gen, dtype=torch.int8)
        return signs.to(device=like.device, dtype=like.dtype) * 2 - 1

    def _latent_shift(self):
        """One random latent shift vector, broadcast over windows and positions."""
        return torch.randn(self.wm.emb_dim, generator=self._gen).to(self.device)

    @torch.no_grad()
    def _set(self, values):
        for p, v in zip(self._params, values):
            p.copy_(v)

    @contextlib.contextmanager
    def _initial_weights(self):
        """Temporarily put the pre-adaptation values back into the adapted tensors."""
        current = [p.detach().clone() for p in self._params]
        self._set(self._init)
        try:
            yield
        finally:
            self._set(current)

    # ------------------------------------------------------------------ other-episode bank
    def _load_bank(self, bank_dir, shapes):
        if not bank_dir or not shapes:
            raise ValueError("probe.mode=other_data needs probe.bank_dir and probe.bank_shapes")
        fs = self.evaluator.frameskip
        banks = []
        for shape in shapes:
            d = np.load(Path(bank_dir) / f"{shape}.npz")
            act = self.preprocessor.normalize_actions(torch.as_tensor(d["env_actions"], dtype=torch.float32))
            act = rearrange(act, "n (t f) d -> n t (f d)", f=fs)
            banks.append({"visual": d["visual"], "proprio": d["proprio"], "act": act})
            assert d["visual"].shape[1] == act.shape[1] + 1, (shape, d["visual"].shape, act.shape)
        return banks

    def _bank_window(self, n):
        """A random n-transition window from another episode, shaped like the own merged buffer."""
        bank = self._bank[int(torch.randint(len(self._bank), (1,), generator=self._gen))]
        T = bank["act"].shape[1]
        assert n <= T, f"buffer holds {n} transitions, bank segments only {T}"
        j = int(torch.randint(bank["act"].shape[0], (1,), generator=self._gen))
        o = int(torch.randint(T - n + 1, (1,), generator=self._gen))
        obs = {k: bank[k][j : j + 1, o : o + n + 1] for k in ("visual", "proprio")}
        return self.preprocessor.transform_obs(obs), bank["act"][j : j + 1, o : o + n].clone()

    # ------------------------------------------------------------------ diagnostics
    def _task_progress(self):
        e_states = self._last_eval[3]
        s = np.asarray(e_states[0, -1], dtype=np.float64)
        g = np.asarray(self.evaluator.state_g[0], dtype=np.float64)
        if g.ndim != 1:  # not a PushT/PushObj state vector (rope: particles); the evaluator logs its own distance
            return {}
        return {
            "dist4": float(np.linalg.norm(g[:4] - s[:4])),  # the success test uses < 20
            "dist_obj": float(np.linalg.norm(g[2:4] - s[2:4])),
            "dist_agent": float(np.linalg.norm(g[:2] - s[:2])),
            "ang": float(np.abs((g[4] - s[4] + np.pi) % (2 * np.pi) - np.pi)),  # success: < pi/9
            "state": [round(float(x), 3) for x in s],
        }

    @torch.no_grad()
    def _predictions(self, obs, act):
        """Teacher-forced predictions for every sliding window, plus the newest-frame one."""
        tr = self.adajepa_trainer
        o, a = tr._prepare_segment(obs, act)
        z = self.wm.encode(o, a)
        T = z.shape[1] - 1
        w = min(self.wm.num_hist, T)
        windows = [(self.wm.predict(z[:, t : t + w]), z[:, t + 1 : t + 1 + w]) for t in range(T - w + 1)]
        return windows

    def _mse(self, pred, tgt, drop):
        return float(torch.mean((pred[..., :-drop] - tgt[..., :-drop]) ** 2))

    def _losses(self, obs, act):
        windows = self._predictions(obs, act)
        pred_last, tgt_last = windows[-1][0][:, -1:], windows[-1][1][:, -1:]
        return {
            "buf": float(np.mean([self._mse(p, t, self._drop_act) for p, t in windows])),
            "new": self._mse(pred_last, tgt_last, self._drop_act),  # newest frame: not trained on yet
            "new_vis": self._mse(pred_last, tgt_last, self._drop_obs),
        }

    def _losses_now_and_init(self, obs, act):
        now = self._losses(obs, act)
        with self._initial_weights():
            init = self._losses(obs, act)
        out = {}
        for k in now:
            out[f"loss_{k}_cur"] = now[k]
            out[f"loss_{k}_init"] = init[k]
        return out

    def _update_stats(self, obs, act, applied, d_own):
        out = {}
        split = self._groups["pred"]
        groups = {"pred": slice(0, split), "enc": slice(split, None)}
        with torch.no_grad():
            for name, sl in groups.items():
                cur = self._params[sl]
                ini = self._init[sl]
                out[f"disp_{name}"] = _norm([p - q for p, q in zip(cur, ini)])
                out[f"step_{name}"] = _norm(applied[sl]) if applied is not None else 0.0
                out[f"cos_own_{name}"] = _cos(applied[sl], d_own[sl]) if applied is not None else float("nan")
            out["cos_own"] = _cos(applied, d_own) if applied is not None else float("nan")
        if applied is None:
            out["func_change"] = 0.0
            return out
        now = self._predictions(obs, act)
        with self._initial_weights():
            init = self._predictions(obs, act)
        out["func_change"] = float(np.mean([self._mse(p1, p0, self._drop_act) for (p1, _), (p0, _) in zip(now, init)]))
        return out


def _norm(tensors):
    return float(torch.sqrt(sum((t.double() ** 2).sum() for t in tensors))) if tensors else 0.0


def _cos(a, b):
    if not a:
        return float("nan")
    dot = sum((x.double() * y.double()).sum() for x, y in zip(a, b))
    denom = _norm(a) * _norm(b)
    return float(dot / denom) if denom > 0 else float("nan")
