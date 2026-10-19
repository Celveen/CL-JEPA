"""Stall-triggered escapes on top of AdaJEPA planning (planner side, separate from the continual line).

Late in an episode AdaJEPA often stalls: the pusher hovers and the object sits still,
short of the goal (reports/cont_S7/diagnostics.txt). This planner detects the stall
from observations only and intervenes on the next replan; everything else is the
AdaJEPA loop of ProbeMPCPlanner (probe.mode=grad).

Stall: from replan `start` on, the pusher (proprio position) moved less than
`stall_px` over the last `window` replans. Escapes are at least `gap` replans apart
(default window + 1, so a check's window never contains an earlier escape), and an
episode gets at most `max_n` escapes. escape.mode decides the intervention:

    none         detect and log only
    kick_random  add a constant offset of `kick_px` pixels per env step, in a random
                 direction, to the actions executed on that replan
    kick_goal    the same offset toward the pusher's goal position (proprio of the goal
                 observation), shortened so it adds up to at most the remaining distance
                 (with proprio_boost > 1: from the episode's first escape on, including that
                 replan's plan, the objective weights its proprio term proprio_boost times
                 more, so the planner holds the pusher at its goal instead of pulling it back)
    restart      plan that replan from `restarts` initialisations in one batch (the usual
                 warm start and restarts - 1 random ones) and execute the plan with the
                 lowest predicted objective
    staged       object first, then pusher: the first `object_escapes` escapes only reweight
                 the objective's proprio term by `object_scale` (0: the planner sees the image
                 alone, so leaving the pusher's goal to push the object is no longer
                 penalised); every later escape is kick_goal with the proprio term at proprio_boost

boost_replans > 0 makes the proprio boost short-lived: it covers the kick replan and the next
boost_replans replans, then the weight returns to the config's (each kick restarts it); 0 keeps it
on for the rest of the episode.

workspace_clip (independent of the mode) keeps the pusher inside the arena. The pusher is a
kinematic body, which the walls do not stop, and actions are unclipped offsets of its target, so a
large planned action carries it out of the arena (the episode is then lost). Before execution, the
pusher's path is predicted exactly from its observed position and velocity (the PD controller alone
moves it); an env step whose path would leave [ws_lo, ws_hi] (centre positions at which the pusher
touches the walls) gets its target pulled back inside, all other steps are left bit-identical.

action_cap_px > 0 bounds every env step's target offset to that many pixels (in the data, 99% of
offsets are below 74 px): GD projects its actions after each optimiser step, and the executed actions
are projected again after any kick, before the workspace clip. Entries within the bound keep their
exact values.

Per replan, per_iter_logs.json gets the stall check (pusher_disp_w, stalled), on escape replans an
`escape` record, and ws_clipped when the workspace clip changed an action; escape_episodes.jsonl gets
one line per episode.
The global RNG is seeded per episode (dropout in the AdaJEPA step), so arms pair with
mode=none exactly until an episode's first escape; compare against a mode=none run,
not against runs of other planners.
"""

import json
import types
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from utils import move_to_device
from .adajepa_probe import ProbeMPCPlanner

MODES = ("none", "kick_random", "kick_goal", "restart", "staged")
ACTION_SCALE = 100.0  # env action = dataset action x 100 = pixel offset of the pusher's target (env/pusht/pusht_env.py)


class EscapeMPCPlanner(ProbeMPCPlanner):
    def __init__(self, escape=None, **kwargs):
        super().__init__(**kwargs)
        e = dict(escape or {})
        self.esc_mode = e.get("mode", "none")
        if self.esc_mode not in MODES:
            raise ValueError(f"escape.mode must be one of {MODES}, got {self.esc_mode!r}")
        self.esc_start = int(e.get("start", 10))
        self.esc_window = int(e.get("window", 3))
        self.esc_stall_px = float(e.get("stall_px", 10.0))
        self.esc_max = int(e.get("max_n", 5))
        self.esc_gap = int(e["gap"]) if e.get("gap") is not None else self.esc_window + 1
        self.kick_px = float(e.get("kick_px", 10.0))
        self.n_restarts = int(e.get("restarts", 8))
        self.esc_seed = int(e.get("seed", 0))
        self.proprio_boost = float(e.get("proprio_boost", 1.0))
        self.object_scale = float(e.get("object_scale", 0.0))
        self.object_escapes = int(e.get("object_escapes", 1))
        self.boost_replans = int(e.get("boost_replans", 0))
        self.workspace_clip = bool(e.get("workspace_clip", False))
        self.ws_lo, self.ws_hi = float(e.get("ws_lo", 22.0)), float(e.get("ws_hi", 489.0))
        self.action_cap_px = float(e.get("action_cap_px", 0.0))
        if self.action_cap_px > 0:
            inner_get_opt = self.sub_planner.get_action_optimizer

            def get_opt(actions):  # project GD's actions after every optimiser step
                opt = inner_get_opt(actions)
                step = opt.step

                def projected_step(_opt, *a, **k):
                    out = step(*a, **k)
                    with torch.no_grad():
                        actions.copy_(self._cap(actions))
                    return out

                opt.step = types.MethodType(projected_step, opt)  # LR schedulers wrap a bound step
                return opt

            self.sub_planner.get_action_optimizer = get_opt
        self._inner_plan = self.sub_planner.plan
        self.sub_planner.plan = self._sub_plan
        self._prop_scale = 1.0  # weight of the objective's proprio term relative to the config's
        if self.proprio_boost != 1.0 or self.esc_mode == "staged":
            self._inner_obj = self.sub_planner.objective_fn
            self.sub_planner.objective_fn = self._objective

    # ------------------------------------------------------------------ episode
    def _plan_single(self, **kw):
        self._pos, self._n_esc, self._last_esc, self._due, self._esc_rec = [], 0, -(10**9), False, None
        self._prop_scale, self._boost_until, self._ws_flag = 1.0, None, False
        self._ep_seed = int(kw["seed_i"][0])
        # The AdaJEPA step runs the predictor in train mode (dropout 0.1), so it draws from the global
        # RNG; seeding it per episode keeps every arm identical to mode=none up to its first escape.
        torch.manual_seed(self.esc_seed * 1_000_003 + self._ep_seed)
        g = np.asarray(kw["obs_g_i"]["proprio"], dtype=np.float64)
        self._goal_pos = g.reshape(-1, g.shape[-1])[-1, :2]
        out = super()._plan_single(**kw)
        recs = [r for r in self._records if r["sample_idx"] == self._sample_idx]
        with Path("escape_episodes.jsonl").open("a") as f:
            f.write(json.dumps({
                "sample_idx": self._sample_idx, "seed": self._ep_seed, "success": bool(self.is_success[0]),
                "replans": self.iter, "n_escapes": self._n_esc,
                "escape_iters": [r["escape"]["iter"] for r in recs if "escape" in r],
                "stalled_iters": [r["step"] for r in recs if r.get("stalled")],
                "ws_clipped_iters": [r["step"] for r in recs if r.get("ws_clipped")],
            }) + "\n")
        return out

    def _post_env_feedback(self, taken_actions, e_obses):
        out = super()._post_env_feedback(taken_actions, e_obses)
        p = np.asarray(e_obses["proprio"][0, -1, :2], dtype=np.float64)  # pusher after this replan
        self._pos.append(p)
        rec = self._records[-1]
        if self._prop_scale != 1.0:
            rec["prop_scale"] = self._prop_scale  # objective weight this replan was planned with
        if self._ws_flag:
            rec["ws_clipped"] = True
            self._ws_flag = False
        if self._esc_rec is not None:  # the escape executed on this replan
            rec["escape"] = {**self._esc_rec, "pusher_after": p.round(2).tolist()}
            self._esc_rec = None
        W, nxt = self.esc_window, self.iter + 1
        if len(self._pos) > W:
            disp = float(np.linalg.norm(self._pos[-1] - self._pos[-1 - W]))
            stalled = (self.esc_start <= nxt < self.max_iter and disp < self.esc_stall_px
                       and nxt - self._last_esc >= self.esc_gap and not self.is_success[0])
            rec["pusher_disp_w"] = disp
            rec["stalled"] = bool(stalled)
            self._due = stalled and self._n_esc < self.esc_max
        return out

    # ------------------------------------------------------------------ escapes
    def _sub_plan(self, obs_0, obs_g, actions=None, step=None):
        if self._boost_until is not None and self.iter > self._boost_until:  # short boost over
            self._prop_scale, self._boost_until = 1.0, None
        if not self._due or self.esc_mode == "none":
            acts, lens = self._inner_plan(obs_0=obs_0, obs_g=obs_g, actions=actions, step=step)
            return self._finish(acts, obs_0), lens
        self._due = False
        self._n_esc += 1
        self._last_esc = self.iter
        p = self._pos[-1]
        rec = {"n": self._n_esc, "iter": self.iter, "mode": self.esc_mode, "pusher_before": p.round(2).tolist()}
        if self.esc_mode == "staged" and self._n_esc <= self.object_escapes:  # object phase: reweight only, no kick
            self._prop_scale = self.object_scale
            rec.update({"phase": "object", "prop_scale": self._prop_scale})
            acts, _ = self._inner_plan(obs_0=obs_0, obs_g=obs_g, actions=actions, step=step)
            self._esc_rec = rec
            return self._finish(acts, obs_0), np.full(acts.shape[0], np.inf)
        if self.proprio_boost != 1.0:
            self._prop_scale = self.proprio_boost
            rec["proprio_boost"] = self.proprio_boost
            if self.boost_replans > 0:
                self._boost_until = self.iter + self.boost_replans
        if self.esc_mode == "staged":
            rec["phase"] = "pusher"
        if self.esc_mode == "restart":
            acts, extra = self._restart_plan(obs_0, obs_g, actions, step)
            rec.update(extra)
        else:
            acts, _ = self._inner_plan(obs_0=obs_0, obs_g=obs_g, actions=actions, step=step)
            offset = self._kick_offset(p)
            acts = self._add_offset(acts, offset)
            rec["offset_px_per_step"] = offset.round(3).tolist()
        self._esc_rec = rec
        return self._finish(acts, obs_0), np.full(acts.shape[0], np.inf)

    def _finish(self, acts, obs_0):
        """Bound the action magnitude (if configured), then keep the pusher inside the arena."""
        if self.action_cap_px > 0:
            acts = self._cap(acts.detach())
        return self._clip_workspace(acts, obs_0)

    def _cap(self, a):
        """Scale every env step whose target offset exceeds action_cap_px back onto the bound."""
        fs = self.evaluator.frameskip
        x = rearrange(a, "b t (f d) -> b t f d", f=fs)
        mean = torch.as_tensor(self.preprocessor.action_mean, device=a.device, dtype=a.dtype)
        std = torch.as_tensor(self.preprocessor.action_std, device=a.device, dtype=a.dtype)
        px = (x * std + mean) * ACTION_SCALE
        norm = px.norm(dim=-1, keepdim=True)
        over = norm > self.action_cap_px
        if not bool(over.any()):
            return a
        capped = (px * (self.action_cap_px / norm.clamp_min(1e-9)) / ACTION_SCALE - mean) / std
        return rearrange(torch.where(over, capped, x), "b t f d -> b t (f d)")

    def _clip_workspace(self, acts, obs_0):
        """Keep the pusher inside the arena (see the module docstring). The pusher is kinematic, so its path
        follows from its observed position and velocity and the env's PD controller alone (contacts do not
        move it); only env steps whose path would leave [ws_lo, ws_hi] get a target pulled back inside."""
        if not self.workspace_clip:
            return acts
        n, fs = self.n_taken_actions, self.evaluator.frameskip
        head = rearrange(acts[:, :n].detach().cpu(), "b t (f d) -> b (t f) d", f=fs)
        off = (self.preprocessor.denormalize_actions(head)[0] * ACTION_SCALE).double().numpy()
        pr = np.asarray(obs_0["proprio"], dtype=np.float64)
        pr = pr.reshape(-1, pr.shape[-1])[-1]
        p, v = pr[:2].copy(), pr[2:4].copy()
        new, changed = off.copy(), np.zeros(len(off), bool)
        for j in range(len(off)):
            p1, v1, lo, hi = self._pd_step(p, v, p + off[j])
            if lo.min() < self.ws_lo or hi.max() > self.ws_hi:
                t = np.clip(p + off[j], self.ws_lo, self.ws_hi)
                for _ in range(20):  # momentum may still carry it out: pull the target in by the overshoot
                    p1, v1, lo, hi = self._pd_step(p, v, t)
                    over = np.maximum(self.ws_lo - lo, 0) - np.maximum(hi - self.ws_hi, 0)
                    if not over.any():
                        break
                    t = t + over
                new[j], changed[j] = t - p, True
            p, v = p1, v1
        if not changed.any():
            return acts
        raw = self.preprocessor.denormalize_actions(head).clone()
        mask = torch.as_tensor(changed)
        raw[0, mask] = torch.as_tensor(new[changed] / ACTION_SCALE, dtype=raw.dtype)
        head = head.clone()
        head[0, mask] = self.preprocessor.normalize_actions(raw)[0, mask].to(head.dtype)
        out = acts.detach().clone()
        out[:, :n] = rearrange(head, "b (t f) d -> b t (f d)", f=fs).to(out.device, out.dtype)
        self._ws_flag = True
        return out

    @staticmethod
    def _pd_step(p, v, target, kp=100.0, kv=20.0, dt=0.01, n_sub=10):
        """One env step of the pusher's PD control (env/pusht/pusht_env.py: k_p, k_v, sim_hz 100, control_hz 10).
        Returns the end position and velocity and the per-axis min / max position along the way."""
        lo, hi = p.copy(), p.copy()
        for _ in range(n_sub):
            v = v + (kp * (target - p) - kv * v) * dt
            p = p + v * dt
            lo, hi = np.minimum(lo, p), np.maximum(hi, p)
        return p, v, lo, hi

    def _objective(self, z_obs_pred, z_obs_tgt, step=None):
        """The configured objective with its proprio MSE term weighted _prop_scale times."""
        if self._prop_scale == 1.0:
            return self._inner_obj(z_obs_pred, z_obs_tgt, step=step)
        s = self._prop_scale ** 0.5
        return self._inner_obj({**z_obs_pred, "proprio": z_obs_pred["proprio"] * s},
                               {**z_obs_tgt, "proprio": z_obs_tgt["proprio"] * s}, step=step)

    def _kick_offset(self, p):
        """Pixel offset per env step for this replan's executed actions."""
        if self.esc_mode == "kick_random":
            rng = np.random.default_rng([self.esc_seed, self._ep_seed, self._n_esc])
            a = rng.uniform(0, 2 * np.pi)
            return self.kick_px * np.array([np.cos(a), np.sin(a)])
        v = self._goal_pos - p
        d = float(np.linalg.norm(v))
        n_env = self.n_taken_actions * self.evaluator.frameskip
        return v / d * min(self.kick_px, d / n_env) if d > 1e-6 else np.zeros(2)

    def _add_offset(self, acts, offset_px):
        n, fs = self.n_taken_actions, self.evaluator.frameskip
        head = rearrange(acts[:, :n].detach().cpu(), "b t (f d) -> b t f d", f=fs)
        raw = self.preprocessor.denormalize_actions(head) + torch.as_tensor(offset_px / ACTION_SCALE, dtype=head.dtype)
        head = rearrange(self.preprocessor.normalize_actions(raw), "b t f d -> b t (f d)")
        out = acts.detach().clone()
        out[:, :n] = head.to(out.device, out.dtype)
        return out

    def _restart_plan(self, obs_0, obs_g, actions, step):
        R = self.n_restarts
        base = self.sub_planner.init_actions(obs_0, actions).cpu()  # the usual warm start
        gen = torch.Generator().manual_seed(self.esc_seed * 1_000_003 + self._ep_seed * 101 + self._n_esc)
        init = torch.cat([base, torch.randn((R - 1,) + tuple(base.shape[1:]), generator=gen)], dim=0)
        rep = lambda o: {k: (v.repeat_interleave(R, 0) if torch.is_tensor(v) else np.repeat(v, R, axis=0))
                         for k, v in o.items()}
        obs_0r, obs_gr = rep(obs_0), rep(obs_g)
        acts, _ = self._inner_plan(obs_0=obs_0r, obs_g=obs_gr, actions=init, step=step)
        acts = acts.detach()
        with torch.no_grad():
            z_g = self.wm.encode_obs(move_to_device(self.preprocessor.transform_obs(obs_gr), self.device))
            z, _ = self.wm.rollout(obs_0=move_to_device(self.preprocessor.transform_obs(obs_0r), self.device), act=acts)
            loss = self.objective_fn(z, z_g, step=step).cpu().numpy()
        best = int(np.argmin(loss))
        return acts[best : best + 1], {"restart_losses": [round(float(x), 5) for x in loss], "restart_best": best}
