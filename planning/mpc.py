import time

import torch
import hydra
import numpy as np
from einops import rearrange
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        reuse_actions=True,
        resim_from_start=True,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        # Warm-start each replan from the unexecuted tail of the previous plan.
        self.reuse_actions = reuse_actions
        # True: every iteration re-simulates all actions so far from the start. False: it simulates only its own
        # actions from where the previous one left off (cheaper for slow simulators such as PyFleX), and the final
        # evaluation reuses the last iteration's (final_eval) instead of replaying the whole plan again.
        self.resim_from_start = resim_from_start
        self.final_eval = None  # (logs, successes) at the final state, set when resim_from_start is False
        # Appended to per-iter names, e.g. "_s3" when a subclass runs per-sample sub-plans.
        self.plan_suffix = ""
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []
        self.t_plan_s = 0.0  # wall-clock seconds of the latest sub-planner call

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def _post_env_feedback(self, taken_actions, e_obses):
        """Hook after each env rollout; subclasses may adapt the model here.
        Returns an optional dict of extra logs for this iteration."""
        return None

    def _record_eval(self, out):
        """Hook with (logs, successes, e_obses, e_states) of the whole trajectory so far, when iterations do not
        re-simulate from the start (eval_actions then only saw the last iteration's actions)."""
        return None

    def plan(self, obs_0, obs_g, actions=None):
        """
        actions is NOT used
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        self.iter = 0
        self.planned_actions = []
        self.final_eval = None
        traj = None  # (e_obses, e_states) executed so far, when not re-simulating from the start
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        memo_actions = None
        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}{self.plan_suffix}"
            _cuda_sync()
            t0 = time.perf_counter()
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=memo_actions,
                step=self.iter,
            )  # (b, t, act_dim)
            _cuda_sync()
            self.t_plan_s = time.perf_counter() - t0
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :] if self.reuse_actions else None
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            if self.resim_from_start:
                action_so_far = torch.cat(self.planned_actions, dim=1)
                self.evaluator.assign_init_cond(
                    obs_0=init_obs_0,
                    state_0=init_state_0,
                )
                logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                    action_so_far,
                    self.action_len,
                    filename=f"plan{self.iter}{self.plan_suffix}",
                    save_video=True,
                )
            else:
                # The evaluator starts where the previous iteration left off: simulate this iteration's actions
                # only and append them to the trajectory so far. Samples that already succeeded execute their
                # zeroed actions, a no-op for relative actions (PushT/PushObj, mazes) but not for absolute ones.
                logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                    taken_actions,
                    None,
                    filename=f"plan{self.iter}{self.plan_suffix}",
                    save_video=True,
                )
                if traj is not None:
                    e_obses = {k: np.concatenate([traj[0][k], v[:, 1:]], axis=1) for k, v in e_obses.items()}
                    e_states = np.concatenate([traj[1], e_states[:, 1:]], axis=1)
                traj = (e_obses, e_states)
                final_logs = dict(logs)
                self._record_eval((logs, successes, e_obses, e_states))
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            extra_logs = self._post_env_feedback(taken_actions, e_obses)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1, **(extra_logs or {})})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            # update evaluator's init conditions with new env feedback
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1

        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )
        if not self.resim_from_start and self.iter > 0:
            final_logs["success_rate"] = float(np.mean(self.is_success))
            self.final_eval = (final_logs, self.is_success.copy())

        return planned_actions, self.action_len
