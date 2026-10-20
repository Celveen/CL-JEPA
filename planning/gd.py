import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device
from tqdm import tqdm

class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        sample_type,
        lr,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        optimizer="adam",
        use_cosine_scheduler=True,
        scheduler_eta_min=0.0,
        adamw_weight_decay=0.01,
        cuda_graph=False,
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
        self.horizon = horizon
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.optimizer_name = str(optimizer).lower()
        self.use_cosine_scheduler = use_cosine_scheduler
        self.scheduler_eta_min = scheduler_eta_min
        self.adamw_weight_decay = adamw_weight_decay
        # cuda_graph: capture rollout + objective + backward once per (step, shapes, model state) and replay it
        # at every optimization step. Batch-1 planning launches ~1.5k tiny kernels per step, so it is bound by
        # CPU launch overhead, not the GPU. The optimizer, scheduler and noise stay eager, so lr changes and
        # RNG consumption are exactly as in the eager loop.
        self.cuda_graph = bool(cuda_graph)
        self._graphs = {}

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        """
        n_evals = obs_0["visual"].shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions

    def get_action_optimizer(self, actions):
        if self.optimizer_name == "sgd":
            return torch.optim.SGD([actions], lr=self.lr)
        if self.optimizer_name == "adam":
            return torch.optim.Adam([actions], lr=self.lr)
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                [actions], lr=self.lr, weight_decay=self.adamw_weight_decay
            )
        raise ValueError(
            f"Invalid optimizer name: {self.optimizer_name}. Expected one of: sgd, adam, adamw."
        )

    def get_scheduler(self, optimizer):
        if self.use_cosine_scheduler:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.opt_steps,
                eta_min=self.scheduler_eta_min,
            )
        return None

    def plan(self, obs_0, obs_g, actions=None, step=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        with torch.no_grad():
            z_obs_g = self.wm.encode_obs(trans_obs_g)

        actions = self.init_actions(obs_0, actions).to(self.device)
        if self.cuda_graph:
            return self._plan_graphed(trans_obs_0, z_obs_g, actions, step)
        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions)
        scheduler = self.get_scheduler(optimizer)
        n_evals = actions.shape[0]

        for i in tqdm(range(self.opt_steps)):
            optimizer.zero_grad()
            i_z_obses, i_zs = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
            loss = self.objective_fn(i_z_obses, z_obs_g, step=step)  # (n_evals, )
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                actions += torch.randn_like(actions) * self.action_noise  # Add Gaussian noise

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1}
            )
            if self.evaluator is not None and self.eval_every != -1 and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
        return actions, np.full(n_evals, np.inf)  # all actions are valid

    def _graph_key(self, trans_obs_0, z_obs_g, actions, step):
        """Everything a captured graph bakes in: the step (the objective may branch on it), input shapes,
        and the model's tensor addresses, requires_grad flags and train/eval modes. Weight *values* may
        change between plans (AdaJEPA updates and learner loads copy in place); the graph reads them live."""
        wm = self.wm
        return (
            step,
            tuple((k, tuple(v.shape), v.dtype) for k, v in sorted(trans_obs_0.items())),
            tuple((k, tuple(v.shape), v.dtype) for k, v in sorted(z_obs_g.items())),
            tuple(actions.shape),
            actions.dtype,
            tuple(t.data_ptr() for t in wm.parameters()),
            tuple(t.data_ptr() for t in wm.buffers()),
            tuple(p.requires_grad for p in wm.parameters()),
            tuple(m.training for m in wm.modules()),
        )

    def _capture(self, trans_obs_0, z_obs_g, actions, step):
        st = {
            "obs_0": {k: v.detach().clone() for k, v in trans_obs_0.items()},
            "z_g": {k: v.detach().clone() for k, v in z_obs_g.items()},
            "act": actions.detach().clone().requires_grad_(True),
        }
        n_evals = actions.shape[0]

        def fwd_bwd():
            i_z_obses, _ = self.wm.rollout(obs_0=st["obs_0"], act=st["act"])
            total_loss = self.objective_fn(i_z_obses, st["z_g"], step=step).mean() * n_evals
            total_loss.backward()
            return total_loss

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up outside capture (allocator, cuBLAS workspaces, lazy caches)
            for _ in range(3):
                st["act"].grad = None
                fwd_bwd()
        torch.cuda.current_stream().wait_stream(side)
        st["act"].grad = None  # backward inside capture allocates a static .grad that every replay overwrites
        st["graph"] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(st["graph"]):
            st["total"] = fwd_bwd()
        return st

    def _plan_graphed(self, trans_obs_0, z_obs_g, actions, step):
        key = self._graph_key(trans_obs_0, z_obs_g, actions, step)
        st = self._graphs.get(key)
        if st is None:
            if len(self._graphs) >= 64:
                self._graphs.clear()
            st = self._graphs[key] = self._capture(trans_obs_0, z_obs_g, actions, step)
        with torch.no_grad():
            for k, v in trans_obs_0.items():
                st["obs_0"][k].copy_(v)
            for k, v in z_obs_g.items():
                st["z_g"][k].copy_(v)
            st["act"].copy_(actions)
        actions = st["act"]
        optimizer = self.get_action_optimizer(actions)
        scheduler = self.get_scheduler(optimizer)
        n_evals = actions.shape[0]
        log_loss = getattr(self.wandb_run, "mode", None) != "disabled"  # .item() would sync every step

        for i in tqdm(range(self.opt_steps)):
            st["graph"].replay()  # rollout + objective + backward: actions.grad <- d total_loss / d actions
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                actions += torch.randn_like(actions) * self.action_noise  # Add Gaussian noise

            if log_loss:
                self.wandb_run.log(
                    {f"{self.logging_prefix}/loss": st["total"].item(), "step": i + 1}
                )
            if self.evaluator is not None and self.eval_every != -1 and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
        return actions.detach().clone(), np.full(n_evals, np.inf)  # the static buffer is reused by the next plan
