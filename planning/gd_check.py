"""Test-only sub-planner for the CUDA-graph GD path (planning/gd.py, cuda_graph=true).

Every plan() runs eagerly and with the CUDA graph from the same RNG state, appends one record per call
to `check_log` (bitwise agreement of the actions and of the RNG state after planning, both times), and
returns the eager result, so the run follows the eager trajectory. Select it with
planner.sub_planner.target=planning.gd_check.GDCheckPlanner +planner.sub_planner.check_log=<path>.
"""
import json
import time

import torch

from .gd import GDPlanner


class GDCheckPlanner(GDPlanner):
    def __init__(self, *args, check_log="graph_check.jsonl", **kwargs):
        super().__init__(*args, **kwargs)
        self.check_log = check_log
        self.n_calls = 0

    def _timed(self, graph, obs_0, obs_g, actions, step):
        self.cuda_graph = graph
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = super().plan(obs_0, obs_g, actions, step)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    def plan(self, obs_0, obs_g, actions=None, step=None):
        cpu0, cuda0 = torch.get_rng_state(), torch.cuda.get_rng_state()
        (a_e, v_e), t_e = self._timed(False, obs_0, obs_g, actions, step)
        cpu_e, cuda_e = torch.get_rng_state(), torch.cuda.get_rng_state()

        torch.set_rng_state(cpu0)
        torch.cuda.set_rng_state(cuda0)
        n_graphs = len(self._graphs)
        (a_g, _), t_g = self._timed(True, obs_0, obs_g, actions, step)
        rng_same = torch.equal(cpu_e, torch.get_rng_state()) and torch.equal(cuda_e, torch.cuda.get_rng_state())
        torch.set_rng_state(cpu_e)  # continue exactly as the eager-only run would
        torch.cuda.set_rng_state(cuda_e)
        self.cuda_graph = False

        a_e_d = a_e.detach()
        rec = {
            "call": self.n_calls, "step": step, "bitwise": bool(torch.equal(a_e_d, a_g)),
            "max_abs_diff": float((a_e_d - a_g).abs().max()), "rng_same": bool(rng_same),
            "t_eager_s": round(t_e, 4), "t_graph_s": round(t_g, 4), "captured": len(self._graphs) > n_graphs,
            "n_graphs": len(self._graphs), "gpu_mem_mb": round(torch.cuda.memory_reserved() / 2**20),
        }
        with open(self.check_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.n_calls += 1
        return a_e, v_e
