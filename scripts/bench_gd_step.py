"""Time one GD planning step (rollout + MSE objective + backward + Adam) of the real world model
at several batch sizes, to see whether a step is GPU-bound or CPU/launch-bound (README_cudagraph.md).

    CUDA_VISIBLE_DEVICES=7 python scripts/bench_gd_step.py
"""
import sys, time
from pathlib import Path
import torch
from omegaconf import OmegaConf

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
import os
os.chdir(repo)
from plan import load_model

ckpt_dir = repo / "checkpoints/pushobj_bias_16k_e3_seed0"
train_cfg = OmegaConf.load(ckpt_dir / "hydra.yaml")
dev = torch.device("cuda")
wm = load_model(ckpt_dir / "checkpoints/model_3.pth", train_cfg, 1, dev)
wm.eval()
for p in wm.parameters():
    p.requires_grad_(False)

act_dim = wm.action_encoder.in_chans if hasattr(wm.action_encoder, "in_chans") else 10
prop_dim = wm.proprio_encoder.in_chans if hasattr(wm.proprio_encoder, "in_chans") else 4
H = 5  # horizon 25 env steps / frameskip 5

def step_time(b, n_steps=30, compute_encoder_once=False):
    obs = {"visual": torch.randn(b, 1, 3, 224, 224, device=dev), "proprio": torch.randn(b, 1, prop_dim, device=dev)}
    with torch.no_grad():
        z_g = wm.encode_obs({"visual": torch.randn(b, 1, 3, 224, 224, device=dev), "proprio": torch.randn(b, 1, prop_dim, device=dev)})
    act = torch.zeros(b, H, act_dim, device=dev, requires_grad=True)
    opt = torch.optim.Adam([act], lr=0.1)
    def one():
        opt.zero_grad()
        z_obses, _ = wm.rollout(obs_0=obs, act=act)
        loss = ((z_obses["visual"][:, -1:] - z_g["visual"]) ** 2).mean(dim=tuple(range(1, z_g["visual"].dim()))) \
             + 0.5 * ((z_obses["proprio"][:, -1:] - z_g["proprio"]) ** 2).mean(dim=tuple(range(1, z_g["proprio"].dim())))
        (loss.mean() * b).backward()
        opt.step()
        return loss.mean().item()  # the real loop syncs here every step too
    for _ in range(5):
        one()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n_steps):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n_steps

print(f"act_dim={act_dim} prop_dim={prop_dim} horizon={H}")
for b in (1, 8, 32, 50):
    ms = 1000 * step_time(b)
    print(f"batch {b:3d}: {ms:7.1f} ms/step  -> {ms/b:6.2f} ms per sample-step")

# how much of a step is the encoder pass over obs_0 (recomputed at every GD step in rollout)?
obs1 = {"visual": torch.randn(1, 1, 3, 224, 224, device=dev), "proprio": torch.randn(1, 1, prop_dim, device=dev)}
with torch.no_grad():
    for _ in range(5): wm.encode_obs(obs1)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(30): wm.encode_obs(obs1)
    torch.cuda.synchronize()
print(f"encoder forward alone (batch 1): {1000*(time.perf_counter()-t)/30:.1f} ms")

# GPU-side time of a batch-1 step (kernel time only) via the profiler
from torch.profiler import profile, ProfilerActivity
act = torch.zeros(1, H, act_dim, device=dev, requires_grad=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        z_obses, _ = wm.rollout(obs_0=obs1, act=act)
        z_obses["visual"].pow(2).mean().backward()
    torch.cuda.synchronize()
ev = prof.key_averages()
cuda_ms = sum(e.self_device_time_total for e in ev) / 1000 / 10
n_kernels = sum(e.count for e in ev if e.device_type.name == "CUDA") / 10
print(f"batch-1 step: GPU kernel time {cuda_ms:.1f} ms, ~{n_kernels:.0f} CUDA kernels launched per step")
