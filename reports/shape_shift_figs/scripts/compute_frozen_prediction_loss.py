"""Frozen prediction MSE on the archived Frozen trajectories (shape shift, H7t3, test seed 2000, sample 0).

No optimizer step and no replanning: the official shape-shift checkpoint is loaded, the Frozen run's executed
actions are replayed in the simulator, and the same loss as the logged AdaJEPA/Ours adaptation loss is evaluated
(sliding-window one-step latent MSE on the most recent five transitions, action dimensions excluded).

The logged adaptation loss is computed with the predictor and encoder in training mode (dropout 0.1), so the plotted
Frozen value uses training mode too, with fixed per-step diagnostic seeds; the deterministic eval-mode value is kept
next to it. Every model buffer is restored after each forward and the model hash is checked before and after.

Checks made before anything is written:
  * replay fidelity: replaying AdaJEPA's executed actions reproduces its saved frames (episode_frames.npz);
  * criterion: the eval-mode function on AdaJEPA's own buffer, with the untouched official weights, reproduces
    AdaJEPA's logged theta0 buffer loss (loss_buf_init) at every step;
  * Frozen replay: the first successful step equals the run's action_len, and the success flag matches.

Run from the CL-JEPA-cudagraph checkout (needs the repo code and the eval outputs):
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=5 python reports/shape_shift_figs/scripts/compute_frozen_prediction_loss.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import gym  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import env  # noqa: E402,F401  (registers pushobj)
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD, STATE_MEAN, STATE_STD  # noqa: E402
from planning.adajepa import AdaJEPATrainer  # noqa: E402
from plan import load_model  # noqa: E402
from preprocessor import Preprocessor  # noqa: E402

EV = REPO / "eval_outputs"
CKPT = REPO / "checkpoints/pushobj_shape_shift"
SHAPES = ["I", "L", "square", "Z", "small_tee", "plus", "T"]
CHUNK = {s: f"chunk_{i:02d}_{s}" for i, s in enumerate(SHAPES)}  # H7t3 test seed 2000
RUNS = {"GD": {"frozen": "cmp_gd_frozen_official/w0", "adajepa": "cmp_gd_official/w0"},
        "CEM": {"frozen": "cmp_cem_frozen_official/w0", "adajepa": "cmp_cem_official/w0"}}
SAMPLE = 0
BUFFER = 5
DEVICE = torch.device("cuda:0")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def digest(model):
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


cfg = OmegaConf.load(CKPT / "hydra.yaml")
fs = int(cfg.frameskip)
wm = load_model(CKPT / "checkpoints/model_latest.pth", cfg, cfg.num_action_repeat, DEVICE)
wm.eval()
for p in wm.parameters():
    p.requires_grad_(False)
initial_digest = digest(wm)
trainer = AdaJEPATrainer(wm, lr=0.0, steps=0, finetune_encoder=True, encoder_lr=0.0,
                         last_layer_only=True, encoder_last_layer_only=True)
prep = Preprocessor(ACTION_MEAN, ACTION_STD, STATE_MEAN[:5], STATE_STD[:5], PROPRIO_MEAN[:4], PROPRIO_STD[:4],
                    hydra.utils.instantiate(cfg.env.dataset.transform))
saved_buffers = [(b, b.detach().clone()) for b in wm.buffers()]
drop = wm.action_dim


def restore():
    with torch.no_grad():
        for b, v in saved_buffers:
            b.copy_(v)
    wm.eval()


def diag_seed(planner, shape, step):
    key = f"frozen-shape-buffer-mse-v1|{planner}|{shape}|{step}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") % (2 ** 31 - 1)


@torch.no_grad()
def buffer_loss(frames, proprio, actions, step, training, seed=None):
    """Loss on the buffer after `step` replans: model-stride frames max(0, step-5)..step and the actions between."""
    first = max(0, step - BUFFER)
    obs = prep.transform_obs({"visual": frames[None, first:step + 1], "proprio": proprio[None, first:step + 1]})
    act = torch.as_tensor(actions[None, first:step], dtype=torch.float32)
    restore()
    if training:
        wm.predictor.train()
        wm.encoder.train()
        base = getattr(wm.encoder, "base_model", None)
        if base is not None:
            base.eval()
    with torch.random.fork_rng(devices=[DEVICE.index or 0]):
        if seed is not None:
            torch.manual_seed(seed)
        o, a = trainer._prepare_segment(obs, act)
        z = wm.encode(o, a)
        if training:
            value = float(trainer._prediction_loss(z, detach_src=True, detach_tgt=True))
        else:  # the planner's logged buffer loss: mean over windows of the per-window MSE
            T = z.shape[1] - 1
            w = min(wm.num_hist, T)
            vals = [float(torch.mean((wm.predict(z[:, t:t + w])[..., :-drop] - z[:, t + 1:t + 1 + w][..., :-drop]) ** 2))
                    for t in range(T - w + 1)]
            value = float(np.mean(vals))
    restore()
    return value


def replay(shape, state_0, actions_norm, n):
    e = gym.make("pushobj", with_velocity=True, with_target=True)
    e.update_env({"shape": "+" if shape == "plus" else shape})
    exec_actions = rearrange(torch.as_tensor(actions_norm[:n]), "t (f d) -> (t f) d", f=fs)
    exec_actions = prep.denormalize_actions(exec_actions).numpy()
    obses, states = e.rollout(SAMPLE * 0 + 1, state_0, exec_actions)  # eval seed of sample 0 is seed * 0 + 1
    e.close()
    return obses, states, e


def per_iter(path):
    rows = [json.loads(l) for l in (path / "per_iter_logs.json").read_text().splitlines()]
    return [r for r in rows if r["sample_idx"] == SAMPLE]


ONLY = [x for x in sys.argv[1:] if "/" in x]  # e.g. GD/I for a pilot; the full run writes nothing partial
cases, checks = [], []
for planner, runs in RUNS.items():
    for shape in SHAPES:
        if ONLY and f"{planner}/{shape}" not in ONLY:
            continue
        fz_dir, ad_dir = EV / runs["frozen"] / CHUNK[shape], EV / runs["adajepa"] / CHUNK[shape]

        # --- check 1+2 on AdaJEPA (official weights): replay fidelity and criterion parity
        ad_rec = per_iter(ad_dir)
        ad_res = np.load(ad_dir / "final_eval_results.npz")
        ad_frames = np.load(ad_dir / "episode_frames.npz")
        vis, prop, act = ad_frames["ep0_visual"], ad_frames["ep0_proprio"], ad_frames["ep0_act"]
        n_ad = len(ad_rec)
        assert act.shape[0] == n_ad and np.allclose(act, ad_res["actions"][SAMPLE, :n_ad], atol=1e-6)
        ob, st, _ = replay(shape, ad_res["state_0"][SAMPLE], ad_res["actions"][SAMPLE], n_ad)
        pix = int(np.abs(ob["visual"][::fs].astype(int) - vis.astype(int)).max())
        prop_err = float(np.abs(ob["proprio"][::fs] - prop).max())
        crit = [abs(buffer_loss(vis, prop, act, k, training=False) - ad_rec[k - 1]["loss_buf_init"])
                for k in range(1, n_ad + 1)]
        checks.append({"planner": planner, "shape": shape, "adajepa_steps": n_ad,
                       "replay_max_pixel_diff": pix, "replay_max_proprio_diff": prop_err,
                       "eval_mode_vs_logged_theta0_buffer_loss_max_abs_diff": max(crit)})
        assert pix == 0 and prop_err < 1e-3, checks[-1]
        assert max(crit) < 1e-5, checks[-1]

        # --- Frozen
        res = np.load(fz_dir / "final_eval_results.npz")
        targets_sha = sha(fz_dir / "plan_targets.pkl")
        assert np.array_equal(res["state_0"][SAMPLE], ad_res["state_0"][SAMPLE])
        assert np.array_equal(res["state_g"][SAMPLE], ad_res["state_g"][SAMPLE])
        al = float(res["action_len"][SAMPLE])
        success = bool(res["successes"][SAMPLE])
        n = int(al) if np.isfinite(al) else res["actions"].shape[1]
        ob, st, e = replay(shape, res["state_0"][SAMPLE], res["actions"][SAMPLE], n)
        sg = res["state_g"][SAMPLE]
        e2 = gym.make("pushobj", with_velocity=True, with_target=True)
        flags = [bool(e2.eval_state(sg, st[k * fs])["success"]) for k in range(1, n + 1)]
        dists = [float(e2.eval_state(sg, st[k * fs])["state_dist"]) for k in range(1, n + 1)]
        e2.close()
        first_success = next((k + 1 for k, f in enumerate(flags) if f), None)
        assert (first_success == n) if success else (first_success is None), (planner, shape, flags, al)
        vis, prop = ob["visual"][::fs].astype(np.uint8), ob["proprio"][::fs].astype(np.float32)
        acts = res["actions"][SAMPLE, :n]
        train_l, eval_l, seeds = [], [], []
        for k in range(1, n + 1):
            sd = diag_seed(planner, shape, k)
            v = buffer_loss(vis, prop, acts, k, training=True, seed=sd)
            assert v == buffer_loss(vis, prop, acts, k, training=True, seed=sd)  # seeded repeat is exact
            train_l.append(v)
            eval_l.append(buffer_loss(vis, prop, acts, k, training=False))
            seeds.append(sd)
        assert digest(wm) == initial_digest
        frame_dir = OUT / "replay" / planner / shape / "frozen"
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(frame_dir / "trajectory.npz", visual=vis, proprio=prop, actions=acts,
                            states=st[::fs].astype(np.float32))
        cases.append({"planner": planner, "shape": shape, "sample_idx": SAMPLE, "method": "frozen",
                      "source": str(fz_dir.relative_to(REPO)),
                      "source_sha256": {"final_eval_results.npz": sha(fz_dir / "final_eval_results.npz"),
                                        "plan_targets.pkl": targets_sha},
                      "steps": list(range(1, n + 1)), "loss": train_l, "eval_mode_loss": eval_l,
                      "diagnostic_dropout_seeds": seeds, "success": success, "success_step": first_success,
                      "replay_success_flags": flags, "replay_state_dist": dists})
        print(json.dumps({"done": f"{planner}/{shape}", "steps": n, "success": success,
                          "loss_first": train_l[0], "loss_last": train_l[-1], **checks[-1]}), flush=True)

if ONLY:
    print("PILOT", json.dumps(checks), flush=True)
    sys.exit(0)
(OUT / "data").mkdir(parents=True, exist_ok=True)
(OUT / "data" / "frozen_prediction_losses.json").write_text(json.dumps({
    "method": "Frozen (official shape-shift checkpoint)",
    "definition": "Sliding-window one-step latent MSE (visual + proprio tokens, action dims excluded) on the most "
                  "recent five executed transitions, the criterion of the logged AdaJEPA/Ours adaptation loss.",
    "mode": "Predictor and encoder in training mode as in the logged adaptation loss; dropout draws use fixed "
            "diagnostic seeds (new draws, not recovered from any run). Buffers restored after every forward; "
            "eval_mode_loss is the deterministic eval-mode value.",
    "checkpoint": str(CKPT.relative_to(REPO)), "checkpoint_sha256": sha(CKPT / "checkpoints/model_latest.pth"),
    "model_state_sha256": initial_digest, "model_state_unchanged": digest(wm) == initial_digest,
    "script_sha256": sha(Path(__file__)), "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
    "verification": checks, "cases": cases,
}, indent=2))
print("COMPLETE", len(cases), "curves", sum(len(c["loss"]) for c in cases), "points", flush=True)
