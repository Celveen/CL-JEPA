# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

AdaJEPA: a latent world model that performs **test-time adaptation inside the MPC loop**. After each replan, the executed environment transition is used as a self-supervised signal to finetune the world model, then planning continues with the updated model. The repo is a fork/extension of [temporal-straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening) (itself built on DINO-WM), so most of the tree (`models/`, `datasets/`, `env/`, `train.py`) is inherited upstream code; the AdaJEPA contribution is concentrated in `planning/adajepa*.py`, `planning/image_corruption.py`, `datasets/diverse_maze_goals.py`, and the `conf/adajepa_plan_*.yaml` configs.

The primary workflow is **evaluation of released checkpoints**, not training. Training code is kept for completeness but world models are trained in the upstream repo.

## Setup

```bash
conda env create -f environment.yaml && conda activate ts
```

- `environment.yaml` is the full conda env (Python 3.9, `ts`); `requirements-pip.txt` is the pip half only, pinned to what the evaluation path actually imports.
- MuJoCo (`mujoco-py`, `d4rl`) is needed **only** for the maze envs. `env/__init__.py` guards that import, so PushT/PushObj work without it — a missing-mujoco warning at import is expected and harmless when running PushObj.
- Dataset configs under `conf/env/*.yaml` resolve `data_path` via `${oc.env:DATASET_DIR}`, so `DATASET_DIR` must be exported for any run that loads a dataset config. Runs with `goal_source=segments` do not touch it (see below).
- The DINOv2 encoder is fetched through `torch.hub.load("facebookresearch/dinov2", ...)`, so first use needs network or a warm torch hub cache.

## Commands

There is **no test suite, linter, or build step**. Verification means running a short planning job and reading the printed success rate.

```bash
# smoke test: 2 episodes, few opt steps, no wandb
python plan.py --config-name adajepa_plan_gd_pushobj.yaml \
    ckpt_base_path=$(pwd)/checkpoints/pushobj_shape_shift \
    eval_data_path=$(pwd)/data/pushobj_eval/val_I/plan_targets.pkl \
    n_evals=2 planner.sub_planner.opt_steps=5 planner.max_iter=2 +wandb_logging=false
```

Full evaluation commands for each distribution-shift condition are in `README.md`. Key invariants:

- `--config-name` selects planner + env: `adajepa_plan_{gd,cem}_{pushobj,maze,diversemaze}.yaml`.
- Frozen baseline: `planner._target_=planning.mpc.MPCPlanner '~planner.adapt'`. This also switches from per-sample sequential planning to one batched plan, so it is much faster than setting `adapt.lr=0`.
- `+wandb_logging=false` needs the `+` (the key is absent from the plan configs); `++env_kwargs_override.foo=bar` needs `++` (the key exists but is `null`).
- Training (rarely needed here): `python train.py env=pusht encoder=dino`, config root `conf/train.yaml`.

## Architecture

### Two-config layering (most important thing to know)

`plan.py` reads **two** configs and they own disjoint concerns:

1. The plan config (`conf/adajepa_plan_*.yaml`) — planner, adaptation hyperparameters, goal source, eval seeds, corruption.
2. `<model_path>/hydra.yaml`, the *training* config shipped inside the checkpoint directory — model architecture, `frameskip`, `num_hist`, `concat_dim`, env name/kwargs, dataset spec.

Model and env are rebuilt from (2); the plan config can only override a narrow set of fields (`env_kwargs_override`, `eval_data_path`, `eval_use_frame_files`). Changing how a model is constructed means editing the checkpoint's `hydra.yaml`, not the plan config.

`model_path` resolution has a sharp edge in `planning_main` ([plan.py:639](plan.py:639)): if `ckpt_base_path` starts with `/` it is used verbatim and `model_name` is ignored; otherwise the path is `<ckpt_base_path>/<model_name>/`. Released-checkpoint runs pass an absolute path and leave `model_name=null`.

Checkpoints pickle **whole `nn.Module` objects**, not state dicts ([train.py:227](train.py:227)). Loading therefore depends on class definitions being importable and structurally unchanged — `load_ckpt` instantiates a `DinoV2Encoder` before `torch.load` for exactly this reason ([plan.py:534](plan.py:534)). Renaming or moving a model class breaks existing checkpoints.

### Planner stack

```
BasePlanner (planning/base_planner.py)   freezes the WM, owns objective_fn + log dumping
  └─ MPCPlanner (planning/mpc.py)        closed loop: sub-plan → execute n_taken_actions → re-plan
       └─ AdaJEPAMPCPlanner (planning/adajepa_mpc.py)
            └─ AdaJEPATrainer (planning/adajepa.py)
  sub_planner: GDPlanner (planning/gd.py) | CEMPlanner (planning/cem.py)
```

The sub-planner optimizes an action sequence against the latent objective (`planning/objectives.py`) with the world model frozen; the MPC loop executes a prefix in the real env and replans from the observed state.

Adaptation hangs off a single hook: `MPCPlanner._post_env_feedback(taken_actions, e_obses)` is a no-op in the base class and is where `AdaJEPAMPCPlanner` buffers the executed segment and finetunes. **Any new test-time-adaptation variant should subclass and override that hook rather than modify the MPC loop.**

Two behaviors of `AdaJEPAMPCPlanner` follow from "each sample adapts independently":
- `plan()` loops over evaluation samples one at a time, calling `AdaJEPATrainer.reset()` before each. Per-sample action sequences have different lengths and are zero-padded to a common `T` before concatenation; `action_len` carries the true lengths.
- A single-sample view of the vectorized env is needed, provided by `_EnvWorkerProxy`, which mirrors the vector-env API with batch-1 arrays for both `SubprocVectorEnv` and `SerialVectorEnv`.

### AdaJEPATrainer

- **Snapshot/reset**: on construction it clones every tensor adaptation may touch (selected params *plus* train-mode buffers such as BN stats); `reset()` restores them. Adding a newly-adapted tensor requires it to be reachable from `_take_snapshot`, or adaptation will leak across samples.
- **Parameter selection** mirrors train-time freezing: predictor → last transformer layer + final norm; encoder → non-`base_model.` params only (the DINO backbone is never adapted), or the projector when one is configured.
- **Loss**: sliding-window 1-step latent MSE over `num_hist` frames, with the action dimensions excluded from the target (the slicing depends on `concat_dim`: extra token when `0`, trailing channels when `1`).
- `_prepare_segment` pads a dummy action onto the last frame so `encode()` sees `T+1` (frame, action) pairs; that dummy never enters the loss.
- Replay buffer modes are parsed from a string: `false` (keep all), `recent<N>`, `hard<N>` (highest prediction loss). `hard` segments are non-contiguous, so they are scored per-segment and never merged; `recent` segments are concatenated into one long sequence.

### Frameskip convention

Config values for `goal_H`, `n_taken_actions`, and sub-planner `horizon` are in **environment steps**; `PlanWorkspace.__init__` divides them by `frameskip` (from the checkpoint's train config) when instantiating the planner, and `action_dim` is `dset.action_dim * frameskip`. Code inside the planners is therefore in model steps. Mixing the two units is the usual source of off-by-frameskip bugs.

### Goal sources

`goal_source` decides how `obs_0`/`obs_g` are built and what data is required:

| value | targets from | needs |
|---|---|---|
| `segments` | pre-sampled `plan_targets.pkl` replayed in env | `eval_data_path` file; builds a stats-only stub dset, so no training data |
| `maze_dist` | BFS distance-controlled start/goal on held-out layouts (`datasets/diverse_maze_goals.py`) | `eval_data_path` dir with `val/`; builds one env per episode with its own `maze_spec` |
| `hard` / `random_state` | random start/goal states, min-separation constants at the top of `plan.py` | dataset (for env_info + normalization stats) |
| `dset` / `random_action` | replayed validation trajectory segments | dataset |
| `file` | a dumped `plan_targets.pkl` verbatim | `goal_file_path` |

Every run dumps its own `plan_targets.pkl`, which is what makes `segments`/`file` reproducible across conditions.

### Env layer

Envs are gym-registered in `env/__init__.py` (`pusht`, `pushobj`, `point_maze`, `point_maze_medium`, `diverse_maze`, `wall`, `deformable_env`). Planning does **not** use the gym step API; it uses a vectorized batch API — `prepare(seed, init_state)`, `rollout(seed, init_state, actions)`, `eval_state(goal_state, cur_state)`, `update_env(env_info)` — implemented by `SubprocVectorEnv` (`env/venv.py`) and `SerialVectorEnv`. `wall` and `deformable_env` are forced onto the serial implementation. Success is defined per-env inside `eval_state` (PushT: position < 20px and angle < π/9; maze: position < 0.5).

The four distribution shifts are each injected at a different layer:

| shift | mechanism |
|---|---|
| visual corruption | `planning/image_corruption.py` applied to observations in `PlanEvaluator`/`plan.py` (`ood_corruption`, `ood_level`) |
| color | `env_kwargs_override.{color,agent_color,goal_color}` → per-sample `env_info` → `update_env` |
| dynamics | `env_kwargs_override.{density_scale,damping_scale}` → `env/pointmaze/maze_model.py` rebuilds the MJCF with scaled density/damping |
| layout | `goal_source=maze_dist`, one held-out `maze_spec` per episode |

### Outputs

Hydra writes to `eval_outputs/<model>/…` (see `hydra.run.dir` in each plan config). A run produces `logs.json` (planner + final eval), `per_iter_logs.json` (per-sample per-MPC-iteration success and timings), `adajepa_losses.csv` (per adaptation step), `plan_targets.pkl`, and rollout `.mp4`/`.png` files. Visualization tags are red for AdaJEPA runs and blue for frozen, set in `PlanWorkspace` by planner type. `decode_for_viz=true` requires a decoder in the checkpoint; checkpoints trained with `model.train_decoder=false` load with `decoder=None` and need `decode_for_viz=false`.

## Reproduction notes (verified on this machine)

The conda env `ts` is **Python 3.9.19**, and that matters:

- `load_ckpt` unconditionally instantiates `DinoV2Encoder('dinov2_vits14', ...)` to warm the torch.hub cache, *regardless of which encoder the checkpoint actually uses*. `torch.hub.load` pulls dinov2's **`main` branch**, which now uses PEP 604 (`float | None`) annotations evaluated at class-definition time — that is a hard `TypeError` on Python 3.9. The warm-up is wrapped in try/except so checkpoints with non-DINO encoders still load; DINO-based checkpoints will need a pinned hub revision or Python >= 3.10.
- `pushobj_shape_shift` uses `models.encoder.resnet.SmallResNetGeM`, not DINO. Since it has no `base_model` attribute, `AdaJEPATrainer` takes the `encoder_last_layer_only` branch and adapts the last submodule (`projection`, 6 tensors) plus the predictor's last transformer layer (13 tensors).
- Released checkpoints unzip to `<name>/hydra.yaml` + `<name>/checkpoints/model_latest.pth`, which is exactly the layout `planning_main` expects. Pass the **absolute** path as `ckpt_base_path` and leave `model_name=null`.

Running two planner variants against one checkpoint: set `hydra.run.dir` explicitly per run. The default `hydra.run.dir` template interpolates `${model_name}` (null in released-checkpoint runs) and `${planner.adapt.lr}` / `${planner.adapt.steps}` (deleted by the frozen baseline's `~planner.adapt`), so an explicit dir both keeps results separate and sidesteps those interpolations:

```bash
# AdaJEPA
CUDA_VISIBLE_DEVICES=4 python plan.py --config-name adajepa_plan_gd_pushobj.yaml \
    ckpt_base_path=$(pwd)/checkpoints/pushobj_shape_shift \
    eval_data_path=$(pwd)/data/pushobj_eval/val_I/plan_targets.pkl \
    n_evals=20 seed=100 hydra.run.dir=eval_outputs/verify_val_I/adajepa +wandb_logging=false

# frozen baseline (batched, ~20x faster)
CUDA_VISIBLE_DEVICES=5 python plan.py --config-name adajepa_plan_gd_pushobj.yaml \
    ckpt_base_path=$(pwd)/checkpoints/pushobj_shape_shift \
    eval_data_path=$(pwd)/data/pushobj_eval/val_I/plan_targets.pkl \
    n_evals=20 seed=100 planner._target_=planning.mpc.MPCPlanner '~planner.adapt' \
    hydra.run.dir=eval_outputs/verify_val_I/frozen +wandb_logging=false
```

`seed` fixes the per-episode eval seeds (`seed*n+1`), so both runs evaluate the same episodes and are directly comparable. Cost asymmetry is large and expected: frozen plans all `n_evals` in one batch (~90s for 20 samples), AdaJEPA plans them one at a time (tens of minutes).

Checkpoints and eval data are not in the repo; they come from the Google Drive folder linked in `README.md` as per-condition zips (`checkpoints/pushobj_shape_shift.zip`, `data/pushobj_eval.zip`, …), so a single condition can be fetched without the whole release.

## Conventions

- `env/wall/data/` is a **source package** (`WallDatasetConfig`, layouts), not a dataset directory — `.gitignore` has explicit negations for it. Don't "clean up" that exception.
- `custom_resolvers.py` registers OmegaConf resolvers (`replace_slash`, `replace_substring`) used inside `hydra.run.dir` interpolations; it must be imported for its side effect (`train.py` does so with a `noqa`).
- Slurm/submitit launcher blocks are present in every config and are inert for local runs (`python plan.py …` runs in-process); `launch_plan_jobs` in `plan.py` is the submitit path.
