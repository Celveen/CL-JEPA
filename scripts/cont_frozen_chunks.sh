#!/bin/bash
# Frozen theta0 (batched MPC, no adaptation) on every chunk of a stream: per-chunk difficulty
# reference, paired with the stream arms (same targets, same order).
#   bash scripts/cont_frozen_chunks.sh S7b /abs/path/to/ckpt_dir 3 0
#   -> eval_outputs/cont_<stream>/frozen/<chunk>/
set -u
STREAM=$1; CKPT=$2; EPOCH=${3:-3}; GPU=${4:-0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PYTHON:-python}
for f in data/cont_streams/$STREAM/chunk_*.pkl; do
  c=$(basename "$f" .pkl); out=eval_outputs/cont_$STREAM/frozen/$c
  if grep -qs "final_eval/success_rate" "$out/logs.json"; then continue; fi
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline OMP_NUM_THREADS=2 SDL_VIDEODRIVER=dummy \
  $PY plan.py --config-name cont_plan_gd_pushobj \
    ckpt_base_path="$CKPT" model_epoch="$EPOCH" \
    eval_data_path="$ROOT/$f" n_evals=20 +wandb_logging=false hydra.run.dir="$out" \
    planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual' \
    > "$out/console.log" 2>&1 \
    && echo "DONE $c $(grep -o '"final_eval/success_rate": [0-9.]*' "$out/logs.json")" || echo "FAIL $c (see $out/console.log)"
done
