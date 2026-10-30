#!/bin/bash
# Frozen theta0 (batched MPC, no adaptation) on every chunk of a stream: per-chunk difficulty
# reference, paired with the stream arms (same targets, same order, same seed as the chunk).
#   bash scripts/cont_frozen_chunks.sh S7b /abs/path/to/ckpt_dir 3 0
#   -> eval_outputs/cont_<stream>/frozen/<chunk>/
# Optional env: CONFIG (default cont_plan_gd_pushobj), N (episodes per chunk, default 20),
#   OUT (output root, default eval_outputs/cont_<stream>), CHUNKS (chunk names to run, in this order; default all),
#   e.g. round 5 of the CEM five-round stream first:
#   CONFIG=cont_plan_cem_pushobj N=50 OUT=eval_outputs/cont_S7t3_cem CHUNKS="chunk_28_I chunk_29_L" \
#     bash scripts/cont_frozen_chunks.sh S7t5 /abs/ckpt 3 7
set -u
STREAM=$1; CKPT=$2; EPOCH=${3:-3}; GPU=${4:-0}
CONFIG=${CONFIG:-cont_plan_gd_pushobj}; N=${N:-20}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${OUT:-eval_outputs/cont_$STREAM}
PY=${PYTHON:-python}
if [ -n "${CHUNKS:-}" ]; then FILES=$(for c in $CHUNKS; do echo data/cont_streams/$STREAM/$c.pkl; done)
else FILES=$(ls data/cont_streams/$STREAM/chunk_*.pkl); fi
for f in $FILES; do
  c=$(basename "$f" .pkl); out=$OUT/frozen/$c
  if grep -qs "final_eval/success_rate" "$out/logs.json"; then continue; fi
  seed=$($PY -c "import json,sys; b=[b for b in json.load(open('data/cont_streams/$STREAM/manifest.json'))['blocks'] if b['chunk']=='$c']; print(f\"seed={b[0]['seed']}\" if b and 'seed' in b[0] else '')")
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline OMP_NUM_THREADS=2 SDL_VIDEODRIVER=dummy \
  $PY plan.py --config-name "$CONFIG" \
    ckpt_base_path="$CKPT" model_epoch="$EPOCH" \
    eval_data_path="$ROOT/$f" n_evals=$N +wandb_logging=false hydra.run.dir="$out" $seed \
    planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual' \
    > "$out/console.log" 2>&1 \
    && echo "DONE $c $(grep -o '"final_eval/success_rate": [0-9.]*' "$out/logs.json")" || echo "FAIL $c (see $out/console.log)"
done
