#!/bin/bash
# H7t3 held-out evaluation (7 shapes x 50 episodes x test seeds 2000/2100/2200) of a round-5 continual state under
# CEM, loaded read-only -- the same settings as eval_outputs/cmp_cem_v6a_s{100,400,700} (gen_logs/cmp_cem_v6a_*.log).
#   bash scripts/run_h7t3_cem.sh <continual_state.pt> <output root> <gpu list>
# Output: <output root>/cls6a_free/chunk_<k>_<shape>/ (21 chunks); log eval_outputs/gen_logs/<basename of root>.log
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
cd "$REPO"
STATE=$(readlink -f "$1") ROOT=$2 GPUS=$3
LOG=eval_outputs/gen_logs/$(basename "$ROOT").log
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline
{
  echo "start $(date '+%F %T') stream=H7t3 root=$ROOT mode=cem state=$STATE gpus=$GPUS"
  /data/zjc/envs/cljepa/bin/python scripts/cont_run_stream.py --stream H7t3 --root "$ROOT" --arms cls6a_free \
    --independent --state-in "$STATE" --no-base-probes --chain-gpus "$GPUS" \
    --extra planner.continual.save_deltas=false planner.sub_planner.target=planning.cem.CEMPlanner \
    +planner.sub_planner.topk=30 +planner.sub_planner.num_samples=200 planner.sub_planner.opt_steps=10
  echo "EXIT=$? $(date '+%F %T')"
} >> "$LOG" 2>&1
