#!/bin/bash
# Granular on its own: our rope DINO-WM (theta0, frozen, batched MPC) planning on 50 granular instances
# (targets/granular_seed301-305.pkl), conf/cont_plan_cem_rope.yaml's CEM, resim_from_start=false, env object granular.
#   bash eval_outputs/rope_granular/granular_single.sh <gpu list>      (from the CL-JEPA-cudagraph worktree)
# Output: eval_outputs/rope_granular/granular_single/seed<k>/, logs/granular_single_seed<k>.log
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
OUT=$REPO/eval_outputs/rope_granular
DATA=/data/zjc/adaJEPA/CL-JEPA/data
PY=/data/zjc/envs/cljepa/bin/python
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
IFS=, read -r -a GPUS <<< "$1"
i=0
for seed in 301 302 303 304 305; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  dir=$OUT/granular_single/seed$seed
  grep -qs "final_eval/" "$dir/logs.json" && { echo "skip seed$seed (done)"; continue; }
  mkdir -p "$dir"
  (
    echo "start $(date '+%F %T') gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name cont_plan_cem_rope.yaml \
      ckpt_base_path=/data/zjc/adaJEPA/CL-JEPA/checkpoints/rope_dino_h1_fs1_e100_seed0 model_epoch=latest \
      +eval_data_path=$DATA/deformable '++env_kwargs_override.object_name=granular' \
      goal_source=file +goal_file_path=$OUT/targets/granular_seed$seed.pkl n_evals=10 seed=$seed +wandb_logging=false \
      planner.resim_from_start=false \
      planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual' \
      hydra.run.dir=$dir
    echo "EXIT=$? $(date '+%F %T')"
  ) > "$OUT/logs/granular_single_seed$seed.log" 2>&1 &
  echo "launched seed$seed on GPU $gpu (pid $!)"
  sleep 15
done
wait
echo "all done $(date '+%F %T')"
