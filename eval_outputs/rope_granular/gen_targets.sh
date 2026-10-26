#!/bin/bash
# Draw granular planning targets for the rope DINO-WM: a validation-set start state of DINO-WM's granular data and the
# goal the dataset's next 5 pushes produce from it, replayed in PyFleX (goal_source=dset, eval_object_name=granular,
# env object granular). Planning in these runs is minimal and discarded: every evaluation plans from the dumped
# files (goal_source=file), so actions keep the model's own (rope) normalization.
#   bash eval_outputs/rope_granular/gen_targets.sh <seed>...     (from the CL-JEPA-cudagraph worktree)
# Output: eval_outputs/rope_granular/targets/granular_seed<k>.pkl (10 instances each), runs in gen/seed<k>/.
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
OUT=$REPO/eval_outputs/rope_granular
DATA=/data/zjc/adaJEPA/CL-JEPA/data
PY=/data/zjc/envs/cljepa/bin/python
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
mkdir -p "$OUT/logs" "$OUT/targets"
GPUS=(1 2 3 4 5 6 7)
i=0
for seed in "$@"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  dir=$OUT/gen/seed$seed tgt=$OUT/targets/granular_seed$seed.pkl
  [ -s "$tgt" ] && { echo "skip seed$seed (targets exist)"; continue; }
  mkdir -p "$dir"
  (
    echo "start $(date '+%F %T') gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name cont_plan_cem_rope.yaml \
      ckpt_base_path=/data/zjc/adaJEPA/CL-JEPA/checkpoints/rope_dino_h1_fs1_e100_seed0 model_epoch=latest \
      +eval_data_path=$DATA/deformable +eval_object_name=granular '++env_kwargs_override.object_name=granular' \
      goal_source=dset n_evals=10 seed=$seed +wandb_logging=false planner.resim_from_start=false \
      planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual' \
      planner.max_iter=1 planner.sub_planner.opt_steps=1 planner.sub_planner.num_samples=30 planner.sub_planner.topk=5 \
      hydra.run.dir=$dir
    rc=$?
    [ $rc = 0 ] && [ -s "$dir/plan_targets.pkl" ] && cp -p "$dir/plan_targets.pkl" "$tgt" && chmod a-w "$tgt"
    echo "EXIT=$rc $(date '+%F %T')"
  ) > "$OUT/logs/gen_seed$seed.log" 2>&1 &
  echo "launched seed$seed on GPU $gpu (pid $!)"
  sleep 15
done
wait
echo "all done $(date '+%F %T')"
