#!/bin/bash
# Held-out test of the single-domain slow learner: does a model that predicts rope better also plan better?
# 50 new rope instances (5 shards x 10, seeds 104-108, not in the 50-episode stream), frozen batched MPC with
# continue-simulation (conf/cont_plan_cem_rope.yaml, resim_from_start=false), two arms per model:
#   theta0   the model as trained
#   learner  theta0's encoder + the rope domain's slow predictor after the 50-episode stream
#            (../<model>/cl/chunk_4/continual_state.pt, planning.continual.LearnerFrozenMPCPlanner)
# dino_ours/theta0 draws each shard (goal_source=dset: validation start state, goal = the dataset's next 5 pushes
# replayed in PyFleX) and dumps it; the file is copied to targets/seed<k>.pkl and every other job plans toward
# exactly those starts and goals (goal_source=file).
#   bash eval_outputs/rope_cl_single/heldout/run.sh          (from the CL-JEPA-cudagraph worktree)
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
OUT=$REPO/eval_outputs/rope_cl_single/heldout
CL=$REPO/eval_outputs/rope_cl_single
CKPTS=/data/zjc/adaJEPA/CL-JEPA/checkpoints
DATA=/data/zjc/adaJEPA/CL-JEPA/data
PY=/data/zjc/envs/cljepa/bin/python
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
mkdir -p "$OUT/logs" "$OUT/targets"

declare -A CKPT=([dino_ours]=rope_dino_h1_fs1_e100_seed0 [B_h3cos_e50]=rope_h3_fs1_cos_e100_seed0)
declare -A EPOCH=([dino_ours]=latest [B_h3cos_e50]=50)
SEEDS=(104 105 106 107 108)
GPUS=(1 2 3 4 5 6)  # GPU 0 renders for every PyFleX process; GPU 7 runs the learner-only chains
JOBS=()
for model in dino_ours B_h3cos_e50; do for arm in theta0 learner; do for seed in "${SEEDS[@]}"; do
  JOBS+=("$model $arm $seed")
done; done; done

i=0
for job in "${JOBS[@]}"; do
  read -r model arm seed <<< "$job"
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  dir=$OUT/$model/$arm/seed$seed log=$OUT/logs/${model}_${arm}_seed$seed.log tgt=$OUT/targets/seed$seed.pkl
  if grep -qs "final_eval/" "$dir/logs.json"; then echo "skip $model $arm seed$seed (done)"; continue; fi
  if [ "$model $arm" = "dino_ours theta0" ]; then goal=(goal_source=dset); else goal=(goal_source=file +goal_file_path=$tgt); fi
  if [ "$arm" = theta0 ]; then
    ARM=(planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual')
  else
    ARM=(planner._target_=planning.continual.LearnerFrozenMPCPlanner planner.continual.tag=rope
         planner.continual.state_in=$CL/$model/cl/chunk_4/continual_state.pt)
  fi
  mkdir -p "$dir"
  (
    if [ "${goal[0]}" = goal_source=file ]; then  # wait for the shard's goals from dino_ours/theta0
      src=$OUT/dino_ours/theta0/seed$seed/plan_targets.pkl
      until [ -s "$tgt" ]; do
        if grep -qs "Dumped plan targets" "$OUT/logs/dino_ours_theta0_seed$seed.log" && [ -s "$src" ]; then
          cp -p "$src" "$tgt" && chmod a-w "$tgt"
        else
          sleep 30
        fi
      done
    fi
    echo "start $(date '+%F %T') gpu=$gpu goals=${goal[*]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name cont_plan_cem_rope.yaml \
      ckpt_base_path=$CKPTS/${CKPT[$model]} model_epoch=${EPOCH[$model]} +eval_data_path=$DATA/deformable \
      "${goal[@]}" n_evals=10 seed=$seed +wandb_logging=false planner.resim_from_start=false \
      "${ARM[@]}" hydra.run.dir=$dir
    echo "EXIT=$? $(date '+%F %T')"
  ) > "$log" 2>&1 &
  echo "launched $model $arm seed$seed on GPU $gpu (pid $!)"
  sleep 15
done
wait
echo "all done $(date '+%F %T')"
