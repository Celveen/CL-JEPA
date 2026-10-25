#!/bin/bash
# Frozen, batched rope planning (conf/cont_plan_cem_rope.yaml, resim_from_start=false) of two arms per model:
#   theta0   the model as trained
#   learner  theta0's encoder + the rope domain's slow predictor at the end of the 50-episode stream
#            (<model>/cl/chunk_4/continual_state.pt, planning.continual.LearnerFrozenMPCPlanner)
# on 5 shards of 10 instances:
#   bash eval_outputs/rope_cl_single/frozen_eval.sh <out dir> new <5 seeds>   new instances: dino_ours/theta0 draws each
#        shard (goal_source=dset) and dumps it to <out dir>/targets/seed<k>.pkl; every other job plans toward it
#   bash eval_outputs/rope_cl_single/frozen_eval.sh <out dir> stream          the stream's own 50 instances
#        (stream/chunk_<k>.pkl, seeds 99-103): the theta0 jobs rerun ../<model>/frozen exactly, so the two runs
#        differ only by run-to-run noise
# Run from the CL-JEPA-cudagraph worktree. Output: <out dir>/<model>/<arm>/seed<k>/, logs in <out dir>/logs/.
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
CL=$REPO/eval_outputs/rope_cl_single
CKPTS=/data/zjc/adaJEPA/CL-JEPA/checkpoints
DATA=/data/zjc/adaJEPA/CL-JEPA/data
PY=/data/zjc/envs/cljepa/bin/python
OUT=$1 MODE=$2
shift 2
if [ "$MODE" = stream ]; then SEEDS=(99 100 101 102 103); else SEEDS=("$@"); fi
[ ${#SEEDS[@]} -eq 5 ] || { echo "need 5 seeds"; exit 1; }
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
mkdir -p "$OUT/logs" "$OUT/targets"
printf '{"mode": "%s", "seeds": [%s]}\n' "$MODE" "$(IFS=,; echo "${SEEDS[*]}")" > "$OUT/meta.json"

declare -A CKPT=([dino_ours]=rope_dino_h1_fs1_e100_seed0 [B_h3cos_e50]=rope_h3_fs1_cos_e100_seed0)
declare -A EPOCH=([dino_ours]=latest [B_h3cos_e50]=50)
GPUS=(1 2 3 4 5 6 7)  # GPU 0 renders for every PyFleX process
i=0
for model in dino_ours B_h3cos_e50; do for arm in theta0 learner; do for j in 0 1 2 3 4; do
  seed=${SEEDS[$j]}
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  dir=$OUT/$model/$arm/seed$seed log=$OUT/logs/${model}_${arm}_seed$seed.log tgt=$OUT/targets/seed$seed.pkl
  if grep -qs "final_eval/" "$dir/logs.json"; then echo "skip $model $arm seed$seed (done)"; continue; fi
  if [ "$MODE" = stream ]; then
    goal=(goal_source=file +goal_file_path=$CL/stream/chunk_$j.pkl) wait_for=""
  elif [ "$model $arm" = "dino_ours theta0" ]; then
    goal=(goal_source=dset) wait_for=""
  else
    goal=(goal_source=file +goal_file_path=$tgt) wait_for=$OUT/dino_ours/theta0/seed$seed/plan_targets.pkl
  fi
  if [ "$arm" = theta0 ]; then
    ARM=(planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual')
  else
    ARM=(planner._target_=planning.continual.LearnerFrozenMPCPlanner planner.continual.tag=rope
         planner.continual.state_in=$CL/$model/cl/chunk_4/continual_state.pt)
  fi
  mkdir -p "$dir"
  (
    if [ -n "$wait_for" ]; then  # wait for the shard's goals from dino_ours/theta0
      until [ -s "$tgt" ]; do
        if grep -qs "Dumped plan targets" "$OUT/logs/dino_ours_theta0_seed$seed.log" && [ -s "$wait_for" ]; then
          cp -p "$wait_for" "$tgt" && chmod a-w "$tgt"
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
done; done; done
wait
echo "all done $(date '+%F %T')"
