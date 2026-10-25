#!/bin/bash
# Rope, dataset-goal protocol (conf/plan_rope_dset.yaml, unchanged), 50 instances = 5 shards x 10 (seeds 99-103).
# dino_ours draws each shard's start states and replays the dataset's next 5 pushes in PyFleX to make the goals
# (goal_source=dset, as in the earlier runs) and dumps them; those files are copied to targets/seed<k>.pkl and ddpwm
# plans toward exactly the same goals (goal_source=file). Replaying the goals separately per model, as the earlier
# 10-instance runs did, gave goals that differ between models on some instances (FleX is not deterministic: on two
# of the 50 instances the two replays ended 0.52 and 1.89 CD apart), which would break the pairing.
#   bash eval_outputs/rope_dset_n50/run.sh [model ...]    (from the CL-JEPA-cudagraph worktree; default: all models)
# Per job: eval_outputs/rope_dset_n50/<model>/seed<k>/, log eval_outputs/rope_dset_n50/logs/<model>_seed<k>.log
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
OUT=$REPO/eval_outputs/rope_dset_n50
CKPTS=/data/zjc/adaJEPA/CL-JEPA/checkpoints
DATA=/data/zjc/adaJEPA/CL-JEPA/data
PY=/data/zjc/envs/cljepa/bin/python
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
mkdir -p "$OUT/logs" "$OUT/targets"

declare -A CKPT=([dino_ours]=rope_dino_h1_fs1_e100_seed0 [ddpwm]=ddpwm_dinowm_rope
                 [A_h1_e100]=rope_h1_fs1_e100_seed0 [B_h3cos_e50]=rope_h3_fs1_cos_e100_seed0)
# B at epoch 50 as in the earlier runs: its validation open-loop rollout error is near its lowest there (0.34) and
# climbs afterwards (0.63 at epoch 60, 0.89 at 100); A's stays low, so A uses its last epoch.
declare -A EPOCH=([dino_ours]=latest [ddpwm]=latest [A_h1_e100]=latest [B_h3cos_e50]=50)
# GPU 0 renders for every PyFleX process (default EGL device), so it gets no planning job.
JOBS=("dino_ours 99 1" "dino_ours 100 2" "dino_ours 101 3" "dino_ours 102 4" "dino_ours 103 5"
      "ddpwm 99 6" "ddpwm 100 7" "ddpwm 101 1" "ddpwm 102 2" "ddpwm 103 3"
      "A_h1_e100 99 1" "A_h1_e100 100 2" "A_h1_e100 101 3" "A_h1_e100 102 4" "A_h1_e100 103 5"
      "B_h3cos_e50 99 1" "B_h3cos_e50 100 2" "B_h3cos_e50 101 3" "B_h3cos_e50 102 4" "B_h3cos_e50 103 5")
WANT=" ${*:-dino_ours ddpwm A_h1_e100 B_h3cos_e50} "

for job in "${JOBS[@]}"; do
  read -r model seed gpu <<< "$job"
  [[ $WANT == *" $model "* ]] || continue
  dir=$OUT/$model/seed$seed log=$OUT/logs/${model}_seed$seed.log tgt=$OUT/targets/seed$seed.pkl
  if grep -qs "final_eval/" "$dir/logs.json"; then echo "skip $model seed$seed (done)"; continue; fi
  if [ "$model" = dino_ours ]; then
    goal=(goal_source=dset)
  else
    goal=(goal_source=file +goal_file_path=$tgt)
  fi
  mkdir -p "$dir"
  (
    if [ "$model" != dino_ours ]; then  # wait for the shard's goals from dino_ours
      src=$OUT/dino_ours/seed$seed/plan_targets.pkl
      until [ -s "$tgt" ]; do
        if grep -qs "Dumped plan targets" "$OUT/logs/dino_ours_seed$seed.log" && [ -s "$src" ]; then
          cp -p "$src" "$tgt" && chmod a-w "$tgt"
        else
          sleep 30
        fi
      done
    fi
    echo "start $(date '+%F %T') gpu=$gpu goals=${goal[*]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name plan_rope_dset.yaml \
      ckpt_base_path=$CKPTS/${CKPT[$model]} model_epoch=${EPOCH[$model]} +eval_data_path=$DATA/deformable \
      seed=$seed n_evals=10 "${goal[@]}" hydra.run.dir=$dir
    echo "EXIT=$? $(date '+%F %T')"
  ) > "$log" 2>&1 &
  echo "launched $model seed$seed on GPU $gpu (pid $!)"
  sleep 20  # stagger model loading and PyFleX scene init
done
