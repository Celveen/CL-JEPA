#!/bin/bash
# DINO-WM's deformable planning protocol, to compare with its reported CD (rope 0.41, granular 0.26):
# goal_source=random_state (start from the validation set; goal = the start transformed by
# FlexEnvWrapper.sample_random_init_goal_states -- granular: scaled by 0.6-0.9 and shifted, rope: rotated and shifted),
# goal_H 5, CEM with conf/planner/cem.yaml (horizon 5, 300 samples, top 30, 30 iterations). eval_every=-1 skips
# simulator rollouts inside CEM (they only feed early stopping, and the env never reports success). 5 seeds x 10
# instances = 50. Protocols:
#   mpc       (default) DINO-WM's Table 8 numbers are MPC (its mpc_cem.yaml: execute 5 pushes, replan; open-loop
#             CEM is "NA" for rope / granular). Its max_iter is null and the env never succeeds, so the paper's
#             number of replans is unknown: here 5 replans (25 pushes), continuing from the current state
#             (resim_from_start=false); the CD after every replan is in the log.
#   openloop  plan once, execute the 5 pushes (the README's example command)
#   bash eval_outputs/granular_ref/eval.sh <case> <gpu list> [mpc|openloop]     (from the CL-JEPA-cudagraph worktree)
# cases: granular_ref        granular DINO-WM (train.sh) on granular
#        rope_on_rope        our rope DINO-WM on rope        (checks the pipeline against DINO-WM's 0.41)
#        rope_on_granular    our rope DINO-WM (theta0) on granular
#        ddpwm_on_rope       the DINO-WM rope checkpoint provided by DDP-WM on rope
# Output: eval_outputs/granular_ref/<protocol>/<case>/seed<k>/, logs eval_outputs/granular_ref/logs/<protocol>_<case>_seed<k>.log
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
OUT=$REPO/eval_outputs/granular_ref
CK=/data/zjc/adaJEPA/CL-JEPA/checkpoints
DATA=/data/zjc/adaJEPA/CL-JEPA/data
PY=/data/zjc/envs/cljepa/bin/python
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
mkdir -p "$OUT/logs"
case=$1
IFS=, read -r -a GPUS <<< "$2"
proto=${3:-mpc}
case $proto in
  mpc)      PROTO=(planner.max_iter=5 planner.n_taken_actions=5 +planner.resim_from_start=false) ;;
  openloop) PROTO=(planner.max_iter=1 planner.n_taken_actions=5) ;;
  *) echo "unknown protocol $proto"; exit 1 ;;
esac
case $case in
  granular_ref)     ARGS=(ckpt_base_path=$CK/granular_dino_h1_fs1_e100_seed0) ;;
  rope_on_rope)     ARGS=(ckpt_base_path=$CK/rope_dino_h1_fs1_e100_seed0) ;;
  ddpwm_on_rope)    ARGS=(ckpt_base_path=$CK/ddpwm_dinowm_rope) ;;
  rope_on_granular) ARGS=(ckpt_base_path=$CK/rope_dino_h1_fs1_e100_seed0 +eval_object_name=granular
                          '++env_kwargs_override.object_name=granular') ;;
  *) echo "unknown case $case"; exit 1 ;;
esac
i=0
for seed in 99 100 101 102 103; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  dir=$OUT/$proto/$case/seed$seed
  grep -qs "final_eval/" "$dir/logs.json" && { echo "skip $case seed$seed (done)"; continue; }
  mkdir -p "$dir"
  (
    echo "start $(date '+%F %T') gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name plan_rope_mpc_cem.yaml "${ARGS[@]}" model_epoch=latest \
      +eval_data_path=$DATA/deformable goal_source=random_state goal_H=5 n_evals=10 seed=$seed \
      "${PROTO[@]}" planner.sub_planner.horizon=5 planner.sub_planner.num_samples=300 \
      planner.sub_planner.topk=30 planner.sub_planner.opt_steps=30 planner.sub_planner.eval_every=-1 \
      decode_for_viz=false hydra.run.dir=$dir
    echo "EXIT=$? $(date '+%F %T')"
  ) > "$OUT/logs/${proto}_${case}_seed$seed.log" 2>&1 &
  echo "launched $case seed$seed on GPU $gpu (pid $!)"
  sleep 15
done
wait
echo "all done $(date '+%F %T')"
