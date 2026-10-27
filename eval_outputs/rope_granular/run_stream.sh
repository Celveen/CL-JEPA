#!/bin/bash
# Multi-domain continual learning on a stream of rope and granular episodes; theta0 = our rope DINO-WM
# (rope_dino_h1_fs1_e100_seed0), so rope is its own domain and granular a new one.
# Stream: 6 chunks of 10 episodes, alternating rope / granular:
#   0 rope seed 99, 1 granular 301, 2 rope 100, 3 granular 302, 4 rope 101, 5 granular 303
# rope chunks are the files of ../rope_cl_single/stream (start state + the dataset's next 5 pushes), granular chunks
# targets/granular_seed<k>.pkl (gen_targets.sh); all plan with goal_source=file, conf/cont_plan_cem_rope.yaml's CEM,
# resim_from_start=false and the chunk's seed, so episodes pair across arms. Arms:
#   frozen  theta0, no adaptation (batched MPC; chunks independent)
#   free    label-free routing: cls6a_free (planning.continual_v7.CLSv7MPCPlanner) with familiar_thr recalibrated for
#           this model, 0.117 = 90th percentile of theta0's familiarity score on rope validation windows (familiarity.json)
#   free_abs  free plus the absolute write check (abs_ratio=20: a domain's module must predict the episode within
#           20 x its median 1-step error on the domain's episodes; offline on free's run, new rope episodes were
#           4-11 x the rope learner's member error, granular 39-58 x); an episode it turns away joins the closest
#           domain within 20 x (abs_join) before the pending pool. chunk 0 is shared with the first run of this arm
#           (no abs_join decision arises in it); that run's fragmented chunks 1-2 are in free_abs_v1/
#   oracle  planning.continual.CLSMPCPlanner write=oracle: one slow learner per true domain (tag rope / granular)
# free and oracle run their chunks as one chain through continual_state.pt.
#   bash eval_outputs/rope_granular/run_stream.sh <arm> <gpu list>     (from the CL-JEPA-cudagraph worktree)
# Output: eval_outputs/rope_granular/<arm>/chunk_<k>_<object>/, logs/<arm>_chunk_<k>_<object>.log. Finished chunks skip.
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
OUT=$REPO/eval_outputs/rope_granular
CKPT=/data/zjc/adaJEPA/CL-JEPA/checkpoints/rope_dino_h1_fs1_e100_seed0
DATA=/data/zjc/adaJEPA/CL-JEPA/data
PY=/data/zjc/envs/cljepa/bin/python
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
mkdir -p "$OUT/logs"

arm=$1
IFS=, read -r -a GPUS <<< "$2"
OBJ=(rope granular rope granular rope granular)
SEED=(99 301 100 302 101 303)
GOAL=($REPO/eval_outputs/rope_cl_single/stream/chunk_0.pkl $OUT/targets/granular_seed301.pkl
      $REPO/eval_outputs/rope_cl_single/stream/chunk_1.pkl $OUT/targets/granular_seed302.pkl
      $REPO/eval_outputs/rope_cl_single/stream/chunk_2.pkl $OUT/targets/granular_seed303.pkl)
case $arm in
  frozen) ARM=(planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual'); CHAIN=0 ;;
  free)   ARM=(planner._target_=planning.continual_v7.CLSv7MPCPlanner planner.continual.write=delta
               planner.continual.tau=0.10 planner.continual.mature_k=3 planner.continual.mature_ratio=0.9
               planner.continual.reject_ratio=1.2 planner.continual.familiar_thr=0.117
               planner.continual.pending_m=3 planner.continual.tau_new=0.05 planner.continual.fp_unit=true
               planner.continual.merge_eps=0.5 planner.continual.merge_k=2
               planner.continual.familiar_read=true planner.continual.join_ratio=0
               planner.continual.learner_win=false planner.continual.absorb_ratio=0.8
               planner.continual.absorb_every=25 planner.continual.absorb_frag=6
               planner.continual.absorb_pending=false); CHAIN=1 ;;
  free_abs) ARM=(planner._target_=planning.continual_v7.CLSv7MPCPlanner planner.continual.write=delta
               planner.continual.tau=0.10 planner.continual.mature_k=3 planner.continual.mature_ratio=0.9
               planner.continual.reject_ratio=1.2 planner.continual.familiar_thr=0.117
               planner.continual.pending_m=3 planner.continual.tau_new=0.05 planner.continual.fp_unit=true
               planner.continual.merge_eps=0.5 planner.continual.merge_k=2
               planner.continual.familiar_read=true planner.continual.join_ratio=0
               planner.continual.learner_win=false planner.continual.absorb_ratio=0.8
               planner.continual.absorb_every=25 planner.continual.absorb_frag=6
               planner.continual.absorb_pending=false +planner.continual.abs_ratio=20); CHAIN=1 ;;
  oracle) ARM=(planner._target_=planning.continual.CLSMPCPlanner planner.continual.write=oracle
               planner.continual.mature_k=3 planner.continual.mature_ratio=0.9); CHAIN=1 ;;
  *) echo "unknown arm $arm"; exit 1 ;;
esac

finished() { grep -qs "final_eval/" "$1/logs.json" && { [ $CHAIN = 0 ] || [ -s "$1/continual_state.pt" ]; }; }

run_chunk() {  # k gpu state_in
  local k=$1 gpu=$2 dir=$OUT/$arm/chunk_$1_${OBJ[$1]} env=() extra=()
  [ "${OBJ[$k]}" = granular ] && env=('++env_kwargs_override.object_name=granular')
  [ $CHAIN = 1 ] && extra=(planner.continual.state_in=$3)
  mkdir -p "$dir"
  echo "start $(date '+%F %T') gpu=$gpu object=${OBJ[$k]} goals=${GOAL[$k]}"
  CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name cont_plan_cem_rope.yaml \
    ckpt_base_path=$CKPT model_epoch=latest +eval_data_path=$DATA/deformable "${env[@]}" \
    goal_source=file +goal_file_path=${GOAL[$k]} n_evals=10 seed=${SEED[$k]} +wandb_logging=false \
    planner.resim_from_start=false planner.continual.tag=${OBJ[$k]} planner.continual.save_deltas=false \
    "${ARM[@]}" "${extra[@]}" hydra.run.dir=$dir
  echo "EXIT=$? $(date '+%F %T')"
}

if [ $CHAIN = 1 ]; then
  state=null
  for k in 0 1 2 3 4 5; do
    dir=$OUT/$arm/chunk_${k}_${OBJ[$k]}
    if ! finished "$dir"; then
      run_chunk $k ${GPUS[0]} $state > "$OUT/logs/${arm}_chunk_${k}_${OBJ[$k]}.log" 2>&1
      finished "$dir" || { echo "chunk $k failed, chain stops"; exit 1; }
    fi
    state=$dir/continual_state.pt
  done
else
  for k in 0 1 2 3 4 5; do
    dir=$OUT/$arm/chunk_${k}_${OBJ[$k]}
    finished "$dir" && { echo "skip chunk $k (done)"; continue; }
    run_chunk $k ${GPUS[$((k % ${#GPUS[@]}))]} null > "$OUT/logs/${arm}_chunk_${k}_${OBJ[$k]}.log" 2>&1 &
    sleep 20
  done
  wait
fi
echo "all done $(date '+%F %T')"
