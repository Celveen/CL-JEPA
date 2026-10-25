#!/bin/bash
# Single-domain continual learning on rope: does planning get better along a stream of rope episodes?
# Stream = the 50 shared rope instances of eval_outputs/rope_dset_n50 (5 chunks x 10, files copied to stream/),
# in a fixed order. Arms (all with planner.resim_from_start=false, conf/cont_plan_cem_rope.yaml's CEM, the chunk's
# seed, so episodes pair across arms one to one):
#   frozen   theta0, no adaptation (batched MPC)                                     chunks independent
#   adajepa  AdaJEPA within each episode, reset at its end (ContinualMPCPlanner w=0)  chunks independent
#   cl       CLSMPCPlanner, single domain (write=oracle, tag rope): a slow copy of the predictor trained on the
#            replay of every earlier episode, read once validated (mature_k 3, ratio 0.9), AdaJEPA on top;
#            chunks chained through continual_state.pt
#   cl_noada the same slow learner without AdaJEPA (planner.adapt.steps=0): the learner's own contribution
#   bash eval_outputs/rope_cl_single/run.sh <model> <arm> <gpu list>     (from the CL-JEPA-cudagraph worktree)
# Output: eval_outputs/rope_cl_single/<model>/<arm>/chunk_<k>/, logs in logs/. Finished chunks are skipped.
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
OUT=$REPO/eval_outputs/rope_cl_single
CKPTS=/data/zjc/adaJEPA/CL-JEPA/checkpoints
DATA=/data/zjc/adaJEPA/CL-JEPA/data
PY=/data/zjc/envs/cljepa/bin/python
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
mkdir -p "$OUT/logs"

model=$1
arm=$2
IFS=, read -r -a GPUS <<< "$3"
declare -A CKPT=([dino_ours]=rope_dino_h1_fs1_e100_seed0 [B_h3cos_e50]=rope_h3_fs1_cos_e100_seed0)
declare -A EPOCH=([dino_ours]=latest [B_h3cos_e50]=50)
SEEDS=(99 100 101 102 103)
case $arm in
  frozen)  ARM=(planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual') ;;
  adajepa) ARM=(planner.continual.w=0.0 planner.continual.state_in=null) ;;
  cl)      ARM=(planner._target_=planning.continual.CLSMPCPlanner planner.continual.write=oracle
                planner.continual.mature_k=3 planner.continual.mature_ratio=0.9) ;;
  cl_noada) ARM=(planner._target_=planning.continual.CLSMPCPlanner planner.continual.write=oracle
                 planner.continual.mature_k=3 planner.continual.mature_ratio=0.9 planner.adapt.steps=0) ;;
  *) echo "unknown arm $arm"; exit 1 ;;
esac
CHAIN=0
case $arm in cl|cl_noada) CHAIN=1 ;; esac  # the slow learner carries over: chunks run in order

finished() { grep -qs "final_eval/" "$1/logs.json" && { [ $CHAIN = 0 ] || [ -s "$1/continual_state.pt" ]; }; }

run_chunk() {  # k gpu state_in
  local k=$1 gpu=$2 dir=$OUT/$model/$arm/chunk_$1 extra=()
  [ $CHAIN = 1 ] && extra=(planner.continual.state_in=$3)
  mkdir -p "$dir"
  echo "start $(date '+%F %T') gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name cont_plan_cem_rope.yaml \
    ckpt_base_path=$CKPTS/${CKPT[$model]} model_epoch=${EPOCH[$model]} +eval_data_path=$DATA/deformable \
    goal_source=file +goal_file_path=$OUT/stream/chunk_$k.pkl n_evals=10 seed=${SEEDS[$k]} +wandb_logging=false \
    planner.resim_from_start=false planner.continual.tag=rope planner.continual.save_deltas=false \
    "${ARM[@]}" "${extra[@]}" hydra.run.dir=$dir
  echo "EXIT=$? $(date '+%F %T')"
}

if [ $CHAIN = 1 ]; then  # one chain: chunk k starts from chunk k-1's state
  state=null
  for k in 0 1 2 3 4; do
    dir=$OUT/$model/$arm/chunk_$k
    if ! finished "$dir"; then
      run_chunk $k ${GPUS[0]} $state > "$OUT/logs/${model}_${arm}_chunk_$k.log" 2>&1
      finished "$dir" || { echo "chunk $k failed, chain stops"; exit 1; }
    fi
    state=$dir/continual_state.pt
  done
else  # chunks are independent: run them in parallel, cycling through the GPUs
  for k in 0 1 2 3 4; do
    dir=$OUT/$model/$arm/chunk_$k
    finished "$dir" && { echo "skip chunk $k (done)"; continue; }
    run_chunk $k ${GPUS[$((k % ${#GPUS[@]}))]} null > "$OUT/logs/${model}_${arm}_chunk_$k.log" 2>&1 &
    sleep 20
  done
  wait
fi
echo "all done $(date '+%F %T')"
