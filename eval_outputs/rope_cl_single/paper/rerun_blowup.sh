#!/bin/bash
# DINO-WM (frozen theta0) on the two granular episodes whose batched run blew up FleX: chunk 1, samples 0 and 9
# (episodes 10 and 19; final CD ~1850 twice, identical on a rerun of the whole chunk -- frozen_blowup/). In the batched
# run sample 9 blew up in MPC iteration 1 and sample 0, the next rollout on the same simulator, in iteration 2: the
# FleX env carries state from one call to the next. Here each episode is planned on its own, in a fresh process,
# with the same model, planner config (run.sh's frozen arm) and chunk seed; the env seed (seed * i + 1) only drives
# random action sampling, never a rollout of given actions, so it does not matter that the lone sample gets i = 0.
#   bash eval_outputs/rope_cl_single/paper/rerun_blowup.sh <gpu for episode 10> <gpu for episode 19>
# Output: granular_ref/frozen_solo/ep<e>/ (final_eval_results.npz, logs.json), logs/granular_ref_frozen_solo_ep<e>.log
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
OUT=$REPO/eval_outputs/rope_cl_single
SOLO=$OUT/granular_ref/frozen_solo
PY=/data/zjc/envs/cljepa/bin/python
DATA=/data/zjc/adaJEPA/CL-JEPA/data
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
mkdir -p "$SOLO/targets"
GOALS=$OUT/../rope_granular/targets/granular_seed302.pkl  # chunk 1 of the granular stream

"$PY" -I - "$GOALS" "$SOLO/targets" <<'EOF'
import pickle, sys
src, out = sys.argv[1], sys.argv[2]
d = pickle.load(open(src, "rb"))
for i in (0, 9):
    one = {k: ({kk: vv[i : i + 1] for kk, vv in v.items()} if isinstance(v, dict) else v[i : i + 1])
           if k != "goal_H" else v for k, v in d.items()}
    pickle.dump(one, open(f"{out}/ep{10 + i}.pkl", "wb"))
EOF

run() {  # episode gpu
  local e=$1 dir=$SOLO/ep$1
  grep -qs "final_eval/" "$dir/logs.json" && { echo "skip ep$e (done)"; return; }
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$2 $PY plan.py --config-name cont_plan_cem_rope.yaml \
    ckpt_base_path=/data/zjc/adaJEPA/CL-JEPA/checkpoints/granular_dino_h1_fs1_e100_seed0 model_epoch=latest \
    +eval_data_path=$DATA/deformable goal_source=file +goal_file_path=$SOLO/targets/ep$e.pkl n_evals=1 seed=302 \
    +wandb_logging=false '++env_kwargs_override.object_name=granular' planner.resim_from_start=false \
    planner.continual.tag=granular planner.continual.save_deltas=false \
    planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual' \
    hydra.run.dir=$dir > "$OUT/logs/granular_ref_frozen_solo_ep$e.log" 2>&1
  echo "ep$e EXIT=$? $(date '+%F %T')"
}
run 10 "$1" &
run 19 "$2" &
wait
