#!/bin/bash
# Smoke test of the continual planner on rope (pipeline check only: few CEM steps, low learner thresholds).
# Chunk a (4 episodes) then chunk b (4 episodes) chained through continual_state.pt; single domain, oracle writes.
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
S=$REPO/eval_outputs/rope_cl_single/smoke
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=/data/zjc/adaJEPA/CL-JEPA/data TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
run() {  # chunk state_in
  CUDA_VISIBLE_DEVICES=6 /data/zjc/envs/cljepa/bin/python plan.py --config-name cont_plan_cem_rope.yaml \
    ckpt_base_path=/data/zjc/adaJEPA/CL-JEPA/checkpoints/rope_dino_h1_fs1_e100_seed0 model_epoch=latest \
    +eval_data_path=/data/zjc/adaJEPA/CL-JEPA/data/deformable goal_source=file +goal_file_path=$S/smoke_$1.pkl \
    n_evals=4 seed=99 +wandb_logging=false \
    planner._target_=planning.continual.CLSMPCPlanner planner.continual.write=oracle planner.continual.tag=rope \
    planner.continual.min_n=2 planner.continual.mature_k=1 planner.continual.state_in=$2 \
    planner.continual.save_deltas=false planner.continual.save_frames=false \
    planner.max_iter=2 planner.sub_planner.opt_steps=3 planner.sub_planner.num_samples=100 \
    hydra.run.dir=$S/run_$1
}
echo "start $(date '+%F %T')"
run a null > $S/a.log 2>&1; echo "a EXIT=$? $(date '+%F %T')"
run b $S/run_a/continual_state.pt > $S/b.log 2>&1; echo "b EXIT=$? $(date '+%F %T')"
