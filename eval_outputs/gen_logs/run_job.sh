#!/usr/bin/env bash
# One S7t3-family job for the generalization batch (v6a GD/CEM on new seed streams + AdaJEPA baselines).
# usage: run_job.sh <log name> <stream> <root> <gd|cem> <cont_run_stream.py args...>
# Output is teed to eval_outputs/gen_logs/<log name>.log, exit code appended as EXIT=<rc>.
name="$1"; stream="$2"; root="$3"; mode="$4"; shift 4
cd /data/zjc/adaJEPA/CL-JEPA-cudagraph || exit 1
source /home/zjc/miniforge3/etc/profile.d/conda.sh && conda activate cljepa || exit 1
if [ "$mode" = gd ]; then
    extra=(planner.sub_planner.cuda_graph=true)
else  # the AdaJEPA paper's CEM sub-planner (conf/adajepa_plan_cem_pushobj.yaml); everything else as the GD runs
    extra=(planner.sub_planner.target=planning.cem.CEMPlanner +planner.sub_planner.topk=30
           +planner.sub_planner.num_samples=200 planner.sub_planner.opt_steps=10)
fi
log="eval_outputs/gen_logs/${name}.log"
echo "start $(date '+%F %T') stream=$stream root=$root mode=$mode args: $*" >> "$log"
python scripts/cont_run_stream.py --stream "$stream" --root "$root" --no-base-probes "$@" \
    --extra planner.continual.save_deltas=false "${extra[@]}" 2>&1 | tee -a "$log"
echo "EXIT=${PIPESTATUS[0]} $(date '+%F %T')" >> "$log"
