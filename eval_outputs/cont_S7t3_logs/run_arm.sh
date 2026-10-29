#!/usr/bin/env bash
# Run one S7t3 job; output is teed to eval_outputs/cont_S7t3_logs/<name>.log, exit code appended as EXIT=<rc>.
# usage: run_arm.sh <log name> <cont_run_stream.py args...>
name="$1"; shift
cd /data/zjc/adaJEPA/CL-JEPA || exit 1
source /home/zjc/miniforge3/etc/profile.d/conda.sh && conda activate cljepa || exit 1
log="eval_outputs/cont_S7t3_logs/${name}.log"
echo "start $(date '+%F %T') args: $*" >> "$log"
python scripts/cont_run_stream.py --stream S7t3 --no-base-probes "$@" \
    --extra planner.continual.save_deltas=false 2>&1 | tee -a "$log"
echo "EXIT=${PIPESTATUS[0]} $(date '+%F %T')" >> "$log"
