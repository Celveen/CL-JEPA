#!/bin/bash
# Train the cos1e-3 ablation of the T-16k model, then evaluate it exactly like the cos1e-2 model.
set -u
cd /home/appuser/workspace/project/multimodal-test/adajepa
export PATH=/home/appuser/anaconda3/envs/ts/bin:$PATH
R=reports/pushobj_T16k_bias
echo "TRAINING START $(date)"
CUDA_VISIBLE_DEVICES=4,5,6,7 WANDB_MODE=offline OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 SDL_VIDEODRIVER=dummy \
  accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 --main_process_port 29519 \
  train.py --config-name train_pushobj_T16k_cos1e3 > $R/train_cos1e3_console.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f checkpoints/pushobj_T16k_bias_cos1e3_e3_seed0/checkpoints/model_3.pth ]; then
  echo "TRAINING FAILED rc=$rc $(date)"; exit 1
fi
echo "TRAINING DONE $(date)"
python scripts/run_eval_jobs.py --jobs $R/eval_cos1e3_seed100_it30_jobs.tsv --gpus 4,5,6,7,4,5,6,7 > $R/eval_cos1e3_seed100_it30.log 2>&1
echo "EVAL exit=$? $(date)"; tail -1 $R/eval_cos1e3_seed100_it30.log
