#!/bin/bash
# Reference model (not theta0): DINO-WM trained on granular only, with the recipe of our rope DINO-WM
# (rope_dino_h1_fs1_e100_seed0: frozen DINOv2 ViT-S/14 patch tokens, ViT predictor, history 1, frameskip 1, 100 epochs,
# batch 32 over 2 GPUs, bf16). Only the object changes. Gives the ceiling for granular planning and checks our
# evaluation against DINO-WM's reported granular CD (0.26) -- eval.sh.
#   bash eval_outputs/granular_ref/train.sh <gpu,gpu>      (from the CL-JEPA-cudagraph worktree)
# Output: /data/zjc/adaJEPA/CL-JEPA/checkpoints/granular_dino_h1_fs1_e100_seed0, log eval_outputs/granular_ref/train.log
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
CKPT=/data/zjc/adaJEPA/CL-JEPA/checkpoints/granular_dino_h1_fs1_e100_seed0
source /home/zjc/miniforge3/etc/profile.d/conda.sh && conda activate cljepa || exit 1
export DATASET_DIR=/data/zjc/adaJEPA/CL-JEPA/data TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1  # as the rope run
cd "$REPO"
echo "start $(date '+%F %T') gpus=$1"
CUDA_VISIBLE_DEVICES=$1 torchrun --standalone --nproc_per_node 2 train.py \
  env=deformable_env env.kwargs.object_name=granular env.dataset.object_name=granular encoder=dino \
  frameskip=1 num_hist=1 model.train_encoder=false training.straighten=False +training.eval_lpips=false \
  training.epochs=100 training.batch_size=32 training.encoder_lr=1e-6 training.save_every_x_epoch=10 \
  training.save_every_x_iterations=0 hydra.run.dir=$CKPT
echo "EXIT=$? $(date '+%F %T')"
