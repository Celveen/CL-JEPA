# 在 RTX 5090 上运行 CL-JEPA

`environment.yaml` 锁的是 torch 2.3.0 + CUDA 12.1，这个组合没有 RTX 5090（Blackwell，sm_120）的内核，不能直接用。本文件说明适配 5090 的环境 `cljepa`，以及权重和数据应该放在哪里。实验本身的用法仍然看 [README_CL-JEPA.md](README_CL-JEPA.md)。

## 环境 `cljepa`

和 `environment.yaml`（环境 `ts`）相比，改动如下：

| 项 | `ts` | `cljepa` | 原因 |
|---|---|---|---|
| Python | 3.9.19 | 3.10 | `load_ckpt` 会先用 torch.hub 拉取 dinov2@main，这份代码在 3.9 上会报 `TypeError`；3.10 上可以正常加载，DINO 系的 checkpoint 也能用 |
| torch / torchvision | 2.3.0 / 0.18.0（cu121） | 2.7.1 / 0.22.1（cu128） | 至少要 CUDA 12.8 才支持 sm_120 |
| `torch.load` | 默认 `weights_only=False` | 环境变量 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | torch ≥ 2.6 默认改成 `weights_only=True`，而 checkpoint 存的是整个 `nn.Module`（`plan.py:573` 加载时没有传 `weights_only`）。这个变量让没传参数的 `torch.load` 恢复旧行为，不需要改代码 |
| setuptools | 70.0.0 | 70.0.0（必须固定） | wandb 0.13.1 要 import `pkg_resources`，新版 setuptools 已经删掉了它 |

其他包和 `requirements-pip.txt` 版本一致（numpy 1.26.4、hydra 1.2.0、gym 0.23.1、pymunk 6.8.0……），完整清单见 [requirements-5090.lock.txt](requirements-5090.lock.txt)。没有装 MuJoCo / d4rl，因为只有 maze 环境用得到。

本机的环境装在 `/data/zjc/envs/cljepa`（`/home` 所在的盘空间不够）。`~/.condarc` 的 `envs_dirs` 里已经加了 `/data/zjc/envs`，所以按名字 `conda activate cljepa` 就能激活。

重建：

```bash
mamba create -y -p /data/zjc/envs/cljepa -c conda-forge python=3.10 pip numpy=1.26.4 pillow=10.3.0 imageio=2.34.1
conda env config vars set -p /data/zjc/envs/cljepa TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
conda config --append envs_dirs /data/zjc/envs   # 只需要做一次，之后可以按名字激活
conda activate cljepa
mkdir -p /data/zjc/tmp && TMPDIR=/data/zjc/tmp pip install --no-cache-dir -r requirements-5090.lock.txt   # 临时文件和缓存都不占 /home
```

使用时注意：
- **先 `conda activate cljepa` 再运行。** 环境变量只在激活后生效；`cont_run_stream.py` 启动子进程时会继承它。如果直接用 `/data/zjc/envs/cljepa/bin/python` 运行，需要自己 `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，否则加载 checkpoint 会报 `UnpicklingError`。
- 第一次运行会通过 torch.hub 下载 dinov2（代码 + `dinov2_vits14_pretrain.pth`），需要联网，本机这一步花了约 4.5 分钟。缓存放在 `~/.cache/torch/hub/`，之后每次加载模型约 2 秒。
- 导入时出现的 mujoco 缺失、gym 停止维护、`pkg_resources` deprecation 等警告都可以忽略。
- `pip check` 会提示 `decord 0.6.0 is not supported on this platform`：它的 wheel 标签是 cp36，但实际能正常导入（原来的 3.9 环境也是这样），只有 maze 和 deformable 的数据集用到它。

## 权重放哪里

```
checkpoints/pushobj_bias_16k_e3_seed0/
├── hydra.yaml                 # 训练配置：模型结构、frameskip、num_hist、env
└── checkpoints/model_3.pth    # 整个 nn.Module 的 pickle（encoder / predictor / proprio_encoder / action_encoder / decoder）
```

- 这是 `scripts/cont_run_stream.py` 默认的 `--ckpt`（配合 `--epoch 3`）。放在别的位置时，用 `--ckpt /绝对路径 --epoch 3` 指定。
- 这份权重是 CL-JEPA 作者自己训练的：配置是 `conf/train_pushobj_bias16k.yaml`，训练形状 T/L/Z/+，epoch 3。**它不在 AdaJEPA 公开的 Google Drive 里，需要向作者要。**
- `familiar_thr = 0.0146` 和 `data/seen_probe/pushobj_bias_16k_e3_seed0.pt` 都是针对这份权重标定的。如果换成 AdaJEPA 公开的 `pushobj_shape_shift`（文件名是 `model_latest.pth`，所以用 `--epoch latest`），两者都要用训练数据的验证集重新标定（见 README_CL-JEPA.md 中的「权重」一节）。

## 数据放哪里

| 路径 | 仓库里是否有 | 什么时候需要 | 来源 |
|---|---|---|---|
| `data/cont_streams/<流>/chunk_*.pkl` + `manifest.json` | ✅ | 跑任意一条流（S2、S7、S7b、S7b3、S7t3、EVAL50……）。用的是 `goal_source=segments`，不读训练数据，也不需要 `DATASET_DIR` | 仓库自带 |
| `data/seen_probe/pushobj_bias_16k_e3_seed0.pt` | ✅ | v6（`cls6_free`） | 仓库自带 |
| `data/pushobj_eval/val_<形状>/plan_targets.pkl` | ❌ | 冻结探针（`cont_run_stream.py` 不加 `--no-base-probes`，或用 `--ws` 跑 w > 0 的组）；建新流（`cont_build_stream.py`、`cont_build_seed_stream.py`）；exp0 / escape 相关脚本 | AdaJEPA Google Drive 中的 `data/pushobj_eval.zip`，解压到 `data/pushobj_eval/` |
| `data/pushobj_bias_16k/{train,val}/` | ❌ | 重新标定 `familiar_thr`（`cont_calibrate_familiarity.py --val-data`）、重新训练见过形状分类器（`cont_train_seen_probe.py`）、训练世界模型 | 用 `scripts/gen_pushobj_dataset.py` 从 DINO-WM 的 `pusht_noise` 生成，具体参数向作者确认 |

只复现 README_CL-JEPA.md 中的 S7t3 实验（AdaJEPA 基线 + v5 + v6，命令里都带 `--no-base-probes`）时，**只需要准备权重**，表中前两项仓库已经自带。

结果写到 `eval_outputs/cont_<流>/<组>/chunk_xx_<形状>/`（已被 gitignore）。

## 本机验证（2026-10-19）

本机配置：8 × RTX 5090、驱动 595.91（CUDA 13.2）、Ubuntu 24.04。下面的测试用的是一份**随机权重**的 checkpoint，结构和 bias16k 相同，只用来检查流水线能不能跑通，成功率没有参考意义：

- GPU 上的 conv / matmul / bf16 autocast / SDPA，前向和反向都正常；
- 所有 `planning.*`、`models.*`、`plan`、`train` 模块都能正常导入；
- PushObj 环境在没有显示器的情况下能渲染 T / I / square / small_tee / + 这几种形状；
- `cont_run_stream.py --stream S2 --arms cls6_free --max-episodes 2`：5 段串行跑完，`continual_state.pt` 在段之间正确传递，领域能建立、学习器能训练也能被读取；
- `--ws 0 --independent`（AdaJEPA 基线，3 张卡并行）、冻结 θ0 的批量 MPC、`cont_rounds.py` 报告：都能跑通。

拿到真实权重后的冒烟测试（每段 1 个 episode）：

```bash
conda activate cljepa
python scripts/cont_run_stream.py --stream S7t3 --arms cls6_free --no-base-probes \
    --chain-gpus 0 --probe-gpus 0 --max-episodes 1 --root /tmp/smoke_v6 \
    --extra planner.continual.save_deltas=false
```
