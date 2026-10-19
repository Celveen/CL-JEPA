# AdaJEPA + 持续学习 v4（两段式无标签路由）推理代码

本包是 S7b 实验中 `cls4_free` 组的完整推理代码。**不含权重和数据集**，只附带两条流的目标文件（`data/cont_streams/S7`、`S7b`，几百 KB）。

## 方法一句话

在 AdaJEPA（每次重规划对 predictor 最后一层和 encoder 投影头做 1 步 Adam，episode 结束重置）之上，加上**按领域分开的慢学习器**：
- 每个领域一份完整 predictor 的拷贝，从 θ0 初始化；
- 用该领域 episode 的回放单独训练（θ0 encoder 的 latent，1 步预测损失，Adam lr 1e-4，每写入一个 episode 训 15 步）；
- episode 开始时按预测误差选用 θ0 或某个慢学习器，快权重照常在它上面做 AdaJEPA。

领域由无标签路由自动划分，v4 分两段：

1. **熟悉度门**：θ0 在 episode 前 3 次转移上的预测误差 ≤ `familiar_thr`，说明 θ0 已经会，不写入任何领域。
2. **只给不熟悉的 episode 分领域**：
   - 用"单位化的快权重改动"作指纹，与某领域指纹的余弦 ≥ `tau`（0.10）就加入；
   - 否则进入待定池，池中有 `pending_m`（3）个互相对齐（余弦 ≥ `tau_new`=0.05）的 episode 才建新领域。

另外两道门：
- **成熟门**：学习器在最近 3 个它没训练过的 episode 上，5 步开环误差相对 θ0 的平均比值 ≤ 0.9 才会被读取；
- **写入门**：验证比 > 1.2 时拒绝写入。

代码在 `planning/continual.py` 的 `CLSMPCPlanner`，各组的超参数在 `scripts/cont_run_stream.py` 的 `NAMED_ARMS`：

| 组 | 说明 |
|---|---|
| `cls4_free` | v4，无标签 |
| `cls3_oracle` | 按真实形状写入，作为路由上限 |
| `cls3_free` | v3，旧路由 |
| `--ws 0` | AdaJEPA 基线 |

## 环境

```bash
conda env create -f environment.yaml && conda activate ts   # Python 3.9
```

- PushObj 不需要 MuJoCo，导入时出现 mujoco 缺失的警告可以忽略。
- 用 `goal_source=segments` 跑流时不读数据集，不需要设置 `DATASET_DIR`。

## 权重

需要一个 PushObj 世界模型的 checkpoint 目录，包含 `hydra.yaml` 和 `checkpoints/model_<epoch>.pth`。我们用的是 bias16k seed0 的 epoch 3。

**换权重时必须重新标定熟悉度门槛。** 0.0146 是我们这份权重在它自己训练分布验证集上的 90 分位：

```bash
python scripts/cont_calibrate_familiarity.py --ckpt /abs/ckpt_dir --epoch 3 --val-data /abs/训练数据/val
# 把输出的 90 分位填进 scripts/cont_run_stream.py 里 cls4_free 的 planner.continual.familiar_thr
```

## 运行

每条链约 4 小时，占一张卡、约 20 个环境进程、约 4 GB 内存；各组可以并行。下面的路径一律用绝对路径。

```bash
# 冒烟测试：每段只跑 1 个 episode
python scripts/cont_run_stream.py --stream S7b --arms cls4_free --no-base-probes \
    --ckpt /abs/ckpt_dir --epoch 3 --chain-gpus 0 --probe-gpus 0 --max-episodes 1 --root /tmp/smoke_v4

# v4 与带标签上限
python scripts/cont_run_stream.py --stream S7b --arms cls4_free cls3_oracle --no-base-probes \
    --ckpt /abs/ckpt_dir --epoch 3 --chain-gpus 0,1 --probe-gpus 0 --keep-states 6 \
    --extra planner.continual.save_deltas=false

# AdaJEPA 基线（w0）
python scripts/cont_run_stream.py --stream S7b --ws 0 --no-base-probes \
    --ckpt /abs/ckpt_dir --epoch 3 --chain-gpus 2 --probe-gpus 2 --keep-states 6

# 冻结 θ0：每段的难度参考，很快
bash scripts/cont_frozen_chunks.sh S7b /abs/ckpt_dir 3 3

# 分析：逐 episode 配对、首访/回访、见过/没见过形状、门控统计
python scripts/cont_rounds.py --stream S7b --arms w0 cls4_free cls3_oracle --frozen frozen --out reports/cont_S7b
```

- 输出在 `eval_outputs/cont_S7b/<组>/chunk_xx_<形状>/`。
- 内存少于 40 GB 的机器加 `--min-free-gb 8`，否则 runner 会一直等内存。
- 已完成的段会自动跳过，中断后重跑同一命令即可续跑。注意 `--keep-states` 会删掉中间段的状态，续跑只能从保留的段接着跑。
- 每个 episode 的路由记录在 `continual_episodes.jsonl`，字段包括 `read`、`write`、`write_rule`（familiar / direction / learner / pending / new_from_pending / oracle / none）、`familiar_score`、`pending_n`、`val_ratio`、`domains`。

## 新建一条流

需要发布数据里的 `data/pushobj_eval/val_<shape>/plan_targets.pkl`，本包没有附带：

```bash
python scripts/cont_build_stream.py --name S7c --seed 2 --exclude-streams S7 S7b \
    --blocks I:20 L:20 square:20 Z:20 small_tee:20 +:20 T:20 square:20 T:20 I:20 +:20 L:20 small_tee:20 Z:20
```

## 离线分析脚本（需要 AdaJEPA 组的 deltas.pt 和 episode_frames.npz）

| 脚本 | 作用 |
|---|---|
| `scripts/cont_route_screen.py` | 熟悉度门和各种指纹的离线筛选 |
| `scripts/cont_route_sweep.py` | 固定参数扫描 |
| `scripts/cont_diagnose.py` | 失败剖析，以及预测改善与成功率的关系 |
