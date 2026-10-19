# AdaJEPA + 持续学习 v5（两段式无标签路由 + 有验证的领域合并）推理代码

本包是 S7b3 实验中 `cls5_free` 组的完整推理代码。**不含权重和数据集**，只附带三条流的目标文件（`data/cont_streams/S7`、`S7b`、`S7b3`，各几百 KB）。

- `S7b3` = S7b 的两轮 + 一轮全新 episode，7 个形状每个访问 3 次，每段 20 个 episode，共 420 个。
- 包里也含同一批实验的其他组：v4、固定 0.003 快慢权重、多步 rollout 损失、重点回放、AdaJEPA 基线。

## 方法

在 AdaJEPA 之上加**按领域分开的慢学习器**。AdaJEPA 本身是：每次重规划对 predictor 最后一层和 encoder 投影头各做 1 步 Adam，episode 结束就重置。慢学习器的做法：
- 每个领域一份完整 predictor 的拷贝，从 θ0 初始化；
- 用该领域 episode 的回放单独训练：θ0 encoder 的 latent，1 步预测损失，Adam lr 1e-4，每写入一个 episode 训 15 步；
- episode 开始的前 3 次重规划，按预测误差在 θ0 和已成熟的学习器之间选一个，快权重照常在它上面做 AdaJEPA。

领域由无标签路由自动划分：

1. **熟悉度门（v4）**：θ0 在前 3 次转移上的预测误差 ≤ `familiar_thr`，说明 θ0 已经会，不写入任何领域。
2. **只给不熟悉的 episode 分领域（v4）**：
   - 指纹是"单位化的快权重改动"，与某领域指纹的余弦 ≥ `tau`（0.10）就加入；
   - 否则进待定池，池中有 `pending_m`（3）个互相对齐（余弦 ≥ `tau_new`=0.05）的 episode 才建新领域。
3. **有验证的合并（v5 新增）**：
   - 每个新的不熟悉 episode，在任何学习器训练它之前，先用**所有**已有学习器做验证：5 步开环误差相对 θ0 的比值。
   - 领域 A、B 在各自最近 `merge_k`（3）个这样的 episode 上**双向满足**两个条件才合并：对方的学习器预测得**比 θ0 好**，并且不比本领域自己的学习器差超过 `merge_eps`（30%）。
   - 合并按"能否迁移"判断，而不是看标签。离线在 S7 上，`merge_eps` 取 0.2–0.5 都只会合并同一形状被拆开的子领域，I 和 square 不会被合并。
   - 合并时保留数据多的学习器，回放取并集后补训。被合并的领域留一个空壳（`merged_into`），保证领域编号不会重复。

另外两道门：
- **成熟门**：学习器在最近 3 个它没训练过的 episode 上，5 步误差相对 θ0 的平均比值 ≤ 0.9，才会被读取；
- **写入门**：比值 > 1.2 时拒绝写入。

代码：

| 文件 | 内容 |
|---|---|
| `planning/continual.py` | v4，即 `CLSMPCPlanner` |
| `planning/continual_merge.py` | v5，即 `CLSMergeMPCPlanner` |
| `planning/continual_fixw.py` | 固定 w 快慢权重 |
| `planning/continual_slow.py` | 多步损失和重点回放 |

各组的超参数在 `scripts/cont_run_stream.py` 的 `NAMED_ARMS`：

| 组 | 说明 |
|---|---|
| `cls5_free` | v5 |
| `cls4_free` | v4 |
| `cls3_oracle` | 按真实形状写入，路由上限 |
| `fixw4_free` | 每个领域 `slow ← slow + 0.003·Δ`，v4 门控 |
| `cls3_oracle_ms` | 真实标签 + 5 步 rollout 损失 |
| `cls3_oracle_pr` | 真实标签 + 重点回放：成功 episode ×2，接触转移加权 |
| `--ws 0` | AdaJEPA 基线 |

## 环境

```bash
conda env create -f environment.yaml && conda activate ts   # Python 3.9，先激活再运行
```

- PushObj 不需要 MuJoCo，导入时出现 mujoco 缺失的警告可以忽略。
- 用 `goal_source=segments` 跑流时不读数据集。

## 权重

需要一个 PushObj 世界模型的 checkpoint 目录，包含 `hydra.yaml` 和 `checkpoints/model_<epoch>.pth`。我们用的是 bias16k seed0 的 epoch 3，训练形状为 T/L/Z/+。

**换权重时必须重新标定熟悉度门槛。** 0.0146 是我们这份权重在它自己训练分布验证集上的 90 分位：

```bash
python scripts/cont_calibrate_familiarity.py --ckpt /abs/ckpt_dir --epoch 3 --val-data /abs/训练数据/val
# 把输出的 90 分位填进 scripts/cont_run_stream.py 里 cls5_free（和 cls4_free、fixw4_free）的 planner.continual.familiar_thr
```

## 运行

每条三轮链约 7 小时，占一张卡、约 20 个环境进程、约 4 GB 内存。GPU 利用率不高，一张卡可以同时跑 2–3 条链。下面的路径一律用绝对路径；内存少于 40 GB 的机器加 `--min-free-gb 8`。

```bash
# 冒烟测试：每段只跑 1 个 episode
python scripts/cont_run_stream.py --stream S7b3 --arms cls5_free --no-base-probes \
    --ckpt /abs/ckpt_dir --epoch 3 --chain-gpus 0 --probe-gpus 0 --max-episodes 1 --root /tmp/smoke_v5

# v5、v4 与带标签上限（三轮）
python scripts/cont_run_stream.py --stream S7b3 --arms cls5_free cls4_free cls3_oracle --no-base-probes \
    --ckpt /abs/ckpt_dir --epoch 3 --chain-gpus 0,1,2 --probe-gpus 0 --keep-states 6 13 \
    --extra planner.continual.save_deltas=false

# AdaJEPA 基线（三轮，逐 episode 配对必需）
python scripts/cont_run_stream.py --stream S7b3 --ws 0 --no-base-probes \
    --ckpt /abs/ckpt_dir --epoch 3 --chain-gpus 3 --probe-gpus 3 --keep-states 6 13

# 冻结 θ0：每段的难度参考，很快
bash scripts/cont_frozen_chunks.sh S7b3 /abs/ckpt_dir 3 3

# 分析：逐 episode 配对、首访/回访、见过/没见过形状、门控统计
python scripts/cont_rounds.py --stream S7b3 --arms w0 cls5_free cls4_free cls3_oracle --frozen frozen --out reports/cont_S7b3
```

- 输出在 `eval_outputs/cont_S7b3/<组>/chunk_xx_<形状>/`。
- 已完成的段会自动跳过；中断后重跑同一命令，会从最后一个还保留状态的段接着跑。
- 每个 episode 的记录在 `continual_episodes.jsonl`，字段包括：
  - `read`、`write`；
  - `write_rule`：familiar / direction / learner / pending / new_from_pending / oracle / none；
  - `familiar_score`、`pending_n`、`val_ratio`、`domains`；
  - v5 另有 `val_all`（所有学习器在该 episode 上的验证比）和 `merges`（本 episode 结束时发生的合并）。

## 离线工具（需要 AdaJEPA 组的 deltas.pt 和 episode_frames.npz）

| 脚本 | 作用 |
|---|---|
| `scripts/cont_route_screen.py` | 熟悉度门和各种指纹的筛选 |
| `scripts/cont_route_sweep.py` | 路由参数扫描 |
| `scripts/cont_merge_screen.py` | 合并规则的离线设计：交叉验证矩阵、不同 eps 下的合并结果 |
| `scripts/cont_diagnose.py` | 失败剖析 |
| `scripts/cont_build_stream.py` | 建新流，需要发布数据里的 `data/pushobj_eval/val_<shape>/plan_targets.pkl` |
