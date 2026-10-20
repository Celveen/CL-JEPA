# CUDA Graph 加速 GD 规划

持续学习的各组（`cls*`）和 AdaJEPA 基线都是一个样本接一个样本地规划：每次重规划要用 batch 为 1 跑 100 步 GD。本分支把这一步的前向和反向捕获成 CUDA Graph，**结果与原实现逐位一致**，规划快约 13 倍，一段的端到端时间快约 5–7 倍。默认关闭，用下面的参数打开：

```bash
python scripts/cont_run_stream.py ... --extra planner.continual.save_deltas=false planner.sub_planner.cuda_graph=true
```

## 瓶颈在哪

S7t3 的 `cls7_free/chunk_00_I`（50 个 episode，954 次重规划，6793 秒）的时间构成：

| 部分 | 耗时 | 占比 |
|---|---|---|
| GD 规划（每次重规划 100 步） | 5988 s | 88% |
| AdaJEPA 快适应 | 161 s | 2.4% |
| 环境回放、日志、持续学习其余部分 | 645 s | 9.5% |

单步 GD 的耗时基本不随 batch 变化（`scripts/bench_gd_step.py`，RTX 5090 + Xeon Platinum 8380）：

| batch | 每步耗时 |
|---|---|
| 1 | 55.0 ms |
| 8 | 61.9 ms |
| 32 | 63.6 ms |
| 50 | 61.8 ms |

batch 为 1 时，GPU kernel 实际只占 5.9 ms，每步却要发射约 1,470 个 kernel。其余时间都耗在 CPU 上：Python、autograd 和 kernel 发射。所以运行时每个主进程都占满一个 CPU 核心，而 GPU 利用率只有 6–7%。持续学习的链必须逐个 episode、逐次重规划地顺序执行，没法靠加大 batch 提速。

## 做法（`planning/gd.py`）

- `cuda_graph=true` 时，`rollout → objective → backward` 第一次先在旁路 stream 上预热，然后捕获成图；之后每步只重放这个图，得到 `actions.grad`。
- Adam 优化步、余弦学习率调度、`randn_like` 加噪声仍在图外照原样执行，所以学习率变化和随机数消耗都和 eager 一样。
- 图按以下内容缓存：MPC 步数（目标函数 `staged` 会随它切换分支）、输入形状、模型各张量的地址、requires_grad 状态、train/eval 模式。其中任何一项变化都会重新捕获。权重数值的变化（AdaJEPA 更新、reset、加载 θ0 或领域学习器）都是原地 `copy_`，图会直接读到最新值，不需要重新捕获。
- 输入（起始观测、目标的隐表示、初始动作）每次规划时拷进固定的静态缓冲区。返回的动作是一份拷贝。
- `planning/objectives.py`：`objective_fn_all` 的系数改为在 GPU 上缓存，因为 CPU 到 GPU 的拷贝不能放进图捕获里。数值不变。

## 验证

**逐次对比**：`planning/gd_check.py` 中的 `GDCheckPlanner` 对每次真实的规划调用都先跑 eager，再从同一个随机数状态跑 CUDA Graph，比较动作和规划结束后的随机数状态，然后返回 eager 的结果（让整条运行沿着 eager 的轨迹走）。

| 运行 | 覆盖的路径 | 调用次数 | 逐位一致 | 随机数一致 |
|---|---|---|---|---|
| `cls7_free`，S7t3 `chunk_00_I`，空状态，3 个 episode | AdaJEPA 快适应、v7 门控 | 47 | 47/47 | 47/47 |
| `cls3_oracle`，`chunk_07_I`，接着第 2 段结束时的状态 | 读取已成熟的 I 学习器 | 21 | 21/21 | 21/21 |
| `cls7_free`，`chunk_07_I`，接着第 2 段结束时的状态 | 读取领域 `d0` 的学习器 | 22 | 22/22 | 22/22 |

**端到端对比**：`cls7_free`、`chunk_00_I`、5 个 episode，纯 eager 和纯 CUDA Graph 各跑一遍：

- `per_iter_logs.json` 中 85 次重规划的记录（去掉计时字段）完全相同；
- `continual_episodes.jsonl`、`adajepa_losses.csv`、最终成功率（0.6）完全相同；
- `continual_state.pt` 中 63 个张量，0 处不同。

```bash
# 逐次对比
python scripts/cont_run_chunk.py cls7_free S7t3 chunk_00_I 3 eval_outputs/cg_check/A 7 null \
    planner.sub_planner.target=planning.gd_check.GDCheckPlanner \
    +planner.sub_planner.check_log=$(pwd)/eval_outputs/cg_check/A/graph_check.jsonl
# 端到端：分别用 cuda_graph=false / true 跑一遍，再比较两个目录的输出
python scripts/cont_run_chunk.py cls7_free S7t3 chunk_00_I 5 eval_outputs/cg_check/E2E_graph 7 null \
    planner.sub_planner.cuda_graph=true
```

## 速度

| 指标 | eager | CUDA Graph | 倍数 |
|---|---|---|---|
| 单次规划（100 步 GD） | 5.88 s | 0.44 s | 13.4× |
| 端到端（5 个 episode，含进程启动） | 539 s | 75.7 s | 7.1× |
| 显存 | 约 1.2 GB | 约 1.8 GB（每个 MPC 步一个图，最多 30 个） | |

按上面 `chunk_00_I` 的时间构成估算，完整的一段（50 个 episode）约从 6800 秒降到 1200 秒，快约 5.5 倍。

## 注意

- 打开后规划变成 GPU 受限（每步约 5.5 ms，约等于 kernel 本身的时间），一张卡上仍然只适合跑一条链。
- 剩下的主要开销是 MPC 每次重规划都从初始状态重放所有动作并保存视频（`PlanEvaluator.eval_actions`），这部分没有改动。
- 因为结果逐位一致，同一个实验中途从 eager 切换到 CUDA Graph（重启 `cont_run_stream.py` 并加上开关，从最后完成的段续跑）不会改变结果。
