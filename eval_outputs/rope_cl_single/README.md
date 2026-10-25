# rope 单领域在线持续学习（2026-10-25）

**问题**：在一串 rope episode 上做在线持续学习，规划效果会不会越来越好？这是在单领域里先验证方法本身：所有 episode 都属于 rope 一个领域，所以领域路由直接用真实标签（`write=oracle`，tag 是 `rope`），不存在路由出错的问题。

**episode 流**：`stream/chunk_<k>.pkl`，共 5 段，每段 10 个 episode，按固定顺序跑。它们就是 `rope_dset_n50` 那 50 个共享实例（`chunk_0`～`chunk_4` 分别对应 `targets/seed99`～`seed103`，逐字节相同）。每段用自己的种子（99～103），三组的种子一样，所以每个 episode 在三组之间是逐一配对的。

| 组 | 做法 | 段与段之间 |
|---|---|---|
| `frozen` | 冻结 θ0，不做适应（批量 MPC） | 互相独立 |
| `adajepa` | 每个 episode 内做 AdaJEPA 快速适应，episode 结束后重置（`ContinualMPCPlanner`，`w=0`） | 互相独立 |
| `cl` | `CLSMPCPlanner`，单领域：为 rope 维护一个慢学习器（预测器的完整副本）。它从 θ0 初始化，用之前所有 episode 的回放来训练：每写入一个 episode 训 15 步，Adam，lr 1e-4，每批 8 个窗口，满 3 个 episode 才建立。在没见过的 episode 上，它的 5 步开环误差和 θ0 之比，最近 3 次的平均 ≤ 0.9，才会被读取。AdaJEPA 照常在它上面做快速适应。 | 用 `continual_state.pt` 串成一条链 |

后来又加了两个实验，都是为了回答「预测更准的模型，规划会不会更好」：
- `cl_noada`：和 `cl` 完全一样，只是去掉了 AdaJEPA（`planner.adapt.steps=0`），单独看慢学习器自己的作用。
- `heldout/`：把流跑完之后的慢学习器（`<模型>/cl/chunk_4/continual_state.pt`）装进模型，在 50 个**新实例**上（种子 104～108）和 θ0 做冻结批量规划对比。用的是新加的 `planning.continual.LearnerFrozenMPCPlanner`，编码器仍然是 θ0 的；结果在 `heldout/RESULTS.md`。

公共设置：
- `conf/cont_plan_cem_rope.yaml`：CEM 300 个样本、取前 30、10 次迭代、horizon 5；MPC 3 次迭代，每次推 5 下。
- `resim_from_start=false`：每次迭代从当前状态接着仿真，三组都一样。

和 `rope_dset_n50` 的唯一区别就是这一点：那边每次迭代都从头重新仿真全部推动。所以冻结组在这里按同样的设置重跑了一遍，三组之间完全可比。

**模型**：先用我们训的 DINO-WM（`rope_dino_h1_fs1_e100_seed0`，epoch 100）。它的编码器是冻结的 DINOv2，没有投影头，所以 AdaJEPA 和慢学习器都只改预测器。之后打算再用 B（`rope_h3_fs1_cos_e100_seed0`，epoch 50）跑一遍，B 的编码器头也会被 AdaJEPA 适应，和 PushObj 上的设置一样。

- 启动：`bash eval_outputs/rope_cl_single/run.sh <model> <arm> <GPU 列表>`。已经完成的段会跳过，`cl` 组会从最后一个完成的段接着跑。
- 日志：`logs/`。冒烟测试（低 CEM 设置，只用来检查流程）在 `smoke/`。
- 为了让 rope 能跑，代码里改了一处：`planning/adajepa_probe.py` 的 `_task_progress` 原本按 PushT 的状态向量算距离，现在遇到 rope 这种粒子状态就直接跳过（rope 的 CD 由评估器自己记录）。
