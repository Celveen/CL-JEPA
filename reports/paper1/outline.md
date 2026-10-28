# 论文一大纲（草稿 v2）

2026-10-22。工作题目：**Selective Consolidation Enables Continual Test-Time Learning in Latent World Models**。按正文 9 页（NeurIPS / ICML 格式）加附录来规划，门控统一用 v6a（`cls6a_free`）。

和 AdaJEPA 原文一样，主结果分形状偏移和视觉偏移两部分，两部分都是正式结果。

相对 v1 的改动：
- 加入视觉线结果（[../visual_v6a/summary.md](../visual_v6a/summary.md)）。
- 参照五篇论文重排图表，主图（大结构图）放在第 2 页。
- 补上逐页版面。

## 1. 参考论文怎么排图（实际页码）

| 论文 | 图 1（概览 / 主图） | 方法图 | 主结果 | 分析与消融 |
|---|---|---|---|---|
| AdaJEPA | 第 2 页：plan–execute–adapt–replan 循环 + 世界模型结构 | 与图 1 合一 | 第 7 页图 2：形状与视觉偏移下的成功率；第 8 页表 1 | 第 8–10 页：轨迹图、实现对比表、定性示例 |
| Temporal Straightening | 第 1 页：预训练编码器的潜轨迹弯曲 | 第 3 页图 3：训练示意 | 第 8 页表 1 + 图 8 | 第 6–7 页：曲率与成功率、距离热图、开环规划对比 |
| JEPA-TTT | 第 2 页：动力学偏移导致规划失败 | 第 3 页图 2：方法概览 | 第 9 页图 5；第 10 页图 6：500 回合曲线 | 第 6、11、12 页：机制、预测误差、更新规则消融表 |
| DINO-WM | 第 3 页：方法概览 | 第 4 页图 2：架构 | 第 6 页表 1–2 | 第 7–8 页：开环 rollout、对比 |
| H-JEPA | 第 1 页：分层抽象示意 | 第 3–4 页图 2–3：训练与规划 | 第 6 页图 6 | 第 5、7–9 页：分析、消融表、真实数据 |

**共同模式：**
1. 第 1–2 页就有一张整宽的概览图，读者看图就能知道"问题是什么、我们怎么做"。
2. 第 3–4 页是方法图，或者与图 1 合并。
3. 主结果在第 6–8 页。
4. 分析和消融放在后面。
5. 定性轨迹放在正文末尾或附录。

## 2. 摘要（英文草稿）

> Latent world models plan by optimizing actions against their own predictions, so prediction errors under distribution shift turn directly into planning failures. AdaJEPA corrects such errors at test time by updating the world model from the transitions it causes inside the model-predictive-control loop, but every episode restarts from the pretrained weights: whatever the model learned about a new object or a new appearance is discarded and must be relearned at the next encounter. We study continual test-time learning, in which an agent faces a stream of episodes whose shifts recur and are interleaved with familiar data, and should improve on recurring shifts without losing what it already does well. We find that test-time adaptation does not accumulate on its own: updates from different episodes are nearly orthogonal, carrying them across episodes degrades planning, and a single model consolidating all experience interferes across shifts. We therefore propose selective consolidation. Within each episode the model adapts as in AdaJEPA; across episodes, experience is consolidated by replay into slow learners for the shifts it discovers, and label-free gates computed from the model's own prediction errors decide whether an episode is consolidated, which learner it updates, and when a learner is trusted for planning. On recurring object-shape shifts (1,050 paired episodes), selective consolidation raises planning success over AdaJEPA by 4.2 points (p < 10⁻³) and recovers two thirds of the gain of an oracle that knows the shift labels; on recurring visual shifts it raises success by 9 to 11 points, and by 15 to 17 points after five rounds. In both settings the gates consolidate almost only the shifts the pretrained model fails on, and performance on familiar data is unchanged.

**还缺实验支撑的说法：**
- "carrying them across episodes degrades planning"：目前只有 S2 上 100 个回合的结果。
- "a single model consolidating all experience interferes"：目前只有离线证据。
- 视觉线的 9–11 个百分点：五轮复用同一批目标，需要每轮换新目标的版本。
- 形状线的 +4.2 来自 CEM；GD 下还没有 v6a 的结果。视觉线 GD 和 CEM 都有。

## 3. 贡献点（英文草稿）

1. **Problem.** We formulate continual test-time learning for latent world models: recurring, heterogeneous shifts interleaved with familiar episodes, evaluated by closed-loop planning with exact per-episode pairing.
2. **Diagnosis.** We show why test-time adaptation does not accumulate on its own: AdaJEPA's gains come from episode-specific information rather than parameter perturbation, updates from different episodes are nearly orthogonal, and naive persistence either degrades planning or interferes across shifts.
3. **Method.** We propose selective consolidation, a fast–slow scheme in the spirit of complementary learning systems, whose label-free gates use the model's own prediction errors to decide what to consolidate, where to store it, and when to use it.
4. **Evidence.** Across object-shape and visual shifts, with gradient-based and sampling-based planners, selective consolidation improves over AdaJEPA (+4.2 points on 1,050 paired shape episodes; +9 to +11 points on visual shifts). It consolidates almost only the shifts the pretrained model fails on, leaves familiar data unchanged, and lets related shifts share one learner.

## 4. 图表规划（按出现顺序）

| 编号 | 页 | 内容 | 数据状态 |
|---|---|---|---|
| **图 1（主图，整宽）** | 2 | **(a)** 一条回合流：AdaJEPA 每回合重置，曲线回到原点；选择性巩固随回访上升。<br>**(b)** 方法结构：回合内 plan–execute–adapt 快循环 → 回合末门控（熟悉：只用 θ0、不写入；新的：路由到领域学习器）→ 回放巩固 → 成熟的学习器在下一回合被读取；合并与吸收。<br>**(c)** 小图：没见过形状按访问次数、视觉偏移按轮次的成功率 | 要画；(c) 的数据已有 |
| 图 2 | 4 | 诊断：(a) 实验 0 各对照的增益；(b) 不同回合的更新几乎正交，熟悉与不熟悉的指纹可分性；(c) 三种直接持续方式 vs AdaJEPA | 部分：实验 0 只有 seed 100；直接持续只有 S2 |
| 算法 1 | 5 | 选择性巩固的伪代码 | 要写 |
| 图 3（整宽） | 7 | 主结果，与 AdaJEPA 图 2 的排法相同：上行形状偏移（按访问次数，熟悉 / 不熟悉分栏），下行视觉偏移（按轮次，六种偏移）；GD 与 CEM | 形状线 CEM 有，GD 缺 v6a；视觉线两种都有，待按原文协议补跑 |
| 表 1 | 7 | 主表：冻结、AdaJEPA、三种直接持续、真实标签路由、v6a；形状与视觉两组列 | 部分：缺冻结、直接持续基线 |
| 图 4 | 8 | 巩固了什么：(a) 视觉线每个视觉域被巩固的比例 vs AdaJEPA 在该域的成功率；(b) 形状线的领域构成，以及 I 与 square 共用学习器的迁移 | 有 |
| 表 2 | 8 | 消融：逐个关掉门控、门槛敏感性、只快 / 只慢 | 缺 v6a 版本 |
| 图 5 | 9 | 分析：预测改善分组 vs 成功率差；剩下的失败是规划器卡住 | S7 上有，S7t3 需重算 |
| 附录 | — | 定性轨迹（解码器可视化）、完整领域表、超参数、开销、早期流 S7 / S7b3 | 大部分有 |

## 5. 逐页版面（正文 9 页）

| 页 | 内容 |
|---|---|
| 1 | 题目、摘要、引言前半：潜空间世界模型 + MPC，AdaJEPA 每回合重置 |
| 2 | **图 1（整宽，页面上半）**；引言后半与贡献点 |
| 3 | 相关工作；问题设定（流、要求、评测） |
| 4 | §4 测试时适应会自己积累吗？＋图 2 |
| 5 | §5.1 快与慢、§5.2 门控；算法 1 |
| 6 | §5.3 开销；§6.1 实验设置：两类偏移、两种规划器、基线、统计 |
| 7 | **图 3（整宽）**；表 1；§6.2 主结果 |
| 8 | 图 4、表 2；§6.3 巩固了什么、§6.4 消融 |
| 9 | §6.5 通用性、§7 分析与局限（图 5 小图）、§8 结论 |

## 6. 章节结构

| 节 | 页数 | 要点 |
|---|---:|---|
| 1 Introduction | 1.25 | AdaJEPA 每回合重置，原文把"与持续学习结合"列为下一步；部署时偏移会反复出现；直接持续为什么不够；System M 作为钩子；我们的发现、方法、结果 |
| 2 Related Work | 0.5 | JEPA 世界模型与潜空间规划；世界模型的测试时适应；持续测试时适应；互补学习系统与记忆 |
| 3 Problem Setting | 0.5 | 流、回合、未知的偏移类型；四条要求；逐回合配对的评测 |
| 4 Does Test-Time Adaptation Accumulate? | 1.0 | 增益来自本回合的信息；更新几乎正交；直接持续不行 |
| 5 Selective Consolidation | 1.5 | 快与慢；学不学、写到哪、何时用、合并与吸收；开销 |
| 6 Experiments | 3.0 | 设置；主结果（形状 + 视觉）；巩固了什么；消融；通用性 |
| 7 Analysis and Limitations | 0.75 | 预测更准不等于规划更好；剩下的失败是规划器卡住（简述）；存储随领域数增长；视觉线的目标复用 |
| 8 Conclusion | 0.25 | — |

## 7. 还缺的实验

**P0：主表必须有**
1. **形状线（S7t3）的基线：** 冻结 θ0、沿用快权重（w=1）、EMA 慢权重（w=0.003）、单一学习器回放全部回合。
2. **形状线 GD 下的 v6a 和真实标签路由：** 各一条链，约 15 小时。
3. **视觉线对齐原文协议：** 原文的三个种子是三批不同的测试目标（3 × 50），现在的视觉线复用同一批目标。需要用测试种子 100 / 200 / 300 各抽一批目标，跑 v6a、AdaJEPA、冻结，并保存逐回合结果做配对检验。

**P1**
1. v6a 门控消融：逐个关掉熟悉度门、成熟门、写入门、合并。
2. 为 T16k 权重重新标定熟悉度门槛，并测门槛敏感性（80 / 90 / 95 分位）。
3. 实验 0 补 seed 200 / 300。

**P2**
1. 官方权重上的结果。
2. 另一种流顺序。
3. 测开销。

## 8. 需要决定的事

已定：和 AdaJEPA 原文一样，论文一的主结果包括形状偏移和视觉偏移两部分，视觉线是正式的第二部分。

1. **主表的规划器：** 视觉线两种都有；形状线要补 GD 下的 v6a。
2. **基础模型怎么写：** bias16k / T16k 权重用了带接触偏置的数据生成，这可能是第二篇的素材，第一篇里只简要说明。
3. **投稿目标与时间线。**
