# 论文一：JEPA 文献速览与题目候选

2026-10-22。检索方式：arXiv 接口按作者 LeCun 取最近 200 篇，再按"JEPA + world model""test-time + world model""continual + world model""continual test-time adaptation"四组关键词检索。AdaJEPA、JEPA-TTT、Dupoux–LeCun–Malik 三篇读了全文，其余读摘要。

门控从今天起统一用 **v6a**（`cls6a_free`）。

## 1. LeCun 组的 JEPA 脉络

### 1.1 蓝图与表示学习

| 年份 | 论文（arXiv） | 要点 |
|---|---|---|
| 2022 | A Path Towards Autonomous Machine Intelligence（LeCun，OpenReview） | 提出 JEPA、分层 JEPA，以及"感知 + 世界模型 + 代价 + 执行器 + 配置器"的整体架构 |
| 2023 | I-JEPA（2301.08243） | 在图像上预测被遮挡区域的表示 |
| 2023 | MC-JEPA（2307.12698） | 同时学运动和内容 |
| 2024 | V-JEPA（2404.08471） | 视频上的特征预测 |
| 2024 | IWM（2403.00504） | 在表示学习里学到并利用世界模型 |
| 2025 | LeJEPA（2511.08544） | SIGReg：嵌入应服从各向同性高斯，只剩一个超参数 |
| 2025 | Gaussian Embeddings（2510.05949） | JEPA 的嵌入隐式地学到了数据密度 |
| 2026 | When Does LeJEPA Learn a World Model?（2605.26379） | 线性可辨识性的理论保证 |
| 2026 | LpWM（2608.22764）、Rectified LpJEPA（2602.01456） | 稀疏表示让动作条件动力学更好建模 |
| 2026 | LeHami（2610.15267） | 辛（Hamiltonian）潜动力学，长 rollout 不收缩 |
| 2026 | Preserving Unstable Modes（2610.07540） | 加逆动力学损失，保留可控的不稳定模态；AdaJEPA 引用它说明分布偏移下会掉性能 |

### 1.2 视频与大规模预训练

| 年份 | 论文 | 要点 |
|---|---|---|
| 2025 | V-JEPA 2（2506.09985） | 100 万小时视频预训练，加少量机器人数据得到 V-JEPA 2-AC，用于规划 |
| 2025 | Intuitive Physics（2502.11831） | 用表示空间的预测误差（"惊讶"）检测违反物理常识的事件 |
| 2025 | Back to the Features（2507.19468） | DINO 特征上的视频世界模型 |
| 2026 | V-JEPA 2.1（2603.14482）、LeVJEPA（2608.27395） | 稠密特征；去掉启发式技巧的视频预训练 |
| 2026 | Latent Action World Models in the Wild（2601.05230） | 从野外视频学潜动作 |

### 1.3 用于规划的 JEPA 世界模型（与我们最相关）

| 年份 | 论文 | 要点 | 与我们的关系 |
|---|---|---|---|
| 2024 | DINO-WM（2411.04983） | 冻结 DINOv2 补丁特征 + 预测器，零样本规划 | 本仓库模型结构的来源 |
| 2024 | Navigation World Models（2412.03572） | 扩散式导航世界模型（CDiT，10 亿参数） | — |
| 2025 | PLDM（2502.14819） | 无奖励离线数据下，JEPA 潜动力学规划与 RL 的系统比较 | — |
| 2025 | What Drives Success…?（2512.24497） | 把这类模型统称 JEPA-WM，系统消融架构、目标和规划器 | 术语"JEPA-WM"可沿用 |
| 2025 | Value-guided planning（2601.00844） | 让潜空间距离逼近负的目标条件价值 | 与第二篇的 cost-to-go 相关 |
| 2025 | Closing the Train-Test Gap（2512.09929） | 训练时合成数据，GD 规划追平 CEM | AdaJEPA 认为它动机最接近，但模型规划时仍固定 |
| 2026 | GRASP（2602.00475） | 并行随机梯度规划，用随机性跳出局部最优 | 第二篇相关 |
| 2026 | Temporal Straightening（2603.12231） | 曲率正则让 GD 规划更稳 | 本仓库的直接上游 |
| 2026 | LeWorldModel（2603.19312） | 第一个从像素端到端稳定训练的 JEPA：两项损失，1500 万参数，规划快 48 倍 | — |
| 2026 | stable-worldmodel（2605.21800） | 可复现评测平台，带可控视觉、几何变化的泛化基准 | 可能用来补第二个环境 |
| 2026 | HWM（2604.03208） | 共享潜空间里多时间尺度的分层 MPC；Franka 抓放 70% 对 0% | — |
| 2026 | H-JEPA（2610.06805） | 端到端分层 JEPA，每层自己的潜空间；Visual AntMaze 18% → 73% | — |
| 2026 | Causal-JEPA（2602.11389） | 遮挡物体级潜变量 | 简称 C-JEPA，与 CL-JEPA 接近 |

### 1.4 适应与持续学习

**AdaJEPA（2606.32026，Wang、Bounou、LeCun、Ren）**，我们的直接基础。
- 每次重规划后，用刚执行的转移做一步自监督更新。
- 原文："each episode starts from the same pretrained model and maintains its own copy and buffer"。
- 数据规模一节自己写了下一步："A natural next step is to combine lightweight test-time adaptation with continual and active learning to expand the world model's coverage over time"。它也承认适应受预训练表示覆盖范围的限制。我们的论文就是这个下一步。

**Why AI Systems Don't Learn and What to Do About It（2603.15381，Dupoux、LeCun、Malik）**，观点文章。
- 三个系统：System A 从观察中学（含世界模型），System B 从行动中学，System M 做元控制。
- System M 监测"认知信号"：置信度、预测误差、学习收益、新颖性。它输出"元动作"：打开或关闭学习、把数据送进情景记忆、回放。文中建议 System M 的核心策略是固定规则。
- 他们认为现有的运行时适应、测试时训练，只是"在僵化系统上的小改动"。
- 对我们的意义：v6a 用的就是这几类信号。θ0 的惊讶决定写不写；学习器在新回合上的验证比决定读不读、合不合并。可以说它是一个具体、无标签、固定规则的 System M。这是引言里最好的钩子，但不建议放进题目。

## 2. 组外的直接相关工作

| 论文 | 做了什么 | 和我们的区别 |
|---|---|---|
| **JEPA-TTT**（2610.00722，2026-09-30） | 跨回合持续更新预测器（编码器和奖励头冻结），dense replay；8 个动力学偏移、500 回合，规划提升 153% | 局限一节自己写明不研究偏移序列、不回到训练动力学、不测保留和灾难性遗忘；只有单一固定偏移，不处理视觉偏移；所有数据都写入，没有"写不写、写到哪"的选择 |
| Sandwich-Residuals（2609.21740） | 冻结模型外加小残差，在 AdaJEPA 基准上达到其 95% 的效果，参数少 97–99% | 单回合适应的参数效率，不跨回合 |
| T2-z（2610.22364） | 测试时优化环境潜变量 | 需要训练时配合，不跨回合积累 |
| CLAW（2609.12278） | 超网络生成 LoRA，几秒数据即可适应 | 需要元训练 |
| World-T3（2610.17434） | 视频生成世界模型里，用 TTT 快权重做持久空间记忆 | "TTT 当持久记忆"这个说法已经有人用 |
| Stratified Retention（2610.03713，观点文章） | 主张按不变性的时间尺度分层保留，提出 differential retention 指标 | 我们"熟悉的数据不写、不读"与"保护不变量"呼应；评测可以报告它建议的指标 |
| Fork ledger（2609.10954） | 反事实地衡量每次更新的价值；一直更新会降低回报 | 支持"必须有选择地更新" |
| One-Step Prediction Fidelity Is a Poor Proxy…（2610.19375） | JEPA 世界模型的一步预测误差与规划成功几乎无关 | 与我们"预测误差降 30–50%，成功率只在改善很大时才涨"一致 |
| TMoW（2601.22647） | LLM 具身智能体：测试时更新多个世界模型之间的路由 | "测试时路由多个世界模型"已有人做，但对象是 LLM 智能体；我们的路由在潜空间、无标签 |
| MoW（2602.01270）、MoWM（2509.21797） | 训练时的世界模型专家混合 | 训练时，不是部署时积累 |
| 持续测试时适应：CoTTA（2203.13591）、RDumb、PeTTA、SloMo-Fast（2511.18468）、GAIN（2609.36655）、When and Where to Reset（2603.03796） | 分类、分割上的持续测试时适应，靠置信度或熵做判断 | "Persistent TTA""快/慢教师"的命名已被占用 |
| 互补学习系统：McClelland 等 1995；Kumaran、Hassabis、McClelland 2016；Continual Learning, Fast and Slow（2209.02370） | 海马快速学习 + 新皮层慢速巩固 | 我们方法的概念来源（代码里就叫 CLS） |

## 3. 写作惯例

**标题有四种格式：**
- 方法名 + 一句话：AdaJEPA: An Adaptive Latent World Model；LeWorldModel: Stable End-to-End JEPA from Pixels；H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning。
- 描述短语：Temporal Straightening for Latent Planning；Hierarchical Planning with Latent World Models。
- 主张句："X Enable(s) Y"，如 DINO-WM 的 World Models on Pre-trained Visual Features enable Zero-shot Planning、V-JEPA 2 的 Self-Supervised Video Models Enable Understanding, Prediction and Planning。
- 问句：What Drives Success in Physical Planning with JEPA-WMs?；When Does LeJEPA Learn a World Model?

**摘要结构：** 现有世界模型冻结或脆弱 → 提出方法 → 一句话机制 → 带数字的结果（如"18% → 73%""as few as one gradient step"）。

**常用术语：** latent world model、JEPA world model（JEPA-WM）、goal-reaching、closed-loop MPC、gradient-based planning、test-time distribution shift、reward-free offline trajectories。

**评测：**
- AdaJEPA 用 3 个测试种子 × 50 回合，报告均值。
- 这一支很少做逐回合配对检验。我们的配对 McNemar 加精确 p 值是优势，正文要说明。

**环境：**
- 常见的有 PushT、PointMaze、Wall、TwoRoom、Reacher、OGBench Cube，加 DROID / Franka 真实数据。
- 我们目前只有 PushObj（AdaJEPA 的基准）。审稿风险是"只有一个环境"，需要第二个环境，视觉线或迷宫都可以。

## 4. 我们的定位

**一句话：** AdaJEPA 让 JEPA 世界模型在回合内适应，但每个回合都从头开始；我们让它跨回合积累，而且只积累它原本不会的东西。

**三个差异：**
1. **相对 AdaJEPA：** 跨回合巩固，回访时不再从零开始。这正是 AdaJEPA 自己写的下一步。
2. **相对 JEPA-TTT：** 面对多个反复出现的偏移，中间穿插训练分布内的数据，并检验没有伤害；门控决定写不写、写到哪、什么时候读；全程无标签。
3. **相对持续测试时适应：** 对象是闭环规划。我们发现直接累积快权重有害，预测改善也不等于规划成功，所以门控依据的是功能证据（学习器在新回合上的验证），而不是置信度。

**目前的证据（S7t3，1050 个配对回合）：**
- CEM 规划器：v6a +4.2 pp（87/43，p=1.4×10⁻⁴）。收益全在没见过的形状上，三次访问分别 +8、+7、+12；见过的形状不变。
- GD 规划器：v6 +3.0 pp（p=0.013）。本机还没有 v6a 在 GD 下的结果。
- 真实标签路由的上限：+6.3 pp。

## 5. 题目候选

### A. 方法名 + 描述（沿用 AdaJEPA 的格式）

**A1（推荐）CL-JEPA: An Adaptive Latent World Model That Consolidates Only What Is New**
- 中文：只巩固新知识的自适应潜空间世界模型。
- 优点：
  - 一句话说出原则：只学 θ0 不会的，和 JEPA-TTT 的"全部写入"拉开距离。
  - 句式直接接上 AdaJEPA 的题目（An Adaptive Latent World Model）。
- 风险：没有体现路由和成熟门，要靠摘要和正文讲清。

**A2 CL-JEPA: Selective Consolidation for Continually Adapting Latent World Models**
- 稳妥、术语化，"selective"就是与 JEPA-TTT 的区别。
- 风险：读起来比较平。

**A3 CL-JEPA: Complementary Learning Systems for Latent World Models**
- 理论锚点清楚。
- 风险：互补学习系统是老概念，审稿人会问新意在哪；题目里没有"选择"。

### B. 主张句

**B1 Surprise-Gated Consolidation Enables Continual Test-Time Learning in Latent World Models**
- "惊讶"是 JEPA 圈的现成词：直觉物理那篇和 LeWorldModel 都把表示空间的预测误差叫作"惊讶"。熟悉度门就是 θ0 的惊讶。
- 风险：门控不只看惊讶，还有验证。

**B2 Selective Consolidation Lets Latent World Models Keep Learning Without Forgetting**
- 风险："Without Forgetting"需要专门的遗忘评测支撑：见过的形状零变化，加冻结探针。

### C. 口号 + 副标题

**C1（备选）Adapt Within Episodes, Consolidate Across Them: Selective Continual Learning for Latent World Models**
- 两个时间尺度一目了然：回合内是 AdaJEPA，跨回合是我们。

**C2 Beyond Episodic Reset: Selective Consolidation for Adaptive Latent World Models**
- 直接点出相对 AdaJEPA 改了什么。

### D. 问句（适合偏分析的写法）

**D1 When Should an Adaptive World Model Remember What It Learns?**
- 如果论文重心放在"巩固什么时候对规划有用"，这个最合适。

### 避免使用

| 词 | 原因 |
|---|---|
| Learning, Fast and Slow | 已有多篇同名论文（2209.02370、2605.12484 等） |
| Persistent Test-Time Training / Adaptation | JEPA-TTT 和 PeTTA 已用 |
| What Should World Models Forget? | 2610.03713 已用，问句结构也太接近 |
| Mixture of World Models | MoW、TMoW、MoWM 已用 |
| Lifelong、never forget | 实验只有 1050 回合的流，说过头了 |
| 形状相关的词 | 方法要能用在只有一个形状的视觉线上 |
| System M | 放题目里像蹭名字，留在引言里用 |

## 6. 待确认

- LeCun 2022 原文里配置器（configurator）的具体表述：本机下载 OpenReview 的 PDF 失败，引用前要核对。
- 方法名 CL-JEPA：既是仓库名，也同时对应 Continual Learning 和 Complementary Learning；arXiv 上没有重名，但与 Causal-JEPA 的简称 C-JEPA 接近。
- 题目里的"Continual"是否成立，取决于是否加入第二个环境（视觉线或迷宫），以及 v6a 在 GD 规划器下的结果。

## 7. 惊讶门（熟悉度门）的定义

曾考虑的题目：Surprise-Gated Consolidation Enables Continual Test-Time Learning in Latent World Models。后续讨论认为惊讶门不是主要创新，不放进题目，见第 8 节。下面的定义保留，作为方法部分的一个组件。

### "惊讶"的定义

- **惊讶**是原模型 θ0 对一个回合的潜空间预测误差：用 θ0 编码，预测最新一帧的潜表示（不含动作维），在回合前 3 次转移上取几何平均。代码见 `CLSMPCPlanner._familiarity`。
- **门槛**是 θ0 在自己训练验证数据上同一统计量的 90 分位（`familiar_thr` = 0.0146）。标定时不看测试流，也不用标签。
- **区分能力：** 区分见过与没见过的形状，AUROC 0.96。在 90 分位门槛下，没见过的形状 68% 判为惊讶，见过的形状误报 4%。

### 门控里的三种惊讶

| 形式 | 决定什么 | v6a 里的规则 |
|---|---|---|
| 绝对惊讶：θ0 惊不惊讶 | 学不学，要不要调用记忆 | 不惊讶的回合只读 θ0（`familiar_read`），永不写入 |
| 相对惊讶：哪个记忆对它最不惊讶 | 读哪个、写到哪、何时启用、何时合并 | 读取：在 θ0 和成熟学习器中取预测误差最小者<br>写入门：比 θ0 差 1.2 倍以上就拒绝<br>成熟门：连续 3 次 ≤ 0.9 倍 θ0<br>合并：互相预测在 1.5 倍以内且优于 θ0<br>碎片吸收：≤ 0.8 |
| 惊讶的方向：快权重改动 Δ | 还没有记忆能解释它时，归到哪一类 | 按单位化 Δ 的余弦归类；确认池里有 3 个互相对齐的回合才建新领域 |

Titans 把惊讶定义为损失的梯度。Δ 是 AdaJEPA 在回合内针对预测误差所做更新的累积，所以可以叫"惊讶的方向"。所有门控只用一个量，即 JEPA 训练时本来就在最小化的潜空间预测误差，不需要标签。

### 文献里的用法（写相关工作用）

- **表示空间的预测误差即惊讶：**
  - Intuitive Physics（2502.11831）。
  - LeWorldModel（2603.19312）的 surprise evaluation。
  - Cambrian-S（2511.04670，LeCun 参与）：用下一帧潜表示的惊讶驱动记忆和事件分割。
- **测试时记忆：** Titans（2501.00663）按惊讶决定写入多少；惊讶即损失梯度，带动量和遗忘机制。
- **视频生成：** Surprise Forcing（2607.18436）里的 "Surprise-Gated Memory Bank"，用于外部记忆的准入。这个词组已经用过，我们是巩固进世界模型参数，表述上要区分开。
- **神经科学：** 新奇信号控制信息能否进入长期记忆（Lisman & Grace 2005, Neuron）。
- **我们的不同之处：**
  - 回合级的硬门控。
  - 用冻结的 θ0 来测，门槛按 θ0 自己的训练分布标定。
  - 作用对象是世界模型参数的巩固，服务于闭环规划。

### 为什么惊讶是对的信号（已有证据）

- **熟悉的数据里没有可巩固的东西：**
  - 见过形状的快权重改动方向随机：同形状余弦 0.014–0.030，跨形状 0.011–0.014，最近邻认对只有 48%。
  - 在见过形状上训练的学习器不比 θ0 准，验证比 1.01–1.42。
  - v6 学了见过的形状，写入多出约 4 倍，成功率不变。
- **新数据里有稳定的修正：**
  - 没见过形状的改动方向能聚类，最近邻认对 92%。
  - 学习器把预测误差降到 θ0 的 0.4–0.7 倍。
- **结果（CEM，S7t3）：**
  - 没见过的形状三次访问 +8.0 / +7.3 / +12.0 pp，见过的形状 +0.0 / +1.5 / +0.0 pp。整体 +4.2 pp（p=1.4×10⁻⁴）。
  - 熟悉的回合只读 θ0，θ0 从不改动，所以这部分不遗忘是由构造保证的。约 4% 的误判由成熟门兜底。

### 题目里 "Enables" 还缺的证据

- **消融一：v6a 去掉惊讶门**（`familiar_thr`=0，其余不变）。
  - 近似证据：S7 上没有熟悉度门的 v3 为 −2.5 pp，分出 29 个碎领域，纯度 0.58。
- **消融二：不加选择的持续学习基线**（类似 JEPA-TTT：一个学习器回放全部回合）。近似证据：
  - S2 上直接沿用快权重（w=1）为 −14 pp。
  - 离线回放中，单一学习器在没见过形状回访上的预测误差比为 0.71 / 0.69；v4 为 0.66 / 0.66，真实标签为 0.62 / 0.64。
- **门槛敏感性：** 80、90、95 分位。
- **第二个环境：** 视觉线只有一个形状，惊讶来自视觉偏移，可以说明惊讶门不依赖形状。

## 8. 题目再讨论（2026-10-22）

**工作题目（已选定，候选 1）：Selective Consolidation Enables Continual Test-Time Learning in Latent World Models**

**结论：惊讶门不是关键创新，不放进题目。**
- **机制常见：** 用预测误差判断新颖，Titans、Cambrian-S、Surprise Forcing 都有类似做法。
- **收益不来自它：** 增益来自没见过形状上的慢学习器，真实标签路由的 +6.3 pp 是上限。惊讶门的作用是保护熟悉的数据、减少领域数。
- **没有证据表明它带来成功率：** 单独加上熟悉度门的 v4，在线成功率与 AdaJEPA 持平（S7b3 −0.2 pp）。
- 第 7 节把几种门都归为"惊讶"，是修辞上的统一，不宜当作贡献。

**真正的关键内容：**
1. **问题：** 从回合内适应到跨回合积累。面对反复出现、互不相同的偏移，中间夹杂熟悉的数据；用闭环规划、逐回合配对来评测。
2. **发现：** 测试时更新本身不会积累。
   - 相邻回合的改动余弦只有 0.06，沿领域均值的能量只占 6–12%。
   - 直接沿用快权重为 −14 pp。
   - 共享一个学习器会干扰其他领域，预测误差变成 1.4–3.6 倍。
3. **方法与结果：** 回合内快适应 + 按领域分开、用回放训练的慢学习器 + 无标签的选择性路由。
   - CEM S7t3 上 +4.2 pp（p=1.4×10⁻⁴），离真实标签上限 2 pp 以内。
   - 收益全在新领域，回访时更大；熟悉的领域不受影响。

**题目候选（按推荐顺序）：**
1. Selective Consolidation Enables Continual Test-Time Learning in Latent World Models
2. Adapt Within Episodes, Consolidate Across Them: Continual Test-Time Learning for Latent World Models
3. CL-JEPA: Complementary Learning Systems for Continually Adapting Latent World Models
4. Test-Time Adaptation Does Not Accumulate on Its Own: Consolidation for Latent World Models（要限定在反复出现的偏移下：JEPA-TTT 在单一固定偏移下直接持续更新是有效的）

**"Enables" 需要的对照：** 直接沿用快权重；单一学习器回放全部回合（类似 JEPA-TTT）。
