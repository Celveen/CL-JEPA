# CL-JEPA：AdaJEPA 上的持续学习（推理阶段）

本仓库在 [AdaJEPA](https://github.com/agentic-learning-ai-lab/adajepa) 的基础上，加入跨 episode 的持续学习：每个领域一份慢学习器，用回放训练；AdaJEPA 的快适应照常运行。领域由门控自动划分。原 AdaJEPA 的说明见 [README.md](README.md)，代码结构见 [CLAUDE.md](CLAUDE.md)。

**仓库里不含权重和数据集**，只附带各条实验流的目标文件（`data/cont_streams/`）和见过形状分类器（`data/seen_probe/`）。

## 主要文件

| 文件 | 内容 |
|---|---|
| `planning/continual.py` | 持续学习基础：`ContinualMPCPlanner`（快慢权重）、`CLSMPCPlanner`（v4：每个领域一份慢学习器 + 无标签路由 + 熟悉度门 + 确认池 + 成熟门 + 写入门） |
| `planning/continual_merge.py` | v5：v4 + 对称验证合并 |
| `planning/continual_v6.py` | v6：v5 + 见过形状按分类器写入各自领域（先学、成熟后才调用）+ 写入与合并稍放宽 |
| `planning/continual_v7.py` | **v7 系列**，同一个规划器的三组配置，都不用形状标签，熟悉的回合照旧不学、也不调用学习器：`cls7_free`（写入优先进已有领域 + 开新领域要 5 个回合 + 定期吸收未成熟领域 + 合并放宽）、`cls7c_free`（同上，但开新领域和合并按 v5，只吸收小碎片）、`cls5a_free`（写入和合并完全按 v5，只加小碎片吸收）、`cls6a_free`（v6 去掉形状分类器：保留 v6 的放宽写入和放宽合并，加小碎片吸收）。离线对比见 `reports/cont_S7b3/v7_offline_replay.md` |
| `planning/continual_fixw.py`、`continual_slow.py`、`continual_v5x.py` | 消融：固定 w 快慢权重、多步损失、重点回放、沿用上一回合领域的早读（`cls5s_free`）、KNN 早读、验证式合并 |
| `planning/adajepa_probe.py` | 实验 0 用的探针规划器（也是持续学习规划器的基类） |
| `planning/escape.py`、`scripts/escape_*.py` | 第二条线：规划器卡住时的脱困策略 |
| `scripts/cont_run_stream.py` | 按段串行跑一条流（各组的超参数在 `NAMED_ARMS`）；`--independent` 让 AdaJEPA 基线各段并行 |
| `scripts/cont_build_seed_stream.py` | 用测试种子的抽样建流（如 S7t3） |
| `scripts/cont_train_seen_probe.py` | 用训练数据的形状标签训练见过形状分类器 |
| `scripts/cont_rounds.py` | 逐 episode 配对分析：按轮次、见过/没见过形状统计，并给出门控统计 |
| `conf/cont_plan_gd_pushobj.yaml` | 持续学习实验的规划配置，所有门控超参数都在 `planner.continual` 下 |

## 环境

```bash
conda env create -f environment.yaml && conda activate ts   # Python 3.9
```

PushObj 不需要 MuJoCo，导入时出现 mujoco 缺失的警告可以忽略。

## 权重

把 PushObj 世界模型的 checkpoint 目录（含 `hydra.yaml` 和 `checkpoints/model_3.pth`）放到 `checkpoints/pushobj_bias_16k_e3_seed0/`，或者运行时用 `--ckpt` 指向它。

熟悉度门槛（`familiar_thr` = 0.0146）和见过形状分类器（只有 v6 用）都是为这份权重标定的。**换权重时要重做：**

```bash
python scripts/cont_calibrate_familiarity.py --ckpt /abs/ckpt --epoch 3 --val-data /abs/训练数据/val   # 取 90 分位
python scripts/cont_train_seen_probe.py --ckpt /abs/ckpt --val-data /abs/训练数据/val --out data/seen_probe/<名字>.pt
```

## 复现 S7t3 实验

流 S7t3：每轮 7 个形状 × 50 个 episode，三轮分别是测试种子 100、200、300 的目标，共 1050 个 episode。

```bash
# AdaJEPA 基线：各段相互独立，可多卡并行
python scripts/cont_run_stream.py --stream S7t3 --ws 0 --independent --no-base-probes \
    --chain-gpus 2,3,4,5,6,7 --probe-gpus 2 --extra planner.continual.save_deltas=false

# v5（同学的参数）、v6、v5 + 早读：持续学习要串行，各占一张卡
python scripts/cont_run_stream.py --stream S7t3 --arms cls5_free --no-base-probes --chain-gpus 1 --probe-gpus 1 \
    --keep-states 6 13 --extra planner.continual.save_deltas=false
python scripts/cont_run_stream.py --stream S7t3 --arms cls6_free --no-base-probes --chain-gpus 0 --probe-gpus 0 \
    --keep-states 6 13 --extra planner.continual.save_deltas=false
# v5 + 沿用上一回合领域的早读（第一次规划前直接用上一回合最后读的学习器）
python scripts/cont_run_stream.py --stream S7t3 --arms cls5s_free --no-base-probes --chain-gpus 3 --probe-gpus 3 \
    --keep-states 6 13 --extra planner.continual.save_deltas=false
# v7 系列：各组可以分别放在不同的卡上同时跑
python scripts/cont_run_stream.py --stream S7t3 --arms cls7_free --no-base-probes --chain-gpus 0 --probe-gpus 0 \
    --keep-states 6 13 --extra planner.continual.save_deltas=false
python scripts/cont_run_stream.py --stream S7t3 --arms cls7c_free --no-base-probes --chain-gpus 1 --probe-gpus 1 \
    --keep-states 6 13 --extra planner.continual.save_deltas=false
python scripts/cont_run_stream.py --stream S7t3 --arms cls5a_free --no-base-probes --chain-gpus 2 --probe-gpus 2 \
    --keep-states 6 13 --extra planner.continual.save_deltas=false
python scripts/cont_run_stream.py --stream S7t3 --arms cls6a_free --no-base-probes --chain-gpus 3 --probe-gpus 3 \
    --keep-states 6 13 --extra planner.continual.save_deltas=false

# 分析：逐 episode 配对、按轮次统计、领域与门控统计
python scripts/cont_rounds.py --stream S7t3 --arms w0 cls5_free cls6_free cls5s_free cls7_free cls7c_free cls5a_free cls6a_free --frozen "" --out reports/cont_S7t3
```

说明：
- 每段 50 个 episode 时，一条持续学习链约需 15–16 小时；AdaJEPA 基线并行后几个小时即可完成。
- `cont_rounds.py` 只比较都跑完的段，各组的输出要放在同一个 `eval_outputs/cont_S7t3/` 下（别的服务器上跑的组，把它的目录拷过来即可）。
- 内存少于 40 GB 的机器加 `--min-free-gb 8`。
- 输出在 `eval_outputs/cont_S7t3/<组>/chunk_xx_<形状>/`。中断后重跑同一命令即可续跑。
- 每个 episode 的路由记录在 `continual_episodes.jsonl`。

冒烟测试（每段只跑 1 个 episode）：在上面的命令里加 `--max-episodes 1 --root /tmp/smoke`。

## 多种子、五轮与冻结评测

**数据流**（`scripts/cont_build_seed_stream.py`，需要作者的测试片段 `data/pushobj_eval/`；生成好的流已在 `data/cont_streams/` 中）：

| 流 | 每轮的测试种子 | 用途 |
|---|---|---|
| `S7t3` / `S7t5` | 100, 200, 300（`S7t5` 再加 1000, 1100） | 原流 / 五轮版本，前 3 轮逐字节相同 |
| `S7t3s400` / `S7t5s400` | 400, 500, 600（再加 1200, 1300） | 第二组种子 |
| `S7t3s700` / `S7t5s700` | 700, 800, 900（再加 1400, 1500） | 第三组种子 |
| `H7t3` | 2000, 2100, 2200 | 留出评测：`--exclude S7t5 S7t5s400 S7t5s700`，与上面所有流的目标都不重合 |

```bash
python scripts/cont_build_seed_stream.py --name S7t5s400 --seeds 400 500 600 1200 1300
python scripts/cont_build_seed_stream.py --name H7t3 --seeds 2000 2100 2200 --exclude S7t5 S7t5s400 S7t5s700
```

**规划器**：GD 加 `--extra planner.sub_planner.cuda_graph=true`（与 eager 逐位一致，见 README_cudagraph.md）；CEM 用论文设置：
`--extra planner.sub_planner.target=planning.cem.CEMPlanner +planner.sub_planner.topk=30 +planner.sub_planner.num_samples=200 planner.sub_planner.opt_steps=10`。CEM 的输出放在另一个 `--root`（如 `eval_outputs/cont_S7t3s400_cem`），以免和 GD 的同名组冲突。

**五轮续跑**：用五轮的流、原来的 `--root`，持续学习链会从第 3 轮结束时保存的状态（`chunk_20`）接着跑：

```bash
python scripts/cont_run_stream.py --stream S7t5s400 --root eval_outputs/cont_S7t3s400 --arms cls6a_free --no-base-probes \
    --chain-gpus 0 --probe-gpus 0 --keep-states 6 13 20 27 --extra planner.continual.save_deltas=false planner.sub_planner.cuda_graph=true
```

**冻结评测（只读）**：`planner.continual.read_only=true` 时，`CLSv7MPCPlanner`（v6a、v7 系列）照常读取学习器、做 AdaJEPA 快适应，但不写入、不训练、不合并，也不保存状态，所以各段互相独立。`cont_run_stream.py --independent --state-in <continual_state.pt>` 会对每一段都从同一个冻结状态出发并行评测：

```bash
# v6a 第 5 轮的冻结权重，在留出流上评测
python scripts/cont_run_stream.py --stream H7t3 --root eval_outputs/cmp_gd_v6a_s400 --arms cls6a_free --independent \
    --state-in eval_outputs/cont_S7t3s400/cls6a_free/chunk_34_T/continual_state.pt --no-base-probes \
    --chain-gpus 0,1 --probe-gpus 0 --extra planner.continual.save_deltas=false planner.sub_planner.cuda_graph=true
# AdaJEPA 官方形状权重（AdaJEPA Google Drive 的 checkpoints/pushobj_shape_shift.zip，解压到 checkpoints/）
python scripts/cont_run_stream.py --stream H7t3 --root eval_outputs/cmp_gd_official --ws 0 --independent \
    --ckpt checkpoints/pushobj_shape_shift --epoch latest --no-base-probes \
    --chain-gpus 2,3 --probe-gpus 2 --extra planner.continual.save_deltas=false planner.sub_planner.cuda_graph=true
```

注意：冻结时学习器若恰好不在成熟状态（最近 `mature_k` 次验证比的平均高于 `mature_ratio`），评测期间不会再验证，整段都只读 θ0。

**出图**（论文风格，PDF + PNG，写到 `reports/figs/`；`reports/figs/fonts/` 下有 TrueType 字体时使用它，例如 Google Fonts 的 Arimo，否则用 DejaVu Sans）。图里我们的方法一律标为 Ours，不出现 v6a 等内部名称。`reports/figs/*.pdf` 纳入版本管理（`reports/` 下其余文件仍只留在本地）；PDF 不写创建时间，所以重画后没变的图不会出现在 `git status` 里：

```bash
python scripts/fig_rounds.py                                 # 三组种子的五轮逐轮成功率：v6a 对 AdaJEPA，GD 与 CEM 各一张
python scripts/cmp_official_stats.py && python scripts/fig_cmp_official.py   # 冻结 v6a 对官方权重：总体与各形状
```

`fig_rounds.py` 默认读 `eval_outputs/cont_<流>`（GD）和 `eval_outputs/cont_<流>_cem`（CEM）；某条流的输出在别处时用 `--root gd:S7t3=/路径` 指定。
