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
