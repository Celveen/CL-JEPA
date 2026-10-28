# 论文表格

由 `make_paper_figs.py` 生成。

## 主表（正文）

最终 Chamfer 距离（CD，越低越好），按两个任务的平均 CD 从高到低排。TD-MPC2、DreamerV3、DINO CLS、IRIS 的数取自 DINO-WM 论文（Table 1、2）；其余是我们测的，均值 ± 标准误，rope 50 个实例、granular 50 个实例。

| Method | Rope CD ↓ | Granular CD ↓ |
|---|---|---|
| TD-MPC2 | 2.52 | 1.21 |
| DreamerV3 | 2.49 | 1.05 |
| DINO CLS | 0.84 | 0.79 |
| IRIS | 1.11 | 0.37 |
| AdaJEPA | 0.63 ± 0.10 | 0.30 ± 0.02 |
| DINO-WM | 0.49 ± 0.06 | 0.30 ± 0.02 |
| Ours | **0.28** ± 0.03 | **0.22** ± 0.01 |

## 附录：分领域的表

CD 越低越好；SR 是成功率（rope：最终 CD < 0.5，granular：最终 CD < 0.2）。ΔCD 是与同一组实例上的 DINO-WM（θ0，冻结）逐 episode 配对的平均差，p 值：CD 用 Wilcoxon 符号秩检验，SR 用精确 McNemar 检验。

两种协议的数值不能直接比较：DINO-WM 论文的协议是 random_state 目标、CEM 迭代 30 次、MPC；我们的协议是 50 个实例的流、CEM 迭代 10 次、MPC 3 次 × 5 推。"DINO-WM, our reproduction" 是我们用 DINO-WM 的协议复现的结果（50 个实例、MPC 5 次）。

Ours：只用慢学习器，流跑完三轮后的结果（第 2、3 轮重访第 1 轮的实例）。Ours†：第 1 轮结束时的慢学习器，在从没见过的新实例上冻结规划。

### Rope

| Method | CD ↓ | SR ↑ | ΔCD vs DINO-WM | p (CD) | p (SR) |
|---|---|---|---|---|---|
| *Reported by DINO-WM (its protocol)* | | | | | |
| IRIS | 1.11 | — | — | — | — |
| DreamerV3 | 2.49 | — | — | — | — |
| TD-MPC2 | 2.52 | — | — | — | — |
| R3M encoder | 1.13 | — | — | — | — |
| ResNet encoder | 1.08 | — | — | — | — |
| DINO CLS encoder | 0.84 | — | — | — | — |
| DINO-WM | 0.41 | — | — | — | — |
| DINO-WM, our reproduction | 0.821 ± 0.112 | 46% | — | — | — |
| *Online stream, 50 instances (our protocol)* | | | | | |
| DINO-WM | 0.491 ± 0.059 | 64% | — | — | — |
| AdaJEPA | 0.629 ± 0.096 | 60% | +0.138 | 0.15 | 0.85 |
| Ours | 0.282 ± 0.033 | 88% | -0.209 | 0.0018 | 0.0075 |
| *Unseen instances, 91 (our protocol, frozen planning)* | | | | | |
| DINO-WM | 0.664 ± 0.072 | 60% | — | — | — |
| Ours† | 0.349 ± 0.042 | 80% | -0.314 | 0.0012 | 0.0051 |

### Granular

| Method | CD ↓ | SR ↑ | ΔCD vs DINO-WM | p (CD) | p (SR) |
|---|---|---|---|---|---|
| *Reported by DINO-WM (its protocol)* | | | | | |
| IRIS | 0.37 | — | — | — | — |
| DreamerV3 | 1.05 | — | — | — | — |
| TD-MPC2 | 1.21 | — | — | — | — |
| R3M encoder | 0.95 | — | — | — | — |
| ResNet encoder | 0.90 | — | — | — | — |
| DINO CLS encoder | 0.79 | — | — | — | — |
| DINO-WM | 0.26 | — | — | — | — |
| DINO-WM, our reproduction | 0.232 ± 0.023 | 64% | — | — | — |
| *Online stream, 50 instances (our protocol)* | | | | | |
| DINO-WM | 0.301 ± 0.020 | 24% | — | — | — |
| AdaJEPA | 0.304 ± 0.019 | 16% | +0.003 | 0.96 | 0.39 |
| Ours | 0.224 ± 0.010 | 40% | -0.077 | 6.5e-05 | 0.039 |

