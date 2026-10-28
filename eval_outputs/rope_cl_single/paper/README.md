# 论文用图表：单领域持续学习（rope / granular）

全部由下面几个脚本生成，重跑即可更新：

```bash
source /data/zjc/adaJEPA/PyFleX/env.sh
# 1. DINO-WM 在 granular 第 10、19 个 episode 上，批量规划时 FleX 炸了：单独重新规划这两个（结果已有，跳过）
bash eval_outputs/rope_cl_single/paper/rerun_blowup.sh 0 1
# 2. 冻结组（DINO-WM θ0）没有存仿真帧：把整段按原运行的顺序在 FleX 里重放，给定性图用
CUDA_VISIBLE_DEVICES=3 /data/zjc/envs/cljepa/bin/python eval_outputs/rope_cl_single/paper/resim_frozen.py dino_ours 0
CUDA_VISIBLE_DEVICES=4 /data/zjc/envs/cljepa/bin/python eval_outputs/rope_cl_single/paper/resim_frozen.py dino_ours 2
CUDA_VISIBLE_DEVICES=5 /data/zjc/envs/cljepa/bin/python eval_outputs/rope_cl_single/paper/resim_frozen.py granular_ref 0   # 0-4 都可以
# 3. 表格和所有图
/data/zjc/envs/cljepa/bin/python -I eval_outputs/rope_cl_single/paper/make_paper_figs.py
```

每张图都有 `.pdf`（矢量，字体已嵌入，直接放进 LaTeX）和 `.png`（300 dpi，预览用）。宽度按双栏论文的通栏 6.75 英寸设计。

## 正文

| 文件 | 内容 |
|---|---|
| `main_qualitative` | 真实仿真结果图（仿 AdaJEPA Figure 4）：上一行 rope、下一行 granular，各两个例子；每个例子是起始状态和 DINO-WM / AdaJEPA / Ours 的最终状态。金黄色是目标形状，右上角 ✓/✗ 是成功/失败 |
| `main_table.tex`（预览 `main_table.png`） | 主表：rope 和 granular 的最终 CD，含 DINO-WM 论文报告的 TD-MPC2、DreamerV3、DINO CLS、IRIS |
| `main_bars` | 主表的柱状图，两个领域各一栏 |

## 附录

| 文件 | 内容 |
|---|---|
| `<领域>_qualitative` | 和正文同样的例子，逐步展开：开始、每 3 推一帧、目标，三种方法各一行，裁得更近 |
| `<领域>_passes` | Ours 随轮数的变化：(a) 预测误差沿流下降；(b) 每轮最终 CD；(c) 每轮成功率 |
| `<领域>_bars` | 分领域的柱状图，带成功率；rope 另有 91 个新实例 |
| `<领域>_steps` | 一个 episode 内 CD 和成功率随推数的变化 |
| `<领域>_paired` | 逐实例配对散点：Ours vs DINO-WM、Ours vs AdaJEPA |
| `tables.tex`（预览 `<领域>_table.png`） | 分领域的表：CD、成功率、和 DINO-WM 的配对差及显著性 |
| `TABLES.md` | 主表和分领域表的 Markdown 版，带 p 值 |
| `data/` | 每张图背后的数字（CSV），可以用别的工具重画 |
| `raw_frames/` | 定性图里每一帧的原图（224 × 224） |
| `logs/` | 冻结组重放的日志：每个 episode 重放出的最终 CD 和运行时记录的对照 |
| `make_paper_figs.py`、`resim_frozen.py`、`rerun_blowup.sh` | 生成脚本 |

## 方法名

- **DINO-WM**：θ0，不做任何适应（冻结组）。主表里用的是我们测的数（rope 0.49、granular 0.30），不是 DINO-WM 论文报告的 0.41、0.26。
- **AdaJEPA**：每个 episode 内做测试时适应，episode 结束就重置。
- **Ours**：只用慢学习器（`cl_noada`），同一条 50 个实例的流连着跑三遍、第 3 轮的结果（`cl_noada_r3`）。第 1、2 轮只出现在随轮数变化的图（`passes`）里。
- **Ours†**：只在附录 rope 的 91 个新实例上出现，是第 1 轮结束时的慢学习器冻结规划的结果。
- **TD-MPC2、DreamerV3、DINO CLS、IRIS**：数字直接取自 DINO-WM 论文 Table 1、2，我们没有重跑；表里不单独标注，正文引用 DINO-WM 时说明出处。

## 写论文时要注意

1. **主表里 TD-MPC2、DreamerV3、DINO CLS、IRIS 的数取自 DINO-WM 论文，评估设置和我们的不完全一样**：它用 CEM 迭代 30 次，我们用 10 次、MPC 3 次 × 5 推。正文介绍实验设置或基线时，写一句这几个数取自 DINO-WM 即可。附录的分领域表把两种设置分开列，还有我们按 DINO-WM 设置复现的结果（rope 0.82、granular 0.23）。
2. **Ours 是第 3 轮的结果，这些实例前两轮已经见过。** 它是"在这 50 个实例上能练到的上限"，不是对新实例的泛化，论文里要写明是流跑三遍之后的结果。对新实例的证据只有 rope 的 Ours†（第 1 轮结束时的学习器）。
3. **granular 的 DINO-WM 有两个 episode 是单独重新规划的。** 批量冻结规划时，第 1 段第 9 个样本在第 1 次 MPC 迭代把 FleX 弄炸了（粒子甩飞，最终 CD 约 1840），紧接着它在同一个仿真器上跑的第 0 个样本也炸了（约 1860）；整段重跑一次结果完全一样。FleX 环境会把状态带到下一次调用，所以这两个 episode 用同样的模型、规划设置和种子各自单独重新规划（`rerun_blowup.sh`，结果在 `../granular_ref/frozen_solo/`）：第 10 个 0.227，第 19 个 0.302，都没有成功。其他方法这两个 episode 都正常，所以现在三组都是 50 个。
4. **定性图里 DINO-WM 那一行是重放出来的。** 批量冻结规划不存帧，所以把它执行过的动作按原运行的顺序重放：新建环境，每次 MPC 迭代（5 推）依次给 10 个样本各仿真一段，每段从该样本上一段结束的粒子位置重新开始。重放的最终 CD 和运行时记录的对照见 `logs/resim_*_chunk*.log`：granular 全部 50 个完全一致（连第 1 段那两个炸掉的值也一样），rope 图里用到的两个例子误差在 0.001 以内。其他两行是规划时直接存下来的帧。图里标的 CD 是运行时记录的最终 CD。
5. **定性图的例子是挑的**：DINO-WM 和 AdaJEPA 都失败、Ours 成功的实例（rope 第 21、2 个，granular 第 6、39 个）。图注要说明这一点，整体效果看主表和柱状图。
6. 主表按两个任务的平均 CD 从高到低排；rope 一栏里 DINO CLS（0.84）排在 IRIS（1.11）前面，是因为它的 granular 差得多。柱状图每栏各自从高到低排。

## 图注草稿（英文）

**Main qualitative.** Real FleX rollouts on rope (top) and granular (bottom), two instances each. Per instance: the start state and the final states of DINO-WM, AdaJEPA and Ours after 15 pushes. Gold: the goal configuration. ✓/✗: success (final Chamfer distance below 0.5 for rope, 0.2 for granular). DINO-WM (frozen) and AdaJEPA (adapting within each episode only) leave the object off the goal, while Ours, whose world model keeps learning across episodes, brings it onto the goal. Instances are chosen where DINO-WM fails.

**Main table.** Planning on deformable objects: final Chamfer distance between the object and the goal after 15 pushes (lower is better), mean ± s.e. over 50 rope and 50 granular instances.

**Main bars.** Final Chamfer distance (lower is better).

**Appendix qualitative.** The instances of the main figure, step by step: the start, every third push, and the goal. The final frame shows the final Chamfer distance. DINO-WM's frames are a replay of its executed actions in the order the planning run simulated them.

**Passes.** Ours over three passes of the stream. (a) Prediction error of the slow learner on the next, not-yet-trained episode, relative to θ0's (5-step open-loop latent error; points: episodes, lines: rolling geometric mean over 10). (b) Mean final Chamfer distance per pass. (c) Success rate per pass. Dashed: frozen DINO-WM; dotted: AdaJEPA. Passes 2 and 3 revisit the instances of pass 1.

**Appendix bars.** (a) Final Chamfer distance. Hatched: numbers reported by DINO-WM under its own protocol. Solid: our protocol on the online stream; for rope, also 91 unseen instances (Ours†: the learner after the first pass, planning frozen). (b) Success rate.

**Steps.** Chamfer distance and success rate after every MPC replan (5 pushes each), mean over the stream's instances. Shaded: ±1 standard error (for the success rate, the 68% Wilson interval).

**Paired.** Final Chamfer distance per instance, Ours against DINO-WM (a) and AdaJEPA (b) on the same instance and seed. Points below the diagonal: Ours reaches a lower CD. Gray lines: the success threshold.
