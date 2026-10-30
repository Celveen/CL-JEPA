# results-shape-v6a 分支：形状偏移（PushObj，7 个形状）v6a 的评测结果和报告（2026-10-28，10-30 更新）

这个分支只放结果，不含代码。代码在 `rope-cl`（`040ba0e`），形状偏移的报告和作图脚本在
`reports/shape_shift_figs/`。路径都相对仓库根目录。

**和 `results` 分支不是同一批运行。** `results` 分支是另一台服务器跑的链，本分支是
跑出 `reports/shape_shift_figs` 那台服务器（8 × RTX 5090）上的链，两边有同名目录
（如 `eval_outputs/cont_S7t3_cem/cls6a_free`），内容不同。Ours（cls6a_free）每段接着上一段的
状态跑，同一条链的 35 段必须来自同一次运行，**不要把两个分支解压到同一个目录里混用**。

## 2026-10-30 更新：CEM 换链

留出评测 H7t3 上，CEM 的 Ours 改用 S7t3、S7t3b、S7t3s700 三条链，不再用 S7t3s400；GD 的三条链不变。
原因：CEM 的 S7t3s400 链第 5 轮结束时没有一个学习器具备成熟资格，在 H7t3 上一次也没有读学习器，
等于只用了基础模型。详见 `reports/shape_shift_figs/说明.txt`。

- `eval_outputs/cont_S7t3b_cem`：S7t3b 链的五轮训练流。这是另一台服务器把 S7t3 流（同一批目标、
  同一批种子）重新跑的结果，取自 `results` 分支的 `eval_outputs/cont_S7t3_cem/cls6a_free`。
  改名放在这里，免得和本机的 `cont_S7t3_cem` 混淆。那边的说明原样附在其中的
  `RESULTS_README_from_results_branch.md`。
- `eval_outputs/cmp_cem_v6a_s100b`：S7t3b 第 5 轮状态在 H7t3 上的只读评测（2026-10-29），
  日志是 `gen_logs/cmp_cem_v6a_s100b.log`。
- `reports/shape_shift_figs` 和 `reports/shape_shift_figs_delivery` 已按新链重新生成（2026-10-29）。
  第一次推送（`7aaa3b4`）里是换链之前的版本。
- S7t3s400 的原始结果（`cmp_cem_v6a_s400`、`cont_S7t3s400_cem`）仍然保留。

## 取出来

    git fetch https://github.com/Celveen/CL-JEPA.git results-shape-v6a
    git archive FETCH_HEAD eval_outputs reports | tar xf - -C <目标目录>

`reports/shape_shift_figs/scripts/collect_data.py` 读 GD 的 S7t3 链时用的是
`../CL-JEPA/eval_outputs/cont_S7t3`（原机器上它在另一个 worktree 里）。本分支把它放在
`eval_outputs/cont_S7t3`，重新汇总前把它放回脚本期望的位置，或者改脚本里的 `EV_MAIN`。

## eval_outputs（每段只有小文件）

### 持续学习流（S7t5 流：5 轮 × 7 形状 × 50 个 episode，35 段）

| 目录 | 规划器 | 流 | 臂 |
|---|---|---|---|
| cont_S7t3 | GD | S7t3 | cls6a_free（Ours）、w0（AdaJEPA）各 35 段；cls3_oracle、cls5a_free、cls7_free、cls7c_free 各 21 段（第 1–3 轮） |
| cont_S7t3s400 | GD | S7t3s400 | cls6a_free、w0 各 35 段 |
| cont_S7t3s700 | GD | S7t3s700 | cls6a_free、w0 各 35 段 |
| cont_S7t3_cem | CEM | S7t3 | cls6a_free、w0 各 35 段 |
| cont_S7t3s400_cem | CEM | S7t3s400 | cls6a_free、w0 各 35 段 |
| cont_S7t3s700_cem | CEM | S7t3s700 | cls6a_free、w0 各 35 段 |
| cont_S7t3b_cem | CEM | S7t3b | cls6a_free 35 段（另一台服务器的运行，见上） |

`cont_S7t3_logs/` 是 GD S7t3 链各臂的控制台日志和启动脚本。

### 留出评测 H7t3（7 形状 × 50 个 episode × 测试种子 2000/2100/2200，21 段）

| 目录 | 方法 |
|---|---|
| cmp_{gd,cem}_frozen_official/w0 | Frozen：AdaJEPA 官方形状偏移权重，不适应（2026-10-28 补跑） |
| cmp_{gd,cem}_official/w0 | AdaJEPA：官方权重，每个 episode 内适应、episode 间重置 |
| cmp_{gd,cem}_v6a_s{100,400,700}/cls6a_free | Ours：三条链第 5 轮结束时的状态，只读评测（cmp_gd_v6a_s100 为 2026-10-28 补跑；cmp_cem_v6a_s400 保留，但报告里已不用） |
| cmp_cem_v6a_s100b/cls6a_free | Ours（CEM）：S7t3b 链第 5 轮结束时的状态，只读评测（2026-10-29），替换 cmp_cem_v6a_s400 |

`gen_logs/` 是这些评测和流的控制台日志、轮次统计（round_success.*）。

每段包含：logs.json、per_iter_logs.json、final_eval_results.npz、plan.log、
continual_episodes.jsonl、adajepa_losses.csv、probe_meta.json、run_metadata.json、.hydra/*.yaml（有的才有）。

未包含：continual_state.pt（每个 30 MB–1.2 GB，大多超过 GitHub 单文件上限）、plan_targets.pkl
（可由 data/cont_streams 重新生成）、episode_frames.npz、console.log。

## reports

| 目录 | 内容 |
|---|---|
| shape_shift_figs | 形状偏移论文图表：20 张图、合集 PDF、英文图注、作图数据（data/）、轨迹帧（replay/）和脚本；见其中的 `说明.txt` |
| shape_shift_figs_delivery | 同一套图表的交付包（PDF 矢量图、附表、预览、绘图源码与数据 zip、SHA256 校验） |
| cmp | 留出评测 H7t3 的对比报告 |
| gen | 六条流（3 条链 × GD/CEM）的逐流统计 |
| gen5 | 五轮持续学习的报告页（report.html）和数据 |
| figs | 五轮实验和官方对比的论文样式图 |
| cont_S7t3 | GD S7t3 链的报告（2026-10-21） |
