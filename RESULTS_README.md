# results 分支：形状线五轮流的评测结果和报告（2026-10-28）

这个分支只放结果，不含代码。代码在 `main`（`6e760d0` 之后）。路径都是相对仓库根目录的，
解压到仓库根目录就能直接被 `scripts/fig_shape_*.py` 读到。

## 合并到另一台服务器（在仓库根目录执行）

    git fetch https://github.com/Celveen/CL-JEPA.git results
    git archive FETCH_HEAD eval_outputs reports | tar xf - --skip-old-files

`--skip-old-files` 不覆盖已有文件。Ours（cls6a_free）每段接着上一段的状态跑，同一条链的
35 段必须来自同一次运行，不要和别处的同名链混在一起。

## eval_outputs（每段只有小文件）

| 根目录 | 臂 | 段数 | 说明 |
|---|---|---|---|
| cont_S7t3（GD，S7t5 流） | w0 | 21 | AdaJEPA，第 1–3 轮 |
| | frozen | 3 | 第 5 轮 I / L / square |
| | cls5_free, cls5s_free, cls6_free | 各 21 | 旧版本，第 1–3 轮 |
| cont_S7t3_cem（CEM，S7t5 流） | cls6a_free | 35 | Ours，第 1–5 轮完整 |
| | w0 | 23 | AdaJEPA，第 1–3 轮 + 第 4 轮 I / L |
| | cls5a_free, cls3_oracle | 各 21 | 第 1–3 轮 |
| cont_S7t3s400_cem（CEM，S7t5s400 流） | frozen | 3 | 第 5 轮 I / L / square |

每段包含：logs.json、per_iter_logs.json、final_eval_results.npz、plan.log、
continual_episodes.jsonl、adajepa_losses.csv、probe_meta.json、run_metadata.json、.hydra/*.yaml。

未包含：continual_state.pt（每个约 565 MB，超过 GitHub 单文件上限）、plan_targets.pkl
（可由 data/cont_streams 重新生成）、episode_frames.npz、console.log。

## reports

本机 reports/ 全部，去掉 6 个超过 5 MB 的训练控制台日志（train_*console.log）。
reports/figs_shape 是本机单链的预览图，数据齐后会重画。
