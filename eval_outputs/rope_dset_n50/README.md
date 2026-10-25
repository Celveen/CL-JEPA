# rope，数据集目标协议，50 个实例（2026-10-25）

只看 10 个实例的旧结果（`eval_outputs/rope_dset/`）时，各模型之间的差距都在 1–2 个标准误以内，看不出谁更好。这次在完全相同的协议下扩大到 50 个实例，先只跑两个 DINO 模型：

| 目录 | 模型 | checkpoint |
|---|---|---|
| `dino_ours/` | 我们自己训的 DINO-WM（DINOv2 ViT-S/14 冻结 + patch 特征，history 1，frameskip 1，100 epoch） | `CL-JEPA/checkpoints/rope_dino_h1_fs1_e100_seed0`，latest = epoch 100 |
| `ddpwm/` | DDP-WM 提供的 DINO-WM，结构和上面相同 | `CL-JEPA/checkpoints/ddpwm_dinowm_rope`，latest |

这次没有跑 A、B 两个 SmallResNetGeM 模型。

协议和旧 run 完全一样，用的是 `conf/plan_rope_dset.yaml`，没有改任何参数（副本见 `plan_rope_dset.yaml`）：
- 起点是验证集里的一个状态，目标是数据集中接下来 5 次推动在 PyFleX 里回放出来的结果；
- CEM：300 个样本，取前 30，10 次迭代，horizon 5；
- MPC 共 3 次迭代，每次推 5 下；
- 指标是最终状态和目标之间的 Chamfer 距离，越低越好。

50 个实例分成 5 份，每份 10 个，分别用种子 99、100、101、102、103（目录 `<模型>/seed<k>/`）。种子 99 这一份就是旧 run 的 10 个实例，可以用来检查结果能否复现；FleX 本身不是确定性的，同一组动作重新仿真，CD 会有大约 0.1 的波动。

**两个模型用的是同一份目标文件**（`targets/seed<k>.pkl`）：
- 先由 `dino_ours` 从验证集抽起点、在 PyFleX 里回放数据集接下来的 5 次推动来生成目标（`goal_source=dset`，和旧 run 相同），生成的目标拷贝到 `targets/`；
- `ddpwm` 直接读这份文件（`goal_source=file`），所以起点和目标都和 `dino_ours` 逐位相同。

为什么不让两个模型各自回放：旧的 10 实例 run 就是各自回放的，但 FleX 不是确定性的。这次第一次启动时也是各自回放，结果 50 个实例里有 2 个两边回放出的目标差得很远，相互之间的 CD 分别是 0.52 和 1.89（中位数 0.007），等于在解不同的题。所以我停掉了 `ddpwm` 的那次启动（只跑了几分钟，输出已删除），改成读同一份目标文件后重新开始。

- 启动：`bash eval_outputs/rope_dset_n50/run.sh`（在 CL-JEPA-cudagraph 下执行）。已经完成的份会自动跳过，所以重跑同一条命令即可续跑。
- 日志：`logs/<模型>_seed<k>.log`。
- 代码版本：`code_HEAD.txt` 加上 `code_uncommitted.diff`（worktree 里还没提交的 rope 改动）。
