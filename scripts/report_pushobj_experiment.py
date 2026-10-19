"""Build a PushObj train+eval report from saved metrics and paired evaluations.

Paths are set with `configure()` (or the CLI flags below) so the same builder
serves every run; run_pushobj_evaluation.py calls it after each finished job.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ("T", "L", "Z", "+", "I", "small_tee", "square")
SEEN = SHAPES[:4]
# Categorical slots 1-3 of the dataviz reference palette (all-pairs validated, light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

REPORT = ROOT / "reports/pushobj_replay_e10"
MODEL = ROOT / "checkpoints/pushobj_replay_2000_e10_seed0"
EVAL = ROOT / "eval_outputs/pushobj_replay_e10"
DATA = ROOT / "data/pushobj_replay_2000"
EPOCHS = 10
PREVIOUS = ROOT / "reports/pushobj_2000_e20"      # earlier run on the same targets, for comparison
PREVIOUS_LABEL = "随机游走数据、20 轮"
TRAIN_CONFIG = "train_pushobj"
RELEASED = ROOT / "eval_outputs/pushobj_released_reference"
# AdaJEPA paper (arXiv:2606.32026). Data-scale study, K=4 shapes x N=4k (App. B.3,
# 3-seed average, main-experiment protocol); hyper-parameter ablation on T/square
# (App. B.2). Main text Sec. 4.1: at most 20 MPC replans; results average 3 test seeds x 50.
PAPER = [
    ("已见形状（数据规模实验 K=4, N=4k）", "seen", "51%", "81%", "3 个训练种子、多组训练形状平均；最多 20 次重规划"),
    ("未见形状（数据规模实验 K=4, N=4k）", "unseen", "36%", "52%", "同上"),
    ("T（超参数消融）", "T", "50%", "81–88%", "最多 20 次重规划；AdaJEPA 为不同超参/回放设置的范围"),
    ("square（超参数消融）", "square", "20%", "35–51%", "同上"),
]


def configure(report=None, model=None, eval_dir=None, data=None, epochs=None,
              previous=None, previous_label=None, train_config=None):
    global REPORT, MODEL, EVAL, DATA, EPOCHS, PREVIOUS, PREVIOUS_LABEL, TRAIN_CONFIG
    REPORT = Path(report).resolve() if report else REPORT
    MODEL = Path(model).resolve() if model else MODEL
    EVAL = Path(eval_dir).resolve() if eval_dir else EVAL
    DATA = Path(data).resolve() if data else DATA
    EPOCHS = int(epochs) if epochs else EPOCHS
    PREVIOUS = Path(previous).resolve() if previous else PREVIOUS
    PREVIOUS_LABEL = previous_label or PREVIOUS_LABEL
    TRAIN_CONFIG = train_config or TRAIN_CONFIG


def add_args(p):
    p.add_argument("--report", default=str(REPORT))
    p.add_argument("--model", default=str(MODEL))
    p.add_argument("--eval-dir", default=str(EVAL))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--previous", default=str(PREVIOUS), help="report dir of an earlier run on the same targets")
    p.add_argument("--previous-label", default=PREVIOUS_LABEL)
    p.add_argument("--train-config", default=TRAIN_CONFIG, help="hydra config name, for the reproduce command")


def configure_from_args(a):
    configure(a.report, a.model, a.eval_dir, a.data, a.epochs, a.previous, a.previous_label, a.train_config)


def display_path(path):
    """Repo-relative path for display; absolute when outside the repo."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a writer may currently be appending its last line
    return result


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return center-radius, center+radius


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def final_log(d):
    log = [r for r in rows(d / "logs.json") if "final_eval/success_rate" in r]
    return log[-1] if log and (d / "final_eval_results.npz").exists() else None


def load_outcome(d):
    with np.load(d / "final_eval_results.npz") as saved:
        return {k: saved[k].copy() for k in ("successes", "state_0", "state_g")}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def loss_figure(metrics, title, decoder_epoch=None):
    panels = [("loss", "Total loss"), ("z_loss", "Latent prediction MSE"),
              ("z_visual_loss", "Visual latent MSE"), ("z_proprio_loss", "Proprio latent MSE"),
              ("curvature_loss_used_for_training", "Curvature (unscaled)"),
              ("decoder_recon_loss_reconstructed", "Decoder reconstruction")]
    panels = [(k, n) for k, n in panels if any(r.get(f"train_{k}") is not None for r in metrics)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (key, name) in zip(axes.flat, panels):
        for prefix, color in (("train", BLUE), ("val", ORANGE)):
            pts = [(r["epoch"], r[f"{prefix}_{key}"]) for r in metrics if r.get(f"{prefix}_{key}") is not None]
            if pts:
                ax.plot(*zip(*pts), color=color, linewidth=2, marker="o", markersize=4, label=prefix)
        if decoder_epoch:
            ax.axvline(decoder_epoch, color=MUTED, alpha=.5, linestyle="--", linewidth=1)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle(title + ("; dashed = decoder starts" if decoder_epoch else ""),
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(REPORT / "loss_curves.png", dpi=160, facecolor="white")
    plt.close(fig)


def success_figure(table):
    series = [("frozen", "This run, frozen", BLUE), ("adajepa", "This run, AdaJEPA", ORANGE),
              ("released", "Released ckpt, frozen", AQUA)]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width, gap = 0.26, 0.02
    x = np.arange(len(SHAPES))
    for i, (key, label, color) in enumerate(series):
        vals = [table.get((s, key), np.nan) for s in SHAPES]
        ax.bar(x + (i - 1) * (width + gap), [v * 100 for v in vals], width, color=color, label=label)
    ax.axvline(3.5, color=MUTED, linewidth=1, linestyle="--")
    ax.text(1.5, 101, "seen shapes", ha="center", va="bottom", color=MUTED, fontsize=9)
    ax.text(5.0, 101, "unseen shapes", ha="center", va="bottom", color=MUTED, fontsize=9)
    ax.set_xticks(x, SHAPES)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Success rate (%)", color=MUTED, fontsize=9)
    style(ax)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.16))
    ax.set_title("Planning success, 50 paired episodes per shape", fontsize=11, color=INK, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(REPORT / "success_by_shape.png", dpi=160, facecolor="white")
    plt.close(fig)


def iteration_figure(iters, metrics, its_per_epoch, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = [(r["epoch"] - 1) + r["iter"] / its_per_epoch for r in iters]
    for ax, key, name in zip(axes, ["z_loss", "z_visual_loss", "z_proprio_loss"],
                             ["Latent prediction MSE", "Visual latent MSE", "Proprio latent MSE"]):
        ax.plot(x, [r[f"train_{key}"] for r in iters], color=BLUE, linewidth=2, marker="o", markersize=3,
                label="train (per save interval)")
        ax.plot([r["epoch"] for r in metrics], [r[f"val_{key}"] for r in metrics], color=ORANGE, linewidth=2,
                marker="o", markersize=5, label="val (epoch end)")
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(REPORT / "iteration_curves.png", dpi=160, facecolor="white")
    plt.close(fig)


def transient_rise(iters, key="train_z_visual_loss", threshold=1.2):
    """Largest rise of the windowed training loss above its running minimum,
    and whether the loss later fell below that minimum again."""
    def ratios(k_key):
        return [(k, iters[k][k_key] / min(q[k_key] for q in iters[:k])) for k in range(1, len(iters))]
    over = [(k, r) for k, r in ratios(key) if r > threshold]
    if not over:
        return None
    k, ratio = max(over, key=lambda x: x[1])
    low = min(q[key] for q in iters[:k])
    rec = next((j for j in range(k + 1, len(iters)) if iters[j][key] < low), None)
    proprio_key = key.replace("visual", "proprio")
    proprio_ratio = dict(ratios(proprio_key)).get(k) if proprio_key in iters[k] else None
    return {"ratio": ratio, "epoch": iters[k]["epoch"], "iter": iters[k]["iter"], "low": low, "peak": iters[k][key],
            "recovered_at": (iters[rec]["epoch"], iters[rec]["iter"]) if rec is not None else None,
            "later_rise": rec is not None and any(j > rec for j, _ in over),
            "proprio_ratio": proprio_ratio}


def training_drift(metrics):
    """Late-epoch rise of the *training* visual latent MSE (instability, not overfitting)."""
    key = "train_z_visual_loss"
    pts = [(r["epoch"], r[key]) for r in metrics if r.get(key) is not None]
    if len(pts) < 3:
        return None
    best_ep, best = min(pts, key=lambda x: x[1])
    last_ep, last = pts[-1]
    if best_ep == last_ep or last < 1.2 * best:
        return None
    return {"best_epoch": best_ep, "best": best, "last_epoch": last_ep, "last": last}


def build_report():
    REPORT.mkdir(parents=True, exist_ok=True)
    metrics = sorted({r["epoch"]: r for r in rows(MODEL / "metrics.jsonl")}.values(), key=lambda r: r["epoch"])
    config = yaml.safe_load((MODEL / "hydra.yaml").read_text()) if (MODEL / "hydra.yaml").exists() else {}
    if config:
        (REPORT / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    gen = json.loads((DATA / "generation_config.json").read_text())
    n_episodes = len(gen["shapes"].split(",")) * (gen["train_per_shape"] + gen["val_per_shape"])
    has_decoder = bool(config.get("has_decoder", True)) and bool(config.get("model", {}).get("train_decoder", True))
    n_gpu = max(1, int(config.get("effective_batch_size", 1)) // max(1, int(config.get("gpu_batch_size", 1))))
    if metrics:
        keys = ["epoch"] + sorted(set().union(*(r.keys() for r in metrics)) - {"epoch"})
        with (REPORT / "losses.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)
        loss_figure(metrics, f"PushObj (pusht_noise replay, {n_episodes:,} episodes): losses per epoch",
                    config.get("training", {}).get("decoder_start_epoch", 1) if has_decoder else None)

    evaluations, outcomes, adaptation_losses = [], {}, []
    for shape in SHAPES:
        for arm in ("frozen", "adajepa"):
            d = EVAL / shape / arm
            log = final_log(d)
            if log is None:
                continue
            outcomes[(shape, arm)] = load_outcome(d)
            success = outcomes[(shape, arm)]["successes"].reshape(-1)
            n, k = len(success), int(success.sum())
            assert n == 50 and abs(k/n - log["final_eval/success_rate"]) < 1e-8
            lo, hi = wilson(k, n)
            evaluation = {"shape": shape, "split": "seen" if shape in SEEN else "unseen", "planner": arm,
                          "n": n, "successes": k, "success_rate": k/n, "ci95_low": lo, "ci95_high": hi, **log}
            if (d / "run_metadata.json").exists():
                evaluation.update(json.loads((d / "run_metadata.json").read_text()))
            evaluations.append(evaluation)
            if arm == "adajepa" and (d / "adajepa_losses.csv").exists():
                with (d / "adajepa_losses.csv").open() as f:
                    adaptation_losses += [{"shape": shape, **row} for row in csv.DictReader(f)]
    if evaluations:
        keys = sorted(set().union(*(r.keys() for r in evaluations)))
        with (REPORT / "evaluation.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(evaluations)
    if adaptation_losses:
        with (REPORT / "adaptation_losses.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(adaptation_losses[0]))
            writer.writeheader()
            writer.writerows(adaptation_losses)

    paired = {}
    for shape in SHAPES:
        if all((shape, arm) in outcomes for arm in ("frozen", "adajepa")):
            a, b = outcomes[(shape, "frozen")], outcomes[(shape, "adajepa")]
            np.testing.assert_array_equal(a["state_0"], b["state_0"])
            np.testing.assert_allclose(a["state_g"], b["state_g"], rtol=0, atol=1e-5)
            sa, sb = a["successes"].reshape(-1), b["successes"].reshape(-1)
            paired[shape] = {"both_success": int((sa & sb).sum()), "frozen_only": int((sa & ~sb).sum()),
                             "adajepa_only": int((~sa & sb).sum()), "both_fail": int((~sa & ~sb).sum())}

    # Same 50 targets per shape: released checkpoint (frozen) and the random-walk run.
    released, previous = {}, {}
    for shape in SHAPES:
        d = RELEASED / shape
        log = final_log(d)
        if log is not None:
            if (shape, "frozen") in outcomes:
                ref = load_outcome(d)
                np.testing.assert_array_equal(ref["state_0"], outcomes[(shape, "frozen")]["state_0"])
            released[shape] = log["final_eval/success_rate"]
    if (PREVIOUS / "evaluation.csv").exists():
        with (PREVIOUS / "evaluation.csv").open() as f:
            for r in csv.DictReader(f):
                previous[(r["shape"], r["planner"])] = int(r["successes"])

    complete = (len(metrics) == EPOCHS and metrics[-1]["epoch"] == EPOCHS and len(evaluations) == 2 * len(SHAPES))
    status = {"complete": complete, "epochs_completed": len(metrics), "evaluations_completed": len(evaluations),
              "updated_utc": datetime.now(timezone.utc).isoformat(), "paired_results": paired}
    (REPORT / "report_status.json").write_text(json.dumps(status, indent=2))

    rate = {(r["shape"], r["planner"]): r["success_rate"] for r in evaluations}
    table = {**rate, **{(s, "released"): v for s, v in released.items()}}
    if evaluations:
        success_figure(table)

    def total(arm, subset, source=None):
        if source is None:
            ks = [r["successes"] for r in evaluations if r["planner"] == arm and r["shape"] in subset]
            return sum(ks) / (50 * len(ks)) if len(ks) == len(subset) else None
        if source == "released":
            vs = [released[s] for s in subset if s in released]
            return sum(vs) / len(vs) if len(vs) == len(subset) else None
        ks = [previous.get((s, arm)) for s in subset]
        return sum(ks) / (50 * len(ks)) if None not in ks else None

    audit_path = REPORT / "dataset_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    tr = config.get("training", {})
    pr = config.get("predictor", {})

    L = [f"# PushObj 重新训练与评估报告（pusht_noise 回放数据 {n_episodes:,} 条，{EPOCHS} 轮）", "",
         f"状态：{'训练与评估已全部完成' if complete else '进行中，下文只包含已完成部分的实测结果'}。"
         f"训练 {len(metrics)}/{EPOCHS} 轮，评估 {len(evaluations)}/{2 * len(SHAPES)} 组。"
         f"生成时间（UTC）：{status['updated_utc'][:19]}。", ""]

    # ---- conclusions (only once everything is in) ----
    if complete:
        best = min(metrics, key=lambda r: r["val_z_loss"])
        last = metrics[-1]
        f_all, a_all = total("frozen", SHAPES), total("adajepa", SHAPES)
        b_all = sum(p["adajepa_only"] for p in paired.values())
        c_all = sum(p["frozen_only"] for p in paired.values())
        L += ["## 主要结论", "",
              f"- 本次模型在 7 种形状、350 个目标上：冻结规划成功率 **{f_all:.1%}**，AdaJEPA **{a_all:.1%}**"
              f"（{(a_all - f_all) * 100:+.1f} 个百分点；配对后只有 AdaJEPA 成功 {b_all} 例、只有冻结成功 {c_all} 例，"
              f"exact McNemar p={mcnemar_exact(b_all, c_all):.3g}）。"]
        for label, subset in (("已见形状 T/L/Z/+", SEEN), ("未见形状 I/small_tee/square", SHAPES[4:])):
            L.append(f"- {label}：冻结 {total('frozen', subset):.1%}，AdaJEPA {total('adajepa', subset):.1%}。")
        prev_f, prev_a = total("frozen", SHAPES, "previous"), total("adajepa", SHAPES, "previous")
        rel = total(None, SHAPES, "released")
        if prev_f is not None:
            L.append(f"- 同一批目标上，上一轮（{PREVIOUS_LABEL}）冻结 {prev_f:.1%} / AdaJEPA {prev_a:.1%}；"
                     f"发布检查点冻结 {rel:.1%}。")
        L += [(f"- 验证潜在预测 MSE 最低出现在第 {best['epoch']} 轮（{best['val_z_loss']:.5f}），"
               f"第 {last['epoch']} 轮为 {last['val_z_loss']:.5f}。" if best["epoch"] != last["epoch"] else
               f"- 验证潜在预测 MSE 逐轮下降，最后一轮最低（{last['val_z_loss']:.5f}）。")
              + f"正式评估固定使用第 {EPOCHS} 轮，没有按测试结果挑选检查点。",
              "- 所有数字来自单个训练种子、每种形状 50 个目标；95% 区间约为 ±13 个百分点，单个形状之间的差异要谨慎解读。"]
        by_rate = {(r["shape"], r["planner"]): r["success_rate"] for r in evaluations}
        gain_t = by_rate[("T", "adajepa")] - by_rate[("T", "frozen")]
        gain_sq = by_rate[("square", "adajepa")] - by_rate[("square", "frozen")]
        L.append(f"- 与论文同协议（最多 20 次重规划）的超参数消融相比：T 上 AdaJEPA 提升 {gain_t * 100:+.0f} 个百分点"
                 f"（论文约 +31 到 +38），square 上 {gain_sq * 100:+.0f} 个百分点（论文约 +15 到 +31）。"
                 + ("冻结模型已接近发布检查点，但在线适应带来的提升明显小于论文。" if gain_t < 0.15 else ""))
        drift = training_drift(metrics)
        if drift:
            L.append(f"- 训练后期不稳定：训练集视觉潜在 MSE 在第 {drift['best_epoch']} 轮最低（{drift['best']:.4f}），"
                     f"到第 {drift['last_epoch']} 轮升到 {drift['last']:.4f}（{drift['last'] / drift['best']:.1f} 倍）。"
                     f"训练损失本身在涨，不是单纯过拟合；第 {EPOCHS} 轮的成功率很可能低估了这份数据能达到的水平，详见“训练后期不稳定”。")
        L.append("")

    # ---- data ----
    L += ["## 训练数据", "",
          "按 AdaJEPA 论文附录 A.2 的方法生成（脚本 `scripts/gen_pushobj_dataset.py`）：",
          "从 DINO-WM 的 PushT 训练轨迹 `pusht_noise`（18,685 条带噪声的专家轨迹）出发，把 T 换成其他形状，从源轨迹的初始状态回放同一组动作。",
          f"- 源轨迹：前 {gen['train_pool']} 条作训练池，其余作为留出池（论文中用于测试），本次验证集取自留出池并只保留有接触的轨迹。",
          "- 回放保真度：形状为 T 时生成的状态与源轨迹逐帧误差为 0.000 px，说明仿真环境与 DINO-WM 一致。",
          "- 接触偏置（论文未给出公式，这是本仓库的实现）：把新形状的重心对齐到 T 在初始位姿下的重心；"
          f"若仍无接触，在该位置加小幅随机扰动重试（最多 {gen['contact_retries']} 次，σ={gen['jitter_px']}px / {gen['jitter_rad']}rad）。",
          f"- 边界处理：agent 是运动学刚体，会穿墙离开画面；剔除了 agent 任意时刻离开 [0,512] 的源轨迹（{gen['source_excluded_agent_leaves_frame']} 条，约 2.9%）。",
          f"- 规模：{'/'.join(gen['shapes'].split(','))} 每种 {gen['train_per_shape']} 条训练 + {gen['val_per_shape']} 条验证。"
          "论文 shape-shift 主实验为每种 4k 条、训练 3 轮；发布检查点配置为 5 轮。"]
    if "max_penetration" in gen:
        L.append(f"- 质量门槛：初始状态要求 agent 与物体穿透不超过 {gen['max_penetration']}px、物体顶点全部在墙内，"
                 "不合法时重新放置，仍不合法则换一条源轨迹；过程中推物体撞墙属于环境真实物理（作者评估数据中比例相近），予以保留，"
                 f"只剔除单步位移超过 {gen['max_block_step']}px 的数值爆炸。")
    L.append("")
    if audit:
        t, v = audit["train"], audit["val"]
        L += ["| 数据检查 | 训练集 | 验证集 |", "|---|---:|---:|",
              f"| 轨迹 / 帧 | {t['episodes']} / {t['frames']:,} | {v['episodes']} / {v['frames']:,} |",
              f"| 轨迹长度（最小/中位/最大） | {t['length_quantiles'][0]:.0f} / {t['length_quantiles'][2]:.0f} / {t['length_quantiles'][4]:.0f} | "
              f"{v['length_quantiles'][0]:.0f} / {v['length_quantiles'][2]:.0f} / {v['length_quantiles'][4]:.0f} |",
              f"| 训练窗口数 | {t['slices']:,} | {v['slices']:,} |",
              f"| agent 出画面的帧 | {t['agent_out_of_frame_frames']} | {v['agent_out_of_frame_frames']} |",
              f"| 有接触的轨迹 | {t['episodes_with_contact']:.0%} | {v['episodes_with_contact']:.0%} |",
              f"| 物体首尾位移 >5px | {t['block_moved_gt5px_fraction']:.1%} | {v['block_moved_gt5px_fraction']:.1%} |",
              f"| agent 位置标准差 (px) | {t['proprio_std'][0]:.0f} / {t['proprio_std'][1]:.0f} | {v['proprio_std'][0]:.0f} / {v['proprio_std'][1]:.0f} |",
              *([f"| 不合法初始状态 | {t['invalid_start_states']} | {v['invalid_start_states']} |"] if "invalid_start_states" in t else []),
              "", "对比上一轮随机游走数据：agent 出画面帧占 36.6%，位置标准差约 215px；"
              "仓库固定的归一化常数为 101/87px。新数据与这些常数基本吻合。", "",
              "![训练数据示例](dataset_examples.png)", ""]

    # ---- training ----
    L += ["## 训练设置", "",
          "架构和优化器与发布检查点 `pushobj_shape_shift/hydra.yaml` 相同，只改了数据和轮数"
          + ("" if has_decoder else "，并关闭了 decoder（它只用于可视化，输入停止梯度，不影响世界模型；规划时需 decode_for_viz=false）")
          + "。完整配置见 `training_config.yaml`。", "",
          "| 项目 | 设置 |", "|---|---|",
          "| 编码器 | SmallResNetGeM，1×384 全局特征，从头训练 |",
          f"| 预测器 | ViT，{pr.get('depth')} 层，{pr.get('heads')} heads，MLP {pr.get('mlp_dim')}，dropout {pr.get('dropout')} |",
          f"| 时间 | frameskip={config.get('frameskip')}，num_hist={config.get('num_hist')}，num_pred={config.get('num_pred')} |",
          f"| 优化 | batch {tr.get('batch_size')}，{tr.get('epochs')} 轮，{tr.get('mixed_precision')}；encoder lr {tr.get('encoder_lr')}，"
          f"predictor/action lr {tr.get('predictor_lr')}"
          + (f"，decoder lr {tr.get('decoder_lr')}（第 {tr.get('decoder_start_epoch')} 轮起）" if has_decoder else "，无 decoder") + " |",
          f"| 正则 | 曲率 {tr.get('straighten')}，stop_grad={tr.get('stop_grad')}，VCReg 关闭 |",
          f"| 硬件 | {n_gpu} 张 RTX 3090，每卡 batch {config.get('gpu_batch_size')}（有效 batch {config.get('effective_batch_size')}），seed {tr.get('seed')} |", ""]
    if metrics:
        L += [("总损失 = 潜在预测 MSE + 0.01×曲率 +（第 3 轮起）解码器重建损失；解码器输入停止梯度，只更新解码器本身。"
               "因此第 3 轮前后的总损失不可直接比较，看潜在预测 MSE 更合适。") if has_decoder else
              "总损失 = 潜在预测 MSE + 0.01×曲率。", "",
              "![损失曲线](loss_curves.png)", "",
              "| Epoch | Train total | Val total | Train latent | Val latent | Val visual | Val proprio | Val curvature | Val recon | 分钟 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        fields = ["train_loss", "val_loss", "train_z_loss", "val_z_loss", "val_z_visual_loss", "val_z_proprio_loss",
                  "val_curvature_loss_used_for_training", "val_decoder_recon_loss_reconstructed"]
        for r in metrics:
            cells = [f"{r[k]:.5f}" if r.get(k) is not None else "—" for k in fields]
            L.append(f"| {r['epoch']} | " + " | ".join(cells) + f" | {r['epoch_seconds'] / 60:.1f} |")
        L += ["", f"训练与逐轮验证合计 {sum(r['epoch_seconds'] for r in metrics) / 3600:.2f} 小时。", ""]
        iters = rows(MODEL / "metrics_iter.jsonl")
        if iters:
            batch = int(config.get("effective_batch_size", tr.get("batch_size", 64)))
            its_per_epoch = math.ceil(audit["train"]["slices"] / batch) if audit else max(r["iter"] for r in iters)
            iteration_figure(iters, metrics, its_per_epoch,
                             f"Training loss every {tr.get('save_every_x_iterations')} iterations "
                             f"({its_per_epoch:,} iterations per epoch) and validation loss per epoch")
            L += ["### 轮内训练曲线", "",
                  f"每 {tr.get('save_every_x_iterations')} 次迭代记录一次这段区间内的平均训练损失（`metrics_iter.jsonl`），"
                  f"并保存检查点 `model_e<轮>_it<迭代>.pth`；每轮 {its_per_epoch:,} 次迭代。", "",
                  "![轮内训练曲线](iteration_curves.png)", ""]
            rise = transient_rise(iters)
            if rise:
                if rise["recovered_at"]:
                    rec = (f"之后在第 {rise['recovered_at'][0]} 轮第 {rise['recovered_at'][1]:,} 次迭代降回该最低值以下，"
                           + ("此后又出现过超过 20% 的回升" if rise["later_rise"] else "此后没有再出现超过 20% 的回升"))
                else:
                    rec = "之后没有恢复到回升前的水平"
                pr = rise["proprio_ratio"]
                pr_txt = ("" if pr is None else
                          f"同一时刻 proprio 潜在 MSE 为其此前最低值的 {pr:.2f} 倍"
                          + ("，没有同步回升。" if pr < 1.1 else "，也同步回升。"))
                L += [f"训练集视觉潜在 MSE 在第 {rise['epoch']} 轮第 {rise['iter']:,} 次迭代附近升到 {rise['peak']:.4f}，"
                      f"是此前最低值 {rise['low']:.4f} 的 {rise['ratio']:.2f} 倍；{rec}。{pr_txt}"
                      "这段区间内保存的检查点不宜使用。", ""]
        drift = training_drift(metrics)
        if drift:
            L += ["### 训练后期不稳定", "",
                  f"训练集视觉潜在 MSE 在第 {drift['best_epoch']} 轮后开始回升，到第 {drift['last_epoch']} 轮达到最低值的 "
                  f"{drift['last'] / drift['best']:.1f} 倍；proprio 潜在 MSE 和解码器重建损失没有同步变差。"
                  "训练损失本身上升，说明问题出在优化而不是过拟合。"]
            diag_path = REPORT / "latent_scale_diagnostic.json"
            if diag_path.exists():
                diag = json.loads(diag_path.read_text())["epochs"]
                eps = sorted(int(e) for e in diag)
                L += ["", "为排除“特征尺度变大导致 MSE 变大”，检查了各轮编码器在 96 帧验证图像上的输出方差"
                      "（`latent_scale_diagnostic.json`）：", "",
                      "| Epoch | " + " | ".join(str(e) for e in eps) + " |", "|---|" + "---:|" * len(eps),
                      "| 样本间方差（384 维平均） | " + " | ".join(f"{diag[str(e)]['between_sample_var']:.3f}" for e in eps) + " |", "",
                      ]
                a_ep = drift["best_epoch"] if str(drift["best_epoch"]) in diag else eps[0]
                b_ep = drift["last_epoch"] if str(drift["last_epoch"]) in diag else eps[-1]
                L.append(f"同样从第 {a_ep} 轮到第 {b_ep} 轮，训练视觉潜在 MSE 变为 {drift['last'] / drift['best']:.1f} 倍，"
                         f"而编码器输出方差只变化 {(diag[str(b_ep)]['between_sample_var'] / diag[str(a_ep)]['between_sample_var'] - 1) * 100:+.0f}%，"
                         "所以不是特征尺度变大造成的，预测本身确实变差了。")
            L += ["", "可能原因是恒定学习率（predictor 5e-4）、无学习率调度、无梯度裁剪下的优化不稳定；这只是推测，本次没有做对照实验。"
                  "发布配置只训 5 轮、论文训 3 轮，或许与此有关。", ""]
        prev_losses = PREVIOUS / "losses.csv"
        if prev_losses.exists():
            with prev_losses.open() as f:
                old = [r for r in csv.DictReader(f) if r.get("val_z_loss")]
            if old:
                o = min(old, key=lambda r: float(r["val_z_loss"]))
                b = min(metrics, key=lambda r: r["val_z_loss"])
                L += [f"参考：本次最低验证潜在 MSE 为 {b['val_z_loss']:.5f}（第 {b['epoch']} 轮）；上一轮（{PREVIOUS_LABEL}）"
                      f"最低为 {float(o['val_z_loss']):.5f}（第 {o['epoch']} 轮）。两轮每轮迭代数和验证集都不同，只作量级参考。", ""]

    # ---- evaluation ----
    L += ["## 规划评估", "",
          "目标来自发布的 `data/pushobj_eval/val_<shape>/plan_targets.pkl`，每种形状 50 例，seed=100；"
          "冻结模型、AdaJEPA、发布检查点和上一轮都用完全相同的起点和目标。",
          "- 规划：GD（Adam，lr 0.1，100 步），horizon 25 环境步，每次执行 5 步，最多 20 次重规划，staged 目标（alpha=1，base=2）。",
          "- AdaJEPA：每次重规划后 1 步梯度更新（predictor lr 5e-4，encoder lr 1e-5），recent5 回放；"
          "只更新预测器最后一层及末端归一化，以及编码器 projection；每个样本开始前重置权重。",
          "- 成功标准：agent 与物体位置的 4 维距离 <20px，且角度差 <π/9。区间为 Wilson 95% 置信区间。", ""]
    if evaluations:
        L += ["![各形状成功率](success_by_shape.png)", "",
              "| Shape | 类别 | 冻结 | AdaJEPA | 冻结 95% CI | AdaJEPA 95% CI | 仅 AdaJEPA 成功 / 仅冻结成功 | 发布检查点冻结 | 上一轮 冻结 / AdaJEPA |",
              "|---|---|---:|---:|---|---|---:|---:|---:|"]
        by = {(r["shape"], r["planner"]): r for r in evaluations}
        for s in SHAPES:
            f, a = by.get((s, "frozen")), by.get((s, "adajepa"))
            cell = lambda r: f"{r['successes']}/50" if r else "待完成"
            ci = lambda r: f"{r['ci95_low']:.0%}–{r['ci95_high']:.0%}" if r else "—"
            pp = f"{paired[s]['adajepa_only']} / {paired[s]['frozen_only']}" if s in paired else "—"
            rel = f"{released[s]:.0%}" if s in released else "—"
            prev = (f"{previous[(s, 'frozen')]}/50 / {previous[(s, 'adajepa')]}/50"
                    if (s, "frozen") in previous and (s, "adajepa") in previous else "—")
            L.append(f"| {s} | {'已见' if s in SEEN else '未见'} | {cell(f)} | {cell(a)} | {ci(f)} | {ci(a)} | {pp} | {rel} | {prev} |")
        if complete:
            L += ["", "| 汇总 | 本次冻结 | 本次 AdaJEPA | 差值 | McNemar p | 发布检查点冻结 | 上一轮 冻结 / AdaJEPA |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
            for label, subset in (("已见形状", SEEN), ("未见形状", SHAPES[4:]), ("全部", SHAPES)):
                fr, ad = total("frozen", subset), total("adajepa", subset)
                b = sum(paired[s]["adajepa_only"] for s in subset)
                c = sum(paired[s]["frozen_only"] for s in subset)
                rel = total(None, subset, "released")
                pf, pa = total("frozen", subset, "previous"), total("adajepa", subset, "previous")
                L.append(f"| {label} | {fr:.1%} | {ad:.1%} | {(ad - fr) * 100:+.1f} pp | {mcnemar_exact(b, c):.3g} | "
                         f"{rel:.1%} | {pf:.1%} / {pa:.1%} |" if rel is not None and pf is not None else
                         f"| {label} | {fr:.1%} | {ad:.1%} | {(ad - fr) * 100:+.1f} pp | {mcnemar_exact(b, c):.3g} | — | — |")
            wall = {arm: [r.get("wall_seconds") for r in evaluations if r["planner"] == arm and r.get("wall_seconds")]
                    for arm in ("frozen", "adajepa")}
            if all(wall.values()):
                L += ["", f"耗时：冻结每种形状平均 {np.mean(wall['frozen']) / 60:.1f} 分钟（50 例一次批量规划），"
                      f"AdaJEPA 平均 {np.mean(wall['adajepa']) / 60:.0f} 分钟（逐例规划并适应）。"]
        L.append("")
    if adaptation_losses:
        L += ["### 在线适应损失", "",
              "每次更新前记录的 1 步潜在预测 MSE。第 1 次重规划与第 10 次及以后的对比可以看出适应是否在降低预测误差，"
              "但后期只剩尚未成功的样本，且回放缓冲区在变化，不能当作固定数据上的收敛曲线。", "",
              "| Shape | 更新次数 | 平均 | 第 1 次重规划 | 第 10 次及以后 |", "|---|---:|---:|---:|---:|"]
        for s in SHAPES:
            rs = [r for r in adaptation_losses if r["shape"] == s]
            if rs:
                losses = np.array([float(r["loss"]) for r in rs])
                assert np.isfinite(losses).all()
                first = [float(r["loss"]) for r in rs if int(r["mpc_iter"]) == 1]
                late = [float(r["loss"]) for r in rs if int(r["mpc_iter"]) >= 10]
                L.append(f"| {s} | {len(losses)} | {losses.mean():.4f} | {np.mean(first):.4f} | "
                         + (f"{np.mean(late):.4f}" if late else "—") + " |")
        L.append("")

    if complete:
        cfg_path = EVAL / "T" / "adajepa" / ".hydra" / "config.yaml"
        max_iter = yaml.safe_load(cfg_path.read_text())["planner"]["max_iter"] if cfg_path.exists() else None
        by = {(r["shape"], r["planner"]): r["success_rate"] for r in evaluations}
        ours = {"seen": (total("frozen", SEEN), total("adajepa", SEEN)),
                "unseen": (total("frozen", SHAPES[4:]), total("adajepa", SHAPES[4:])),
                **{s: (by[(s, "frozen")], by[(s, "adajepa")]) for s in SHAPES}}
        L += ["## 与论文数字对照", "",
              f"本次评估最多 {max_iter} 次重规划、每种形状 50 个目标、单个训练种子。论文数字来自不同的平均方式和协议（见最后一列），"
              "只能作为量级参考；冻结一侧可以用同一协议下的发布检查点做严格对照（见上表）。", "",
              "| 指标 | 论文冻结 | 论文 AdaJEPA | 本次冻结 | 本次 AdaJEPA | 论文数字的条件 |", "|---|---:|---:|---:|---:|---|"]
        for label, key, pf, pa, note in PAPER:
            f, a = ours[key]
            L.append(f"| {label} | {pf} | {pa} | {f:.0%} | {a:.0%} | {note} |")
        L.append("")

    L += ["## 局限", "",
          "- 单个训练种子、每种形状 50 个目标，没有重复实验。",
          "- 接触偏置的具体做法是本仓库的实现，论文没有给出细节。"
          + (f"数据规模（每种 {gen['train_per_shape']} 条）比论文主实验（每种 4k 条）小约 {4000 / gen['train_per_shape']:.0f} 倍。"
             if gen["train_per_shape"] < 3000 else f"数据规模每种 {gen['train_per_shape']} 条，与论文主实验（每种 4k 条）相当。"),
          "- 发布的评估目标来自作者自己的测试集。它们不是 pusht_noise 的逐帧回放（动作序列无法精确匹配），"
          "所以无法严格排除本次训练源轨迹与作者测试集源轨迹之间的重叠。",
          *([f"- 正式评估固定用第 {EPOCHS} 轮；训练后期不稳定，这个检查点不是本次训练中最好的模型。"]
            if metrics and training_drift(metrics) else []),
          "", "## 建议的下一步", ""]
    best_ep = min(metrics, key=lambda r: r["val_z_loss"])["epoch"] if metrics else EPOCHS
    if best_ep != EPOCHS:
        L.append(f"- 验证损失最低的是第 {best_ep} 轮，可以对它补一次冻结评估（4 卡约 5 分钟），确认早期检查点是否规划得更好。")
    if metrics and training_drift(metrics):
        L.append("- 训练稳定性：加学习率衰减或梯度裁剪，或者按验证损失早停后再评估。")
    if complete and gain_t < 0.15:
        L.append("- 在 T 和 square 上用发布检查点跑同协议的 AdaJEPA（2 卡约 1 小时）：如果发布检查点的提升也远小于论文，"
                 "差距来自评估协议或实现；如果它能达到论文幅度，问题在本次模型对在线适应的响应上。")
    if gen["train_per_shape"] < 3000:
        L.append(f"- 数据规模：论文主实验每种形状 4k 条，本次 {gen['train_per_shape']} 条；数据量是与发布检查点差距的另一个候选原因。")
    L.append("- 多训练种子（至少 3 个）重复，才能对小于 ±13 个百分点的差异下结论。")
    L += ["",
          "## 文件与复现", "",
          f"- 数据：`{display_path(DATA)}`（审计 `dataset_audit.json`）；模型：`{display_path(MODEL)}`（第 {EPOCHS} 轮 `checkpoints/model_{EPOCHS}.pth`）；"
          f"评估输出：`{display_path(EVAL)}/<shape>/<frozen|adajepa>`。",
          "- 本目录：`losses.csv`、`evaluation.csv`、`adaptation_losses.csv`、`training_config.yaml`、`pipeline_status.json`。", "",
          "```bash", "conda activate ts",
          f"python scripts/gen_pushobj_dataset.py --out-dir {gen['out_dir']} --train-per-shape {gen['train_per_shape']} "
          f"--val-per-shape {gen['val_per_shape']} --n-workers {gen['n_workers']}",
          (f"WANDB_MODE=offline SDL_VIDEODRIVER=dummy accelerate launch --multi_gpu --num_processes {n_gpu} "
           f"--mixed_precision bf16 train.py --config-name {TRAIN_CONFIG}") if n_gpu > 1 else
          f"WANDB_MODE=offline SDL_VIDEODRIVER=dummy python train.py --config-name {TRAIN_CONFIG}",
          f"python scripts/run_pushobj_evaluation.py --epochs {EPOCHS} --model {display_path(MODEL)} --data {display_path(DATA)} "
          f"--report {display_path(REPORT)} --eval-dir {display_path(EVAL)} --gpus <ids>", "```"]
    (REPORT / "report.md").write_text("\n".join(L) + "\n")
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_args(p)
    a = p.parse_args()
    configure_from_args(a)
    print(json.dumps(build_report(), indent=2))
