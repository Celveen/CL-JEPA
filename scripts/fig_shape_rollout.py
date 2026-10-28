"""Shape shifts: one episode side by side for Frozen / AdaJEPA / Ours, with each method's latent prediction loss
per MPC step. Counterpart of the visual-shift rollout pages; one page per (shape, planner).

All three methods plan the same target (sample --sample of the shape's chunk in round --round). Frames come from
replaying each method's executed actions (final_eval_results.npz) in the simulator from the shared start state;
the replay is checked against the logged success. Columns are MPC steps 0..13 and 30, then the goal; after an
episode terminates its cells hold the terminal frame, greyed.

The loss is the replay-buffer prediction MSE the planner logs before each adaptation step: teacher-forced 1-step
latent prediction over the up-to-five most recent transitions, visual + proprioceptive channels (action channels
excluded). AdaJEPA and Ours log it with their current weights (per_iter_logs.json, loss_buf_cur). Frozen does not
adapt and logs nothing, so its curve is the same quantity computed here with the pretrained weights on its replayed
transitions; --check verifies that computation against AdaJEPA's logged loss under the pretrained weights
(loss_buf_init).

    CUDA_VISIBLE_DEVICES=7 python scripts/fig_shape_rollout.py --planner cem --root eval_outputs/cont_S7t3_cem
    # -> reports/figs_shape/rollout_cem/<shape>_cem.{pdf,png} and rollout_cem.pdf (all shapes, one page each)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fig_shape_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

METHODS = ["frozen", "w0", "cls6a_free"]
COLS = list(range(14)) + [30]
FRAMESKIP = 5
BUF = 5  # replay_buffer: recent5


# ---------------------------------------------------------------- model (Frozen's loss)
class Theta0:
    """Pretrained world model + the planner's preprocessing, for the replay-buffer loss of the Frozen rollout."""

    def __init__(self, ckpt="checkpoints/pushobj_bias_16k_e3_seed0", epoch=3):
        import hydra  # noqa: F401
        from omegaconf import OmegaConf

        from datasets.img_transforms import default_transform
        from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD, STATE_MEAN, STATE_STD
        from plan import load_model
        from preprocessor import Preprocessor

        cfg = OmegaConf.load(C.ROOT / ckpt / "hydra.yaml")
        tcfg = cfg.env.dataset.get("transform", None)
        transform = hydra.utils.instantiate(tcfg) if tcfg is not None else default_transform()
        self.pre = Preprocessor(action_mean=ACTION_MEAN, action_std=ACTION_STD, state_mean=STATE_MEAN[:5],
                                state_std=STATE_STD[:5], proprio_mean=PROPRIO_MEAN[:4], proprio_std=PROPRIO_STD[:4],
                                transform=transform)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.wm = load_model(C.ROOT / ckpt / "checkpoints" / f"model_{epoch}.pth", cfg, cfg.num_action_repeat, self.device)
        self.wm.eval()

    @torch.no_grad()
    def buffer_losses(self, frames, proprio, actions):
        """frames (L+1, H, W, C) and proprio (L+1, 4) at the model stride, actions (L, 10) normalized.
        Returns the loss after each step t = 1..L on the last min(t, BUF) transitions, as ProbeMPCPlanner._losses."""
        wm = self.wm
        obs_all = self.pre.transform_obs({"visual": frames[None], "proprio": proprio[None]})
        act_all = torch.as_tensor(actions, dtype=torch.float32)[None]
        drop = wm.action_dim
        out = []
        for t in range(1, len(actions) + 1):
            s = max(0, t - BUF)
            obs = {k: v[:, s: t + 1].to(self.device) for k, v in obs_all.items()}
            act = act_all[:, s:t].to(self.device)
            act = torch.cat([act, torch.zeros_like(act[:, :1])], dim=1)  # dummy action for the last frame
            z = wm.encode(obs, act)
            T = z.shape[1] - 1
            w = min(wm.num_hist, T)
            ls = []
            for i in range(T - w + 1):
                pred, tgt = wm.predict(z[:, i: i + w]), z[:, i + 1: i + 1 + w]
                ls.append(float(torch.mean((pred[..., :-drop] - tgt[..., :-drop]) ** 2)))
            out.append(float(np.mean(ls)))
        return np.array(out)


# ---------------------------------------------------------------- replay
def replay(env, pre, state_0, actions, n_steps):
    """Model-stride frames/proprio/states (n_steps + 1) from replaying the first n_steps MPC actions."""
    act = torch.as_tensor(actions[:n_steps], dtype=torch.float32)
    exec_act = rearrange(act[None], "b t (f d) -> b (t f) d", f=FRAMESKIP)
    exec_act = pre.denormalize_actions(exec_act).numpy()[0]
    obses, states = env.rollout(0, state_0, exec_act)
    idx = np.arange(0, n_steps * FRAMESKIP + 1, FRAMESKIP)
    return obses["visual"][idx], obses["proprio"][idx], states[idx]


def load_episode(d, k):
    res = np.load(d / "final_eval_results.npz")
    n = res["action_len"][k]  # MPC steps executed; inf when the episode never succeeded (all 30 steps run)
    n = int(n) if np.isfinite(n) else res["actions"].shape[1]
    logs = C.per_iter(d).get(k, []) if (d / "per_iter_logs.json").exists() else []
    return {"success": bool(res["successes"][k]), "n": n, "actions": res["actions"][k], "state_0": res["state_0"][k],
            "state_g": res["state_g"][k], "logs": logs}


# ---------------------------------------------------------------- drawing
def faded(img):
    g = np.full_like(img, 216)
    return (0.25 * img + 0.75 * g).astype(np.uint8)


def draw_page(shape, planner, sample, eps, frames, losses, goal_img):
    fig = plt.figure(figsize=(14.6, 5.9))
    n_cols = len(COLS) + 1
    left, right, top = 0.075, 0.985, 0.90
    cw = (right - left) / (n_cols + 0.25)
    row_h, gap = 0.155, 0.04
    fig.text(left, 0.965, f"{C.shape_name(shape)} / {C.PLANNER[planner]}", fontsize=13, va="center")
    fig.text(right, 0.965, f"Fixed sample {sample} | Simulator action replay", fontsize=10, ha="right", va="center",
             color="#444444")
    for r, m in enumerate(METHODS):
        y0 = top - (r + 1) * row_h - r * gap
        ep, fr = eps[m], frames[m]
        color = C.ROW_COLOR[m]
        for c, step in enumerate(COLS):
            x0 = left + c * cw
            ax = fig.add_axes([x0 + cw * 0.04, y0 + 0.01, cw * 0.92, row_h - 0.02])
            img = fr[min(step, ep["n"])]
            ax.imshow(img if step <= ep["n"] else faded(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#c8c8c8")
                sp.set_linewidth(0.6)
            if r == 0:
                ax.set_title(str(step), fontsize=10, pad=3)
            if c == 0:
                ax.text(0.0, 1.0, "Simulator", transform=ax.transAxes, fontsize=6.5, color="white", va="top",
                        ha="left", fontweight="bold", bbox=dict(boxstyle="square,pad=0.15", fc=color, ec="none"))
        fig.patches.append(Rectangle((left, y0), cw * len(COLS), row_h, transform=fig.transFigure, fill=False,
                                     ec=color, lw=1.4))
        fig.text(left - 0.012, y0 + row_h / 2, C.ARMS[m], rotation=90, ha="center", va="center", color=color,
                 fontsize=11, fontweight="bold")
        end = f"Success @ {ep['n']}" if ep["success"] else f"Failure @ {ep['n']}"
        fig.text(left + cw * len(COLS), y0 - 0.006, end, ha="right", va="top", fontsize=7, color=color)
        gx = left + len(COLS) * cw + cw * 0.25
        gax = fig.add_axes([gx + cw * 0.04, y0 + 0.01, cw * 0.92, row_h - 0.02])
        gax.imshow(goal_img)
        gax.set_xticks([])
        gax.set_yticks([])
        for sp in gax.spines.values():
            sp.set_color("#c8c8c8")
        gax.text(0.83, 0.13, "✓" if ep["success"] else "✕", transform=gax.transAxes, ha="center",
                 va="center", fontsize=9, fontweight="bold", color="#1f8a4c" if ep["success"] else "#e2382c",
                 bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#1f8a4c" if ep["success"] else "#e2382c",
                           lw=1.1))
        if r == 0:
            gax.set_title("Goal", fontsize=10, pad=3)
    # loss panel
    y_loss = top - 3 * row_h - 2 * gap - 0.25
    ax = fig.add_axes([left + 0.02, y_loss, cw * len(COLS) - 0.02, 0.19])
    ax.set_facecolor("#efefef")
    style = {"frozen": dict(ls="-", marker="o", ms=1.8), "w0": dict(ls=(0, (5, 2, 1, 2)), marker="o", ms=1.8),
             "cls6a_free": dict(ls="-")}
    for m in METHODS:
        y = losses[m]
        if y is not None and len(y):
            ax.plot(np.arange(1, len(y) + 1), y, color=C.ROW_COLOR[m], lw=1.4, label=C.ARMS[m], **style[m])
    ax.set_xlim(0, 30)
    ax.set_xticks(range(0, 31, 2))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(True, axis="y", color="white", ls="-", lw=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("MPC replanning steps")
    ax.set_ylabel("Latent prediction\nloss (MSE)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig.text(left + 0.02, y_loss - 0.115, "Prediction MSE on up to five recent transitions. Visual + proprioceptive "
             "latents; action dimensions excluded. Frozen: computed with the pretrained weights on its replayed "
             "transitions.", fontsize=8.5, color="#444444")
    fig.text(left + 0.02, y_loss - 0.145, "Gray frames hold the terminal state. Curves stop at termination.",
             fontsize=8.5, color="#444444")
    return fig


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["gd", "cem"], required=True)
    ap.add_argument("--root", required=True, help="chain root with frozen/, w0/, cls6a_free/")
    ap.add_argument("--stream", default="S7t5")
    ap.add_argument("--round", type=int, default=5)
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--shapes", nargs="*", default=C.SHAPES)
    ap.add_argument("--out", default="reports/figs_shape")
    ap.add_argument("--check", action="store_true", help="compare the Frozen loss computation with AdaJEPA's log")
    ap.add_argument("--list", action="store_true",
                    help="print each sample's outcome (F/A/O = Frozen/AdaJEPA/Ours, 1 success) and exit")
    a = ap.parse_args()
    C.setup_style()
    root = Path(a.root) if Path(a.root).is_absolute() else C.ROOT / a.root
    chunks = C.round_chunks(a.stream, a.round)
    if a.list:
        for shape in a.shapes:
            dirs = {m: root / m / chunks[shape] for m in METHODS}
            if not all(C.finished(d) for d in dirs.values()):
                print(f"{shape}: unfinished")
                continue
            s = {m: np.load(d / "final_eval_results.npz")["successes"].astype(int) for m, d in dirs.items()}
            only = [i for i in range(len(s["frozen"])) if s["cls6a_free"][i] and not s["w0"][i] and not s["frozen"][i]]
            print(f"{shape:9s} F/A/O per sample: " + " ".join(f"{i}:{s['frozen'][i]}{s['w0'][i]}{s['cls6a_free'][i]}"
                                                         for i in range(len(s["frozen"]))))
            print(f"{'':9s} Ours succeeds where both fail: {only}")
        return
    model = Theta0()
    out = C.ROOT / a.out / f"rollout_{a.planner}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    with PdfPages(out / f"rollout_{a.planner}.pdf", metadata={"CreationDate": None}) as pdf:
        for shape in a.shapes:
            dirs = {m: root / m / chunks[shape] for m in METHODS}
            missing = [m for m, d in dirs.items() if not C.finished(d)]
            if missing:
                print(f"skip {shape}: unfinished {missing}")
                continue
            env = C.load_env(shape)
            eps, frames, losses = {}, {}, {}
            s0 = None
            for m, d in dirs.items():
                ep = load_episode(d, a.sample)
                if s0 is None:
                    s0 = ep["state_0"]
                assert np.allclose(ep["state_0"], s0), f"{shape}: {m} planned a different start state"
                fr, pr, st = replay(env, model.pre, ep["state_0"], ep["actions"], ep["n"])
                ok = bool(env.eval_state(ep["state_g"], st[-1])["success"])
                if ok != ep["success"]:
                    print(f"WARNING {shape}/{m}: replay success {ok} != logged {ep['success']}")
                eps[m], frames[m] = ep, fr
                if m == "frozen":
                    losses[m] = model.buffer_losses(fr, pr, ep["actions"][: ep["n"]])
                else:
                    losses[m] = np.array([r["loss_buf_cur"] for r in ep["logs"]])[: ep["n"]]
                if a.check and m == "w0":
                    mine = model.buffer_losses(fr, pr, ep["actions"][: ep["n"]])
                    logged = np.array([r["loss_buf_init"] for r in ep["logs"]])[: ep["n"]]
                    print(f"check {shape}: max |offline - logged theta0 loss| = {np.abs(mine - logged).max():.2e} "
                          f"(logged mean {logged.mean():.4f})")
            goal, _ = env.prepare(0, eps["frozen"]["state_g"])
            fig = draw_page(shape, a.planner, a.sample, eps, frames, losses, goal["visual"])
            name = f"{C.shape_name(shape)}_{a.planner}"
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
            fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=150)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            meta[shape] = {m: {"success": eps[m]["success"], "steps": eps[m]["n"]} for m in METHODS}
            print(shape, meta[shape])
    (out / "rollout_meta.json").write_text(json.dumps({"round": a.round, "sample": a.sample, "episodes": meta},
                                                      indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
