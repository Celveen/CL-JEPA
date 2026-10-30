"""Shared data loading and style for the shape-shift paper figures (scripts/fig_shape_*.py).

A run root is one chain of the five-round shape stream for one planner, laid out by scripts/cont_run_stream.py:
<root>/<arm>/<chunk>/ with arm cls6a_free (ours), w0 (AdaJEPA) and frozen (scripts/cont_frozen_chunks.sh).
Every chunk is 50 episodes of one shape; a round is the 7 shapes once. Chains are given as
--run planner:stream=root (stream = the 5-round manifest under data/cont_streams/); without --run the defaults
below are used and missing roots are skipped, so the same command works on a machine that has only some chains.

The style follows the visual-shift figures: GD blue / CEM green, Frozen dashed, AdaJEPA dash-dot, Ours solid.
"""
import json
import pickle
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ["I", "L", "square", "Z", "small_tee", "+", "T"]  # stream order
SHAPE_LABEL = {"small_tee": "smallT"}
SEEN = {"T", "L", "Z", "+"}  # training shapes of pushobj_bias_16k
ARMS = {"frozen": "Frozen", "w0": "AdaJEPA", "cls6a_free": "Ours"}  # figure labels: no internal arm names
PLANNER = {"gd": "GD", "cem": "CEM"}
COLOR = {"gd": "#6b86c8", "cem": "#7cb98b"}
LS = {"frozen": (0, (4, 2.5)), "w0": (0, (5, 2, 1, 2)), "cls6a_free": "-"}
ROW_COLOR = {"frozen": "#2c63d6", "w0": "#e2382c", "cls6a_free": "#1f8a4c"}  # rollout rows
DEFAULT_RUNS = [
    ("gd", "S7t5", "eval_outputs/cont_S7t3"), ("gd", "S7t5s400", "eval_outputs/cont_S7t3s400"),
    ("gd", "S7t5s700", "eval_outputs/cont_S7t3s700"),
    ("cem", "S7t5", "eval_outputs/cont_S7t3_cem"), ("cem", "S7t5s400", "eval_outputs/cont_S7t3s400_cem"),
    ("cem", "S7t5s700", "eval_outputs/cont_S7t3s700_cem"),
]
MAX_STEPS = 30


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.linewidth": 0.7,
        "axes.edgecolor": "#444444", "grid.color": "#b8b8b8", "grid.linestyle": ":", "grid.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def save(fig, out, name):
    """out/<name>.pdf (no creation date, so an unchanged redraw is byte-identical) and out/<name>.png."""
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out / name}.pdf")


def parse_runs(specs):
    """--run planner:stream=root entries -> [(planner, stream, Path)]; defaults when none are given."""
    runs = []
    for s in specs or []:
        key, root = s.split("=", 1)
        planner, stream = key.split(":")
        runs.append((planner, stream, Path(root)))
    if not specs:
        runs = [(p, s, Path(r)) for p, s, r in DEFAULT_RUNS]
    return [(p, s, r if r.is_absolute() else ROOT / r) for p, s, r in runs]


def blocks(stream):
    return json.loads((ROOT / "data/cont_streams" / stream / "manifest.json").read_text())["blocks"]


def round_chunks(stream, rnd):
    """{shape: chunk name} of round rnd (1-based)."""
    bs = blocks(stream)
    per = len(SHAPES)
    return {b["shape"]: b["chunk"] for b in bs[(rnd - 1) * per: rnd * per]}


def finished(d):
    logs = d / "logs.json"
    return (d / "final_eval_results.npz").exists() and logs.exists() and "final_eval/success_rate" in logs.read_text()


def per_iter(d):
    """{sample_idx: [records by step]} from per_iter_logs.json (one JSON object per line)."""
    out = {}
    for line in (d / "per_iter_logs.json").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault(int(r["sample_idx"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["step"])
    return out


def success_by_step(d):
    """(n_episodes, MAX_STEPS + 1) 0/1: success reached at or before each MPC step (step 0 = start).

    The final evaluation (final_eval_results.npz) is the ground truth for the end point; per_iter_logs gives the
    step at which an episode first succeeded."""
    final = np.load(d / "final_eval_results.npz")["successes"].astype(bool)
    first = np.full(len(final), MAX_STEPS + 1)
    logs = per_iter(d) if (d / "per_iter_logs.json").exists() else {}
    for i, recs in logs.items():
        hit = [r["step"] for r in recs if r.get("success")]
        if hit:
            first[i] = min(hit)
    if not logs:  # no per-step log: only the end point is known
        first[final] = MAX_STEPS
    first[final & (first > MAX_STEPS)] = MAX_STEPS
    steps = np.arange(MAX_STEPS + 1)
    return (first[:, None] <= steps[None, :]).astype(float)


def shape_name(s):
    return SHAPE_LABEL.get(s, s)


def load_env(shape):
    """A PushObj env (the checkpoint's env kwargs) set to one shape; used for thumbnails and action replay."""
    from omegaconf import OmegaConf

    from env.pusht.pusht_wrapper import PushTWrapper

    cfg = OmegaConf.load(ROOT / "checkpoints/pushobj_bias_16k_e3_seed0/hydra.yaml")
    kw = dict(cfg.env.kwargs) if cfg.env.kwargs else {}
    env = PushTWrapper(**kw)
    env.update_env({"shape": shape})
    return env


_THUMBS = {}


def thumbnail(shape, state=(360, 318, 340, 272, 0.25, 0, 0), crop=0.15):
    """Rendered scene of a shape (fixed green target, grey object to its right, blue agent), lightly cropped."""
    if shape not in _THUMBS:
        env = load_env(shape)
        obs, _ = env.prepare(0, np.array(state, dtype=np.float64))
        img = obs["visual"]
        h = img.shape[0]
        c = int(h * crop)
        _THUMBS[shape] = img[c: h - c, c + int(h * 0.04): h - c + int(h * 0.04)]
    return _THUMBS[shape]


def chunk_targets(d):
    with open(d / "plan_targets.pkl", "rb") as f:
        return pickle.load(f)
