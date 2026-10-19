#!/usr/bin/env python
"""Generate a PushObj training set following AdaJEPA (arXiv:2606.32026), App. A.2.

Paper recipe: start from DINO-WM's PushT training trajectories (`pusht_noise`:
expert demos replayed with noise), replace the T block with another shape,
replay the same action sequences, and add a contact bias because T-shaped
actions often miss the new geometry. 16k source trajectories are used for
training; the remaining ones are held out and, for testing, filtered to those
with at least one contact.

This script implements that recipe:

* Source: `<source>/train/{states,rel_actions,velocities}.pth`, `seq_lengths.pkl`
  from DINO-WM's pusht_noise (only these tensors are read; no videos). The
  first `--train-pool` (16000) episodes feed the train split; the rest are held
  out and feed the val split, which keeps only contact episodes (as the paper
  does for its test set).
* Replay: the env state is set to the source episode's first state and the
  source pixel-delta actions are replayed (env does agent_pos + action/100*100).
  For shape T this reproduces the source trajectory exactly.
* Contact bias (the paper gives no formula; this is our implementation): the new
  block is placed so its centre of gravity coincides with the T's centre of
  gravity at the source start pose (same angle), i.e. where the expert was
  pushing. If a replay still has no contact, it is retried with a small random
  jitter of that placement (up to `--contact-retries` times); if none makes
  contact the aligned replay is kept (training data is not filtered).
* Boundary handling: the agent is a kinematic body, so the walls do not stop it
  and it can leave the rendered frame. Source episodes in which the agent ever
  leaves [0, 512] (2.9% of pusht_noise train) are excluded from the pool.
* Action-level contact bias (optional, `--action-bias-px`, `--action-noise-px`):
  at every step the source action gets a push of b px toward the block's centre
  of gravity plus isotropic Gaussian noise of sigma px, and the *executed* action
  is what the dataset stores. Calibrated against the authors' released test
  segments (scripts/calibrate_contact_bias.py), whose actions deviate from their
  nearest pusht_noise window toward the block. Applied to every shape, T included.
  `--placement source` keeps the source block pose instead of the CoG alignment.
* Quality gate: a start state is accepted only if the agent penetrates the block
  by at most `--max-penetration` px and every block vertex is inside the walls.
  About 4% of pusht_noise T start states fail this even before any shape swap
  (agent centre inside the block, block overlapping a wall); the first physics
  step then ejects the block, which closed-loop evaluation never sees. Invalid
  placements are re-drawn with the same jitter as the contact bias; if none is
  valid the (shape, source) pair is dropped and another source takes its place.
  Pushing the block into a wall mid-episode is kept: it is real env dynamics and
  appears in the released eval targets at similar rates. Only numerical
  blow-ups (non-finite state or a block step above `--max-block-step` px; the
  released targets peak at 57 px) are rejected.

* Source budget (`--max-draws`): every (shape, source) pair is replayed once by
  default. When a split needs more episodes than its pool supplies (16k T-only
  episodes from the ~15.5k eligible train sources), `--max-draws 2` adds a second
  pass that replays sources again under a different job seed, so action noise and
  retries differ. First-pass episodes are taken first; repeats only fill the gap.

Output layout matches datasets/pusht_dset.py (and pusht_noise): per split,
`obses/episode_XXX.mp4`, zero-padded `states.pth` (N, T_max, 5),
`rel_actions.pth` (pixel deltas), `velocities.pth`, `seq_lengths.pkl`,
`shapes.pkl`; plus `provenance.json` recording each episode's source index,
block placement and contact count. state[t]/frame[t] precede action[t].
"""
import argparse
import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import sys

import imageio
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

WINDOW = 512
AGENT_RADIUS = 15.0
WALL_LO, WALL_HI = 7.0, 504.0  # wall segments at 5 / 506 px with radius 2
# Block centre of gravity in body-local coordinates (env/pusht/pusht_env.py add_*).
COG = {"T": (0.0, 45.0), "L": (30.0, 10.0), "Z": (20.0, 0.0), "+": (0.0, 0.0),
       "I": (0.0, 0.0), "small_tee": (0.0, 40.0), "square": (0.0, 0.0)}


class _PlainUnpickler(pickle.Unpickler):
    """seq_lengths.pkl is a list of numpy ints; refuse anything else."""
    ALLOWED = {("numpy.core.multiarray", "scalar"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"unexpected global {module}.{name}")


def load_source(root):
    root = Path(root) / "train"
    states = torch.load(root / "states.pth", weights_only=True).double().numpy()
    actions = torch.load(root / "rel_actions.pth", weights_only=True).double().numpy()
    vels = torch.load(root / "velocities.pth", weights_only=True).double().numpy()
    with open(root / "seq_lengths.pkl", "rb") as f:
        lengths = np.asarray(_PlainUnpickler(f).load(), dtype=int)
    return states, actions, vels, lengths


def cog_aligned(state, shape):
    """Move the block so `shape`'s centre of gravity sits where the T's was."""
    s = state.copy()
    c, si = np.cos(s[4]), np.sin(s[4])
    dx, dy = np.subtract(COG["T"], COG[shape])
    s[2] += c * dx - si * dy
    s[3] += si * dx + c * dy
    return s


_ENV = None


def _init_worker(render=True):
    global _ENV
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gym
    import env  # noqa: F401  # registers pushobj
    _ENV = gym.make("pushobj", with_velocity=True, with_target=True).unwrapped
    if not render:  # physics only (calibration); observations carry no image
        _ENV._render_frame = lambda mode: None


def _toward_block():
    agent = np.array(tuple(_ENV.agent.position))
    cog = np.array(tuple(_ENV.block.local_to_world(_ENV.block.center_of_gravity)))
    d = cog - agent
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.zeros(2)


def _replay(shape, init_state, actions, seed, bias_px=0.0, noise_px=0.0, noise_seed=0):
    """Replay `actions` (pixel deltas) with the optional action-level contact bias.
    Returns frames, states, contact steps and the executed actions (T, 2)."""
    _ENV.update_env({"shape": shape})
    obs, state = _ENV.prepare(seed, init_state)
    rs = np.random.RandomState(noise_seed)
    frames, states, executed, contact_steps = [obs["visual"]], [state], [], 0
    for t, a in enumerate(actions):
        a = np.asarray(a, dtype=np.float64)
        if bias_px or noise_px:
            a = a + bias_px * _toward_block() + noise_px * rs.randn(2)
        executed.append(a)
        if t == len(actions) - 1:
            break  # the last action's outcome is never stored
        obs, _, _, info = _ENV.step(a / 100.0)
        frames.append(obs["visual"])
        states.append(info["state"])
        contact_steps += int(info["n_contacts"] > 0)
    return frames, np.stack(states), contact_steps, np.stack(executed)


def _placement_ok(shape, init_state, seed, max_penetration):
    """Start state is physical: agent barely touches the block, block inside the walls."""
    _ENV.update_env({"shape": shape})
    _ENV.prepare(seed, init_state)
    gap = min(s.point_query(tuple(_ENV.agent.position)).distance for s in _ENV.block.shapes)
    verts = np.array([tuple(_ENV.block.local_to_world(v)) for s in _ENV.block.shapes for v in s.get_vertices()])
    return gap >= AGENT_RADIUS - max_penetration and verts.min() >= WALL_LO and verts.max() <= WALL_HI


def _run(job):
    (src_state, actions, shape, seed, retries, jitter_px, jitter_rad,
     max_penetration, max_block_step, video_path, fps, bias_px, noise_px, placement) = job
    rs = np.random.RandomState(seed)
    base = cog_aligned(src_state, shape) if placement == "cog" else np.array(src_state, dtype=np.float64)
    chosen, fallback, rejected, jitter = None, None, 0, False
    # Attempt 0 uses the base placement. The block placement is only jittered after
    # an invalid start state or a replay without contact; a replay rejected for the
    # agent leaving the frame or a blow-up is retried with fresh action noise only.
    # The first valid replay with contact wins; otherwise the first valid one is kept.
    for attempt in range(retries + 1):
        init = base.copy()
        if jitter:
            init[2:4] += rs.randn(2) * jitter_px
            init[4] += rs.randn() * jitter_rad
            init[2:4] = np.clip(init[2:4], 60.0, WINDOW - 60.0)  # keep the block off the walls
        if not _placement_ok(shape, init, seed, max_penetration):
            rejected += 1
            jitter = True
            continue
        frames, states, contacts, executed = _replay(shape, init, actions, seed, bias_px, noise_px,
                                                     noise_seed=seed * 31 + attempt)
        steps = np.linalg.norm(np.diff(states[:, 2:4], axis=0), axis=1)
        agent_out = ((states[:, :2] < 0) | (states[:, :2] > WINDOW)).any()
        if not np.isfinite(states).all() or agent_out or (len(steps) and steps.max() > max_block_step):
            rejected += 1
            continue
        if contacts > 0:
            chosen = (frames, states, contacts, init, attempt, executed)
            break
        if fallback is None:
            fallback = (frames, states, contacts, init, attempt, executed)
        jitter = True
    chosen = chosen or fallback
    if chosen is None:
        return {"valid": False, "rejected": rejected}
    frames, states, contacts, init, attempt, executed = chosen
    if video_path is not None:
        writer = imageio.get_writer(video_path, fps=fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
    block = states[:, 2:4]
    return {"valid": True, "states": states[:, :5].astype(np.float32), "vels": states[:, 5:7].astype(np.float32),
            "actions": executed.astype(np.float32),
            "contact_steps": contacts, "attempt": attempt, "rejected": rejected, "init_state": init[:5].tolist(),
            "block_disp": float(np.linalg.norm(block[-1] - block[0])),
            "max_block_step": float(np.linalg.norm(np.diff(block, axis=0), axis=1).max(initial=0.0))}


def draw_plan(pool, n, max_draws, rs):
    """Up to n candidates as (source, draw): every source once in random order, then
    further passes over fresh permutations while the pool is smaller than n and
    max_draws allows. With max_draws=1 this is a single rs.permutation(pool)[:n]."""
    out = []
    for d in range(max_draws):
        if len(out) >= n:
            break
        out += [(int(s), d) for s in rs.permutation(pool)[: n - len(out)]]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/pusht_noise_meta", help="DINO-WM pusht_noise root")
    p.add_argument("--out-dir", default="data/pushobj_replay_2000")
    p.add_argument("--shapes", default="T,L,Z,+")
    p.add_argument("--train-per-shape", type=int, default=450)
    p.add_argument("--val-per-shape", type=int, default=50)
    p.add_argument("--train-pool", type=int, default=16000, help="first K source episodes are trainable")
    p.add_argument("--contact-retries", type=int, default=4)
    p.add_argument("--jitter-px", type=float, default=20.0)
    p.add_argument("--jitter-rad", type=float, default=0.3)
    p.add_argument("--max-penetration", type=float, default=3.0, help="agent/block overlap allowed at t=0 (px)")
    p.add_argument("--max-block-step", type=float, default=80.0, help="per-step block move treated as blow-up (px)")
    p.add_argument("--action-bias-px", type=float, default=0.0, help="per-step push toward the block (px)")
    p.add_argument("--action-noise-px", type=float, default=0.0, help="per-axis Gaussian action noise (px)")
    p.add_argument("--placement", choices=("cog", "source"), default="cog",
                   help="cog: align the new shape's centre of gravity with the T's; source: keep the source pose")
    p.add_argument("--train-oversample", type=float, default=1.4, help="candidates per wanted train episode")
    p.add_argument("--val-oversample", type=float, default=2.0, help="val also drops no-contact episodes")
    p.add_argument("--max-draws", type=int, default=1, help="passes over a source pool that is too small")
    p.add_argument("--n-workers", type=int, default=32)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    unknown = set(shapes) - set(COG)
    if unknown:
        raise ValueError(f"unknown shapes {unknown}")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset: {out}")

    S, A, V, L = load_source(args.source)
    frame_idx = np.arange(S.shape[1])[None]
    agent = S[..., :2]
    leaves = ((((agent < 0) | (agent > WINDOW)).any(-1)) & (frame_idx < L[:, None])).any(1)
    idx = np.arange(len(L))
    pools = {"train": idx[(idx < args.train_pool) & ~leaves],
             "val": idx[(idx >= args.train_pool) & ~leaves]}
    print(f"source episodes {len(L)}; agent leaves frame in {leaves.sum()} (excluded); "
          f"pool train={len(pools['train'])} held-out={len(pools['val'])}", flush=True)

    rs = np.random.RandomState(args.seed)
    want = {"train": args.train_per_shape, "val": args.val_per_shape}
    # Every shape replays its own random subset of the pool, like subsetting the
    # paper's full set in which every source trajectory is replayed per shape.
    # Over-sample candidates: the quality gate drops some, and val also drops
    # episodes without contact.
    factor = {"train": args.train_oversample, "val": args.val_oversample}
    plan = {split: {sh: draw_plan(pools[split], int(np.ceil(want[split] * factor[split])), args.max_draws, rs)
                    for sh in shapes} for split in pools}
    for split in pools:
        for sh in shapes:
            if len(plan[split][sh]) < want[split]:
                raise ValueError(f"{split} pool too small for {want[split]} {sh} episodes (see --max-draws)")

    results = {"train": [], "val": []}
    with Pool(args.n_workers, initializer=_init_worker) as pool:
        for split in ("train", "val"):
            (out / split / "obses").mkdir(parents=True, exist_ok=True)
            tmp_dir = out / split / "_candidates"
            tmp_dir.mkdir(exist_ok=True)
            jobs, meta = [], []
            for sh in shapes:
                for src, draw in plan[split][sh]:
                    k = len(jobs)
                    # draw 0 keeps the original seeds; later draws are offset past every src*7+shape
                    jobs.append((np.concatenate([S[src, 0], V[src, 0]]), A[src, :L[src]], sh,
                                 int(args.seed * 1_000_003 + src * 7 + shapes.index(sh) + draw * 140_000),
                                 args.contact_retries, args.jitter_px, args.jitter_rad,
                                 args.max_penetration, args.max_block_step, str(tmp_dir / f"cand_{k:05d}.mp4"), args.fps,
                                 args.action_bias_px, args.action_noise_px, args.placement))
                    meta.append((sh, src, draw))
            for k, r in enumerate(pool.imap(_run, jobs, chunksize=2)):
                results[split].append((meta[k], r, tmp_dir / f"cand_{k:05d}.mp4"))
                if (k + 1) % 1000 == 0:
                    print(f"  {split}: {k + 1}/{len(jobs)} rendered", flush=True)

    summary = {}
    for split in ("train", "val"):
        chosen, dropped = [], {}
        for sh in shapes:
            rows = [x for x in results[split] if x[0][0] == sh]
            dropped[sh] = sum(not x[1]["valid"] for x in rows)
            rows = [x for x in rows if x[1]["valid"]]
            if split == "val":  # paper: held-out trajectories keep contact episodes only
                rows = [x for x in rows if x[1]["contact_steps"] > 0]
            if len(rows) < want[split]:
                raise RuntimeError(f"only {len(rows)} usable {split} {sh} episodes")
            chosen += rows[: want[split]]
        root = out / split
        n, t_max = len(chosen), max(len(x[1]["states"]) for x in chosen)
        states = torch.zeros(n, t_max, 5)
        actions = torch.zeros(n, t_max, 2)
        vels = torch.zeros(n, t_max, 2)
        lengths, shape_list, prov = [], [], []
        for i, ((sh, src, draw), r, video) in enumerate(chosen):
            T = len(r["states"])
            states[i, :T] = torch.from_numpy(r["states"])
            vels[i, :T] = torch.from_numpy(r["vels"])
            actions[i, :T] = torch.from_numpy(r["actions"])  # executed actions (bias/noise included)
            lengths.append(T)
            shape_list.append(sh)
            video.rename(root / "obses" / f"episode_{i:03d}.mp4")
            prov.append({"episode": i, "shape": sh, "source_index": src, "draw": draw, "length": T,
                         "contact_steps": r["contact_steps"], "retry_used": r["attempt"],
                         "rejected_placements": r["rejected"], "init_state": r["init_state"],
                         "block_displacement": r["block_disp"], "max_block_step": r["max_block_step"]})
        for leftover in (root / "_candidates").glob("*.mp4"):
            leftover.unlink()
        (root / "_candidates").rmdir()
        torch.save(states, root / "states.pth")
        torch.save(actions, root / "rel_actions.pth")
        torch.save(vels, root / "velocities.pth")
        with open(root / "seq_lengths.pkl", "wb") as f:
            pickle.dump(lengths, f)
        with open(root / "shapes.pkl", "wb") as f:
            pickle.dump(shape_list, f)
        (root / "provenance.json").write_text(json.dumps(prov))
        contact = np.array([x["contact_steps"] for x in prov])
        summary[split] = {
            "episodes": n, "frames": int(sum(lengths)), "max_len": t_max,
            "shapes": {sh: shape_list.count(sh) for sh in shapes},
            "episodes_with_contact": float((contact > 0).mean()),
            "mean_contact_steps": float(contact.mean()),
            "retry_used": int(sum(x["retry_used"] > 0 for x in prov)),
            "unique_sources": len({x["source_index"] for x in prov}),
            "repeat_draws_kept": int(sum(x["draw"] > 0 for x in prov)),
            "candidates": {sh: sum(x[0][0] == sh for x in results[split]) for sh in shapes},
            "candidates_dropped_by_quality_gate": dropped,
            "kept_with_rejected_placement": int(sum(x["rejected_placements"] > 0 for x in prov)),
            "max_block_step": float(max(x["max_block_step"] for x in prov)),
        }
        print(f"  {split}: {json.dumps(summary[split])}", flush=True)

    overlap = set(p["source_index"] for p in json.loads((out / "train/provenance.json").read_text())) & \
        set(p["source_index"] for p in json.loads((out / "val/provenance.json").read_text()))
    assert not overlap, "train/val share source trajectories"
    with (out / "generation_config.json").open("w") as f:
        json.dump({**vars(args), "method": "AdaJEPA App. A.2: pusht_noise replay with swapped shape "
                   "+ contact bias (placement: " + args.placement + ", action bias/noise px: "
                   f"{args.action_bias_px}/{args.action_noise_px}) + quality gate", "source_episodes": int(len(L)),
                   "source_excluded_agent_leaves_frame": int(leaves.sum()),
                   "action_units": "pixel delta (env input = action / 100)",
                   "state_alignment": "observation/state before action at same index",
                   "summary": summary}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
