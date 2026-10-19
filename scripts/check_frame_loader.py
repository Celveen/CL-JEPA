"""Check that PushTDataset.load_visual_frames (the slicer's fast path) returns exactly
what the old path returned: get_frames over the whole window, then [::frameskip].

Builds the train/val slices with the same arguments as train.py, reads random windows
both ways (the old path by hiding load_visual_frames from the class) and requires
bit-identical visual, proprio, state and action tensors. Also times both paths.

    python scripts/check_frame_loader.py --data data/pushobj_bias_16k --train 2000 --val 500
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(slices, indices):
    return [slices[int(i)] for i in indices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pushobj_bias_16k")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--time-windows", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from datasets import pusht_dset
    from datasets.img_transforms import default_transform
    slices, _ = pusht_dset.load_pusht_slice_train_val(default_transform(224), n_rollout=None, data_path=str(ROOT / a.data),
                                                      normalize_action=True, split_ratio=0.9, num_hist=3, num_pred=1,
                                                      frameskip=5, with_velocity=True)
    rs = np.random.RandomState(0)
    fast_fn = pusht_dset.PushTDataset.load_visual_frames
    total = 0
    for split, n in (("train", a.train), ("valid", a.val)):
        s = slices[split]
        idx = rs.choice(len(s), n, replace=False)
        fast = read(s, idx)
        del pusht_dset.PushTDataset.load_visual_frames  # old path: get_frames over the window, then [::frameskip]
        try:
            old = read(s, idx)
        finally:
            pusht_dset.PushTDataset.load_visual_frames = fast_fn
        for k, ((fo, fa, fs), (oo, oa, os_)) in enumerate(zip(fast, old)):
            same = (set(fo) == set(oo) and all(torch.equal(fo[key], oo[key]) for key in fo)
                    and torch.equal(fa, oa) and torch.equal(fs, os_))
            if not same:
                raise SystemExit(f"MISMATCH in {split} window {idx[k]}")
        total += n
        print(f"{split}: {n} random windows identical (visual {tuple(fast[0][0]['visual'].shape)}, "
              f"proprio {tuple(fast[0][0]['proprio'].shape)}, action {tuple(fast[0][1].shape)}, state {tuple(fast[0][2].shape)})", flush=True)
    s = slices["train"]
    idx = rs.choice(len(s), a.time_windows, replace=False)
    t = time.perf_counter(); read(s, idx); fast_t = time.perf_counter() - t
    del pusht_dset.PushTDataset.load_visual_frames
    try:
        t = time.perf_counter(); read(s, idx); old_t = time.perf_counter() - t
    finally:
        pusht_dset.PushTDataset.load_visual_frames = fast_fn
    print(f"all {total} windows bit-identical; 1 thread: old {old_t / a.time_windows * 1000:.1f} ms/window, "
          f"new {fast_t / a.time_windows * 1000:.1f} ms/window ({old_t / fast_t:.1f}x)")


if __name__ == "__main__":
    main()
