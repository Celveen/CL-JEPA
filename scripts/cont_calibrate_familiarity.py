"""Calibrate the familiarity threshold of the v4 router (planner.continual.familiar_thr) for a checkpoint.

The statistic is theta0's newest-frame latent error over the first 3 transitions (geometric
mean), computed on windows of the checkpoint's own training-distribution validation data at
the model stride. familiar_thr = a percentile of it (v4 uses the 90th): an episode at or
below it is predicted by theta0 as well as its training data, so it is never written into a
domain. No stream labels are involved. The value depends on the checkpoint -- recalibrate
whenever the weights change.

    python scripts/cont_calibrate_familiarity.py --ckpt /abs/path/to/ckpt_dir --epoch 3 \
        --val-data /abs/path/to/training_data/val
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cont_offline import load_world  # noqa: E402
from cont_route_screen import calibration  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (hydra.yaml + checkpoints/model_<epoch>.pth)")
    ap.add_argument("--epoch", default="3")
    ap.add_argument("--val-data", required=True, help="validation split of the training data (PushTDataset layout)")
    ap.add_argument("--n", type=int, default=100, help="windows per training shape")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre, _ = load_world(Path(a.ckpt), a.epoch, dev)
    vals, _ = calibration(wm, pre, Path(a.val_data), a.n, fs=5)
    print(f"{len(vals)} windows; median {np.median(vals):.4f}")
    for q in (80, 85, 90, 95, 99):
        print(f"  {q}th percentile: {np.percentile(vals, q):.4f}" + ("   <- v4 familiar_thr" if q == 90 else ""))


if __name__ == "__main__":
    main()
