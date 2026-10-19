"""Train the seen-shape probe used by v6: a linear classifier over the training shapes.

Inputs are theta0 encoder latents (the 384-d visual token the world model uses) of frames from the
base model's own training-distribution validation data, labelled with the shapes it was trained on
(shapes.pkl of the training data -- no deployment labels). v6 applies it to the first 3 executed
frames of an episode (mean softmax): a familiar episode (theta0 already predicts it) whose shape the
probe names with confidence >= seen_conf is written to that training shape's domain; anything
else is routed label-free as in v5. On S7 the probe names the trained shapes' episodes 160/160
correctly (confidence median ~0.995); unseen shapes get ~0.6-0.75.

    python scripts/cont_train_seen_probe.py --ckpt checkpoints/pushobj_bias_16k_e3_seed0 \
        --val-data data/pushobj_bias_16k/val --out data/seen_probe/pushobj_bias_16k_e3_seed0.pt
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cont_offline import load_world  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
    ap.add_argument("--epoch", default="3")
    ap.add_argument("--val-data", default="data/pushobj_bias_16k/val")
    ap.add_argument("--per-shape", type=int, default=300, help="episodes per training shape (4 frames each)")
    ap.add_argument("--out", default="data/seen_probe/pushobj_bias_16k_e3_seed0.pt")
    a = ap.parse_args()
    from datasets.pusht_dset import PushTDataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre, _ = load_world(ROOT / a.ckpt, a.epoch, dev)
    ds = PushTDataset(data_path=str(ROOT / a.val_data), transform=pre.transform)
    classes = sorted(set(ds.shapes), key=lambda s: ["T", "L", "Z", "+"].index(s) if s in "TLZ+" else 9)
    rng = random.Random(0)
    X, y = [], []
    for c, s in enumerate(classes):
        idx = [i for i, t in enumerate(ds.shapes) if t == s]
        for i in rng.sample(idx, min(a.per_shape, len(idx))):
            frames = sorted(rng.sample(range(ds.get_seq_length(i)), 4))
            with torch.no_grad():
                X.append(wm.encoder(ds.load_visual_frames(i, frames).to(dev)).squeeze(1).float().cpu())
            y += [c] * len(frames)
    X, y = torch.cat(X), torch.tensor(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    torch.manual_seed(0)
    net = torch.nn.Linear(X.shape[1], len(classes))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xn = (X - mu) / sd
    for _ in range(300):
        perm = torch.randperm(len(Xn))
        for s in range(0, len(Xn), 256):
            bi = perm[s : s + 256]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(Xn[bi]), y[bi]).backward()
            opt.step()
    with torch.no_grad():
        acc = float((net(Xn).argmax(1) == y).float().mean())
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"W": net.weight.detach(), "b": net.bias.detach(), "mu": mu, "sd": sd, "classes": classes,
                "ckpt": a.ckpt, "val_data": a.val_data, "train_acc": acc, "n_frames": len(y)}, out)
    print(f"saved {out}: classes {classes}, {len(y)} frames, training accuracy {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
