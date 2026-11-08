#!/usr/bin/env python3
"""Offline LPIPS for one saved snapshot directory.

usage: eval_snapshot_lpips.py <snapshot_dir> [--device cpu]

The snapshot directory is the one written by renderAndRecordAllKeyframes(),
i.e. it contains image/ and image_gt/. LPIPS is computed on CPU by default and
is never part of the mapped time.
"""
import os
import statistics
import sys


def main(argv):
    if not argv:
        print("usage: eval_snapshot_lpips.py <snapshot_dir> [--device cpu]")
        return 2
    snap = argv[0]
    device = "cpu"
    if "--device" in argv:
        device = argv[argv.index("--device") + 1]
    img_dir = os.path.join(snap, "image")
    gt_dir = os.path.join(snap, "image_gt")
    if not (os.path.isdir(img_dir) and os.path.isdir(gt_dir)):
        print("MISSING image/ or image_gt/ in", snap)
        return 1
    try:
        import numpy as np
        import torch
        import lpips
        from PIL import Image
    except Exception as e:
        print(f"LPIPS UNAVAILABLE ({e}) -> reported as MISSING, not substituted.")
        return 1
    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    vals, pairs = [], 0
    for name in sorted(os.listdir(img_dir)):
        if not name.endswith(".jpg"):
            continue
        gt_name = name[:-4] + "_gt.jpg"
        gtp = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gtp):
            continue
        a = np.asarray(Image.open(os.path.join(img_dir, name)).convert("RGB"), dtype="float32") / 255.0
        b = np.asarray(Image.open(gtp).convert("RGB"), dtype="float32") / 255.0
        if a.shape != b.shape:
            continue
        ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
        with torch.no_grad():
            vals.append(float(net(ta.to(device), tb.to(device)).item()))
        pairs += 1
    if not vals:
        print("NO PAIRS in", snap)
        return 1
    print(f"lpips={statistics.mean(vals):.6f} pairs={pairs} n_pairs_missing="
          f"{len([n for n in os.listdir(img_dir) if n.endswith('.jpg')]) - pairs}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
