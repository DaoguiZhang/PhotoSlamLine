#!/usr/bin/env python3
"""Offline LPIPS for gba-recovery-v1.

Reads the render/GT pairs the C++ experiment already saved under
<root>/<arm>/eval_step<N>/{image,image_gt}. LPIPS is never part of the
measured recovery time.

usage: lpips_gba_recovery.py <root> [--device cpu|cuda]
"""
import csv
import os
import statistics
import sys

ARMS = ["PoseOnly", "PoseAndMap", "PoseAndMap_r2", "PoseOnly_r2"]


def main(argv):
    root = argv[0] if argv and not argv[0].startswith("--") else "abl_out/gba_recovery_v1"
    device = "cuda"
    if "--device" in argv:
        device = argv[argv.index("--device") + 1]

    try:
        import torch
        import lpips
        from PIL import Image
        import numpy as np
    except Exception as e:
        print(f"LPIPS UNAVAILABLE ({e}) -> reported as MISSING, not substituted.")
        return 1

    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def pair(p1, p2):
        a = np.asarray(Image.open(p1).convert("RGB"), dtype="float32") / 255.0
        b = np.asarray(Image.open(p2).convert("RGB"), dtype="float32") / 255.0
        if a.shape != b.shape:
            return None
        ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
        with torch.no_grad():
            return float(net(ta.to(device), tb.to(device)).item())

    out_path = os.path.join(root, "lpips.csv")
    rows = []
    for arm in ARMS:
        arm_dir = os.path.join(root, arm)
        if not os.path.isdir(arm_dir):
            continue
        steps = []
        for d in sorted(os.listdir(arm_dir)):
            if not d.startswith("eval_step"):
                continue
            step = int(d[len("eval_step"):])
            img = os.path.join(arm_dir, d, "image")
            gt = os.path.join(arm_dir, d, "image_gt")
            if not (os.path.isdir(img) and os.path.isdir(gt)):
                continue
            vals = []
            for nm in sorted(os.listdir(img)):
                g = os.path.join(gt, nm)
                if os.path.exists(g):
                    v = pair(os.path.join(img, nm), g)
                    if v is not None:
                        vals.append(v)
            if vals:
                rows.append((arm, step, statistics.fmean(vals), len(vals)))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "lpips", "n"])
        for r in rows:
            w.writerow(r)
    for arm in ARMS:
        vals = [(s, v, n) for (a, s, v, n) in rows if a == arm]
        if vals:
            print(arm + ":")
            for s, v, n in vals:
                print(f"  step {s:>4}: LPIPS={v:.6f} (n={n})")
    print("wrote " + out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
