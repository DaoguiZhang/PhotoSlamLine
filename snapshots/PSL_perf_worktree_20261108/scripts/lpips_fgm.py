#!/usr/bin/env python3
"""Offline LPIPS for the fixed-geometry mapper convergence experiment.

LPIPS is computed here, after the run, on images the run already saved. It is
never part of the measured mapping time.

Two sources are supported, in order of preference:

  1. `abl_out/<label>/<tag>/eval_step<N>/image/*.png` + `.../image_gt/*.png`
     (per-evaluation-point renders, exact same views and mask as PSNR/SSIM)
  2. `abl_out/<label>/<tag>/<iter>_shutdown/image/<iter>_<fid>.jpg`
     (all keyframes at the final state) paired with the dataset ground truth
     `.../replica/<scene>/results/frame<fid:06d>.jpg`

usage: lpips_fgm.py <label> [<label> ...] [--scene office0] [--device cpu|cuda]
"""
import os
import statistics
import sys

ROOT = "abl_out"
DATASET = "/workspace/code/SEGS-SLAM/datasets/replica"


def make_scorer(device):
    import lpips
    import torch
    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return net, torch


def lpips_pair(net, torch, p1, p2, device):
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(p1).convert("RGB"), dtype="float32") / 255.0
    b = np.asarray(Image.open(p2).convert("RGB"), dtype="float32") / 255.0
    if a.shape != b.shape:
        return None                      # resolutions must match exactly
    ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
    tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
    with torch.no_grad():
        return float(net(ta.to(device), tb.to(device)).item())


def eval_step_pairs(tag_dir):
    """Yield (step, list_of_(render, gt))."""
    out = []
    if not os.path.isdir(tag_dir):
        return out
    for d in sorted(os.listdir(tag_dir)):
        if not d.startswith("eval_step"):
            continue
        step = int(d[len("eval_step"):])
        img = os.path.join(tag_dir, d, "image")
        gt = os.path.join(tag_dir, d, "image_gt")
        if not (os.path.isdir(img) and os.path.isdir(gt)):
            continue
        pairs = []
        for nm in sorted(os.listdir(img)):
            g = os.path.join(gt, nm)
            if os.path.exists(g):
                pairs.append((os.path.join(img, nm), g))
        if pairs:
            out.append((step, pairs))
    return out


def shutdown_pairs(tag_dir, scene):
    out = []
    if not os.path.isdir(tag_dir):
        return out
    for d in sorted(os.listdir(tag_dir)):
        if not d.endswith("_shutdown"):
            continue
        img = os.path.join(tag_dir, d, "image")
        if not os.path.isdir(img):
            continue
        pairs = []
        for nm in os.listdir(img):
            stem = os.path.splitext(nm)[0]
            if "_" not in stem:
                continue
            fid = int(stem.split("_")[-1])
            g = os.path.join(DATASET, scene, "results", "frame%06d.jpg" % fid)
            if os.path.exists(g):
                pairs.append((os.path.join(img, nm), g))
        if pairs:
            out.append((d, pairs))
    return out


def main(argv):
    labels = [a for a in argv if not a.startswith("--")]
    scene = "office0"
    if "--scene" in argv:
        scene = argv[argv.index("--scene") + 1]
    device = "cuda"
    if "--device" in argv:
        device = argv[argv.index("--device") + 1]
    if not labels:
        print(__doc__)
        return 2

    try:
        net, torch = make_scorer(device)
    except Exception as e:
        print(f"LPIPS UNAVAILABLE ({e}) -> reported as MISSING, not substituted.")
        return 1
    print(f"LPIPS net=alex device={device} scene={scene}\n")

    results = {}
    for lbl in labels:
        for tag in ("B", "C", "E", "F", "G", "B0", "B1", "J", "K", "Q0", "Q25",
                    "Q50", "Q100", "RA", "RC"):
            tag_dir = os.path.join(ROOT, lbl, tag)
            if not os.path.isdir(tag_dir):
                continue
            rows = eval_step_pairs(tag_dir)
            src = "eval_step"
            if not rows:
                rows = shutdown_pairs(tag_dir, scene)
                src = "shutdown"
            if not rows:
                print(f"{lbl}/{tag}: no saved images -> MISSING")
                continue
            print(f"{lbl}/{tag}  source={src}")
            for step, pairs in rows:
                vals = []
                for a, b in pairs:
                    v = lpips_pair(net, torch, a, b, device)
                    if v is not None:
                        vals.append(v)
                if not vals:
                    print(f"   step {step}: no usable pairs (resolution mismatch?)")
                    continue
                m = statistics.fmean(vals)
                results[(lbl, tag, step)] = m
                print(f"   step {step:>6}: LPIPS={m:.5f}  (n={len(vals)})")

    # paired ON-OFF where both exist
    print()
    for lbl in labels:
        common = sorted({s for (l, t, s) in results if l == lbl and t == "C"}
                        & {s for (l, t, s) in results if l == lbl and t == "B"})
        if common:
            print(f"{lbl}: paired dLPIPS (K - B0), lower is better for B0")
            for s in common:
                b = results[(lbl, "B", s)]
                c = results[(lbl, "K", s)]
                print(f"   step {s:>6}: B={b:.5f} C={c:.5f} d={c-b:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
