#!/usr/bin/env python3
"""Scale-corrected (Sim(3)-on-translation) ATE evaluator for Mono trajectories.

Mirrors ORB-SLAM3/evaluation/evaluate_ate_scale.py (Horn alignment + scale on the
translation part) but is Python 3 and also writes a per-frame APE file.

Usage:
    eval_ate_sim3.py <gt_tum> <est_tum> <ate_out> <ape_out>

Both files use TUM format: `timestamp tx ty tz qx qy qz qw` (rotation ignored for
alignment, same as evaluate_ate_scale.py).
"""
import sys

import numpy as np


def load_tum(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ts = float(parts[0])
            data[ts] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    return data


def associate(gt, est, max_diff=0.01):
    """Return lists of (gt_ts, est_ts) matched by nearest timestamp."""
    est_ts = sorted(est.keys())
    pairs = []
    for t in sorted(gt.keys()):
        best = min(est_ts, key=lambda e: abs(e - t))
        if abs(best - t) <= max_diff:
            pairs.append((t, best))
    return pairs


def umeyama(src, dst):
    """Sim(3)-on-translation Umeyama alignment (rotation + scale + translation).

    src/dst: 3xN matrices. Returns R (3x3), t (3x1), scale (float).
    """
    n = src.shape[1]
    src_m = src.mean(axis=1, keepdims=True)
    dst_m = dst.mean(axis=1, keepdims=True)
    src_c = src - src_m
    dst_c = dst - dst_m
    cov = (dst_c @ src_c.T) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_src = np.sum(src_c ** 2) / n
    scale = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 0 else 1.0
    t = dst_m - scale * (R @ src_m)
    return R, t, scale


def main():
    gt_path, est_path, ate_path, ape_path = sys.argv[1:5]

    gt = load_tum(gt_path)
    est = load_tum(est_path)

    pairs = associate(gt, est)
    if len(pairs) < 2:
        # fallback: exact integer timestamps
        pairs = [(t, t) for t in sorted(set(gt.keys()) & set(est.keys()))]
    if len(pairs) < 2:
        print("ERROR: <2 matched timestamps between GT and EST", file=sys.stderr)
        sys.exit(1)

    gt_xyz = np.array([gt[t] for t, _ in pairs]).T  # 3xN
    est_xyz = np.array([est[e] for _, e in pairs]).T  # 3xN

    R, t, scale = umeyama(est_xyz, gt_xyz)
    est_aligned = scale * (R @ est_xyz) + t
    err = np.linalg.norm(est_aligned - gt_xyz, axis=0)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mean = float(np.mean(err))
    median = float(np.median(err))
    std = float(np.std(err))
    mn = float(np.min(err))
    mx = float(np.max(err))
    argmax = pairs[int(np.argmax(err))][0]

    lines = [
        "APE w.r.t. translation part (m) (with Sim(3) scale-corrected Umeyama alignment)",
        f"       max      {mx:.6f}",
        f"      mean      {mean:.6f}",
        f"    median      {median:.6f}",
        f"       min      {mn:.6f}",
        f"      rmse      {rmse:.6f}",
        f"       std      {std:.6f}",
        f"     scale      {scale:.6f}",
        f"matched_pairs   {len(pairs)}",
        f"max_ape_frame   {argmax:.0f}",
    ]
    text = "\n".join(lines)
    with open(ate_path, "w") as f:
        f.write(text + "\n")
    print(text)

    with open(ape_path, "w") as f:
        f.write("# gt_ts  est_ts  ape(m)\n")
        for (t, e), a in zip(pairs, err):
            f.write(f"{t:.6f} {e:.6f} {a:.6f}\n")

    # first significant deviation from GT (APE > 2x median, first occurrence)
    thr = max(2.0 * median, 0.02)
    dev_ts = None
    for (t, _), a in zip(pairs, err):
        if a > thr:
            dev_ts = t
            break
    if dev_ts is not None:
        print(f"first_deviation_frame_(ape>{thr:.3f}m)  {dev_ts:.0f}")


if __name__ == "__main__":
    main()
