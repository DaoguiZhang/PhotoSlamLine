#!/usr/bin/env python3
"""SE(3) ATE for TUM-format trajectories (no scale).

Reads TUM-format GT (timestamp tx ty tz qx qy qz qw, Twc) and an estimated
trajectory in the same format, associates timestamps, aligns translations with
a rigid Umeyama alignment (scale FIXED to 1) and reports the translation RMSE.

Use this for RGB-D/Stereo runs where metric scale is observable. For monocular
runs use scripts/eval_tum_sim3.py (Sim(3), scale + rotation + translation).

Usage:
    python3 scripts/eval_tum_se3.py <gt.txt> <est.txt> [out.txt]
"""
import sys
import numpy as np


def load_tum(path):
    ts, T = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            ts.append(float(p[0]))
            T.append([float(x) for x in p[1:8]])
    return np.array(ts), np.array(T)


def associate(ts_a, ts_b, max_dt=0.02):
    """Return index pairs (ia, ib) with the closest timestamp match."""
    pairs = []
    j = 0
    for ia, ta in enumerate(ts_a):
        while j + 1 < len(ts_b) and abs(ts_b[j + 1] - ta) <= abs(ts_b[j] - ta):
            j += 1
        if abs(ts_b[j] - ta) <= max_dt:
            pairs.append((ia, j))
    return pairs


def umeyama_rigid(src, dst):
    """Rigid alignment: dst ~= R @ src + t (scale fixed to 1)."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = dst_c.T @ src_c / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_d - R @ mu_s
    return R, t


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    gt_path, est_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    ts_gt, T_gt = load_tum(gt_path)
    ts_est, T_est = load_tum(est_path)
    pairs = associate(ts_gt, ts_est)
    if len(pairs) < 3:
        print(f"[SE3 ATE] not enough associations: n={len(pairs)}")
        return 2

    p_gt = T_gt[[i for i, _ in pairs], 0:3]
    p_est = T_est[[j for _, j in pairs], 0:3]

    R, t = umeyama_rigid(p_est, p_gt)
    p_aligned = (R @ p_est.T).T + t
    err = np.linalg.norm(p_aligned - p_gt, axis=1)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    line = (f"[SE3 ATE] rmse={rmse:.6f} m mean={err.mean():.6f} "
            f"median={np.median(err):.6f} max={err.max():.6f} n={len(pairs)}")
    print(line)
    if out_path:
        with open(out_path, "w") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
