#!/usr/bin/env python3
"""Sim(3) ATE for TUM monocular trajectories.

Reads TUM-format GT (timestamp tx ty tz qx qy qz qw, Twc) and an estimated
trajectory (same format), matches timestamps, aligns with Sim(3) Umeyama
(scale + rotation + translation) on the whole trajectory, and reports the
translation RMSE (ATE) and the recovered scale.

This mirrors the Sim(3)-on-translation ATE used in the Replica monocular
diagnosis (single global scale, no per-frame scale).
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


def se3_from_tum(row):
    t = row[0:3]
    q = row[3:7]
    # normalize quaternion
    q = q / np.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])
    return R, t


def sim3_umeyama(pts_est, pts_gt):
    """Align est -> gt with Sim(3) (Umeyama with scale)."""
    mu_e = pts_est.mean(0)
    mu_g = pts_gt.mean(0)
    e = pts_est - mu_e
    g = pts_gt - mu_g
    sigma_e = (e ** 2).sum() / e.shape[0]
    cov = g.T @ e / e.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma_e if sigma_e > 1e-12 else 1.0
    t = mu_g - s * R @ mu_e
    return s, R, t


def main():
    if len(sys.argv) != 4:
        print("usage: eval_tum_sim3.py <gt_tum.txt> <est_tum.txt> <out.txt>")
        return 2
    gt_ts, gt_T = load_tum(sys.argv[1])
    est_ts, est_T = load_tum(sys.argv[2])

    # Match by nearest timestamp
    idx = []
    for t in est_ts:
        j = int(np.argmin(np.abs(gt_ts - t)))
        idx.append(j)

    est_pos = np.array([se3_from_tum(r)[1] for r in est_T])
    gt_pos = np.array([se3_from_tum(gt_T[j])[1] for j in idx])

    s, R, t = sim3_umeyama(est_pos, gt_pos)
    aligned = (s * (R @ est_pos.T).T) + t
    err = np.linalg.norm(aligned - gt_pos, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    mean_err = float(err.mean())
    median_err = float(np.median(err))

    with open(sys.argv[3], "w") as f:
        f.write(f"scale={s:.6f}\n")
        f.write(f"ate_rmse_m={rmse:.6f}\n")
        f.write(f"ate_mean_m={mean_err:.6f}\n")
        f.write(f"ate_median_m={median_err:.6f}\n")
        f.write(f"n_matches={len(est_pos)}\n")

    print(f"[Sim3 ATE] scale={s:.4f} rmse={rmse:.4f} m mean={mean_err:.4f} "
          f"median={median_err:.4f} n={len(est_pos)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
