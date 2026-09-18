#!/usr/bin/env python3
"""TUM trajectory evaluation for P6: SE(3) ATE, Sim(3) ATE and translation RPE.

Usage: p6_eval_tum.py <gt_tum.txt> <est_tum.txt> <out_metrics.txt>

Both files are TUM format (timestamp tx ty tz qx qy qz qw, pose Twc).
- SE(3) ATE: Umeyama alignment without scale (rotation + translation).
- Sim(3) ATE: Umeyama alignment with scale.
- RPE: frame-to-frame relative translation drift (RMSE + mean + drift %).
"""
import sys
import numpy as np


def load_tum(path):
    ts, rows = [], []
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            try:
                ts.append(float(p[0]))
                rows.append([float(x) for x in p[1:8]])
            except ValueError:
                continue
    return np.array(ts), np.array(rows)


def se3_from_tum(row):
    t = row[0:3]
    q = row[3:7] / np.linalg.norm(row[3:7])
    # TUM quaternion order: qx qy qz qw
    x, y, z, w = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])
    return R, t


def umeyama(a, b, with_scale):
    mu_a = a.mean(0)
    mu_b = b.mean(0)
    A = a - mu_a
    B = b - mu_b
    cov = B.T @ A / A.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = 1.0
    if with_scale:
        var_a = (A ** 2).sum() / A.shape[0]
        s = np.trace(np.diag(D) @ S) / var_a if var_a > 1e-12 else 1.0
    t = mu_b - s * R @ mu_a
    return s, R, t


def rmse(x):
    return float(np.sqrt((np.asarray(x) ** 2).mean()))


def main():
    if len(sys.argv) != 4:
        print("usage: p6_eval_tum.py <gt_tum.txt> <est_tum.txt> <out_metrics.txt>")
        return 2
    gt_ts, gt_rows = load_tum(sys.argv[1])
    est_ts, est_rows = load_tum(sys.argv[2])
    if len(est_ts) == 0 or len(gt_ts) == 0:
        print("no valid data")
        return 1

    # Match each estimate to nearest GT timestamp
    idx = [int(np.argmin(np.abs(gt_ts - t))) for t in est_ts]
    est_pos = np.array([se3_from_tum(r)[1] for r in est_rows])
    gt_pos = np.array([se3_from_tum(gt_rows[j])[1] for j in idx])

    # Sim(3) ATE
    s, R, t = umeyama(est_pos, gt_pos, True)
    aligned = (s * (R @ est_pos.T).T) + t
    sim3_ate = rmse(np.linalg.norm(aligned - gt_pos, axis=1))

    # SE(3) ATE
    _, R2, t2 = umeyama(est_pos, gt_pos, False)
    aligned2 = (R2 @ est_pos.T).T + t2
    se3_ate = rmse(np.linalg.norm(aligned2 - gt_pos, axis=1))

    # Frame-to-frame translation RPE (consecutive estimates)
    rel_errs = []
    for i in range(1, len(est_pos)):
        d_est = np.linalg.norm(est_pos[i] - est_pos[i-1])
        d_gt = np.linalg.norm(gt_pos[i] - gt_pos[i-1])
        rel_errs.append(abs(d_est - d_gt))
    rpe_rmse = rmse(rel_errs) if rel_errs else float("nan")
    rpe_mean = float(np.mean(rel_errs)) if rel_errs else float("nan")
    total_gt = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))
    drift_pct = (rpe_rmse * len(rel_errs) / total_gt * 100.0) if (rel_errs and total_gt > 1e-9) else float("nan")

    with open(sys.argv[3], "w") as f:
        f.write(f"se3_ate_rmse_m={se3_ate:.6f}\n")
        f.write(f"sim3_ate_rmse_m={sim3_ate:.6f}\n")
        f.write(f"sim3_scale={s:.6f}\n")
        f.write(f"rpe_rmse_m={rpe_rmse:.6f}\n")
        f.write(f"rpe_mean_m={rpe_mean:.6f}\n")
        f.write(f"rpe_drift_pct={drift_pct:.4f}\n")
        f.write(f"n_matches={len(est_pos)}\n")

    print(f"[Eval] SE3_ATE={se3_ate:.4f}m Sim3_ATE={sim3_ate:.4f}m scale={s:.4f} "
          f"RPE={rpe_rmse:.4f}m drift={drift_pct:.3f}% n={len(est_pos)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
