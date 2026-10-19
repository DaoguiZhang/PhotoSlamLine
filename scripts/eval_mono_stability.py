#!/usr/bin/env python3
"""Independent monocular trajectory evaluation (Sim(3)-primary + diagnostics).

Usage:
  eval_mono_stability.py <gt_tum.txt> <est_tum.txt> <out_prefix>

GT and EST are TUM format: `ts tx ty tz qx qy qz qw` (camera-to-world, qw last).

Reports (written to <out_prefix>.txt and printed as one-line JSON):
  - timestamp matching: n_matches, coverage (est matched / est total, gt matched
    / gt total), first/last matched timestamp
  - Sim(3) Umeyama alignment (single global scale): scale, ate_rmse, ate_mean,
    ate_median, ate_max, argmax frame index + timestamp
  - SE(3) Umeyama alignment (no scale): ate_rmse_se3 (scale-anomaly diagnostic)
  - RPE (relative pose error, translation) at delta=1/10/30 frames
  - NaN/Inf counts in est
  - jump detection: max consecutive translation delta of the ALIGNED Sim(3) est
"""
import sys
import json
import numpy as np


def load_tum(path):
    ts, rows = [], []
    with open(path) as f:
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
    t = row[0:3].astype(np.float64)
    q = row[3:7].astype(np.float64)  # qx qy qz qw
    n = np.linalg.norm(q)
    if n < 1e-12:
        q = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        q = q / n
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ])
    return R, t


def umeyama_sim3(est, gt):
    mu_e = est.mean(0)
    mu_g = gt.mean(0)
    e = est - mu_e
    g = gt - mu_g
    sigma_e = (e ** 2).sum() / max(e.shape[0], 1)
    cov = g.T @ e / max(e.shape[0], 1)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma_e if sigma_e > 1e-12 else 1.0
    t = mu_g - s * R @ mu_e
    return s, R, t


def umeyama_se3(est, gt):
    # rigid alignment (no scale)
    return umeyama_sim3(est, gt)  # then ignore scale; but better do Kabsch


def kabsch(est, gt):
    mu_e = est.mean(0)
    mu_g = gt.mean(0)
    e = est - mu_e
    g = gt - mu_g
    cov = g.T @ e
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_g - R @ mu_e
    return R, t


def align(est, s, R, t):
    return (s * (R @ est.T)).T + t


def rpe_translation(est_aligned, gt, delta):
    if delta <= 0 or delta >= len(est_aligned):
        return float("nan")
    # relative motion magnitude of gt vs est between frame i and i+delta
    d_est = np.linalg.norm(est_aligned[delta:] - est_aligned[:-delta], axis=1)
    d_gt = np.linalg.norm(gt[delta:] - gt[:-delta], axis=1)
    err = np.abs(d_est - d_gt)
    return float(np.sqrt((err ** 2).mean()))


def main():
    if len(sys.argv) != 4:
        print("usage: eval_mono_stability.py <gt_tum.txt> <est_tum.txt> <out_prefix>")
        return 2
    gt_ts, gt_rows = load_tum(sys.argv[1])
    est_ts, est_rows = load_tum(sys.argv[2])
    if len(est_rows) == 0 or len(gt_rows) == 0:
        print(json.dumps({"error": "empty input", "n_est": len(est_rows),
                          "n_gt": len(gt_rows)}))
        return 1

    # NaN/Inf check
    n_nan = int(np.isnan(est_rows).sum())
    n_inf = int(np.isinf(est_rows).sum())

    # Match by nearest timestamp
    gt_idx = []
    for t in est_ts:
        gt_idx.append(int(np.argmin(np.abs(gt_ts - t))))
    gt_idx = np.array(gt_idx)

    est_pos = np.array([se3_from_tum(r)[1] for r in est_rows])
    gt_pos = np.array([se3_from_tum(gt_rows[j])[1] for j in gt_idx])

    s, R, t = umeyama_sim3(est_pos, gt_pos)
    aligned = align(est_pos, s, R, t)
    err = np.linalg.norm(aligned - gt_pos, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    mean_e = float(err.mean())
    median_e = float(np.median(err))
    max_e = float(err.max())
    argmax = int(np.argmax(err))
    max_ts = float(est_ts[argmax])

    Rr, tr = kabsch(est_pos, gt_pos)
    aligned_se3 = align(est_pos, 1.0, Rr, tr)
    err_se3 = np.linalg.norm(aligned_se3 - gt_pos, axis=1)
    rmse_se3 = float(np.sqrt((err_se3 ** 2).mean()))

    # coverage
    n_est = len(est_rows)
    n_gt = len(gt_rows)
    cov_est = n_est / n_gt if n_gt else 0.0
    # matched coverage of gt (fraction of gt frames covered by matched est)
    matched_gt = len(set(gt_idx.tolist()))
    cov_gt = matched_gt / n_gt if n_gt else 0.0

    # jump detection on aligned est (max consecutive delta)
    if len(aligned) > 1:
        deltas = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
        max_jump = float(deltas.max())
        max_jump_idx = int(np.argmax(deltas))
        max_jump_ts = float(est_ts[max_jump_idx + 1])
    else:
        max_jump = 0.0
        max_jump_idx = -1
        max_jump_ts = float("nan")

    rpe = {}
    for d in (1, 10, 30):
        rpe[f"rpe_{d}"] = rpe_translation(aligned, gt_pos, d)

    res = {
        "n_est": n_est,
        "n_gt": n_gt,
        "n_matches": n_est,
        "matched_gt_frames": matched_gt,
        "coverage_est_vs_gt": round(cov_est, 6),
        "coverage_gt": round(cov_gt, 6),
        "first_est_ts": float(est_ts[0]) if len(est_ts) else float("nan"),
        "last_est_ts": float(est_ts[-1]) if len(est_ts) else float("nan"),
        "first_gt_ts": float(gt_ts[0]) if len(gt_ts) else float("nan"),
        "last_gt_ts": float(gt_ts[-1]) if len(gt_ts) else float("nan"),
        "scale": round(s, 6),
        "ate_rmse_m": round(rmse, 6),
        "ate_mean_m": round(mean_e, 6),
        "ate_median_m": round(median_e, 6),
        "ate_max_m": round(max_e, 6),
        "ate_max_frame": argmax,
        "ate_max_ts": max_ts,
        "ate_rmse_se3_m": round(rmse_se3, 6),
        "rpe": {k: round(v, 6) for k, v in rpe.items()},
        "nan_count": n_nan,
        "inf_count": n_inf,
        "max_jump_m": round(max_jump, 6),
        "max_jump_frame": max_jump_idx,
        "max_jump_ts": max_jump_ts,
    }

    out = sys.argv[3]
    with open(out + ".txt", "w") as f:
        for k, v in res.items():
            if k == "rpe":
                for rk, rv in v.items():
                    f.write(f"{rk}={rv}\n")
            else:
                f.write(f"{k}={v}\n")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
