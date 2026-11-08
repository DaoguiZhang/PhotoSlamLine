#!/usr/bin/env python3
"""Compare two monocular runs against GT to find the first systematic divergence.

Usage:
  compare_runs.py <gt_tum.txt> <estA_tum.txt> <estB_tum.txt> <out_prefix> [thresh_m]

Both trajectories are aligned to GT with Sim(3) independently, then compared
frame-by-frame (by GT-matched frame index). Reports the first frame where the
two runs' aligned positions differ by more than `thresh_m` (default 0.05 m),
and prints per-frame divergence summary.
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


def se3_pos(row):
    t = np.array(row[0:3], dtype=np.float64)
    return t


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


def align_series(est_ts, est_rows, gt_ts, gt_rows):
    # match each est to nearest gt
    idx = [int(np.argmin(np.abs(gt_ts - t))) for t in est_ts]
    est_pos = np.array([se3_pos(r) for r in est_rows])
    gt_pos = np.array([se3_pos(gt_rows[j]) for j in idx])
    s, R, t = umeyama_sim3(est_pos, gt_pos)
    aligned = (s * (R @ est_pos.T)).T + t
    return est_ts, aligned, idx, s


def main():
    if len(sys.argv) < 5:
        print("usage: compare_runs.py <gt> <estA> <estB> <out_prefix> [thresh_m]")
        return 2
    thresh = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    gt_ts, gt_rows = load_tum(sys.argv[1])
    a_ts, a_rows = load_tum(sys.argv[2])
    b_ts, b_rows = load_tum(sys.argv[3])

    a_ts, a_aligned, a_idx, a_scale = align_series(a_ts, a_rows, gt_ts, gt_rows)
    b_ts, b_aligned, b_idx, b_scale = align_series(b_ts, b_rows, gt_ts, gt_rows)

    # Build per-frame-id divergence table
    a_by_id = {int(t): a_aligned[i] for i, t in enumerate(a_ts)}
    b_by_id = {int(t): b_aligned[i] for i, t in enumerate(b_ts)}
    common_ids = sorted(set(a_by_id) & set(b_by_id))

    divs = []
    first = None
    for fid in common_ids:
        d = float(np.linalg.norm(a_by_id[fid] - b_by_id[fid]))
        divs.append((fid, d))
        if first is None and d > thresh:
            first = (fid, d)

    res = {
        "n_a": len(a_ts), "n_b": len(b_ts), "n_common": len(common_ids),
        "scale_a": round(a_scale, 6), "scale_b": round(b_scale, 6),
        "first_divergence_frame": first[0] if first else None,
        "first_divergence_m": round(first[1], 6) if first else None,
        "thresh_m": thresh,
        "max_divergence": round(max(d for _, d in divs), 6) if divs else None,
        "max_divergence_frame": max(divs, key=lambda x: x[1])[0] if divs else None,
    }
    with open(sys.argv[4] + ".txt", "w") as f:
        for fid, d in divs:
            f.write(f"{fid}\t{d:.6f}\n")
    with open(sys.argv[4] + ".json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
