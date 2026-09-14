#!/usr/bin/env python3
"""Independent cross-check of eval_ate_sim3.py.

Implements the alignment from scratch with a DIFFERENT code path than
eval_ate_sim3.py:
  * Horn + scale (Sim(3)-on-translation), via explicit cross-covariance formula
    (reference method, same math as ORB-SLAM3/evaluation/evaluate_ate_scale.py)
  * SE(3) no-scale (translation-only Horn), to show the scale correction matters.

Usage: eval_crosscheck.py <gt_tum> <est_tum>
Prints: matched_pairs, scale, rmse_sim3, rmse_se3.
"""
import sys
import numpy as np


def load_tum(path):
    t, q = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            t.append(float(p[0]))
            q.append(np.array([float(x) for x in p[1:8]]))  # tx ty tz qx qy qz qw
    return t, q


def associate(tgt, test, max_diff=0.01):
    """Nearest-timestamp association."""
    pairs = []
    for gt_ts in tgt:
        best = min(test, key=lambda e: abs(e - gt_ts))
        if abs(best - gt_ts) <= max_diff:
            pairs.append((gt_ts, best))
    return pairs


def horn(src, dst, with_scale=True):
    """src, dst: 3xN. Returns R (3x3), t (3x1), s."""
    n = src.shape[1]
    sm = src.mean(1, keepdims=True)
    dm = dst.mean(1, keepdims=True)
    sc = src - sm
    dc = dst - dm
    # cross-covariance: H = sum(dst_i * src_i^T)  (align src -> dst)
    H = dc @ sc.T / n
    U, S_, Vt = np.linalg.svd(H)
    # correct reflection
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = np.sum(sc ** 2) / n
        s = float(np.sum(S_) / var_src) if var_src > 0 else 1.0
    else:
        s = 1.0
    t = dm - s * R @ sm
    return R, t, s


def main():
    gt_ts, gt_q = load_tum(sys.argv[1])
    es_ts, es_q = load_tum(sys.argv[2])
    pairs = associate(gt_ts, es_ts)
    if len(pairs) < 2:
        print("ERROR <2 pairs", file=sys.stderr)
        sys.exit(1)
    # reconstruct in pair order
    gt_map = {ts: q for ts, q in zip(gt_ts, gt_q)}
    es_map = {ts: q for ts, q in zip(es_ts, es_q)}
    g = np.array([[gt_map[t][0], gt_map[t][1], gt_map[t][2]] for t, _ in pairs]).T
    e = np.array([[es_map[tt][0], es_map[tt][1], es_map[tt][2]] for _, tt in pairs]).T

    R, t, s = horn(e, g, with_scale=True)
    ea = s * R @ e + t
    err = np.linalg.norm(ea - g, axis=0)
    rmse_sim3 = float(np.sqrt(np.mean(err ** 2)))

    R2, t2, _ = horn(e, g, with_scale=False)
    ea2 = R2 @ e + t2
    err2 = np.linalg.norm(ea2 - g, axis=0)
    rmse_se3 = float(np.sqrt(np.mean(err2 ** 2)))

    print(f"matched_pairs {len(pairs)}")
    print(f"scale {s:.6f}")
    print(f"rmse_sim3 {rmse_sim3:.6f}")
    print(f"rmse_se3  {rmse_se3:.6f}")


if __name__ == "__main__":
    main()
