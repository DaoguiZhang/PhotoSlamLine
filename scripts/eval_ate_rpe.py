#!/usr/bin/env python3
"""Scale-corrected ATE + fixed-interval RPE for Mono trajectories.

- Global Sim(3) alignment (Horn on translation) applied ONCE to the estimated
  trajectory; the SAME scale/rotation is used for all RPE segments (no
  per-segment scale fitting).
- ATE translation error after global alignment.
- RPE translation (m) and rotation (deg) for intervals 1 and 10 frames.

Usage: eval_ate_rpe.py <gt_tum> <est_tum> <ate_out> <rpe_out> <ape_out>
TUM format: timestamp tx ty tz qx qy qz qw (Twc, camera-to-world).
"""
import sys
import numpy as np


def load_tum(path):
    ts, trans, rot = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            ts.append(float(p[0]))
            trans.append(np.array([float(p[1]), float(p[2]), float(p[3])]))
            q = np.array([float(p[4]), float(p[5]), float(p[6]), float(p[7])])  # qx qy qz qw
            rot.append(quat_to_mat(q))
    return ts, trans, rot


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def umeyama(src, dst):
    n = src.shape[1]
    sm, dm = src.mean(1, keepdims=True), dst.mean(1, keepdims=True)
    sc, dc = src - sm, dst - dm
    H = dc @ sc.T / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = np.sum(sc ** 2) / n
    s = float(np.trace(np.diag(D) @ S) / var) if var > 0 else 1.0
    t = dm - s * R @ sm
    return R, t, s


def rel_err(T1, T2):
    """Relative pose error between two 4x4 (Twc). Returns (trans_err, rot_err_deg)."""
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    dR = R2.T @ R1
    dt = t2 - t1
    return np.linalg.norm(dt), rot_angle(dR) * 180.0 / np.pi


def rot_angle(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return np.arccos(c)


def main():
    gt_path, est_path, ate_path, rpe_path, ape_path = sys.argv[1:6]
    gt_ts, gt_t, gt_R = load_tum(gt_path)
    es_ts, es_t, es_R = load_tum(est_path)

    pairs = [(t, min(es_ts, key=lambda e: abs(e - t)))
             for t in gt_ts if abs(min(es_ts, key=lambda e: abs(e - t)) - t) <= 0.01]
    if len(pairs) < 2:
        print("ERROR <2 pairs", file=sys.stderr)
        sys.exit(1)

    gt_idx = {t: i for i, t in enumerate(gt_ts)}
    es_idx = {t: i for i, t in enumerate(es_ts)}

    g_t = np.array([gt_t[gt_idx[t]] for t, _ in pairs]).T
    e_t = np.array([es_t[es_idx[e]] for _, e in pairs]).T

    R, t, s = umeyama(e_t, g_t)
    e_t_a = s * R @ e_t + t  # aligned translation (3xN)

    err = np.linalg.norm(e_t_a - g_t, axis=0)
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # build aligned 4x4 poses (Twc) for RPE
    def T4(Rm, tt):
        M = np.eye(4)
        M[:3, :3] = Rm
        M[:3, 3] = tt
        return M

    tflat = t[:, 0]  # flatten (3,1) -> (3,) for the T4 helper
    gT = [T4(gt_R[gt_idx[t]], gt_t[gt_idx[t]]) for t, _ in pairs]
    eT = [T4(R @ es_R[es_idx[e]], s * R @ es_t[es_idx[e]] + tflat) for _, e in pairs]

    lines = [
        "APE w.r.t. translation part (m) (with global Sim(3) scale-corrected Umeyama)",
        f"       max      {err.max():.6f}",
        f"      mean      {err.mean():.6f}",
        f"    median      {np.median(err):.6f}",
        f"       min      {err.min():.6f}",
        f"      rmse      {rmse:.6f}",
        f"       std      {err.std():.6f}",
        f"     scale      {s:.6f}",
        f"matched_pairs   {len(pairs)}",
    ]
    with open(ate_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    with open(ape_path, "w") as f:
        f.write("# gt_ts  est_ts  ape(m)\n")
        for (t, e), a in zip(pairs, err):
            f.write(f"{t:.6f} {e:.6f} {a:.6f}\n")

    rpe_lines = ["RPE (fixed interval; single global Sim(3) scale applied to all segments)"]
    for delta in (1, 10):
        t_errs, r_errs = [], []
        for i in range(len(pairs) - delta):
            te, re_ = rel_err(gT[i], gT[i + delta])
            te2, re2 = rel_err(eT[i], eT[i + delta])
            # relative translation error = |delta_t_est - delta_t_gt|
            t_errs.append(np.linalg.norm(
                (eT[i + delta][:3, 3] - eT[i][:3, 3]) - (gT[i + delta][:3, 3] - gT[i][:3, 3])))
            # relative rotation error = angle( dR_est^T dR_gt )
            dR_gt = gT[i + delta][:3, :3] @ gT[i][:3, :3].T
            dR_es = eT[i + delta][:3, :3] @ eT[i][:3, :3].T
            r_errs.append(rot_angle(dR_es.T @ dR_gt) * 180.0 / np.pi)
        t_errs = np.array(t_errs)
        r_errs = np.array(r_errs)
        rpe_lines.append(f"interval={delta}: trans_mean={t_errs.mean():.6f} "
                         f"trans_rmse={np.sqrt(np.mean(t_errs**2)):.6f} "
                         f"rot_mean_deg={r_errs.mean():.4f} "
                         f"rot_rmse_deg={np.sqrt(np.mean(r_errs**2)):.4f} "
                         f"n={len(t_errs)}")
    with open(rpe_path, "w") as f:
        f.write("\n".join(rpe_lines) + "\n")
    print("\n".join(rpe_lines))


if __name__ == "__main__":
    main()
